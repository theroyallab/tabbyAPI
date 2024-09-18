"""The main tabbyAPI module. Contains the FastAPI server and endpoints."""

import asyncio
import os
import pathlib
import platform
import signal
from loguru import logger
from typing import Optional

from common import gen_logging, sampling, model
from common.args import convert_args_to_dict, init_argparser
from common.auth import load_auth_keys
from common.actions import branch_to_actions
from common.logger import setup_logger
from common.networking import is_port_in_use
from common.signals import signal_handler
from common.tabby_config import config
from endpoints.server import start_api
from endpoints.utils import do_export_openapi

if not do_export_openapi:
    from backends.exllamav2.utils import check_exllama_version


async def entrypoint_async():
    """Async entry function for program startup"""

    host = config.network.host
    port = config.network.port

    # Check if the port is available and attempt to bind a fallback
    if is_port_in_use(port):
        fallback_port = port + 1

        if is_port_in_use(fallback_port):
            logger.error(
                f"Ports {port} and {fallback_port} are in use by different services.\n"
                "Please free up those ports or specify a different one.\n"
                "Exiting."
            )

            return
        else:
            logger.warning(
                f"Port {port} is currently in use. Switching to {fallback_port}."
            )

            port = fallback_port

    # Initialize auth keys
    await load_auth_keys(config.network.disable_auth)

    gen_logging.broadcast_status()

    # Set sampler parameter overrides if provided
    sampling_override_preset = config.sampling.override_preset
    if sampling_override_preset:
        try:
            await sampling.overrides_from_file(sampling_override_preset)
        except FileNotFoundError as e:
            logger.warning(str(e))

    # If an initial model name is specified, create a container
    # and load the model
    model_name = config.model.model_name
    if model_name:
        model_path = pathlib.Path(config.model.model_dir)
        model_path = model_path / model_name

        # TODO: remove model_dump()
        await model.load_model(
            model_path.resolve(),
            **config.model.model_dump(),
            draft=config.draft_model.model_dump(),
        )

        # Load loras after loading the model
        if config.lora.loras:
            lora_dir = pathlib.Path(config.lora.lora_dir)
            # TODO: remove model_dump()
            await model.container.load_loras(
                lora_dir.resolve(), **config.lora.model_dump()
            )

    # If an initial embedding model name is specified, create a separate container
    # and load the model
    embedding_model_name = config.embeddings.embedding_model_name
    if embedding_model_name:
        embedding_model_path = pathlib.Path(config.embeddings.embedding_model_dir)
        embedding_model_path = embedding_model_path / embedding_model_name

        try:
            # TODO: remove model_dump()
            await model.load_embedding_model(
                embedding_model_path, **config.embeddings.model_dump()
            )
        except ImportError as ex:
            logger.error(ex.msg)

    await start_api(host, port)


def entrypoint(arguments: Optional[dict] = None):
    setup_logger()

    # Set up signal aborting
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse and override config from args
    if arguments is None:
        parser = init_argparser()
        arguments = convert_args_to_dict(parser.parse_args(), parser)

    # load config
    config.load(arguments)

    # branch to default paths if required
    if branch_to_actions():
        return

    # Check exllamav2 version and give a descriptive error if it's too old
    # Skip if launching unsafely
    if config.developer.unsafe_launch:
        logger.warning(
            "UNSAFE: Skipping ExllamaV2 version check.\n"
            "If you aren't a developer, please keep this off!"
        )
    else:
        check_exllama_version()

    # Enable CUDA malloc backend
    if config.developer.cuda_malloc_backend:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        logger.warning("EXPERIMENTAL: Enabled the pytorch CUDA malloc backend.")

    # Use Uvloop/Winloop
    if config.developer.uvloop:
        if platform.system() == "Windows":
            from winloop import install
        else:
            from uvloop import install

        # Set loop event policy
        install()

        logger.warning("EXPERIMENTAL: Running program with Uvloop/Winloop.")

    # Set the process priority
    if config.developer.realtime_process_priority:
        import psutil

        current_process = psutil.Process(os.getpid())
        if platform.system() == "Windows":
            current_process.nice(psutil.REALTIME_PRIORITY_CLASS)
        else:
            current_process.nice(psutil.IOPRIO_CLASS_RT)

        logger.warning(
            "EXPERIMENTAL: Process priority set to Realtime. \n"
            "If you're not running on administrator/sudo, the priority is set to high."
        )

    # Enter into the async event loop
    asyncio.run(entrypoint_async())


if __name__ == "__main__":
    entrypoint()
