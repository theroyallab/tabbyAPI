"""The main tabbyAPI module. Contains the FastAPI server and endpoints."""

import asyncio
import json
import os
import pathlib
import platform
import signal
from loguru import logger
from typing import Optional

from common import gen_logging, sampling, model
from common.args import convert_args_to_dict, init_argparser
from common.auth import auth
from common.logger import setup_logger
from common.networking import is_port_in_use
from common.signals import signal_handler
from common.tabby_config import config
from common.utils import unwrap
from endpoints.server import export_openapi, start_api
from endpoints.utils import do_export_openapi

if not do_export_openapi:
    from backends.exllamav2.utils import check_exllama_version


async def entrypoint_async():
    """Async entry function for program startup"""

    host = unwrap(config.network.get("host"), "127.0.0.1")
    port = unwrap(config.network.get("port"), 5000)

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
    auth.load(unwrap(config.network.get("disable_auth"), False))

    # Override the generation log options if given
    if config.logging:
        gen_logging.update_from_dict(config.logging)

    gen_logging.broadcast_status()

    # Set sampler parameter overrides if provided
    sampling_override_preset = config.sampling.get("override_preset")
    if sampling_override_preset:
        try:
            sampling.overrides_from_file(sampling_override_preset)
        except FileNotFoundError as e:
            logger.warning(str(e))

    # If an initial model name is specified, create a container
    # and load the model
    model_name = config.model.get("model_name")
    if model_name:
        model_path = pathlib.Path(unwrap(config.model.get("model_dir"), "models"))
        model_path = model_path / model_name

        await model.load_model(model_path.resolve(), **config.model)

        # Load loras after loading the model
        if config.lora.get("loras"):
            lora_dir = pathlib.Path(unwrap(config.lora.get("lora_dir"), "loras"))
            await model.container.load_loras(lora_dir.resolve(), **config.lora)

    # If an initial embedding model name is specified, create a separate container
    # and load the model
    embedding_model_name = config.embeddings.get("embedding_model_name")
    if embedding_model_name:
        embedding_model_path = pathlib.Path(
            unwrap(config.embeddings.get("embedding_model_dir"), "models")
        )
        embedding_model_path = embedding_model_path / embedding_model_name

        try:
            await model.load_embedding_model(embedding_model_path, **config.embeddings)
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

    if do_export_openapi:
        openapi_json = export_openapi()

        with open("openapi.json", "w") as f:
            f.write(json.dumps(openapi_json))
            logger.info("Successfully wrote OpenAPI spec to openapi.json")

        return

    # Check exllamav2 version and give a descriptive error if it's too old
    # Skip if launching unsafely
    if unwrap(config.developer.get("unsafe_launch"), False):
        logger.warning(
            "UNSAFE: Skipping ExllamaV2 version check.\n"
            "If you aren't a developer, please keep this off!"
        )
    else:
        check_exllama_version()

    # Enable CUDA malloc backend
    if unwrap(config.developer.get("cuda_malloc_backend"), False):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        logger.warning("EXPERIMENTAL: Enabled the pytorch CUDA malloc backend.")

    # Use Uvloop/Winloop
    if unwrap(config.developer.get("uvloop"), False):
        if platform.system() == "Windows":
            from winloop import install
        else:
            from uvloop import install

        # Set loop event policy
        install()

        logger.warning("EXPERIMENTAL: Running program with Uvloop/Winloop.")

    # Set the process priority
    if unwrap(config.developer.get("realtime_process_priority"), False):
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
