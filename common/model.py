"""
Manages the storage and utility of model containers.

Containers exist as a common interface for backends.
"""

import aiofiles
import asyncio
import pathlib
from enum import Enum
from fastapi import HTTPException
from common.logger import xlogger
from ruamel.yaml import YAML
from typing import Optional

from common.errors import ContextLengthExceededError, ContextLengthHTTPException
from common.logger import get_loading_progress_bar
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import handle_request_error
from common.sampling import BaseSamplerRequest
from common.tabby_config import config
from common.optional_dependencies import dependencies
from common.transformers_utils import HFModel
from common.utils import deep_merge_dict, unwrap

if dependencies.exllamav3:
    from backends.exllamav3.model import ExllamaV3Container

# Global variables for model container
container: Optional["ExllamaV3Container"] = None
embeddings_container = None

# Serializes model loads and swaps. The container's load_lock is per-instance
# and can't order operations that span two containers.
load_lock = asyncio.Lock()


if dependencies.extras:
    from backends.infinity.model import InfinityContainer

    embeddings_container: Optional[InfinityContainer] = None


class ModelType(Enum):
    MODEL = "model"
    DRAFT = "draft"
    EMBEDDING = "embedding"
    VISION = "vision"


def load_progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


def validate_backend(backend: Optional[str], hf_model: HFModel):
    """Check that the requested model can be loaded with the exllamav3 backend."""

    if backend == "exllamav2":
        raise ValueError("The exllamav2 backend is no longer supported. Please use exllamav3.")
    elif backend and backend != "exllamav3":
        raise ValueError(f"Invalid backend '{backend}'. Available backends: ['exllamav3']")

    quant_method = hf_model.quant_method()
    if quant_method in {"exl2", "gptq"}:
        raise ValueError(
            f"Models quantized with '{quant_method}' require the exllamav2 backend, "
            "which is no longer supported. Please use an exl3 or unquantized model."
        )

    if not dependencies.exllamav3:
        raise ValueError(
            "The exllamav3 backend is selected, but required dependencies are not installed."
        )


async def apply_load_defaults(model_path: pathlib.Path, **kwargs):
    """
    Applies model load overrides.
    Sources are from inline config and use_as_default.
    Currently agnostic due to different schemas for API and config.
    """

    override_config_path = model_path / "tabby_config.yml"

    # Initialize overrides dict
    overrides = {"draft_model": {}}
    model_inline_config = None

    if override_config_path.exists():
        async with aiofiles.open(
            override_config_path, "r", encoding="utf8"
        ) as override_config_file:
            contents = await override_config_file.read()

            # Create a temporary YAML parser
            yaml = YAML(typ="safe")
            inline_config = unwrap(yaml.load(contents), {})

            # Check for inline model overrides and merge config defaults
            model_inline_config = unwrap(inline_config.get("model"), {})
            if model_inline_config:
                overrides = {**overrides, **model_inline_config}
            else:
                xlogger.warning(
                    "Cannot find inline model overrides. "
                    'Make sure they are nested under a "model:" key'
                )

            # Merge draft overrides beforehand and merge config defaults
            draft_inline_config = unwrap(inline_config.get("draft_model"), {})
            if draft_inline_config:
                overrides["draft_model"] = {
                    **overrides.get("draft_model"),
                    **draft_inline_config,
                }

    # Add use_as_default
    overrides = {**overrides, **config.model_defaults}
    overrides["draft_model"] = {
        **overrides.get("draft_model"),
        **config.draft_model_defaults,
    }

    # Merge the override and model kwargs
    # No need to preserve the original overrides dict
    merged_kwargs = deep_merge_dict(overrides, kwargs)

    xlogger.debug(
        "Applying load defaults",
        {
            "kwargs": kwargs,
            "model_inline_config": model_inline_config,
            "overrides": overrides,
            "merged_kwargs": merged_kwargs,
        },
    )
    return merged_kwargs


async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    """Unloads a model"""
    global container

    await container.unload(skip_wait=skip_wait, shutdown=shutdown)
    container = None


async def load_model_gen(model_path: pathlib.Path, **kwargs):
    """Generator to load a model"""
    global container

    async with load_lock:
        # Check if the model is already loaded
        if container and container.model:
            loaded_model_name = container.model_dir.name

            if loaded_model_name == model_path.name and container.loaded:
                xlogger.info(f'Model "{loaded_model_name}" is already loaded')
                return

            if container.loaded:
                xlogger.info("Unloading existing model.")
                await unload_model()

        # Reset to prepare for a new container
        container = None

        # Model_dir is already provided
        if "model_dir" in kwargs:
            kwargs.pop("model_dir")

        # Merge with config and inline defaults
        # TODO: Figure out a way to do this with Pydantic validation
        # and ModelLoadRequest. Pydantic doesn't have async validators
        kwargs = await apply_load_defaults(model_path, **kwargs)

        # Fetch the extra HF configuration options
        hf_model = await HFModel.from_directory(model_path)

        # Override the max sequence length based on user
        max_seq_len = kwargs.get("max_seq_len")
        if max_seq_len == -1:
            kwargs["max_seq_len"] = hf_model.hf_config.get_max_position_embeddings()

        # Check model compatibility and dependencies before creating a container
        validate_backend(kwargs.get("backend"), hf_model)

        new_container = await ExllamaV3Container.create(model_path.resolve(), hf_model, **kwargs)

        # Add possible types of models that can be loaded
        model_type = [ModelType.MODEL]

        if new_container.use_draft_model:
            model_type.insert(0, ModelType.DRAFT)

        if new_container.use_vision:
            model_type.insert(0, ModelType.VISION)

        load_status = new_container.load_gen(load_progress, **kwargs)

        progress = get_loading_progress_bar()
        progress.start()

        try:
            index = 0
            async for module, modules in load_status:
                current_model_type = model_type[index].value
                if module == 0:
                    loading_task = progress.add_task(
                        f"[cyan]Loading {current_model_type} modules", total=modules
                    )
                else:
                    progress.advance(loading_task)

                yield module, modules, current_model_type

                if module == modules:
                    # Switch to model progress if the draft model is loaded
                    if index == len(model_type):
                        progress.stop()
                    else:
                        index += 1

            container = new_container
        finally:
            progress.stop()


async def load_model(model_path: pathlib.Path, **kwargs):
    async for _ in load_model_gen(model_path, **kwargs):
        pass


async def load_loras(lora_dir, **kwargs):
    """Wrapper to load loras."""
    if len(container.get_loras()) > 0:
        await unload_loras()

    return await container.load_loras(lora_dir, **kwargs)


async def unload_loras():
    """Wrapper to unload loras"""
    await container.unload(loras_only=True)


async def load_embedding_model(model_path: pathlib.Path, **kwargs):
    global embeddings_container

    # Break out if infinity isn't installed
    if not dependencies.extras:
        raise ImportError(
            "Skipping embeddings because infinity-emb is not installed.\n"
            "Please run the following command in your environment "
            "to install extra packages:\n"
            "pip install -U .[extras]"
        )

    # Check if the model is already loaded
    if embeddings_container and embeddings_container.engine:
        loaded_model_name = embeddings_container.model_dir.name

        if loaded_model_name == model_path.name and embeddings_container.loaded:
            raise ValueError(f'Embeddings model "{loaded_model_name}" is already loaded! Aborting.')

        xlogger.info("Unloading existing embeddings model.")
        await unload_embedding_model()

    # Reset to prepare for a new container
    embeddings_container = None

    new_embeddings_container = InfinityContainer(model_path)
    await new_embeddings_container.load(**kwargs)

    embeddings_container = new_embeddings_container


async def unload_embedding_model():
    global embeddings_container

    await embeddings_container.unload()
    embeddings_container = None


async def check_model_container():
    """FastAPI depends that checks if a model isn't loaded or currently loading."""

    if container is None:
        error_message = handle_request_error(
            "No models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(503, error_message)


async def check_embeddings_container():
    """
    FastAPI depends that checks if an embeddings model is loaded.

    This is the same as the model container check, but with embeddings instead.
    """

    if embeddings_container is None:
        error_message = handle_request_error(
            "No embedding models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(503, error_message)


def check_context_length(
    prompts: str | list[str],
    params: BaseSamplerRequest,
    mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
):
    """Reject oversized prompts before a streaming response commits HTTP 200."""

    if isinstance(prompts, str):
        prompts = [prompts]

    try:
        for prompt in prompts:
            container.validate_context_length(prompt, params, mm_embeddings)
    except ContextLengthExceededError as exc:
        error_message = handle_request_error(str(exc), exc_info=False).error.message
        raise ContextLengthHTTPException(error_message) from exc
