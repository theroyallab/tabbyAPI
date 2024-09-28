"""
Manages the storage and utility of model containers.

Containers exist as a common interface for backends.
"""

import pathlib
from enum import Enum
from fastapi import HTTPException
from loguru import logger
from typing import Optional

from backends.exllamav2.types import DraftModelInstanceConfig, ModelInstanceConfig
from common.logger import get_loading_progress_bar
from common.networking import handle_request_error
from common.optional_dependencies import dependencies
from common.tabby_config import config

if dependencies.exllamav2:
    from backends.exllamav2.model import ExllamaV2Container

    # Global model container
    container: Optional[ExllamaV2Container] = None
    embeddings_container = None


if dependencies.extras:
    from backends.infinity.model import InfinityContainer

    embeddings_container: Optional[InfinityContainer] = None


class ModelType(Enum):
    MODEL = "model"
    DRAFT = "draft"
    EMBEDDING = "embedding"


def load_progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    """Unloads a model"""
    global container

    await container.unload(skip_wait=skip_wait, shutdown=shutdown)
    container = None


async def load_model_gen(
    model: ModelInstanceConfig,
    draft: Optional[DraftModelInstanceConfig] = None,
    skip_wait: bool = False,
):
    """Generator to load a model"""
    global container

    # Check if the model is already loaded
    if container and container.model:
        loaded_model_name = container.model_dir.name

        if loaded_model_name == model.model_name and container.model_loaded:
            raise ValueError(
                f'Model "{loaded_model_name}" is already loaded! Aborting.'
            )

        logger.info("Unloading existing model.")
        await unload_model()

    # Merge with config defaults
    model = model.model_copy(update=config.model_defaults)
    model.model_validate(model, strict=True)

    # Create a new container
    draft = draft or DraftModelInstanceConfig()

    container = await ExllamaV2Container.create(
        model=model, draft=draft, quiet=False
    )

    model_type = "draft" if container.draft_config else "model"
    load_status = container.load_gen(load_progress, skip_wait)

    progress = get_loading_progress_bar()
    progress.start()

    try:
        async for module, modules in load_status:
            if module == 0:
                loading_task = progress.add_task(
                    f"[cyan]Loading {model_type} modules", total=modules
                )
            else:
                progress.advance(loading_task)

            yield module, modules, model_type

            if module == modules:
                # Switch to model progress if the draft model is loaded
                if model_type == "draft":
                    model_type = "model"
                else:
                    progress.stop()
    finally:
        progress.stop()


async def load_model(
    model: ModelInstanceConfig, draft: Optional[DraftModelInstanceConfig] = None
):
    async for _ in load_model_gen(model=model, draft=draft):
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

        if loaded_model_name == model_path.name and embeddings_container.model_loaded:
            raise ValueError(
                f'Embeddings model "{loaded_model_name}" is already loaded! Aborting.'
            )

        logger.info("Unloading existing embeddings model.")
        await unload_embedding_model()

    embeddings_container = InfinityContainer(model_path)
    await embeddings_container.load(**kwargs)


async def unload_embedding_model():
    global embeddings_container

    await embeddings_container.unload()
    embeddings_container = None


async def check_model_container():
    """FastAPI depends that checks if a model isn't loaded or currently loading."""

    if container is None or not (container.model_is_loading or container.model_loaded):
        error_message = handle_request_error(
            "No models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)


async def check_embeddings_container():
    """
    FastAPI depends that checks if an embeddings model is loaded.

    This is the same as the model container check, but with embeddings instead.
    """

    if embeddings_container is None or not (
        embeddings_container.model_is_loading or embeddings_container.model_loaded
    ):
        error_message = handle_request_error(
            "No embedding models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)
