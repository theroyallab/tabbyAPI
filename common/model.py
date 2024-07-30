"""
Manages the storage and utility of model containers.

Containers exist as a common interface for backends.
"""

import pathlib
from fastapi import HTTPException
from loguru import logger
from typing import Optional

from common import config
from common.logger import get_loading_progress_bar
from common.networking import handle_request_error
from common.utils import unwrap
from endpoints.utils import do_export_openapi

if not do_export_openapi:
    from backends.exllamav2.model import ExllamaV2Container

    # Global model container
    container: Optional[ExllamaV2Container] = None


def load_progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


async def unload_model(skip_wait: bool = False):
    """Unloads a model"""
    global container

    await container.unload(skip_wait=skip_wait)
    container = None


async def load_model_gen(model_path: pathlib.Path, **kwargs):
    """Generator to load a model"""
    global container

    # Check if the model is already loaded
    if container and container.model:
        loaded_model_name = container.get_model_path().name

        if loaded_model_name == model_path.name and container.model_loaded:
            raise ValueError(
                f'Model "{loaded_model_name}" is already loaded! Aborting.'
            )

    # Unload the existing model
    if container and container.model:
        logger.info("Unloading existing model.")
        await unload_model()

    container = ExllamaV2Container(model_path.resolve(), False, **kwargs)

    model_type = "draft" if container.draft_config else "model"
    load_status = container.load_gen(load_progress, **kwargs)

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


def get_config_default(key, fallback=None, is_draft=False):
    """Fetches a default value from model config if allowed by the user."""

    model_config = config.model_config()
    default_keys = unwrap(model_config.get("use_as_default"), [])
    if key in default_keys:
        # Is this a draft model load parameter?
        if is_draft:
            draft_config = config.draft_model_config()
            return unwrap(draft_config.get(key), fallback)
        else:
            return unwrap(model_config.get(key), fallback)
    else:
        return fallback


async def check_model_container():
    """FastAPI depends that checks if a model isn't loaded or currently loading."""

    if container is None or not (container.model_is_loading or container.model_loaded):
        error_message = handle_request_error(
            "No models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)
