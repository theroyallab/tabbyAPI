"""
Manages the storage and utility of model containers.

Containers exist as a common interface for backends.
"""

import pathlib
from loguru import logger
from typing import Optional

from backends.exllamav2.model import ExllamaV2Container
from common.logger import get_loading_progress_bar
from common.utils import load_progress


# Global model container
container: Optional[ExllamaV2Container] = None


async def unload_model():
    """Unloads a model"""
    global container

    container.unload()
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
    load_status = container.load_gen(load_progress)

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
            if module == modules:
                # Switch to model progress if the draft model is loaded
                if model_type == "draft":
                    model_type = "model"
                else:
                    progress.stop()

            yield module, modules, model_type
    finally:
        progress.stop()


async def load_model(model_path: pathlib.Path, **kwargs):
    async for _ in load_model_gen(model_path, **kwargs):
        pass


async def load_loras(lora_dir, **kwargs):
    """Wrapper to load loras."""
    if len(container.active_loras) > 0:
        unload_loras()

    return await container.load_loras(lora_dir, **kwargs)


def unload_loras():
    """Wrapper to unload loras"""
    container.unload(loras_only=True)
