"""
Manages the storage and utility of model containers.

Containers exist as a common interface for backends.
"""

import aiofiles
import pathlib
from enum import Enum
from fastapi import HTTPException
from loguru import logger
from ruamel.yaml import YAML
from typing import Optional

from backends.base_model_container import BaseModelContainer
from common.logger import get_loading_progress_bar
from common.networking import handle_request_error
from common.tabby_config import config
from common.optional_dependencies import dependencies
from common import sampling
from common.utils import unwrap

# Global variables for model container
container: Optional[BaseModelContainer] = None
embeddings_container = None

# FIXME: Possibly use this solely when creating the model
if dependencies.exllamav2:
    from backends.exllamav2.model import ExllamaV2Container


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


async def apply_inline_overrides(model_dir: pathlib.Path, **kwargs):
    """Sets overrides from a model folder's config yaml."""

    override_config_path = model_dir / "tabby_config.yml"

    if not override_config_path.exists():
        return kwargs

    async with aiofiles.open(
        override_config_path, "r", encoding="utf8"
    ) as override_config_file:
        contents = await override_config_file.read()

        # Create a temporary YAML parser
        yaml = YAML(typ="safe")
        override_args = unwrap(yaml.load(contents), {})

        # Merge draft overrides beforehand
        draft_override_args = unwrap(override_args.get("draft_model"), {})
        if draft_override_args:
            kwargs["draft_model"] = {
                **draft_override_args,
                **unwrap(kwargs.get("draft_model"), {}),
            }

        # Merge the override and model kwargs
        merged_kwargs = {**override_args, **kwargs}
        return merged_kwargs


async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    """Unloads a model"""
    global container

    await container.unload(skip_wait=skip_wait, shutdown=shutdown)
    container = None


async def load_model_gen(model_path: pathlib.Path, **kwargs):
    """Generator to load a model"""
    global container
    from common.tabby_config import TabbyConfig  # import TabbyConfig for later use

    # Check if the model is already loaded
    if container and container.model:
        loaded_model_name = container.model_dir.name

        if loaded_model_name == model_path.name and container.loaded:
            raise ValueError(
                f'Model "{loaded_model_name}" is already loaded! Aborting.'
            )

        logger.info("Unloading existing model.")
        await unload_model()

    # Check for model-specific config and apply sampler override if it exists
    config_path = model_path / "tabby_config.yml"
    if config_path.exists():
        config_parser = TabbyConfig()
        config_data = config_parser._from_file(config_path)
        preset_name = config_data.get('model_sampler_preset')
        if preset_name:
            await sampling.overrides_from_file(preset_name)

    # Reset to prepare for a new container
    container = None

    # Model_dir is already provided
    if "model_dir" in kwargs:
        kwargs.pop("model_dir")

    # Merge with config and inline defaults
    # TODO: Figure out a way to do this with Pydantic validation
    # and ModelLoadRequest. Pydantic doesn't have async validators
    kwargs = {**config.model_defaults, **kwargs}
    kwargs = await apply_inline_overrides(model_path, **kwargs)

    # Create a new container
    new_container = await ExllamaV2Container.create(
        model_path.resolve(), False, **kwargs
    )

    # Add possible types of models that can be loaded
    model_type = [ModelType.MODEL]

    if new_container.use_vision:
        model_type.insert(0, ModelType.VISION)

    if new_container.draft_config:
        model_type.insert(0, ModelType.DRAFT)

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
            raise ValueError(
                f'Embeddings model "{loaded_model_name}" is already loaded! Aborting.'
            )

        logger.info("Unloading existing embeddings model.")
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
