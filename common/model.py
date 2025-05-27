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
from typing import Dict, Optional

from backends.base_model_container import BaseModelContainer
from common.logger import get_loading_progress_bar
from common.networking import handle_request_error
from common.tabby_config import config
from common.optional_dependencies import dependencies
from common.transformers_utils import HFModel
from common.utils import unwrap

# Global variables for model container
container: Optional[BaseModelContainer] = None
embeddings_container = None


_BACKEND_REGISTRY: Dict[str, BaseModelContainer] = {}

if dependencies.exllamav2:
    from backends.exllamav2.model import ExllamaV2Container

    _BACKEND_REGISTRY["exllamav2"] = ExllamaV2Container


if dependencies.exllamav3:
    from backends.exllamav3.model import ExllamaV3Container

    _BACKEND_REGISTRY["exllamav3"] = ExllamaV3Container


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


def detect_backend(hf_model: HFModel) -> str:
    """Determine the appropriate backend based on model files and configuration."""

    quant_method = hf_model.quant_method()

    if quant_method == "exl3":
        return "exllamav3"
    else:
        return "exllamav2"


async def apply_inline_overrides(model_dir: pathlib.Path, **kwargs):
    """Sets overrides from a model folder's config yaml."""

    override_config_path = model_dir / "tabby_config.yml"

    if not override_config_path.exists():
        return kwargs

    # Initialize overrides dict
    overrides = {}

    async with aiofiles.open(
        override_config_path, "r", encoding="utf8"
    ) as override_config_file:
        contents = await override_config_file.read()

        # Create a temporary YAML parser
        yaml = YAML(typ="safe")
        inline_config = unwrap(yaml.load(contents), {})

        # Check for inline model overrides
        model_inline_config = unwrap(inline_config.get("model"), {})
        if model_inline_config:
            overrides = {**model_inline_config}
        else:
            logger.warning(
                "Cannot find inline model overrides. "
                "Make sure they are nested under a \"model:\" key"
            )

        # Merge draft overrides beforehand
        draft_inline_config = unwrap(inline_config.get("draft_model"), {})
        if draft_inline_config:
            overrides["draft_model"] = {**draft_inline_config}

        # Merge the override and model kwargs
        merged_kwargs = {**overrides, **kwargs}
        return merged_kwargs


async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    """Unloads a model"""
    global container

    await container.unload(skip_wait=skip_wait, shutdown=shutdown)
    container = None


async def load_model_gen(model_path: pathlib.Path, **kwargs):
    """Generator to load a model"""
    global container

    # Check if the model is already loaded
    if container and container.model:
        loaded_model_name = container.model_dir.name

        if loaded_model_name == model_path.name and container.loaded:
            raise ValueError(
                f'Model "{loaded_model_name}" is already loaded! Aborting.'
            )

        logger.info("Unloading existing model.")
        await unload_model()

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

    # Fetch the extra HF configuration options
    hf_model = await HFModel.from_directory(model_path)

    # Create a new container and check if the right dependencies are installed
    backend = unwrap(kwargs.get("backend"), detect_backend(hf_model))
    container_class = _BACKEND_REGISTRY.get(backend)

    if not container_class:
        available_backends = list(_BACKEND_REGISTRY.keys())
        if backend in available_backends:
            raise ValueError(
                f"Backend '{backend}' selected, but required dependencies "
                "are not installed."
            )
        else:
            raise ValueError(
                f"Invalid backend '{backend}'. Available backends: {available_backends}"
            )

    logger.info(f"Using backend {backend}")
    new_container: BaseModelContainer = await container_class.create(
        model_path.resolve(), hf_model, **kwargs
    )

    # Add possible types of models that can be loaded
    model_type = [ModelType.MODEL]

    if new_container.use_vision:
        model_type.insert(0, ModelType.VISION)

    if new_container.use_draft_model:
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
