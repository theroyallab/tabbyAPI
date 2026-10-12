"""Manages the lifecycle of the global model container and embeddings container."""

import aiofiles
import asyncio
import copy
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
from common import sampling
from common.sampling import BaseSamplerRequest
from common.status_display import status_display
from common.tabby_config import config
from common.optional_dependencies import dependencies
from common.transformers_utils import HFModel
from common.utils import deep_merge_dicts, unwrap

if dependencies.exllamav3:
    from backends.exllamav3.model import ExllamaV3Container

# Global variables for model container
container: Optional["ExllamaV3Container"] = None
embeddings_container = None

# Serializes model loads and swaps. The container's load_lock is per-instance
# and can't order operations that span two containers.
load_lock = asyncio.Lock()


# A broken infinity-emb install (e.g. missing transitive dependencies) only
# disables embeddings instead of preventing server startup
_extras_import_error: Optional[str] = None
if dependencies.extras:
    try:
        from backends.infinity.model import InfinityContainer

        embeddings_container: Optional[InfinityContainer] = None
    except Exception as exc:
        _extras_import_error = str(exc)
        xlogger.warning(f"Embeddings are disabled because infinity-emb failed to import: {exc}")


class ModelType(Enum):
    MODEL = "model"
    DRAFT = "draft"
    VISION = "vision"
    WARMUP = "warmup"


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


MODEL_CONFIG_FILE = "tabby_config.yml"


async def read_model_config_file(model_path: pathlib.Path) -> Optional[dict]:
    """
    The load overrides in a model folder's tabby_config.yml, as
    {"model": {...}, "draft_model": {...}}, or None if the folder has none.
    """

    config_path = model_path / MODEL_CONFIG_FILE
    if not config_path.exists():
        return None

    async with aiofiles.open(config_path, "r", encoding="utf8") as config_file:
        contents = await config_file.read()

    file_config = unwrap(YAML(typ="safe").load(contents), {})
    sections = {}
    for section in ("model", "draft_model"):
        sections[section] = unwrap(
            file_config.get(section) if isinstance(file_config, dict) else None, {}
        )
        if not isinstance(file_config, dict) or not isinstance(sections[section], dict):
            raise ValueError(
                f"{config_path} must hold mappings of options under model: and draft_model:"
            )

    return sections


def use_as_default_settings() -> dict:
    """
    The config.yml values that use_as_default carries over to every load, as
    {model keys..., "draft_model": {...}}. A copy, since merging modifies it.
    """

    return copy.deepcopy({**config.model_defaults, "draft_model": config.draft_model_defaults})


async def apply_load_defaults(model_path: pathlib.Path, request_wins: bool = False, **kwargs):
    """
    Applies model load overrides.

    Priority, lowest to highest: use_as_default keys, then the load kwargs and
    the model folder's tabby_config.yml. For a startup load the file wins,
    since the kwargs hold every config.yml value, set or not. An API request
    (request_wins) only carries the options the client set, and those win
    over the file.
    """

    file_config = await read_model_config_file(model_path)

    # Inline overrides from the model folder's tabby_config.yml
    inline_overrides = {"draft_model": {}}
    if file_config is not None:
        if file_config["model"]:
            inline_overrides.update(file_config["model"])
        else:
            xlogger.warning(
                'Cannot find inline model overrides. Make sure they are nested under a "model:" key'
            )

        if file_config["draft_model"]:
            inline_overrides["draft_model"] = file_config["draft_model"]

        xlogger.info(f"Applying model overrides from {model_path / MODEL_CONFIG_FILE}")

    defaults = use_as_default_settings()
    if request_wins:
        merged_kwargs = deep_merge_dicts(defaults, inline_overrides, kwargs)
    else:
        merged_kwargs = deep_merge_dicts(defaults, kwargs, inline_overrides)

    xlogger.debug(
        "Applying load defaults",
        {
            "kwargs": kwargs,
            "defaults": defaults,
            "inline_overrides": inline_overrides,
            "request_wins": request_wins,
            "merged_kwargs": merged_kwargs,
        },
    )
    return merged_kwargs


async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    """Unloads a model"""
    global container

    await container.unload(skip_wait=skip_wait, shutdown=shutdown)
    container = None
    sampling.clear_model_overrides()


async def load_model_gen(model_path: pathlib.Path, request_wins: bool = False, **kwargs):
    """
    Generator to load a model. request_wins lets the kwargs override the model
    folder's tabby_config.yml; set it when they hold only options a client
    chose (see apply_load_defaults).
    """
    global container

    async with load_lock:
        # Check if the model is already loaded. Compare full resolved paths:
        # quant-style layouts give unrelated models the same directory
        # basename (e.g. <model_a>/exl3/4.00bpw vs <model_b>/exl3/4.00bpw),
        # and a name comparison would silently skip the swap.
        if container and container.model:
            loaded_model_dir = container.model_dir

            if loaded_model_dir.resolve() == model_path.resolve() and container.loaded:
                xlogger.info(f'Model "{loaded_model_dir.name}" is already loaded')

                # Emit a terminal progress event so API clients always
                # see a "finished" status even when no load was needed
                yield 1, 1, ModelType.MODEL.value
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
        kwargs = await apply_load_defaults(model_path, request_wins=request_wins, **kwargs)

        # Fetch the extra HF configuration options
        hf_model = await HFModel.from_directory(model_path)

        # Override the max sequence length based on user
        max_seq_len = kwargs.get("max_seq_len")
        if max_seq_len == -1:
            kwargs["max_seq_len"] = hf_model.hf_config.get_max_position_embeddings()

        # Check model compatibility and dependencies before creating a container
        validate_backend(kwargs.get("backend"), hf_model)

        # Check the model's own sampling section before the expensive load, so
        # a bad preset name or malformed inline entry fails here rather than
        # after the weights are in VRAM
        sampler_preset, sampler_inline = sampling.split_sampling_section(
            kwargs.get("sampling"), "the model's sampling config"
        )
        if sampler_preset and sampling.resolve_preset_path(sampler_preset) is None:
            raise ValueError(
                f'Sampler override preset "{sampler_preset}" (model sampling section) was '
                "not found in the sampler_overrides folder."
            )

        new_container = await ExllamaV3Container.create(model_path.resolve(), hf_model, **kwargs)

        # Add possible types of models that can be loaded
        model_type = [ModelType.MODEL]

        if new_container.use_draft_model:
            model_type.insert(0, ModelType.DRAFT)

        if new_container.use_vision:
            model_type.insert(0, ModelType.VISION)

        # Warmup runs after the weights are loaded and reports its own progress
        if getattr(new_container, "warmup_enabled", False):
            model_type.append(ModelType.WARMUP)

        load_status = new_container.load_gen(load_progress, **kwargs)

        # The live status line must not be showing while the loading bars run
        async with status_display.suspended():
            progress = get_loading_progress_bar()
            progress.start()

            try:
                index = 0
                async for module, modules in load_status:
                    current_model_type = model_type[index].value
                    if module == 0:
                        description = (
                            "[cyan]Warming up"
                            if current_model_type == ModelType.WARMUP.value
                            else f"[cyan]Loading {current_model_type} modules"
                        )
                        loading_task = progress.add_task(description, total=modules)
                    else:
                        progress.advance(loading_task)

                    yield module, modules, current_model_type

                    if module == modules:
                        # Move on to the next component; the last one ends the bars
                        index += 1
                        if index == len(model_type):
                            progress.stop()

                container = new_container
            finally:
                progress.stop()

        # Sampler overrides for this model, replacing the previous model's layer
        if sampler_preset or sampler_inline:
            await sampling.set_model_overrides(sampler_preset, sampler_inline)
        else:
            sampling.clear_model_overrides()


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

    # Break out if infinity is installed but not importable
    if _extras_import_error is not None:
        raise ImportError(
            f"Embeddings are disabled because infinity-emb failed to import: {_extras_import_error}"
        )

    # Check if the model is already loaded (by full path, same as load_model_gen)
    if embeddings_container and embeddings_container.engine:
        loaded_model_dir = embeddings_container.model_dir

        if loaded_model_dir.resolve() == model_path.resolve() and embeddings_container.loaded:
            raise ValueError(
                f'Embeddings model "{loaded_model_dir.name}" is already loaded! Aborting.'
            )

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
