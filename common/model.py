"""
Manages the storage and utility of model containers.

Containers exist as a common interface for backends.
"""

import asyncio
import gc
import pathlib
from enum import Enum
from fastapi import HTTPException
from loguru import logger
from typing import Optional

from common.logger import get_loading_progress_bar
from common.networking import handle_request_error
from common.tabby_config import config
from common.optional_dependencies import dependencies

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
    VISION = "vision"


def load_progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    """Unloads a model"""
    global container

    if container is None:
        return
    
    # Log the current state before unloading
    model_name = container.model_dir.name if hasattr(container, 'model_dir') and container.model_dir else "Unknown"
    logger.info(f"Attempting to unload model {model_name}, skip_wait={skip_wait}, shutdown={shutdown}")
    
    # Check for active generations before unloading
    if not shutdown and not skip_wait and hasattr(container, 'active_generations') and container.active_generations > 0:
        logger.warning(
            f"Cannot unload model with {container.active_generations} active generations. "
            "Wait for generations to complete or use skip_wait=True to force unload."
        )
        return
        
    logger.info(f"Proceeding with unload of model {model_name}, active_generations={getattr(container, 'active_generations', 0)}")
    
    try:
        if shutdown:
            # Only terminate jobs if we're shutting down the entire application
            await container.unload(skip_wait=True, shutdown=shutdown)
        else:
            # Otherwise respect the skip_wait parameter but don't force it to True
            await container.unload(skip_wait=skip_wait, shutdown=shutdown)
    finally:
        # Always set container to None to ensure references are cleared
        container = None
        
        # Force garbage collection to free memory
        gc.collect()


async def load_model_gen(model_path: pathlib.Path, **kwargs):
    """Generator to load a model"""
    global container

    logger.info(f"Request to load model: {model_path.name}")
    
    # Check if the model is already loaded
    if container and container.model:
        loaded_model_name = container.model_dir.name
        logger.info(f"Current loaded model: {loaded_model_name}, model_loaded={container.model_loaded}, active_generations={getattr(container, 'active_generations', 0)}")

        if loaded_model_name == model_path.name and container.model_loaded:
            logger.info(f"Model {loaded_model_name} is already loaded, aborting load request")
            raise ValueError(
                f'Model "{loaded_model_name}" is already loaded! Aborting.'
            )
            
        # Check for active generations before unloading the current model
        if hasattr(container, 'state_manager') and container.state_manager and container.state_manager.active_generations > 0:
            logger.warning(f"Cannot load a new model while {container.state_manager.active_generations} generations are active")
            raise ValueError(
                f"Cannot load a new model while {container.state_manager.active_generations} generations are active. "
                "Wait for generations to complete before loading a new model."
            )

        logger.info(f"Unloading existing model {loaded_model_name} before loading {model_path.name}")
        await unload_model()

    # Merge with config defaults
    kwargs = {**config.model_defaults, **kwargs}

    # Create a new container
    container = await ExllamaV2Container.create(model_path.resolve(), False, **kwargs)

    # Add possible types of models that can be loaded
    model_type = [ModelType.MODEL]

    if container.use_vision:
        model_type.insert(0, ModelType.VISION)

    if container.draft_config:
        model_type.insert(0, ModelType.DRAFT)

    # Get the load status generator
    load_status = container.load_gen(load_progress, **kwargs)

    # Create the progress bar and associated lock for non-blocking operations
    progress = None
    progress_lock = asyncio.Lock()
    
    try:
        # Attempt to acquire the progress lock with a timeout for initialization
        try:
            await asyncio.wait_for(progress_lock.acquire(), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout acquiring progress lock during progress bar initialization; skipping progress visualization.")
            async for module, modules in load_status:
                yield module, modules, model_type[0].value
            return
        try:
            progress = get_loading_progress_bar()
            progress.start()
        except Exception as e:
            logger.warning(f"Could not start progress visualization: {str(e)}")
            async for module, modules in load_status:
                yield module, modules, model_type[0].value
            return
        finally:
            progress_lock.release()
            
        index = 0
        async for module, modules in load_status:
            current_model_type = model_type[index].value
            
            # Update progress bar in a non-blocking manner using the lock with timeout
            try:
                await asyncio.wait_for(progress_lock.acquire(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout acquiring progress lock during progress update; skipping progress update.")
            else:
                try:
                    if module == 0:
                        loading_task = progress.add_task(
                            f"[cyan]Loading {current_model_type} modules", total=modules
                        )
                    else:
                        progress.advance(loading_task)
                except Exception as e:
                    logger.warning(f"Progress visualization error: {str(e)}")
                finally:
                    progress_lock.release()
                    
            yield module, modules, current_model_type

            if module == modules:
                # Switch to model progress if the draft model is loaded
                if index == len(model_type):
                    try:
                        await asyncio.wait_for(progress_lock.acquire(), timeout=1.0)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout acquiring progress lock while stopping progress bar; proceeding without stopping.")
                    else:
                        try:
                            progress.stop()
                            progress = None  # Clear reference after stopping
                        except Exception as e:
                            logger.warning(f"Error stopping progress bar: {str(e)}")
                        finally:
                            progress_lock.release()
                else:
                    index += 1
    finally:
        # Always ensure progress is stopped, without blocking the main event loop
        if progress:
            try:
                await asyncio.wait_for(progress_lock.acquire(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout acquiring progress lock during final progress bar stop; proceeding without stopping.")
            else:
                try:
                    progress.stop()
                    progress = None
                except Exception as e:
                    logger.warning(f"Error stopping progress bar: {str(e)}")
                finally:
                    progress_lock.release()


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
