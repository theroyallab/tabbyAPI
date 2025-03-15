"""
Model utility functions.

This module provides utility functions for common model operations,
reducing redundancy in the codebase.
"""

import asyncio
from typing import Dict, Any, Union

from fastapi import HTTPException, Request
from loguru import logger

from common import model
from common.networking import handle_request_error
from common.tabby_config import config


async def validate_model_load_permissions(model_name: str, request: Request) -> bool:
    """
    Validates permissions for model loading and checks if the model is a dummy model.

    Args:
        model_name: The name of the model to load
        request: The FastAPI request object

    Returns:
        True if loading should proceed, False if it should be skipped

    Raises:
        HTTPException: If the user doesn't have permission to load the model
    """
    from common.auth import get_key_permission

    # Return if inline loading is disabled
    if not config.model.inline_model_loading:
        if get_key_permission(request) == "admin":
            logger.warning(
                f"Unable to switch model to {model_name} because "
                '"inline_model_loading" is not True in config.yml.'
            )
        return False

    is_dummy_model = (
        config.model.use_dummy_models and model_name in config.model.dummy_model_names
    )

    # Error if an invalid key is passed
    # If a dummy model is provided, don't error
    if get_key_permission(request) != "admin":
        if not is_dummy_model:
            error_message = handle_request_error(
                f"Unable to switch model to {model_name} because "
                + "an admin key isn't provided",
                exc_info=False,
            ).error.message
            raise HTTPException(401, error_message)
        else:
            return False

    # Skip if the model is a dummy
    if is_dummy_model:
        logger.warning(f"Dummy model {model_name} provided. Skipping inline load.")
        return False

    return True


async def handle_model_unloading_error(
    request_id: str, operation: str
) -> Dict[str, Any]:
    """
    Handle the case where a model is being unloaded during an operation.

    Args:
        request_id: The ID of the request
        operation: The operation being performed (e.g., "generation", "completion")

    Returns:
        A dictionary with error information
    """
    logger.warning(
        f"Model was unloaded during {operation} for request " f"{request_id}"
    )
    return {
        "error": f"Model was unloaded during {operation}",
        "finish_reason": "model_unloaded",
    }


async def check_model_before_operation(
    request_id: str, operation: str
) -> Union[Dict[str, Any], None]:
    """
    Check if the model is available before performing an operation.

    Args:
        request_id: The ID of the request
        operation: The operation to perform (e.g., "generation", "completion")

    Returns:
        None if the model is available, otherwise a dictionary
        with error information
    """
    if model.container is None or getattr(model.container, "model_is_unloading", False):
        logger.warning(
            f"Model is being unloaded, cannot start {operation} for request "
            f"{request_id}"
        )
        return {"error": "Model unavailable", "finish_reason": "model_unloaded"}
    return None


async def track_generation_start(request_id: str, **kwargs):
    """
    Track the start of a generation.

    Args:
        request_id: The ID of the request
        **kwargs: Additional parameters to track
    """
    if model.container and hasattr(model.container, "active_generations_lock"):
        async with model.container.active_generations_lock:
            if hasattr(model.container, "increment_active_generations"):
                await model.container.increment_active_generations(request_id, **kwargs)
            else:
                # Fallback for older container implementations
                model.container.active_generations += 1
                if hasattr(model.container, "no_active_generations_event"):
                    model.container.no_active_generations_event.clear()

                # Track generation details if possible
                if hasattr(model.container, "active_generation_info"):
                    model.container.active_generation_info[request_id] = {
                        "start_time": asyncio.get_event_loop().time(),
                        "model": model.container.model_dir.name
                        if model.container.model_dir
                        else None,
                        "params": {
                            k: v for k, v in kwargs.items() if k != "embeddings"
                        },  # Exclude large objects
                    }

                logger.info(
                    f"Starting generation {request_id}, active generations: "
                    f"{model.container.active_generations}"
                )


async def track_generation_end(request_id: str):
    """
    Track the end of a generation.

    Args:
        request_id: The ID of the request
    """
    if model.container and hasattr(model.container, "active_generations_lock"):
        async with model.container.active_generations_lock:
            if hasattr(model.container, "decrement_active_generations"):
                await model.container.decrement_active_generations(request_id)
            else:
                # Fallback for older container implementations
                model.container.active_generations -= 1

                # Log generation completion with timing information
                if (
                    hasattr(model.container, "active_generation_info")
                    and request_id in model.container.active_generation_info
                ):
                    generation_info = model.container.active_generation_info[request_id]
                    duration = asyncio.get_event_loop().time() - generation_info.get(
                        "start_time", 0
                    )
                    logger.info(
                        f"Finished generation {request_id} in {duration:.2f}s, "
                        f"active generations: {model.container.active_generations}"
                    )

                    # Remove from tracking
                    del model.container.active_generation_info[request_id]
                else:
                    logger.info(
                        f"Finished generation {request_id}, active generations: "
                        f"{model.container.active_generations}"
                    )

                # If no more active generations, set the event
                if model.container.active_generations == 0 and hasattr(
                    model.container, "no_active_generations_event"
                ):
                    model.container.no_active_generations_event.set()
