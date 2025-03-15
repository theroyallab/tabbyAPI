"""
Model lifecycle management module.

This module provides classes and utilities for managing the lifecycle of models,
including loading, unloading, and switching between models.
"""

import asyncio
import time
from enum import Enum
from typing import Dict

from loguru import logger


class ModelState(Enum):
    """Enum representing the possible states of a model."""

    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    SERVING = "serving"
    UNLOADING = "unloading"
    FAILED = "failed"


class ModelSwitchError(Exception):
    """Base class for model switching errors."""

    pass


class ModelLoadTimeoutError(ModelSwitchError):
    """Raised when a model load times out."""

    pass


class ModelUnloadError(ModelSwitchError):
    """Raised when a model fails to unload properly."""

    pass


async def wait_for_event(
    event: asyncio.Event, timeout: float = 300, description: str = "event"
) -> bool:
    """
    Wait for an event to be set with a timeout.

    Args:
        event: The event to wait for
        timeout: Maximum time to wait in seconds
        description: Description of the event for logging

    Returns:
        True if the event was set, False if a timeout occurred
    """
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning(f"Timed out waiting for {description} after {timeout} seconds")
        return False


def is_model_available(container) -> bool:
    """
    Check if a model is available for use.

    Args:
        container: The model container to check

    Returns:
        True if the model is available, False otherwise
    """
    return (
        container
        and container.model_loaded
        and not container.model_is_loading
        and not getattr(container, "model_is_unloading", False)
    )


class ModelStateManager:
    """
    Manages the state of models and provides events for synchronization.

    This class centralizes event management for model lifecycle operations,
    reducing redundancy in the codebase.
    """

    def __init__(self):
        # State tracking
        self.current_state = ModelState.IDLE
        self.current_model_name = None
        self.currently_loading_model = None

        # Events for signaling state changes
        self.load_complete_event = asyncio.Event()
        self.ready_for_switch_event = asyncio.Event()
        self.no_active_generations_event = asyncio.Event()

        # Initialize events as set (no pending operations)
        self.load_complete_event.set()
        self.ready_for_switch_event.set()
        self.no_active_generations_event.set()

        # Active generation tracking
        self.active_generations = 0
        self.active_generation_info = {}  # request_id -> info dict

    def set_state(self, state: ModelState):
        """Set the current state of the model."""
        logger.info(
            f"Model state changing: {self.current_state.value} -> {state.value}"
        )
        self.current_state = state

    async def wait_for_no_active_generations(self, timeout: float = 300) -> bool:
        """
        Wait for all active generations to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if there are no active generations, False if a timeout occurred
        """
        if self.active_generations == 0:
            return True

        logger.info(
            f"Waiting for {self.active_generations} active generations to complete"
        )
        return await wait_for_event(
            self.no_active_generations_event,
            timeout=timeout,
            description="active generations to complete",
        )

    async def increment_active_generations(
        self, request_id: str, model_name: str = None, params: Dict = None
    ):
        """
        Increment the active generations counter and update tracking information.

        Args:
            request_id: The ID of the request
            model_name: The name of the model being used
            params: Additional parameters to track
        """
        self.active_generations += 1

        # If this is the first active generation, clear the event
        if self.active_generations == 1:
            self.no_active_generations_event.clear()

        # Track generation details
        self.active_generation_info[request_id] = {
            "start_time": time.time(),
            "model": model_name,
            "params": params or {},
        }

        logger.info(
            f"Starting generation {request_id}, active generations: "
            f"{self.active_generations}"
        )

    async def decrement_active_generations(self, request_id: str):
        """
        Decrement the active generations counter and update tracking information.

        Args:
            request_id: The ID of the request
        """
        self.active_generations -= 1

        # Log generation completion with timing information
        if request_id in self.active_generation_info:
            generation_info = self.active_generation_info[request_id]
            duration = time.time() - generation_info["start_time"]
            logger.info(
                f"Finished generation {request_id} in {duration:.2f}s, "
                f"active generations: {self.active_generations}"
            )

            # Remove from tracking
            del self.active_generation_info[request_id]
        else:
            logger.info(
                f"Finished generation {request_id}, active generations: "
                f"{self.active_generations}"
            )

        # If no more active generations, set the event
        if self.active_generations == 0:
            self.no_active_generations_event.set()
