"""
Model lifecycle manager implementation.

This module provides the ModelLifecycleManager class that handles model loading,
unloading, and switching operations.
"""

import asyncio
import gc
import pathlib
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Tuple, Union

from fastapi import HTTPException, Request
from loguru import logger

from common import model
from common.model_lifecycle import (
    ModelState,
    ModelStateManager,
    ModelSwitchError,
    ModelLoadTimeoutError,
    ModelUnloadError,
    is_model_available,
    wait_for_event,
)
from common.networking import handle_request_error
from common.tabby_config import config


class ModelLifecycleManager:
    """
    Manages the lifecycle of models, including loading, unloading, and switching.

    This class provides a centralized interface for model operations, ensuring
    proper synchronization and error handling.
    """

    def __init__(self, ready_timeout: float = 5.0):
        """
        Initialize the model lifecycle manager.

        Args:
            ready_timeout: Time to wait after loading before allowing model switching
        """
        # State manager for event coordination
        self.state_manager = ModelStateManager()

        # Locks and synchronization primitives
        self.load_lock = asyncio.Lock()
        self.switch_queue = asyncio.Queue()
        self.switch_task = None

        # Configuration
        self.ready_timeout = ready_timeout

        # Logging level for model switch operations
        self.log_level = "INFO"

    def log_model_switch(self, message: str, level: str = None):
        """
        Structured logging for model switching operations.

        Args:
            message: The message to log
            level: The log level to use
        """
        if level is None:
            level = self.log_level

        getattr(logger, level.lower())(f"[MODEL_SWITCH] {message}")

    def _is_model_fully_loaded_and_stable(self, model_name: str) -> bool:
        """
        Check if the specified model is fully loaded and in a stable state.

        Args:
            model_name: The name of the model to check

        Returns:
            True if the model is the current model, fully loaded, and not in a transitional state
        """
        return (
            self.state_manager.current_model_name == model_name
            and model.container
            and model.container.model_loaded
            and not model.container.model_is_loading
            and not getattr(model.container, "model_is_unloading", False)
        )

    def _is_expected_model_loaded(self, model_name: str) -> bool:
        """
        Check if the expected model is loaded.

        Args:
            model_name: The name of the model to check

        Returns:
            True if the container exists, is loaded, and has the expected model
        """
        return (
            model.container
            and model.container.model_loaded
            and model.container.model_dir.name == model_name
        )

    def _is_model_in_transition(self) -> bool:
        """
        Check if the model is in a transitional state (loading or unloading).

        Returns:
            True if the model is loading or unloading
        """
        return model.container.model_is_loading or getattr(
            model.container, "model_is_unloading", False
        )

    def start_processor(self):
        """Start the background task to process model switch requests."""
        # Log the current state of the model switch processor
        self.log_model_switch(
            f"Starting processor, ready_for_switch={self.state_manager.ready_for_switch_event.is_set()}, "
            f"ready_timeout={self.ready_timeout}"
        )

        if self.switch_task is None or self.switch_task.done():
            self.switch_task = asyncio.create_task(self.process_switch_queue())
            self.log_model_switch("Started queue processor")
        else:
            self.log_model_switch("Queue processor already running")

    async def process_switch_queue(self):
        """Background task to process model switch requests."""
        self.log_model_switch("Queue processor started")

        while True:
            item = None  # Will hold the tuple (model_name, request, future) once successfully fetched.
            try:
                self.log_model_switch("Waiting for next model switch request")
                item = (
                    await self.switch_queue.get()
                )  # May raise CancelledError or other exceptions.
                model_name, request, future = item
                self.log_model_switch(
                    f"Processing request for {model_name}, current model: {self.state_manager.current_model_name}, "
                    f"currently loading: {self.state_manager.currently_loading_model}"
                )
                try:
                    # If the model is already loaded and stable, skip switching.
                    if self._is_model_fully_loaded_and_stable(model_name):
                        self.log_model_switch(
                            f"Skipping switch since {model_name} is already the current model and fully loaded"
                        )
                        if not future.done():
                            future.set_result(True)
                        continue

                    # If another model is loading, wait for that load event to complete.
                    if self.state_manager.currently_loading_model is not None:
                        self.log_model_switch(
                            f"Waiting for currently loading model {self.state_manager.currently_loading_model} "
                            f"to complete before switching to {model_name}"
                        )
                        try:
                            await asyncio.wait_for(
                                self.state_manager.load_complete_event.wait(),
                                timeout=300,
                            )
                        except asyncio.TimeoutError:
                            error_msg = f"Timed out waiting for model {self.state_manager.currently_loading_model} to load"
                            self.log_model_switch(error_msg, "ERROR")
                            if not future.done():
                                future.set_exception(ModelLoadTimeoutError(error_msg))
                            continue

                    # Wait for the current model to be ready for switching.
                    if not self.state_manager.ready_for_switch_event.is_set():
                        self.log_model_switch(
                            f"Waiting for current model to be ready for switching before switching to {model_name}"
                        )
                        try:
                            await asyncio.wait_for(
                                self.state_manager.ready_for_switch_event.wait(),
                                timeout=60,
                            )
                        except asyncio.TimeoutError:
                            self.log_model_switch(
                                f"Timed out waiting for model to be ready for switching, proceeding with switch to {model_name}",
                                "WARNING",
                            )

                    # Wait until there are no active generations.
                    await self._wait_for_no_active_generations(model_name)

                    # Mark this model as currently loading and clear events.
                    self.state_manager.currently_loading_model = model_name
                    self.state_manager.load_complete_event.clear()
                    self.state_manager.ready_for_switch_event.clear()

                    try:
                        # Perform the model switch.
                        await self._perform_model_switch(model_name)
                        self.state_manager.current_model_name = model_name
                        self.state_manager.load_complete_event.set()
                        # Schedule setting the model ready after a delay.
                        asyncio.create_task(
                            self._set_model_ready_after_delay(model_name)
                        )
                        if not future.done():
                            future.set_result(True)
                    except Exception as e:
                        if not future.done():
                            future.set_exception(e)
                    finally:
                        if self.state_manager.currently_loading_model == model_name:
                            self.state_manager.currently_loading_model = None
                except Exception as inner_exc:
                    if not future.done():
                        future.set_exception(inner_exc)
                finally:
                    # Only call task_done() if an item was successfully retrieved.
                    if item is not None:
                        self.switch_queue.task_done()
            except Exception as outer_exc:
                self.log_model_switch(
                    f"Unexpected error in queue processor: {str(outer_exc)}", "ERROR"
                )
                logger.error(traceback.format_exc())
                try:
                    if item is not None and not future.done():
                        future.set_exception(outer_exc)
                except Exception:
                    pass
                # Delay briefly before continuing.
                await asyncio.sleep(1)
            await asyncio.sleep(0)  # Yield control to the event loop.

    async def _wait_for_no_active_generations(self, model_name: str):
        """
        Wait for all active generations to complete using event-based approach when possible.

        Args:
            model_name: The name of the model to switch to
        """
        # Log the current state before checking for active generations
        if model.container:
            # Make sure the container has a state manager
            if (
                not hasattr(model.container, "state_manager")
                or model.container.state_manager is None
            ):
                self.log_model_switch(
                    f"Container does not have a state manager, cannot check active generations",
                    "WARNING",
                )
                return

            self.log_model_switch(
                f"Before active generations check: model={model.container.model_dir.name if model.container.model_dir else 'None'}, "
                f"active_generations={model.container.state_manager.active_generations}"
            )

            # Always use the state manager's event for waiting
            self.log_model_switch(
                f"Using state manager for waiting for active generations"
            )
            try:
                await asyncio.wait_for(
                    model.container.state_manager.no_active_generations_event.wait(),
                    timeout=300,
                )
                self.log_model_switch(
                    f"No active generations event triggered, safe to proceed with model switch"
                )
            except asyncio.TimeoutError:
                self.log_model_switch(
                    f"Timed out waiting for active generations to complete", "ERROR"
                )

    async def _perform_model_switch(self, model_name: str):
        """
        Perform the actual model switch operation.

        Args:
            model_name: The name of the model to switch to
        """
        model_path = pathlib.Path(config.model.model_dir) / model_name

        # Flag the container as unloading
        if model.container and hasattr(model.container, "model_is_unloading"):
            model.container.model_is_unloading = True

            # Get active generations safely
            active_gens = 0
            if (
                hasattr(model.container, "state_manager")
                and model.container.state_manager
            ):
                active_gens = model.container.state_manager.active_generations

            self.log_model_switch(
                f"Setting model_is_unloading flag, active_generations={active_gens}"
            )

            # Reset current_model_name when unloading to prevent incorrect model state reporting
            if self.state_manager.current_model_name == model.container.model_dir.name:
                self.log_model_switch(
                    f"Resetting current_model_name during unload of {self.state_manager.current_model_name}"
                )
                self.state_manager.current_model_name = None

        try:
            # Unload the current model
            if model.container:
                self.log_model_switch(
                    f"Unloading existing model to switch to {model_name}"
                )
                await model.unload_model(skip_wait=False)

                # Force garbage collection to free memory
                gc.collect()

                # Add a small delay to ensure resources are freed
                await asyncio.sleep(0.5)

            # Load the new model
            self.log_model_switch(f"Loading model {model_name}")
            try:
                await model.load_model(
                    model_path,
                    draft_model=config.draft_model.model_dump(
                        include={"draft_model_dir"}
                    ),
                )
                self.log_model_switch(f"Successfully loaded model {model_name}")
            except Exception as e:
                self.log_model_switch(
                    f"Error loading model {model_name}: {str(e)}", "ERROR"
                )
                # Make sure events are set to avoid deadlocks
                self.state_manager.load_complete_event.set()
                self.state_manager.ready_for_switch_event.set()
                raise
        except Exception as e:
            self.log_model_switch(
                f"Error during model switch operation: {str(e)}", "ERROR"
            )
            logger.error(traceback.format_exc())
            # Make sure events are set to avoid deadlocks
            self.state_manager.load_complete_event.set()
            self.state_manager.ready_for_switch_event.set()
            raise

    async def queue_model_switch(
        self, model_name: str, request: Request
    ) -> asyncio.Future:
        """
        Queue a model switch and return a future that will be completed when the switch is done.

        Args:
            model_name: The name of the model to switch to
            request: The FastAPI request object

        Returns:
            A future that will be completed when the switch is done
        """
        self.log_model_switch(f"Queuing switch to {model_name}")

        # Create a future to wait for the model switch to complete
        future = asyncio.Future()

        # Add the model switch request to the queue
        await self.switch_queue.put((model_name, request, future))

        return future

    async def load_model(self, model_name: str, request: Request):
        """
        Load a model, either directly or via the queue if there are active generations.

        Args:
            model_name: The name of the model to load
            request: The FastAPI request object
        """
        # Log the requested model and the currently loaded model
        current_model = (
            model.container.model_dir.name
            if model.container and model.container.model_loaded
            else "(none)"
        )
        self.log_model_switch(
            f"Request for model: {model_name}, Currently loaded: {current_model}, "
            f"current_model_name: {self.state_manager.current_model_name}, "
            f"currently_loading: {self.state_manager.currently_loading_model}"
        )
        self.log_model_switch(
            f"Model ready state: load_complete_event={self.state_manager.load_complete_event.is_set()}, "
            f"ready_for_switch={self.state_manager.ready_for_switch_event.is_set()}"
        )

        # Log active generations if available
        if model.container:
            active_gens = 0
            if (
                hasattr(model.container, "state_manager")
                and model.container.state_manager
            ):
                active_gens = model.container.state_manager.active_generations
            self.log_model_switch(
                f"Active generations for current model: {active_gens}"
            )

        # Return if the model container already exists and the model is fully loaded
        if (
            model.container
            and model.container.model_dir.name == model_name
            and model.container.model_loaded
        ):
            self.log_model_switch(
                f"Model {model_name} is already loaded, proceeding with request"
            )
            self.state_manager.current_model_name = (
                model_name  # Update current model name
            )
            return

        # Check if we're currently loading this exact model
        if self.state_manager.currently_loading_model == model_name:
            self.log_model_switch(
                f"Model {model_name} is currently being loaded, waiting for it to complete"
            )
            try:
                # Wait for the current model load to complete
                await asyncio.wait_for(
                    self.state_manager.load_complete_event.wait(), timeout=300
                )  # 5 minute timeout

                # After loading completes, check if the loaded model matches what we requested
                if (
                    model.container
                    and model.container.model_dir.name == model_name
                    and model.container.model_loaded
                ):
                    self.log_model_switch(
                        f"Model {model_name} has finished loading, proceeding with request"
                    )
                    self.state_manager.current_model_name = model_name
                    return
                else:
                    self.log_model_switch(
                        f"Expected model {model_name} to be loaded, but found "
                        f"{model.container.model_dir.name if model.container else 'None'}",
                        "WARNING",
                    )
            except asyncio.TimeoutError:
                self.log_model_switch(
                    f"Timed out waiting for model {model_name} to load", "ERROR"
                )
                error_message = handle_request_error(
                    f"Timed out waiting for model {model_name} to load. Please try again later.",
                    exc_info=False,
                ).error.message
                raise HTTPException(503, error_message)

        # Check if there are active generations or if another model is currently loading
        active_gens = 0
        if (
            model.container
            and hasattr(model.container, "state_manager")
            and model.container.state_manager
        ):
            active_gens = model.container.state_manager.active_generations

        if (model.container and active_gens > 0) or (
            self.state_manager.currently_loading_model is not None
            and self.state_manager.currently_loading_model != model_name
        ):
            # If there are active generations or another model is loading, queue the model switch
            self.log_model_switch(
                f"Queuing switch to {model_name} (active generations: "
                f"{active_gens}, "
                f"currently loading: {self.state_manager.currently_loading_model})"
            )

            # Add the model switch request to the queue and get the future
            future = await self.queue_model_switch(model_name, request)

            # Wait for the model switch to complete before proceeding
            self.log_model_switch(
                f"Waiting for model switch to {model_name} to complete before processing request"
            )
            try:
                # Set a reasonable timeout to avoid hanging indefinitely
                await asyncio.wait_for(future, timeout=300)  # 5 minute timeout
                self.log_model_switch(
                    f"Model switch to {model_name} completed, proceeding with request"
                )
                return
            except asyncio.TimeoutError:
                self.log_model_switch(
                    f"Timed out waiting for model switch to {model_name}", "ERROR"
                )
                error_message = handle_request_error(
                    f"Timed out waiting for model switch to {model_name}. Please try again later.",
                    exc_info=False,
                ).error.message
                raise HTTPException(503, error_message)

        # If there are no active generations, we can switch immediately
        # Use a lock to prevent concurrent model loading
        async with self.load_lock:
            try:
                # Mark this model as currently loading and clear the completion event
                self.state_manager.currently_loading_model = model_name
                self.state_manager.load_complete_event.clear()
                self.state_manager.ready_for_switch_event.clear()  # Mark that the model is not ready for switching

                self.log_model_switch(f"Starting to load model {model_name}")

                # Wait for any ongoing operations to complete at the model level
                if (
                    model.container
                    and hasattr(model.container, "load_lock")
                    and model.container.load_lock.locked()
                ):
                    self.log_model_switch(
                        "Waiting for existing model operations to complete..."
                    )

                # Flag the container as unloading first so streaming operations can detect it
                if model.container and hasattr(model.container, "model_is_unloading"):
                    model.container.model_is_unloading = True
                    self.log_model_switch(
                        f"Setting model_is_unloading flag, "
                        f"active_generations={model.container.state_manager.active_generations}"
                    )

                    # Reset current_model_name when unloading to prevent incorrect model state reporting
                    if (
                        self.state_manager.current_model_name
                        == model.container.model_dir.name
                    ):
                        self.log_model_switch(
                            f"Resetting current_model_name during unload of {self.state_manager.current_model_name}"
                        )
                        self.state_manager.current_model_name = None

                # Give ongoing operations time to detect the unloading flag
                await asyncio.sleep(0.5)

                # Then unload the model - without terminating jobs
                if model.container:
                    self.log_model_switch("Unloading existing model.")
                    await model.unload_model(
                        skip_wait=False
                    )  # Changed to False to not terminate jobs

                    # Force garbage collection to free memory
                    gc.collect()

                    # Add a small delay to ensure resources are freed
                    await asyncio.sleep(0.5)

                # Load the model and also add draft dir with better error handling
                try:
                    await model.load_model(
                        pathlib.Path(config.model.model_dir) / model_name,
                        draft_model=config.draft_model.model_dump(
                            include={"draft_model_dir"}
                        ),
                    )
                    self.state_manager.current_model_name = (
                        model_name  # Update current model name
                    )

                    # Signal that the model load is complete
                    self.log_model_switch(
                        f"Model {model_name} successfully loaded, signaling completion"
                    )
                    self.state_manager.load_complete_event.set()

                    # Store the currently loaded model name to avoid race conditions
                    loaded_model_name = model_name

                    # Schedule setting the model ready for switch event after a delay
                    # This gives the model time to serve at least one request before being unloaded
                    # Use a separate variable to avoid race conditions with model_name changing
                    asyncio.create_task(
                        self._set_model_ready_after_delay(loaded_model_name)
                    )
                except RuntimeError as e:
                    if "Insufficient VRAM" in str(e):
                        error_message = handle_request_error(
                            f"Unable to load model {model_name} due to insufficient VRAM. "
                            "Try unloading other models first or use a smaller model.",
                            exc_info=False,
                        ).error.message
                        raise HTTPException(503, error_message) from e
                    raise
            except Exception as e:
                self.log_model_switch(f"Error during model loading: {str(e)}", "ERROR")
                logger.error(traceback.format_exc())
                raise HTTPException(500, f"Failed to load model: {str(e)}") from e
            finally:
                # Reset the currently loading model if it's still set to this model
                # This handles the case where an exception occurred during loading
                if self.state_manager.currently_loading_model == model_name:
                    self.state_manager.currently_loading_model = None

                    # Only log and set these events if we had an exception
                    # (successful loads already set the load_complete_event)
                    if "e" in locals() and isinstance(e, Exception):
                        # Make sure the events are set so other requests don't wait forever
                        if not self.state_manager.load_complete_event.is_set():
                            self.log_model_switch(
                                f"Model load failed or was interrupted, resetting load complete event"
                            )
                            self.state_manager.load_complete_event.set()
                        if not self.state_manager.ready_for_switch_event.is_set():
                            self.log_model_switch(
                                f"Model load failed or was interrupted, resetting model ready for switch event"
                            )
                            self.state_manager.ready_for_switch_event.set()

    async def _set_model_ready_after_delay(self, model_name: str):
        """
        Set the model ready for switch event after a delay and when there are no active generations.

        Args:
            model_name: The name of the model that was loaded
        """
        self.log_model_switch(
            f"Waiting {self.ready_timeout} seconds before allowing model switch"
        )
        await asyncio.sleep(self.ready_timeout)

        # Check if the model is still the expected one and in a stable state
        if not self._is_expected_model_loaded(model_name):
            self.log_model_switch(
                f"Not setting model ready because model {model_name} is no longer loaded"
            )
            return

        # Check if model is in a transitional state
        if self._is_model_in_transition():
            self.log_model_switch(
                f"Not setting model ready because model is loading={model.container.model_is_loading} or "
                f"unloading={getattr(model.container, 'model_is_unloading', False)}"
            )
            return

        # Check if the container has a state manager
        if (
            not hasattr(model.container, "state_manager")
            or model.container.state_manager is None
        ):
            self.log_model_switch(
                f"Container does not have a state manager, assuming no active generations",
                "WARNING",
            )
            self.state_manager.ready_for_switch_event.set()
            return

        # Check if there are active generations before setting ready
        if model.container.state_manager.active_generations > 0:
            self.log_model_switch(
                f"Not setting model ready yet because there are {model.container.state_manager.active_generations} active generations"
            )

            # Wait for the state manager's no_active_generations_event
            self.log_model_switch(
                f"Waiting for no_active_generations_event before setting model ready"
            )
            try:
                # Wait for the event with a timeout to avoid hanging indefinitely
                await asyncio.wait_for(
                    model.container.state_manager.no_active_generations_event.wait(),
                    timeout=300,
                )

                # After event is triggered, verify the model is still the one we expect
                if (
                    not model.container
                    or not model.container.model_loaded
                    or model.container.model_dir.name != model_name
                ):
                    self.log_model_switch(
                        f"Model changed during wait for active generations to complete"
                    )
                    return

                # Also verify the model is not loading or unloading
                if model.container.model_is_loading or getattr(
                    model.container, "model_is_unloading", False
                ):
                    self.log_model_switch(
                        f"Model is now loading or unloading after waiting for active generations"
                    )
                    return

                # Now safe to set the ready event
                self.log_model_switch(
                    f"No active generations event triggered, model {model_name} is now ready for switching"
                )
                self.state_manager.ready_for_switch_event.set()
                return
            except asyncio.TimeoutError:
                self.log_model_switch(
                    f"Timed out waiting for active generations to complete", "WARNING"
                )
                # Set the ready event anyway after timeout to avoid deadlock
                self.state_manager.ready_for_switch_event.set()
                return

        self.log_model_switch(f"Model {model_name} is now ready for switching")
        self.state_manager.ready_for_switch_event.set()

    async def _check_model_ready_once(self, model_name: str):
        """
        Check once if the model is ready for switching without creating additional tasks.

        Args:
            model_name: The name of the model to check
        """
        # Only check if the container exists and is the expected model
        if not self._is_expected_model_loaded(model_name):
            return

        # Check if model is in the process of loading or unloading
        if self._is_model_in_transition():
            return

        # Check if the container has a state manager
        if (
            not hasattr(model.container, "state_manager")
            or model.container.state_manager is None
        ):
            self.log_model_switch(
                f"Container does not have a state manager, assuming no active generations",
                "WARNING",
            )
            self.state_manager.ready_for_switch_event.set()
            return

        # Check if there are active generations
        if model.container.state_manager.active_generations == 0:
            self.log_model_switch(
                f"Model {model_name} is now ready for switching (delayed check)"
            )
            self.state_manager.ready_for_switch_event.set()

    async def check_model_ready_state(self) -> bool:
        """
        Check if the model is ready to serve requests before allowing a switch.

        Returns:
            True if the model is ready, False otherwise
        """
        # First check if model container exists and is fully loaded
        if not model.container or not model.container.model_loaded:
            self.log_model_switch(
                "Model ready check: model not loaded or container doesn't exist"
            )
            return False

        # Check if model is in a transitional state
        if self._is_model_in_transition():
            self.log_model_switch(
                f"Model ready check: model is loading={model.container.model_is_loading} or "
                f"unloading={getattr(model.container, 'model_is_unloading', False)}"
            )
            return False

        # Check if the container has a state manager
        if (
            not hasattr(model.container, "state_manager")
            or model.container.state_manager is None
        ):
            self.log_model_switch(
                f"Container does not have a state manager, assuming no active generations",
                "WARNING",
            )
            return True

        # Check active generations
        self.log_model_switch(
            f"Model ready check: active_generations={model.container.state_manager.active_generations}"
        )
        return model.container.state_manager.active_generations == 0


# Create a global instance of the model lifecycle manager
model_lifecycle_manager = ModelLifecycleManager()


def start_model_switch_processor():
    """Start the background task to process model switch requests."""
    model_lifecycle_manager.start_processor()


async def check_model_ready_state():
    """
    Check if the model is ready to serve requests before allowing a switch.

    Returns:
        True if the model is ready, False otherwise
    """
    return await model_lifecycle_manager.check_model_ready_state()


async def load_model(model_name: str, request: Request):
    """
    Load a model, either directly or via the queue if there are active generations.

    Args:
        model_name: The name of the model to load
        request: The FastAPI request object
    """
    return await model_lifecycle_manager.load_model(model_name, request)
