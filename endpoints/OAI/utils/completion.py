"""
Completion utilities for OAI server.

Also serves as a common module for completions and chat completions.
"""

import traceback
import asyncio
import pathlib
from asyncio import CancelledError
from fastapi import HTTPException, Request
from typing import List, Union

from loguru import logger

from common import model
from common.model_lifecycle_manager import load_model
from common.model_utils import (
    validate_model_load_permissions,
    check_model_before_operation,
    handle_model_unloading_error,
    track_generation_start,
    track_generation_end,
)
from common.networking import (
    get_generator_error,
    handle_request_disconnect,
    handle_request_error,
    request_disconnect_loop,
)
from common.tabby_config import config
from common.utils import unwrap
from endpoints.OAI.types.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionRespChoice,
    CompletionLogProbs,
)
from endpoints.OAI.types.common import UsageStats


# Re-export the start_model_switch_processor function for backwards compatibility


def _create_response(
    request_id: str, generations: Union[dict, List[dict]], model_name: str = ""
):
    """Create a completion response from the provided choices."""

    # Convert the single choice object into a list
    if not isinstance(generations, list):
        generations = [generations]

    choices: List[CompletionRespChoice] = []
    for index, generation in enumerate(generations):
        logprob_response = None

        token_probs = unwrap(generation.get("token_probs"), {})
        if token_probs:
            logprobs = unwrap(generation.get("logprobs"), [])
            offset = unwrap(generation.get("offset"), [])

            logprob_response = CompletionLogProbs(
                text_offset=offset if isinstance(offset, list) else [offset],
                token_logprobs=token_probs.values(),
                tokens=token_probs.keys(),
                top_logprobs=logprobs if isinstance(logprobs, list) else [logprobs],
            )

        # The index can be located in the generation itself
        choice = CompletionRespChoice(
            index=unwrap(generation.get("index"), index),
            finish_reason=generation.get("finish_reason"),
            text=unwrap(generation.get("text"), ""),
            logprobs=logprob_response,
        )

        choices.append(choice)

    prompt_tokens = unwrap(generations[-1].get("prompt_tokens"), 0)
    completion_tokens = unwrap(generations[-1].get("generated_tokens"), 0)

    response = CompletionResponse(
        id=f"cmpl-{request_id}",
        choices=choices,
        model=model_name,
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


async def _stream_collector(
    task_idx: int,
    gen_queue: asyncio.Queue,
    prompt: str,
    request_id: str,
    abort_event: asyncio.Event,
    **kwargs,
):
    """Collects a stream and places results in a common queue"""

    try:
        # Check if model is being unloaded before starting generation
        error_dict = await check_model_before_operation(request_id, "generation")
        if error_dict:
            await gen_queue.put(error_dict)
            return

        # Track the start of the generation
        await track_generation_start(request_id, model=kwargs.get("model"))

        try:
            new_generation = model.container.generate_gen(
                prompt, request_id, abort_event, **kwargs
            )
            async for generation in new_generation:
                # Check if model was unloaded during generation
                if model.container is None or model.container.model_is_unloading:
                    error_dict = await handle_model_unloading_error(
                        request_id, "generation"
                    )
                    await gen_queue.put(error_dict)
                    break

                # Ensure generation is a dictionary before modifying
                if isinstance(generation, dict):
                    generation["index"] = task_idx

                await gen_queue.put(generation)

                # Only check for finish_reason on dict generations
                if isinstance(generation, dict) and "finish_reason" in generation:
                    break
        finally:
            # Track the end of the generation
            await track_generation_end(request_id)
    except Exception as e:
        logger.error(f"Error in _stream_collector: {str(e)}")
        logger.error(traceback.format_exc())
        await gen_queue.put(e)
        raise  # Propagate the exception so that pending tasks can be cleaned up


async def load_inline_model(model_name: str, request: Request):
    """Load a model from the data.model parameter"""
    # Check permissions and validate model path before attempting to load
    if not await validate_model_load_permissions(model_name, request):
        return

    # Validate model path exists
    model_path = pathlib.Path(config.model.model_dir) / model_name
    if not model_path.exists():
        logger.warning(
            f"Could not find model path {str(model_path)}. Skipping inline model load."
        )
        return

    # Use the model lifecycle manager to handle the model loading
    await load_model(model_name, request)


async def stream_generate_completion(
    data: CompletionRequest, request: Request, model_path: pathlib.Path
):
    """Streaming generation for completions."""

    abort_event = asyncio.Event()
    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    try:
        logger.info(f"Received streaming completion request {request.state.id}")

        for n in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)

            gen_task = asyncio.create_task(
                _stream_collector(
                    n,
                    gen_queue,
                    data.prompt,
                    request.state.id,
                    abort_event,
                    **task_gen_params.model_dump(exclude={"prompt"}),
                )
            )

            gen_tasks.append(gen_task)

        # Consumer loop
        while True:
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect(
                    f"Completion generation {request.state.id} cancelled by user."
                )

            try:
                # Use a timeout when getting from the queue to periodically
                # check model state
                try:
                    generation = await asyncio.wait_for(gen_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check model state again after timeout
                    error_dict = await check_model_before_operation(
                        request.state.id, "completion"
                    )
                    if error_dict:
                        logger.warning(
                            f"Model was unloaded while waiting for generation "
                            f"results for {request.state.id}"
                        )
                        yield get_generator_error(
                            f"Completion aborted: {error_dict['error']}"
                        )
                        break
                    continue  # Try again

                # Stream collector will push an exception to the queue if it fails
                if isinstance(generation, Exception):
                    raise generation

                # Check for special error indicator
                if isinstance(generation, dict) and "error" in generation:
                    logger.error(f"Generation error: {generation['error']}")
                    yield get_generator_error(
                        f"Completion error: {generation['error']}"
                    )
                    break

                response = _create_response(
                    request.state.id, generation, model_path.name
                )
                yield response.model_dump_json()
            except Exception as e:
                if not isinstance(e, asyncio.CancelledError):
                    logger.error(f"Error in stream_generate_completion: {str(e)}")
                    logger.error(traceback.format_exc())
                    yield get_generator_error(f"Completion error: {str(e)}")
                    break
                raise

            # Check if all tasks are completed
            if all(task.done() for task in gen_tasks) and gen_queue.empty():
                yield "[DONE]"
                logger.info(f"Finished streaming completion request {request.state.id}")
                break
    except CancelledError:
        # Get out if the request gets disconnected

        if not disconnect_task.done():
            abort_event.set()
            handle_request_disconnect(
                f"Completion generation {request.state.id} cancelled by user."
            )
    except Exception:
        yield get_generator_error(
            f"Completion {request.state.id} aborted. Please check the server console."
        )


async def generate_completion(
    data: CompletionRequest, request: Request, model_path: pathlib.Path
):
    """Non-streaming generate for completions"""

    gen_tasks: List[asyncio.Task] = []

    try:
        logger.info(f"Received completion request {request.state.id}")

        # Check if model is being unloaded before starting generation
        error_dict = await check_model_before_operation(request.state.id, "completion")
        if error_dict:
            error_message = handle_request_error(
                f"Completion aborted: {error_dict['error']}",
                exc_info=False,
            ).error.message
            raise HTTPException(503, error_message)

        # Track the start of the generation
        await track_generation_start(request.state.id, model=data.model)

        try:
            for _ in range(0, data.n):
                task_gen_params = data.model_copy(deep=True)

                gen_tasks.append(
                    asyncio.create_task(
                        model.container.generate(
                            data.prompt,
                            request.state.id,
                            **task_gen_params.model_dump(exclude={"prompt"}),
                        )
                    )
                )

            generations = await asyncio.gather(*gen_tasks)
            response = _create_response(request.state.id, generations, model_path.name)

            logger.info(f"Finished completion request {request.state.id}")

            return response
        finally:
            # Track the end of the generation
            await track_generation_end(request.state.id)
    except Exception as exc:
        # Cancel any remaining tasks
        for task in gen_tasks:
            if not task.done():
                task.cancel()

        # Check if the exception is related to model unloading
        if model.container is None or getattr(
            model.container, "model_is_unloading", False
        ):
            error_message = handle_request_error(
                f"Completion {request.state.id} aborted because the model was "
                f"unloaded during generation.",
                exc_info=False,
            ).error.message
        else:
            error_message = handle_request_error(
                f"Completion {request.state.id} aborted. "
                f"Please check the server console.",
                exc_info=True,
            ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc
