"""
Completion utilities for OAI server.

Also serves as a common module for completions and chat completions.
"""

import asyncio
import pathlib
from asyncio import CancelledError
from copy import deepcopy
from fastapi import HTTPException, Request
from typing import List, Union

from loguru import logger

from backends.exllamav2.types import ModelInstanceConfig
from common import model
from common.auth import get_key_permission
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
        new_generation = model.container.generate_gen(
            prompt, request_id, abort_event, **kwargs
        )
        async for generation in new_generation:
            generation["index"] = task_idx

            await gen_queue.put(generation)

            if "finish_reason" in generation:
                break
    except Exception as e:
        await gen_queue.put(e)


async def load_inline_model(model_name: str, request: Request):
    """Load a model from the data.model parameter"""

    # Return if the model container already exists and the model is fully loaded
    if (
        model.container
        and model.container.model_dir.name == model_name
        and model.container.model_loaded
    ):
        return

    # Inline model loading isn't enabled or the user isn't an admin
    if not get_key_permission(request) == "admin":
        error_message = handle_request_error(
            f"Unable to switch model to {model_name} because "
            + "an admin key isn't provided",
            exc_info=False,
        ).error.message

        raise HTTPException(401, error_message)

    if not config.model.inline_model_loading:
        logger.warning(
            f"Unable to switch model to {model_name} because "
            '"inline_model_loading" is not True in config.yml.'
        )

        return

    # Model path doesn't exist
    # if not model_path.exists():
    #     logger.warning(
    #         f"Could not find model path {str(model_path)}." +
    #         "Skipping inline model load."
    #     )

    #     return

    # Load the model
    await model.load_model(ModelInstanceConfig(model_name=model_name))


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

        gen_params = data.to_gen_params()

        for n in range(0, data.n):
            if n > 0:
                task_gen_params = deepcopy(gen_params)
            else:
                task_gen_params = gen_params

            gen_task = asyncio.create_task(
                _stream_collector(
                    n,
                    gen_queue,
                    data.prompt,
                    request.state.id,
                    abort_event,
                    **task_gen_params,
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

            generation = await gen_queue.get()

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            response = _create_response(request.state.id, generation, model_path.name)
            yield response.model_dump_json()

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
    gen_params = data.to_gen_params()

    try:
        logger.info(f"Recieved completion request {request.state.id}")

        for n in range(0, data.n):
            # Deepcopy gen params above the first index
            # to ensure nested structures aren't shared
            if n > 0:
                task_gen_params = deepcopy(gen_params)
            else:
                task_gen_params = gen_params

            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        data.prompt, request.state.id, **task_gen_params
                    )
                )
            )

        generations = await asyncio.gather(*gen_tasks)
        response = _create_response(request.state.id, generations, model_path.name)

        logger.info(f"Finished completion request {request.state.id}")

        return response
    except Exception as exc:
        error_message = handle_request_error(
            f"Completion {request.state.id} aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc
