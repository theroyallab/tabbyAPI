"""Completion utilities for OAI server."""

import asyncio
import pathlib
from asyncio import CancelledError
from copy import deepcopy
from fastapi import HTTPException, Request
from typing import List, Optional

from common import model
from common.networking import (
    get_generator_error,
    handle_request_disconnect,
    handle_request_error,
    request_disconnect_loop,
)
from common.utils import unwrap
from endpoints.OAI.types.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionRespChoice,
    CompletionLogProbs,
)
from endpoints.OAI.types.common import UsageStats


def _create_response(generations: List[dict], model_name: Optional[str]):
    """Create a completion response from the provided text."""

    prompt_tokens = unwrap(generations[-1].get("prompt_tokens"), 0)
    completion_tokens = unwrap(generations[-1].get("generated_tokens"), 0)

    choices = []
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

        choice = CompletionRespChoice(
            index=index,
            finish_reason=generation.get("finish_reason"),
            text=unwrap(generation.get("text"), ""),
            logprobs=logprob_response,
        )

        choices.append(choice)

    response = CompletionResponse(
        choices=choices,
        model=unwrap(model_name, ""),
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


async def stream_generate_completion(
    data: CompletionRequest, request: Request, model_path: pathlib.Path
):
    """Streaming generation for completions."""

    abort_event = asyncio.Event()

    try:
        new_generation = model.container.generate_gen(
            data.prompt, abort_event, **data.to_gen_params()
        )

        # Create a background task to avoid blocking the loop
        disconnect_task = asyncio.create_task(request_disconnect_loop(request))

        async for generation in new_generation:
            # Sometimes this fires, and sometimes a CancelledError will fire
            # Keep both implementations in to avoid the headache
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect("Completion generation cancelled by user.")

            response = _create_response([generation], model_path.name)
            yield response.model_dump_json()

            # Break if the generation is finished
            if "finish_reason" in generation:
                yield "[DONE]"
                break
    except CancelledError:
        # Get out if the request gets disconnected

        abort_event.set()
        handle_request_disconnect("Completion generation cancelled by user.")
    except Exception:
        yield get_generator_error(
            "Completion aborted. Please check the server console."
        )


async def generate_completion(data: CompletionRequest, model_path: pathlib.Path):
    """Non-streaming generate for completions"""

    gen_tasks: List[asyncio.Task] = []
    gen_params = data.to_gen_params()

    try:
        for n in range(0, data.n):

            # Deepcopy gen params above the first index
            # to ensure nested structures aren't shared
            if n > 0:
                task_gen_params = deepcopy(gen_params)
            else:
                task_gen_params = gen_params

            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(data.prompt, **task_gen_params)
                )
            )

        generations = await asyncio.gather(*gen_tasks)
        response = _create_response(generations, model_path.name)

        return response
    except Exception as exc:
        error_message = handle_request_error(
            "Completion aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc
