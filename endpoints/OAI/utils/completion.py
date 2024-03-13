"""Completion utilities for OAI server."""

import pathlib
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from typing import Optional

from common import model
from common.generators import release_semaphore
from common.utils import get_generator_error, handle_request_error, unwrap
from endpoints.OAI.types.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionRespChoice,
    CompletionLogProbs,
)
from endpoints.OAI.types.common import UsageStats


def _create_response(generation: dict, model_name: Optional[str]):
    """Create a completion response from the provided text."""

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
        finish_reason="Generated",
        text=unwrap(generation.get("text"), ""),
        logprobs=logprob_response,
    )

    prompt_tokens = unwrap(generation.get("prompt_tokens"), 0)
    completion_tokens = unwrap(generation.get("generated_tokens"), 0)

    response = CompletionResponse(
        choices=[choice],
        model=unwrap(model_name, ""),
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


async def stream_generate_completion(
    request: Request, data: CompletionRequest, model_path: pathlib.Path
):
    """Streaming generation for completions."""

    try:
        new_generation = model.container.generate_gen(
            data.prompt, **data.to_gen_params()
        )
        for generation in new_generation:
            # Get out if the request gets disconnected
            if await request.is_disconnected():
                release_semaphore()
                logger.error("Completion generation cancelled by user.")
                return

            response = _create_response(generation, model_path.name)

            yield response.model_dump_json()

        # Yield a finish response on successful generation
        yield "[DONE]"
    except Exception:
        yield get_generator_error(
            "Completion aborted. Please check the server console."
        )


async def generate_completion(data: CompletionRequest, model_path: pathlib.Path):
    """Non-streaming generate for completions"""

    try:
        generation = await run_in_threadpool(
            model.container.generate, data.prompt, **data.to_gen_params()
        )

        response = _create_response(generation, model_path.name)
        return response
    except Exception as exc:
        error_message = handle_request_error(
            "Completion aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc
