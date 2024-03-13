"""Chat completion utilities for OAI server."""

import pathlib
from typing import Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from jinja2 import TemplateError
from loguru import logger

from common import model
from common.generators import release_semaphore
from common.templating import get_prompt_from_template
from common.utils import get_generator_error, handle_request_error, unwrap
from endpoints.OAI.types.chat_completion import (
    ChatCompletionLogprobs,
    ChatCompletionLogprob,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionRespChoice,
    ChatCompletionStreamChunk,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
)
from endpoints.OAI.types.common import UsageStats


def _create_response(generation: dict, model_name: Optional[str]):
    """Create a chat completion response from the provided text."""

    message = ChatCompletionMessage(
        role="assistant", content=unwrap(generation.get("text"), "")
    )

    logprob_response = None

    token_probs = unwrap(generation.get("token_probs"), {})
    if token_probs:
        logprobs = unwrap(generation.get("logprobs"), [])

        collected_token_probs = []
        for index, token in enumerate(token_probs.keys()):
            top_logprobs = [
                ChatCompletionLogprob(token=token, logprob=logprob)
                for token, logprob in logprobs[index].items()
            ]

            collected_token_probs.append(
                ChatCompletionLogprob(
                    token=token,
                    logprob=token_probs[token],
                    top_logprobs=top_logprobs,
                )
            )

        logprob_response = ChatCompletionLogprobs(content=collected_token_probs)

    choice = ChatCompletionRespChoice(
        finish_reason="Generated", message=message, logprobs=logprob_response
    )

    prompt_tokens = unwrap(generation.get("prompt_tokens"), 0)
    completion_tokens = unwrap(generation.get("completion_tokens"), 0)

    response = ChatCompletionResponse(
        choices=[choice],
        model=unwrap(model_name, ""),
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


def _create_stream_chunk(
    const_id: str,
    generation: Optional[dict] = None,
    model_name: Optional[str] = None,
    finish_reason: Optional[str] = None,
):
    """Create a chat completion stream chunk from the provided text."""

    logprob_response = None

    if finish_reason:
        message = {}
    else:
        message = ChatCompletionMessage(
            role="assistant", content=unwrap(generation.get("text"), "")
        )

        token_probs = unwrap(generation.get("token_probs"), {})
        if token_probs:
            logprobs = unwrap(generation.get("logprobs"), {})
            top_logprobs = [
                ChatCompletionLogprob(token=token, logprob=logprob)
                for token, logprob in logprobs.items()
            ]

            generated_token = next(iter(token_probs))
            token_prob_response = ChatCompletionLogprob(
                token=generated_token,
                logprob=token_probs[generated_token],
                top_logprobs=top_logprobs,
            )

            logprob_response = ChatCompletionLogprobs(content=[token_prob_response])

    # The finish reason can be None
    choice = ChatCompletionStreamChoice(
        finish_reason=finish_reason, delta=message, logprobs=logprob_response
    )

    chunk = ChatCompletionStreamChunk(
        id=const_id, choices=[choice], model=unwrap(model_name, "")
    )

    return chunk


def format_prompt_with_template(data: ChatCompletionRequest):
    try:
        special_tokens_dict = model.container.get_special_tokens(
            unwrap(data.add_bos_token, True),
            unwrap(data.ban_eos_token, False),
        )

        return get_prompt_from_template(
            data.messages,
            model.container.prompt_template,
            data.add_generation_prompt,
            special_tokens_dict,
        )
    except KeyError as exc:
        raise HTTPException(
            400,
            "Could not find a Conversation from prompt template "
            f"'{model.container.prompt_template.name}'. "
            "Check your spelling?",
        ) from exc
    except TemplateError as exc:
        raise HTTPException(
            400,
            f"TemplateError: {str(exc)}",
        ) from exc


async def stream_generate_chat_completion(
    prompt: str, request: Request, data: ChatCompletionRequest, model_path: pathlib.Path
):
    """Generator for the generation process."""
    try:
        const_id = f"chatcmpl-{uuid4().hex}"

        new_generation = model.container.generate_gen(prompt, **data.to_gen_params())
        for generation in new_generation:
            # Get out if the request gets disconnected
            if await request.is_disconnected():
                release_semaphore()
                logger.error("Chat completion generation cancelled by user.")
                return

            response = _create_stream_chunk(const_id, generation, model_path.name)

            yield response.model_dump_json()

        # Yield a finish response on successful generation
        finish_response = _create_stream_chunk(const_id, finish_reason="stop")

        yield finish_response.model_dump_json()
    except Exception:
        yield get_generator_error(
            "Chat completion aborted. Please check the server console."
        )


async def generate_chat_completion(
    prompt: str, request: Request, data: ChatCompletionRequest, model_path: pathlib.Path
):
    try:
        generation = await run_in_threadpool(
            model.container.generate,
            prompt,
            **data.to_gen_params(),
        )
        response = _create_response(generation, model_path.name)

        return response
    except Exception as exc:
        error_message = handle_request_error(
            "Chat completion aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc
