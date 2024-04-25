"""Chat completion utilities for OAI server."""

from asyncio import CancelledError
import pathlib
import threading
from typing import Optional
from uuid import uuid4

from fastapi import HTTPException
from jinja2 import TemplateError
from loguru import logger

from common import model
from common.networking import (
    get_generator_error,
    handle_request_disconnect,
    handle_request_error,
)
from common.utils import unwrap
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
        finish_reason=generation.get("finish_reason"),
        message=message,
        logprobs=logprob_response,
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
):
    """Create a chat completion stream chunk from the provided text."""

    logprob_response = None

    if "finish_reason" in generation:
        choice = ChatCompletionStreamChoice(
            finish_reason=generation.get("finish_reason")
        )
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

        choice = ChatCompletionStreamChoice(
            delta=message,
            logprobs=logprob_response,
        )

    chunk = ChatCompletionStreamChunk(
        id=const_id, choices=[choice], model=unwrap(model_name, "")
    )

    return chunk


def format_prompt_with_template(data: ChatCompletionRequest):
    """
    Compile the prompt and get any additional stop strings from the template.
    Template stop strings can be overriden by sampler overrides if force is true.
    """

    try:
        special_tokens_dict = model.container.get_special_tokens(
            unwrap(data.add_bos_token, True),
            unwrap(data.ban_eos_token, False),
        )

        # Overwrite any protected vars with their values
        data.template_vars.update(
            {
                "messages": data.messages,
                "add_generation_prompt": data.add_generation_prompt,
                **special_tokens_dict,
            }
        )

        prompt, template_stop_strings = model.container.prompt_template.render(
            data.template_vars
        )

        # Append response prefix if present
        if data.response_prefix:
            if data.add_generation_prompt:
                prompt += data.response_prefix
            else:
                logger.warning(
                    "Could not add response prefix because "
                    "add_generation_prompt is False"
                )

        # Removes the starting BOS token if present
        # This is to prevent add_bos_token from adding multiple bos tokens
        bos_token = special_tokens_dict.get("bos_token")
        if bos_token and prompt.startswith(bos_token):
            prompt = prompt.removeprefix(bos_token)

        # Append template stop strings
        if isinstance(data.stop, str):
            data.stop = [data.stop] + template_stop_strings
        else:
            data.stop += template_stop_strings

        return prompt

    except KeyError as exc:
        error_message = handle_request_error(
            "Could not find a Conversation from prompt template "
            f"'{model.container.prompt_template.name}'. "
            "Check your spelling?",
        ).error.message

        raise HTTPException(400, error_message) from exc
    except TemplateError as exc:
        error_message = handle_request_error(f"TemplateError: {str(exc)}").error.message

        raise HTTPException(400, error_message) from exc


async def stream_generate_chat_completion(
    prompt: str, data: ChatCompletionRequest, model_path: pathlib.Path
):
    """Generator for the generation process."""
    try:
        const_id = f"chatcmpl-{uuid4().hex}"
        abort_event = threading.Event()

        new_generation = model.container.generate_gen(
            prompt, abort_event, **data.to_gen_params()
        )
        async for generation in new_generation:
            response = _create_stream_chunk(const_id, generation, model_path.name)

            yield response.model_dump_json()

            # Break if the generation is finished
            if "finish_reason" in generation:
                break
    except CancelledError:
        # Get out if the request gets disconnected

        abort_event.set()
        handle_request_disconnect("Chat completion generation cancelled by user.")
    except Exception:
        yield get_generator_error(
            "Chat completion aborted. Please check the server console."
        )


async def generate_chat_completion(
    prompt: str, data: ChatCompletionRequest, model_path: pathlib.Path
):
    try:
        generation = await model.container.generate(
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
