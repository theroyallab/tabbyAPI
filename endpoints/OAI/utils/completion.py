""" Utility functions for the OpenAI server. """
from typing import Optional

from common.utils import unwrap
from endpoints.OAI.types.chat_completion import (
    ChatCompletionLogprobs,
    ChatCompletionLogprob,
    ChatCompletionMessage,
    ChatCompletionRespChoice,
    ChatCompletionStreamChunk,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
)
from endpoints.OAI.types.completion import (
    CompletionResponse,
    CompletionRespChoice,
    CompletionLogProbs,
)
from endpoints.OAI.types.common import UsageStats


def create_completion_response(generation: dict, model_name: Optional[str]):
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


def create_chat_completion_response(generation: dict, model_name: Optional[str]):
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


def create_chat_completion_stream_chunk(
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
