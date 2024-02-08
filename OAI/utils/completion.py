""" Utility functions for the OpenAI server. """
from typing import Optional

from common.utils import unwrap
from OAI.types.chat_completion import (
    ChatCompletionMessage,
    ChatCompletionRespChoice,
    ChatCompletionStreamChunk,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
)
from OAI.types.completion import (
    CompletionResponse,
    CompletionRespChoice,
    CompletionLogProbs,
)
from OAI.types.common import UsageStats


def create_completion_response(**kwargs):
    """Create a completion response from the provided text."""

    token_probs = unwrap(kwargs.get("token_probs"), {})
    logprobs = unwrap(kwargs.get("logprobs"), [])
    offset = unwrap(kwargs.get("offset"), [])

    logprob_response = CompletionLogProbs(
        text_offset=offset if isinstance(offset, list) else [offset],
        token_logprobs=token_probs.values(),
        tokens=token_probs.keys(),
        top_logprobs=logprobs if isinstance(logprobs, list) else [logprobs],
    )

    choice = CompletionRespChoice(
        finish_reason="Generated",
        text=unwrap(kwargs.get("text"), ""),
        logprobs=logprob_response,
    )

    prompt_tokens = unwrap(kwargs.get("prompt_tokens"), 0)
    completion_tokens = unwrap(kwargs.get("completion_tokens"), 0)

    response = CompletionResponse(
        choices=[choice],
        model=unwrap(kwargs.get("model_name"), ""),
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


def create_chat_completion_response(
    text: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    model_name: Optional[str],
):
    """Create a chat completion response from the provided text."""
    message = ChatCompletionMessage(role="assistant", content=unwrap(text, ""))

    choice = ChatCompletionRespChoice(finish_reason="Generated", message=message)

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
    text: Optional[str] = None,
    model_name: Optional[str] = None,
    finish_reason: Optional[str] = None,
):
    """Create a chat completion stream chunk from the provided text."""
    if finish_reason:
        message = {}
    else:
        message = ChatCompletionMessage(role="assistant", content=text)

    # The finish reason can be None
    choice = ChatCompletionStreamChoice(finish_reason=finish_reason, delta=message)

    chunk = ChatCompletionStreamChunk(
        id=const_id, choices=[choice], model=unwrap(model_name, "")
    )

    return chunk
