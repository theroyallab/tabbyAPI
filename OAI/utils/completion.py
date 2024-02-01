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
from OAI.types.completion import CompletionResponse, CompletionRespChoice
from OAI.types.common import UsageStats


def create_completion_response(
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    model_name: Optional[str],
):
    """Create a completion response from the provided text."""
    choice = CompletionRespChoice(finish_reason="Generated", text=text)

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


def create_chat_completion_response(
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    model_name: Optional[str],
):
    """Create a chat completion response from the provided text."""
    message = ChatCompletionMessage(role="assistant", content=text)

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
