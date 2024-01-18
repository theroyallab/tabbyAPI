""" Utility functions for the OpenAI server. """
import pathlib
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
from OAI.types.lora import LoraList, LoraCard
from OAI.types.model import ModelList, ModelCard


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


def get_model_list(model_path: pathlib.Path, draft_model_path: Optional[str] = None):
    """Get the list of models from the provided path."""

    # Convert the provided draft model path to a pathlib path for
    # equality comparisons
    if draft_model_path:
        draft_model_path = pathlib.Path(draft_model_path).resolve()

    model_card_list = ModelList()
    for path in model_path.iterdir():
        # Don't include the draft models path
        if path.is_dir() and path != draft_model_path:
            model_card = ModelCard(id=path.name)
            model_card_list.data.append(model_card)  # pylint: disable=no-member

    return model_card_list


def get_lora_list(lora_path: pathlib.Path):
    """Get the list of Lora cards from the provided path."""
    lora_list = LoraList()
    for path in lora_path.iterdir():
        if path.is_dir():
            lora_card = LoraCard(id=path.name)
            lora_list.data.append(lora_card)  # pylint: disable=no-member

    return lora_list
