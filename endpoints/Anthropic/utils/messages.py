"""Message utilities for the Anthropic server."""

from typing import List, Optional, Tuple

from common import model
from common.utils import unwrap
from endpoints.Anthropic.types.messages import (
    CountTokensRequest,
    CountTokensResponse,
    MessagesRequest,
    MessagesResponse,
    ResponseContentBlock,
    ResponseTextBlock,
    ResponseThinkingBlock,
    Usage,
)
from endpoints.Anthropic.utils.convert import convert_count_tokens_request
from endpoints.OAI.types.chat_completion import ChatCompletionRespChoice, ChatCompletionResponse
from endpoints.OAI.utils.chat_completion import apply_chat_template


def _stop_reason(
    choice: ChatCompletionRespChoice, stop_sequences: Optional[List[str]]
) -> Tuple[str, Optional[str]]:
    """
    Map a finished choice onto an Anthropic stop reason.

    The stop_sequence field only reports sequences the client asked for. A
    prompt template contributes its own stop strings, and surfacing one of
    those as a stop_sequence would name a string the client never sent.
    """

    if choice.finish_reason == "tool_calls":
        return "tool_use", None

    if choice.finish_reason == "length":
        return "max_tokens", None

    if choice.eos_reason == "stop_string" and choice.stop_str in (stop_sequences or []):
        return "stop_sequence", choice.stop_str

    return "end_turn", None


def convert_response(
    completion: ChatCompletionResponse,
    data: MessagesRequest,
    model_name: str,
) -> MessagesResponse:
    """Convert a chat completion response into an Anthropic message."""

    choice = completion.choices[0]
    message = choice.message

    # Reasoning precedes the answer it produced, matching the block order the
    # Anthropic API emits
    content: List[ResponseContentBlock] = []
    if message.reasoning_content:
        content.append(ResponseThinkingBlock(thinking=message.reasoning_content))
    if message.content:
        content.append(ResponseTextBlock(text=message.content))

    stop_reason, stop_sequence = _stop_reason(choice, data.stop_sequences)

    usage = completion.usage
    return MessagesResponse(
        content=content,
        model=model_name,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=Usage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
    )


async def count_tokens(data: CountTokensRequest) -> CountTokensResponse:
    """
    Count the tokens an equivalent Messages request would consume.

    The prompt is rendered through the same template path as generation, so
    the count includes the template's own structure and generation prompt.
    """

    converted = convert_count_tokens_request(data)
    prompt, mm_embeddings = await apply_chat_template(converted)

    raw_tokens = model.container.encode_tokens(prompt, embeddings=mm_embeddings)

    return CountTokensResponse(input_tokens=len(unwrap(raw_tokens, [])))
