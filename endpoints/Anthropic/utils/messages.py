"""Message utilities for the Anthropic server."""

import json
from typing import List, Optional, Tuple

from common import model
from common.logger import xlogger
from common.utils import unwrap
from endpoints.Anthropic.types.messages import (
    CountTokensRequest,
    CountTokensResponse,
    MessagesRequest,
    MessagesResponse,
    ResponseContentBlock,
    ResponseTextBlock,
    ResponseThinkingBlock,
    ResponseToolUseBlock,
    Usage,
)
from endpoints.Anthropic.utils.convert import convert_count_tokens_request
from endpoints.OAI.types.chat_completion import ChatCompletionResponse
from endpoints.OAI.utils.chat_completion import apply_chat_template


def tool_call_input(name: str, arguments: str) -> dict:
    """
    Parse tool call arguments into the object a tool_use block carries.

    The pipeline hands back arguments as a JSON string, per the OAI shape the
    tool call parsers emit. A parser that produced something unparseable would
    otherwise fail the whole response, so the call is surfaced with empty
    input and a warning instead: the tool name is the useful part.
    """

    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if not isinstance(parsed, dict):
        xlogger.warning(
            "Tool call arguments could not be parsed into an object",
            {"name": name, "arguments": arguments},
        )

        return {}

    return parsed


def stop_reason(
    finish_reason: Optional[str],
    eos_reason: Optional[str],
    stop_str: Optional[str],
    stop_sequences: Optional[List[str]],
) -> Tuple[str, Optional[str]]:
    """
    Map a finished generation onto an Anthropic stop reason.

    Takes the raw fields rather than a response object so the streaming and
    non-streaming paths share one implementation; a stop reason that differs
    between them is a class of bug worth designing out.

    The stop_sequence field only reports sequences the client asked for. A
    prompt template contributes its own stop strings, and surfacing one of
    those as a stop_sequence would name a string the client never sent.
    """

    if finish_reason == "tool_calls":
        return "tool_use", None

    if finish_reason == "length":
        return "max_tokens", None

    if eos_reason == "stop_string" and stop_str in (stop_sequences or []):
        return "stop_sequence", stop_str

    return "end_turn", None


def convert_response(
    completion: ChatCompletionResponse,
    data: MessagesRequest,
    model_name: str,
) -> MessagesResponse:
    """Convert a chat completion response into an Anthropic message."""

    choice = completion.choices[0]
    message = choice.message

    # Reasoning precedes the answer it produced, and tool calls follow the
    # text introducing them, matching the block order the Anthropic API emits
    content: List[ResponseContentBlock] = []
    if message.reasoning_content:
        content.append(ResponseThinkingBlock(thinking=message.reasoning_content))
    if message.content:
        content.append(ResponseTextBlock(text=message.content))
    for tool_call in message.tool_calls or []:
        content.append(
            ResponseToolUseBlock(
                id=tool_call.id,
                name=tool_call.function.name,
                input=tool_call_input(tool_call.function.name, tool_call.function.arguments),
            )
        )

    reason, stop_sequence = stop_reason(
        choice.finish_reason, choice.eos_reason, choice.stop_str, data.stop_sequences
    )

    usage = completion.usage
    return MessagesResponse(
        content=content,
        model=model_name,
        stop_reason=reason,
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
