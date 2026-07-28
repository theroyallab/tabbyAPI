"""Translation between Anthropic Messages requests and TabbyAPI's internal
chat completion request.

Anthropic requests are converted rather than served by a separate inference
path, so the Anthropic API inherits chat templating, reasoning tag parsing,
samplers and context length handling from the chat completion pipeline.
"""

import json
from typing import List, NamedTuple, Optional, Union

from common.logger import xlogger
from endpoints.Anthropic.errors import request_error
from endpoints.Anthropic.types.messages import (
    AnthropicMessage,
    CountTokensRequest,
    MessagesRequest,
    RedactedThinkingBlock,
    RequestContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolChoice,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest
from endpoints.OAI.types.common import ChatCompletionStreamOptions
from endpoints.OAI.types.tools import (
    Function,
    NamedToolChoice,
    NamedToolFunction,
    Tool,
    ToolCall,
    ToolSpec,
)

# Anthropic clients routinely split one turn across several text blocks (a
# system prompt and an environment block, say), which are separate pieces of
# context rather than a continuing sentence. Joining on a blank line keeps
# them apart in the flat string a chat template renders.
BLOCK_SEPARATOR = "\n\n"

# Anthropic tool_choice modes mapped onto their OAI equivalents
TOOL_CHOICE_MODES = {"auto": "auto", "any": "required", "none": "none"}


def _system_prompt(system: Optional[Union[str, List[TextBlock]]]) -> Optional[str]:
    """Flatten the system field into a single prompt string."""

    if system is None:
        return None

    if isinstance(system, str):
        return system or None

    texts = [block.text for block in system if block.text]

    return BLOCK_SEPARATOR.join(texts) or None


def _tool_result_text(block: ToolResultBlock) -> str:
    """
    Flatten a tool result into the string a chat template can render.

    Chat templates have no concept of a failed tool call, so is_error is
    folded into the text: the model can only act on the failure if it can
    read it.
    """

    if block.content is None:
        text = ""
    elif isinstance(block.content, str):
        text = block.content
    else:
        parts = []
        for inner in block.content:
            if isinstance(inner, TextBlock):
                parts.append(inner.text)
            else:
                raise request_error(
                    400,
                    f"Content block type '{inner.type}' inside a tool_result is not "
                    "supported. This server currently accepts text blocks there.",
                )
        text = BLOCK_SEPARATOR.join(parts)

    if block.is_error:
        text = f"Error: {text}" if text else "Error"

    return text


class MessageParts(NamedTuple):
    """One Anthropic message, split along the axes a chat message needs."""

    text: Optional[str]
    reasoning: Optional[str]
    tool_calls: List[ToolCall]
    tool_results: List[ToolResultBlock]


def _split_content(content: Union[str, List[RequestContentBlock]]) -> MessageParts:
    """
    Split one message's content by block kind.

    Raises for block types this server does not handle yet, so an unsupported
    request fails with a readable message instead of silently dropping the
    part of the conversation the client cared about.
    """

    if isinstance(content, str):
        return MessageParts(content or None, None, [], [])

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    tool_results: List[ToolResultBlock] = []

    for block in content:
        if isinstance(block, TextBlock):
            if block.text:
                text_parts.append(block.text)
        elif isinstance(block, ThinkingBlock):
            if block.thinking:
                reasoning_parts.append(block.thinking)
        elif isinstance(block, RedactedThinkingBlock):
            # Encrypted by the Anthropic API and unreadable here. Dropping it
            # loses nothing a local model could have used.
            continue
        elif isinstance(block, ToolUseBlock):
            # Templates render OAI-shaped tool calls, whose arguments are a
            # JSON string; format_messages_with_template parses it back for
            # the templates that want a mapping
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    function=Tool(name=block.name, arguments=json.dumps(block.input)),
                )
            )
        elif isinstance(block, ToolResultBlock):
            tool_results.append(block)
        else:
            raise request_error(
                400,
                f"Content block type '{block.type}' is not supported. This server "
                "currently accepts text, thinking, tool_use and tool_result blocks.",
            )

    return MessageParts(
        BLOCK_SEPARATOR.join(text_parts) or None,
        BLOCK_SEPARATOR.join(reasoning_parts) or None,
        tool_calls,
        tool_results,
    )


def build_chat_messages(
    system: Optional[Union[str, List[TextBlock]]],
    messages: List[AnthropicMessage],
) -> List[ChatCompletionMessage]:
    """Convert an Anthropic system prompt and message list for templating."""

    chat_messages: List[ChatCompletionMessage] = []

    system_prompt = _system_prompt(system)
    if system_prompt:
        chat_messages.append(ChatCompletionMessage(role="system", content=system_prompt))

    for message in messages:
        parts = _split_content(message.content)

        # Anthropic packs every tool result for a turn into one user message,
        # but chat templates expect one tool message per result, so a single
        # message can fan out. They lead the turn, which is also the order
        # Anthropic requires them in.
        for result in parts.tool_results:
            chat_messages.append(
                ChatCompletionMessage(
                    role="tool",
                    content=_tool_result_text(result),
                    tool_call_id=result.tool_use_id,
                )
            )

        # An assistant turn that only called tools carries no text, and a user
        # turn of nothing but tool results adds no message of its own
        if parts.text or parts.reasoning or parts.tool_calls:
            chat_messages.append(
                ChatCompletionMessage(
                    role=message.role,
                    content=parts.text,
                    reasoning_content=parts.reasoning,
                    tool_calls=parts.tool_calls or None,
                )
            )

    return chat_messages


def convert_tools(tools: Optional[List[ToolDefinition]]) -> Optional[List[ToolSpec]]:
    """
    Convert tool definitions into the OAI shape chat templates render.

    Every tool call format in the pipeline is driven from that shape, so
    translating here means the Anthropic API inherits all of them.
    """

    if not tools:
        return None

    specs = []

    for tool in tools:
        # Anthropic's own server-side tools run on their infrastructure and
        # have no local equivalent
        if tool.type and tool.type != "custom":
            raise request_error(
                400,
                f"Server-side tool '{tool.type}' is not supported. This server only "
                "serves client-defined tools.",
            )

        if not tool.name or tool.input_schema is None:
            raise request_error(400, "Each tool needs a name and an input_schema.")

        specs.append(
            ToolSpec(
                type="function",
                function=Function(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.input_schema,
                ),
            )
        )

    return specs


def convert_tool_choice(choice: Optional[ToolChoice]):
    """
    Convert tool_choice, returning the choice and the parallel call setting.

    Returns (None, None) when the request left it unset, so the pipeline's own
    defaults apply.
    """

    if choice is None:
        return None, None

    parallel = None
    if choice.disable_parallel_tool_use is not None:
        parallel = not choice.disable_parallel_tool_use

    if choice.type == "tool":
        if not choice.name:
            raise request_error(400, "A tool_choice of type 'tool' needs a tool name.")

        return NamedToolChoice(function=NamedToolFunction(name=choice.name)), parallel

    mode = TOOL_CHOICE_MODES.get(choice.type)
    if mode is None:
        raise request_error(
            400,
            f"Unknown tool_choice type '{choice.type}'. Expected auto, any, tool or none.",
        )

    return mode, parallel


def _sampler_params(data: MessagesRequest) -> dict:
    """
    Collect the sampler fields the request actually set.

    Unset fields are left out entirely so the model's sampler defaults and
    any configured overrides still apply.
    """

    params = {"max_tokens": data.max_tokens}

    if data.stop_sequences:
        params["stop"] = list(data.stop_sequences)
    if data.temperature is not None:
        params["temperature"] = data.temperature
    if data.top_p is not None:
        params["top_p"] = data.top_p
    if data.top_k is not None:
        params["top_k"] = data.top_k
    if data.metadata and data.metadata.user_id:
        params["user"] = data.metadata.user_id

    return params


def convert_messages_request(data: MessagesRequest) -> ChatCompletionRequest:
    """Convert an Anthropic Messages request into a chat completion request."""

    template_vars = {}
    if data.thinking is not None:
        template_vars["enable_thinking"] = data.thinking.type != "disabled"

        if data.thinking.budget_tokens is not None:
            # Thinking length is a property of the model and its template
            # here, not something a request can allocate
            xlogger.debug("thinking.budget_tokens is not supported; ignoring.")

    tool_choice, parallel_tool_calls = convert_tool_choice(data.tool_choice)

    optional = {}
    if parallel_tool_calls is not None:
        optional["parallel_tool_calls"] = parallel_tool_calls

    return ChatCompletionRequest(
        messages=build_chat_messages(data.system, data.messages),
        model=data.model,
        template_vars=template_vars,
        tools=convert_tools(data.tools),
        tool_choice=tool_choice,
        **optional,
        # Anthropic returns usage on every response, and the chat completion
        # pipeline only assembles it when asked
        stream_options=ChatCompletionStreamOptions(include_usage=True),
        # The Messages API has no multi-choice concept
        n=1,
        **_sampler_params(data),
    )


def convert_count_tokens_request(data: CountTokensRequest) -> ChatCompletionRequest:
    """
    Convert a token counting request into a chat completion request.

    max_tokens is a placeholder: nothing is generated, but the prompt is
    rendered through the same path a real request would take so the count
    reflects what generation would actually see.
    """

    return ChatCompletionRequest(
        messages=build_chat_messages(data.system, data.messages),
        model=data.model,
        tools=convert_tools(data.tools),
        max_tokens=1,
        n=1,
    )
