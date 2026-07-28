"""Translation between Anthropic Messages requests and TabbyAPI's internal
chat completion request.

Anthropic requests are converted rather than served by a separate inference
path, so the Anthropic API inherits chat templating, reasoning tag parsing,
samplers and context length handling from the chat completion pipeline.
"""

from typing import List, Optional, Tuple, Union

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
)
from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest
from endpoints.OAI.types.common import ChatCompletionStreamOptions

# Anthropic clients routinely split one turn across several text blocks (a
# system prompt and an environment block, say), which are separate pieces of
# context rather than a continuing sentence. Joining on a blank line keeps
# them apart in the flat string a chat template renders.
BLOCK_SEPARATOR = "\n\n"


def _system_prompt(system: Optional[Union[str, List[TextBlock]]]) -> Optional[str]:
    """Flatten the system field into a single prompt string."""

    if system is None:
        return None

    if isinstance(system, str):
        return system or None

    texts = [block.text for block in system if block.text]

    return BLOCK_SEPARATOR.join(texts) or None


def _split_content(
    content: Union[str, List[RequestContentBlock]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Split one message's content into visible text and reasoning text.

    Raises for block types this server does not handle yet, so an unsupported
    request fails with a readable message instead of silently dropping the
    part of the conversation the client cared about.
    """

    if isinstance(content, str):
        return content or None, None

    text_parts: List[str] = []
    reasoning_parts: List[str] = []

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
        else:
            raise request_error(
                400,
                f"Content block type '{block.type}' is not supported. This server "
                "currently accepts text and thinking blocks.",
            )

    return (
        BLOCK_SEPARATOR.join(text_parts) or None,
        BLOCK_SEPARATOR.join(reasoning_parts) or None,
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
        content, reasoning_content = _split_content(message.content)
        chat_messages.append(
            ChatCompletionMessage(
                role=message.role,
                content=content,
                reasoning_content=reasoning_content,
            )
        )

    return chat_messages


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

    return ChatCompletionRequest(
        messages=build_chat_messages(data.system, data.messages),
        model=data.model,
        template_vars=template_vars,
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
        max_tokens=1,
        n=1,
    )
