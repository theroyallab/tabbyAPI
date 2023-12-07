import os, pathlib
from OAI.types.completion import CompletionResponse, CompletionRespChoice, UsageStats
from OAI.types.chat_completion import (
    ChatCompletionMessage,
    ChatCompletionRespChoice,
    ChatCompletionStreamChunk,
    ChatCompletionResponse,
    ChatCompletionStreamChoice
)
from OAI.types.common import UsageStats
from OAI.types.model import ModelList, ModelCard
from packaging import version
from typing import Optional, List

# Check fastchat
try:
    import fastchat
    from fastchat.model.model_adapter import get_conversation_template
    from fastchat.conversation import SeparatorStyle
    _fastchat_available = True
except ImportError:
    _fastchat_available = False

def create_completion_response(text: str, prompt_tokens: int, completion_tokens: int, model_name: Optional[str]):
    choice = CompletionRespChoice(
        finish_reason = "Generated",
        text = text
    )

    response = CompletionResponse(
        choices = [choice],
        model = model_name or "",
        usage = UsageStats(prompt_tokens = prompt_tokens,
                           completion_tokens = completion_tokens,
                           total_tokens = prompt_tokens + completion_tokens)
    )

    return response

def create_chat_completion_response(text: str, prompt_tokens: int, completion_tokens: int, model_name: Optional[str]):
    message = ChatCompletionMessage(
        role = "assistant",
        content = text
    )

    choice = ChatCompletionRespChoice(
        finish_reason = "Generated",
        message = message
    )

    response = ChatCompletionResponse(
        choices = [choice],
        model = model_name or "",
        usage = UsageStats(prompt_tokens = prompt_tokens,
                           completion_tokens = completion_tokens,
                           total_tokens = prompt_tokens + completion_tokens)
    )

    return response

def create_chat_completion_stream_chunk(const_id: str,
                                        text: Optional[str] = None,
                                        model_name: Optional[str] = None,
                                        finish_reason: Optional[str] = None):
    if finish_reason:
        message = {}
    else:
        message = ChatCompletionMessage(
            role = "assistant",
            content = text
        )

    # The finish reason can be None
    choice = ChatCompletionStreamChoice(
        finish_reason = finish_reason,
        delta = message
    )

    chunk = ChatCompletionStreamChunk(
        id = const_id,
        choices = [choice],
        model = model_name or ""
    )

    return chunk

def get_model_list(model_path: pathlib.Path, draft_model_path: Optional[str]):

    # Convert the draft model path to a pathlib path for equality comparisons
    if draft_model_path:
        draft_model_path = pathlib.Path(draft_model_path).resolve()

    model_card_list = ModelList()
    for path in model_path.iterdir():

        # Don't include the draft models path
        if path.is_dir() and path != draft_model_path:
            model_card = ModelCard(id = path.name)
            model_card_list.data.append(model_card)

    return model_card_list

def get_chat_completion_prompt(model_path: str, messages: List[ChatCompletionMessage]):

    # Check if fastchat is available
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "Fastchat must be installed to parse these chat completion messages.\n"
            "Please run the following command: pip install fschat[model_worker]"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            "Parsing these chat completion messages requires fastchat 0.2.23 or greater. "
            f"Current version: {fastchat.__version__}\n"
            "Please upgrade fastchat by running the following command: "
            "pip install -U fschat[model_worker]"
        )

    conv = get_conversation_template(model_path)
    if conv.sep_style is None:
        conv.sep_style = SeparatorStyle.LLAMA2

    for message in messages:
        msg_role = message.role
        if msg_role == "system":
            conv.set_system_message(message.content)
        elif msg_role == "user":
            conv.append_message(conv.roles[0], message.content)
        elif msg_role == "assistant":
            conv.append_message(conv.roles[1], message.content)
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt
