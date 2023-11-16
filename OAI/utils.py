import pathlib
from fastchat.model.model_adapter import get_conversation_template, Conversation
from OAI.types.completion import CompletionResponse, CompletionRespChoice
from OAI.types.chat_completion import (
    ChatCompletionMessage,
    ChatCompletionRespChoice,
    ChatCompletionStreamChunk,
    ChatCompletionResponse,
    ChatCompletionStreamChoice
)
from OAI.types.common import UsageStats
from OAI.types.model import ModelList, ModelCard
from typing import Optional, List

def create_completion_response(text: str, model_name: Optional[str]):
    # TODO: Add method to get token amounts in model for UsageStats

    choice = CompletionRespChoice(
        finish_reason = "Generated",
        text = text
    )

    response = CompletionResponse(
        choices = [choice],
        model = model_name or ""
    )

    return response

def create_chat_completion_response(text: str, model_name: Optional[str]):
    # TODO: Add method to get token amounts in model for UsageStats

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
        model = model_name or ""
    )

    return response

def create_chat_completion_stream_chunk(const_id: str, text: str, model_name: Optional[str]):
    # TODO: Add method to get token amounts in model for UsageStats

    message = ChatCompletionMessage(
        role = "assistant",
        content = text
    )

    choice = ChatCompletionStreamChoice(
        finish_reason = "Generated",
        delta = message
    )

    chunk = ChatCompletionStreamChunk(
        id = const_id,
        choices = [choice],
        model = model_name or ""
    )

    return chunk

def get_model_list(model_path: pathlib.Path):
    model_card_list = ModelList()
    for path in model_path.iterdir():
        if path.is_dir():
            model_card = ModelCard(id = path.name)
            model_card_list.data.append(model_card)

    return model_card_list

def get_chat_completion_prompt(model_path: str, messages: List[ChatCompletionMessage]):
    conv = get_conversation_template(model_path)
    for message in messages:
        msg_role = message.role
        if msg_role == "system":
            conv.system_message = message.content
        elif msg_role == "user":
            conv.append_message(conv.roles[0], message.content)
        elif msg_role == "assistant":
            conv.append_message(conv.roles[1], message.content)
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt
