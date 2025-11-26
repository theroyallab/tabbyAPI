from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, field_validator
from time import time
from typing import Literal
from uuid import uuid4

from endpoints.OAI.types.common import UsageStats, CommonCompletionRequest
from endpoints.OAI.types.tools import ToolSpec, ToolCall


class ChatCompletionLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: list[ChatCompletionLogprob] | None = Field(default_factory=list)


class ChatCompletionLogprobs(BaseModel):
    content: list[ChatCompletionLogprob] = Field(default_factory=list)


class ChatCompletionImageUrl(BaseModel):
    url: str


class ChatCompletionMessagePart(BaseModel):
    type: Literal["text", "image_url"] = "text"
    text: str | None = None
    image_url: ChatCompletionImageUrl | None = None


class ChatCompletionMessage(BaseModel):
    role: str = "user"
    content: str | list[ChatCompletionMessagePart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRespChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: str | None = None

    # let's us understand why it stopped and if we need to generate a tool_call
    stop_str: str | None = None
    message: ChatCompletionMessage
    logprobs: ChatCompletionLogprobs | None = None


class ChatCompletionStreamChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: str | None = None
    delta: ChatCompletionMessage | dict = {}
    logprobs: ChatCompletionLogprobs | None = None


# Inherited from common request
class ChatCompletionRequest(CommonCompletionRequest):
    messages: list[ChatCompletionMessage]
    prompt_template: str | None = None
    add_generation_prompt: bool | None = True
    template_vars: dict | None = Field(
        default={},
        validation_alias=AliasChoices("template_vars", "chat_template_kwargs"),
        description="Aliases: chat_template_kwargs",
    )
    response_prefix: str | None = None
    model: str | None = None

    # tools is follows the format OAI schema, functions is more flexible
    # both are available in the chat template.

    tools: list[ToolSpec] | None = None
    functions: list[dict] | None = None

    # Chat completions requests do not have a BOS token preference. Backend
    # respects the tokenization config for the individual model.
    add_bos_token: bool | None = None

    @field_validator("add_bos_token", mode="after")
    def force_bos_token(cls, v):
        """Always disable add_bos_token with chat completions."""
        return None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    choices: list[ChatCompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "chat.completion"
    usage: UsageStats | None = None


class ChatCompletionStreamChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    choices: list[ChatCompletionStreamChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "chat.completion.chunk"
    usage: UsageStats | None = None
