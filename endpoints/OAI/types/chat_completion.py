from pydantic import AliasChoices, BaseModel, Field, field_validator
from time import time
from typing import Literal, Union, List, Optional, Dict
from uuid import uuid4

from endpoints.OAI.types.common import UsageStats, CommonCompletionRequest
from endpoints.OAI.types.tools import ToolSpec, ToolCall


class ChatCompletionLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[List["ChatCompletionLogprob"]] = Field(default_factory=list)


class ChatCompletionLogprobs(BaseModel):
    content: List[ChatCompletionLogprob] = Field(default_factory=list)


class ChatCompletionImageUrl(BaseModel):
    url: str


class ChatCompletionMessagePart(BaseModel):
    type: Literal["text", "image_url"] = "text"
    text: Optional[str] = None
    image_url: Optional[ChatCompletionImageUrl] = None


class ChatCompletionMessage(BaseModel):
    role: str = "user"
    content: Optional[Union[str, List[ChatCompletionMessagePart]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRespChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None

    # let's us understand why it stopped and if we need to generate a tool_call
    stop_str: Optional[str] = None
    message: ChatCompletionMessage
    logprobs: Optional[ChatCompletionLogprobs] = None


class ChatCompletionStreamChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None
    delta: Union[ChatCompletionMessage, dict] = {}
    logprobs: Optional[ChatCompletionLogprobs] = None


# Inherited from common request
class ChatCompletionRequest(CommonCompletionRequest):
    messages: List[ChatCompletionMessage]
    prompt_template: Optional[str] = None
    add_generation_prompt: Optional[bool] = True
    template_vars: Optional[dict] = Field(
        default={},
        validation_alias=AliasChoices("template_vars", "chat_template_kwargs"),
        description="Aliases: chat_template_kwargs",
    )
    response_prefix: Optional[str] = None
    model: Optional[str] = None

    # tools is follows the format OAI schema, functions is more flexible
    # both are available in the chat template.

    tools: Optional[List[ToolSpec]] = None
    functions: Optional[List[Dict]] = None

    # Chat completions requests do not have a BOS token preference. Backend
    # respects the tokenization config for the individual model.
    add_bos_token: Optional[bool] = None

    @field_validator("add_bos_token", mode="after")
    def force_bos_token(cls, v):
        """Always disable add_bos_token with chat completions."""
        return None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    choices: List[ChatCompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "chat.completion"
    usage: Optional[UsageStats] = None


class ChatCompletionStreamChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    choices: List[ChatCompletionStreamChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "chat.completion.chunk"
    usage: Optional[UsageStats] = None
