from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema
from time import time
from typing import Literal, Union, List, Optional, Dict
from uuid import uuid4

from endpoints.OAI.types.common import UsageStats, CommonCompletionRequest
from endpoints.OAI.types.tools import ToolSpec, ToolCall, tool_call_schema


class ChatCompletionLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[List["ChatCompletionLogprob"]] = None


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
    tool_calls_json: SkipJsonSchema[Optional[str]] = None


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
    # Messages
    # Take in a string as well even though it's not part of the OAI spec
    # support messages.content as a list of dict

    # WIP this can probably be tightened, or maybe match the OAI lib type
    # in openai\types\chat\chat_completion_message_param.py
    messages: List[ChatCompletionMessage] = Field(default_factory=list)
    prompt_template: Optional[str] = None
    add_generation_prompt: Optional[bool] = True
    template_vars: Optional[dict] = {}
    response_prefix: Optional[str] = None
    model: Optional[str] = None

    # tools is follows the format OAI schema, functions is more flexible
    # both are available in the chat template.

    tools: Optional[List[ToolSpec]] = None
    functions: Optional[List[Dict]] = None

    # Typically collected from Chat Template.
    # Don't include this in the OpenAPI docs
    # TODO: Use these custom parameters
    tool_call_start: SkipJsonSchema[Optional[List[Union[str, int]]]] = None
    tool_call_end: SkipJsonSchema[Optional[str]] = None
    tool_call_schema: SkipJsonSchema[Optional[dict]] = tool_call_schema

    # Chat completions requests do not have a BOS token preference. Backend
    # respects the tokenization config for the individual model.
    add_bos_token: Optional[bool] = Field(default=None)

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
