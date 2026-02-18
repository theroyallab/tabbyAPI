from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from time import time
from typing import Literal, Union, List, Optional, Dict
from uuid import uuid4

from endpoints.OAI.types.common import UsageStats, CommonCompletionRequest
from endpoints.OAI.types.tools import NamedToolChoice, ToolSpec, ToolCall


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
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None
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
    delta: Union[ChatCompletionMessage, dict] = Field(default_factory=dict)
    logprobs: Optional[ChatCompletionLogprobs] = None


# Inherited from common request
class ChatCompletionRequest(CommonCompletionRequest):
    messages: List[ChatCompletionMessage]
    prompt_template: Optional[str] = None
    add_generation_prompt: Optional[bool] = True
    template_vars: Optional[dict] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("template_vars", "chat_template_kwargs"),
        description="Aliases: chat_template_kwargs",
    )
    enable_thinking: Optional[bool] = None
    thinking: Optional[bool] = None
    response_prefix: Optional[str] = None
    model: Optional[str] = None
    include_reasoning: Optional[bool] = True

    # tools is follows the format OAI schema, functions is more flexible
    # both are available in the chat template.

    tools: Optional[List[ToolSpec]] = None
    functions: Optional[List[Dict]] = None
    tool_choice: Optional[
        Union[Literal["none", "auto", "required"], NamedToolChoice]
    ] = None
    parallel_tool_calls: Optional[bool] = True

    # Chat completions requests do not have a BOS token preference. Backend
    # respects the tokenization config for the individual model.
    add_bos_token: Optional[bool] = None

    @field_validator("add_bos_token", mode="after")
    def force_bos_token(cls, v):
        """Always disable add_bos_token with chat completions."""
        return None

    @model_validator(mode="after")
    def apply_thinking_aliases(self):
        """Support clients that send thinking flags at the top-level."""
        template_vars = dict(self.template_vars or {})

        if self.enable_thinking is not None and "enable_thinking" not in template_vars:
            template_vars["enable_thinking"] = self.enable_thinking

        if self.thinking is not None and "thinking" not in template_vars:
            template_vars["thinking"] = self.thinking

        self.template_vars = template_vars
        return self


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
