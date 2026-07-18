from pydantic import AliasChoices, BaseModel, Field, field_validator
from time import time
from typing import Literal, Union, List, Optional, Dict
from uuid import uuid4

from endpoints.OAI.types.common import UsageStats, CommonCompletionRequest
from endpoints.OAI.types.tools import NamedToolChoice, ToolSpec, ToolCall


class ChatCompletionLogprobLeaf(BaseModel):
    token: str
    token_id: Optional[int] = None  # not standard but widely adopted
    logprob: float


class ChatCompletionLogprob(BaseModel):
    token: str
    token_id: Optional[int] = None  # not standard but widely adopted
    logprob: float
    top_logprobs: Optional[List[ChatCompletionLogprobLeaf]] = Field(default_factory=list)


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
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRespChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None
    eos_reason: Optional[str] = None

    # Lets us understand why it stopped and if we need to generate a tool_call
    stop_str: Optional[str] = None
    message: ChatCompletionMessage
    logprobs: Optional[ChatCompletionLogprobs] = None


class ChatCompletionStreamChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None
    eos_reason: Optional[str] = None
    delta: Union[ChatCompletionMessage, dict] = {}
    logprobs: Optional[ChatCompletionLogprobs] = None


class ReasoningOptions(BaseModel):
    """
    OpenRouter / OpenAI Responses style reasoning options. Unknown keys are
    ignored. The flat top-level reasoning_effort and enable_thinking fields
    take precedence over the equivalent keys here.
    """

    effort: Optional[str] = None
    enabled: Optional[bool] = None
    max_tokens: Optional[int] = None


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
    reasoning_effort: Optional[str] = Field(
        default=None,
        description=(
            "Reasoning effort hint, forwarded to the chat template as the "
            "reasoning_effort variable. Accepted values depend on the model. "
            "An explicit reasoning_effort in template_vars takes precedence."
        ),
    )
    enable_thinking: Optional[bool] = Field(
        default=None,
        description=(
            "Reasoning toggle, forwarded to the chat template as the "
            "enable_thinking variable. An explicit enable_thinking in "
            "template_vars takes precedence."
        ),
    )
    verbosity: Optional[str] = Field(
        default=None,
        description=(
            "Response verbosity hint, forwarded to the chat template as the "
            "verbosity variable. An explicit verbosity in template_vars "
            "takes precedence."
        ),
    )
    reasoning: Optional[ReasoningOptions] = Field(
        default=None,
        description=(
            "OpenRouter / OpenAI Responses style reasoning options. "
            "reasoning.effort and reasoning.enabled map to the reasoning_effort "
            "and enable_thinking template variables; the flat top-level fields "
            "take precedence. reasoning.max_tokens is not supported and is "
            "ignored."
        ),
    )
    response_prefix: Optional[str] = None

    continue_final_message: Optional[bool] = False
    model: Optional[str] = None

    # tools is follows the format OAI schema, functions is more flexible
    # both are available in the chat template.

    tools: Optional[List[ToolSpec]] = None
    functions: Optional[List[Dict]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], NamedToolChoice]] = None
    parallel_tool_calls: Optional[bool] = True

    # Chat completions requests do not have a BOS token preference. Backend
    # respects the tokenization config for the individual model.
    add_bos_token: Optional[bool] = None

    # Accept json_schema as top-level argument
    json_schema: Optional[object] = None

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
