from pydantic import BaseModel, Field
from time import time
from typing import Union, List, Optional, Dict
from uuid import uuid4

from endpoints.OAI.types.common import UsageStats, CommonCompletionRequest
from endpoints.OAI.types.tools import ToolSpec, ToolCall, openai_tool_call_schema

class ChatCompletionLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[List["ChatCompletionLogprob"]] = None


class ChatCompletionLogprobs(BaseModel):
    content: List[ChatCompletionLogprob] = Field(default_factory=list)


class ChatCompletionMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    


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
    messages: Union[
        str, List
    ]
    prompt_template: Optional[str] = None
    add_generation_prompt: Optional[bool] = True
    template_vars: Optional[dict] = {}
    response_prefix: Optional[str] = None

    # Tools is follows the format OAI schema, functions allows a list of function
    # schemas to be passed. Chat template will determine which is used.

    # TODO ensure tools matches the oai format then ensure in util/chat_comp its
    # properly converted to json/str for template
    tools: Optional[List[ToolSpec]] = None
    functions: Optional[List[Dict]] = None

    # Typically collected from Chat Template.
    tool_call_start: Optional[List[Union[str, int]]] = None  # toks or strs preceeding
    # string that might be placed after the tool calls
    tool_call_end: Optional[str] = None
    # schema to be used when generating tool calls
    tool_call_schema: Optional[dict] = openai_tool_call_schema


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
