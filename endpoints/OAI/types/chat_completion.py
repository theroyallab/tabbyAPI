from pydantic import BaseModel, Field
from time import time
from typing import Union, List, Optional, Dict
from uuid import uuid4

from endpoints.OAI.types.common import UsageStats, CommonCompletionRequest
from endpoints.OAI.types.tools import Tool, default_tool_call_schema

class ChatCompletionLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[List["ChatCompletionLogprob"]] = None


class ChatCompletionLogprobs(BaseModel):
    content: List[ChatCompletionLogprob] = Field(default_factory=list)


class ChatCompletionMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None # WIP # WIP Could match this to the OAI type in 
                                            # \Lib\site-packages\openai\types\chat\chat_completion_message_tool_call_param.py

class ChatCompletionRespChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None
    stop_str: Optional[str] = None # let's us understand why it stopped and if we need to generate a tool_call
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
    messages: Union[str, List[Dict]] # WIP this can probably be tightened, or maybe match the OAI lib type 
                                     # in \Lib\site-packages\openai\types\chat\chat_completion_message_param.py
    prompt_template: Optional[str] = None
    add_generation_prompt: Optional[bool] = True
    template_vars: Optional[dict] = {}
    response_prefix: Optional[str] = None

    # Tools is follows the format OAI schema, functions allows a list of function schemas to be passed. Chat template will determine which is used.
    tools: Optional[List[Dict]] = None # WIP Could match OAI Type here too, undecided.
    functions: Optional[List[Dict]] = None

    # Typically collected from Chat Template.
    tool_call_start: Optional[str] = None # string/token that precedes tool calls
    tool_call_end: Optional[str] = None # string/token that might be placed after the tool calls
    tool_call_schema: Optional[dict] = default_tool_call_schema # schema to be used when generating tool calls


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
