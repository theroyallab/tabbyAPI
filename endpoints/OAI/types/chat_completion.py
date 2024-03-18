from pydantic import BaseModel, Field
from time import time
from typing import Union, List, Optional, Dict
from uuid import uuid4

from endpoints.OAI.types.common import UsageStats, CommonCompletionRequest


class ChatCompletionLogprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: Optional[List["ChatCompletionLogprob"]] = None


class ChatCompletionLogprobs(BaseModel):
    content: List[ChatCompletionLogprob] = Field(default_factory=list)


class ChatCompletionMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionRespChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None
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
    messages: Union[str, List[Dict[str, str]]]
    prompt_template: Optional[str] = None
    add_generation_prompt: Optional[bool] = True


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
