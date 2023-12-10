from uuid import uuid4
from time import time
from pydantic import BaseModel, Field
from typing import Union, List, Optional
from OAI.types.common import UsageStats, CommonCompletionRequest

class ChatCompletionMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionRespChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: str
    message: ChatCompletionMessage

class ChatCompletionStreamChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str]
    delta: Union[ChatCompletionRespChoice, dict] = {}

# Inherited from common request
class ChatCompletionRequest(CommonCompletionRequest):
    # Messages
    # Take in a string as well even though it's not part of the OAI spec
    messages: Union[str, List[ChatCompletionMessage]]
    prompt_template: Optional[str] = None

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
