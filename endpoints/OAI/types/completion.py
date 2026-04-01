"""Completion API protocols"""

from pydantic import BaseModel, Field
from time import time
from typing import Dict, List, Optional, Union
from uuid import uuid4

from endpoints.OAI.types.chat_completion import ChatCompletionLogprobs
from endpoints.OAI.types.common import CommonCompletionRequest, UsageStats


class CompletionLogprobs(BaseModel):
    tokens: List[str] = Field(default_factory=list)
    token_logprobs: List[float] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)
    text_offset: List[int] = Field(default_factory=list)


def chat_logprobs_to_completion_logprobs(
    chat_logprobs: ChatCompletionLogprobs,
) -> CompletionLogprobs:
    tokens = []
    token_logprobs = []
    top_logprobs = []
    text_offset = []
    offset = 0

    for entry in chat_logprobs.content:
        tokens.append(entry.token)
        token_logprobs.append(entry.logprob)
        top_logprobs.append(
            {tp.token: tp.logprob for tp in entry.top_logprobs} if entry.top_logprobs else None
        )
        text_offset.append(offset)
        offset += len(entry.token)

    return CompletionLogprobs(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=text_offset,
    )


class CompletionRespChoice(BaseModel):
    """Represents a single choice in a completion response."""

    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None
    logprobs: Optional[CompletionLogprobs] = None
    text: str


# Inherited from common request
class CompletionRequest(CommonCompletionRequest):
    """Represents a completion request."""

    # Prompt can also contain token ids, but that's out of scope
    # for this project.
    prompt: Union[str, List[str]]


class CompletionResponse(BaseModel):
    """Represents a completion response."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid4().hex}")
    choices: List[CompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "text_completion"
    usage: Optional[UsageStats] = None
