""" Completion API protocols """
from pydantic import BaseModel, Field
from time import time
from typing import Dict, List, Optional, Union
from uuid import uuid4

from OAI.types.common import CommonCompletionRequest, UsageStats


class CompletionLogProbs(BaseModel):
    """Represents log probabilities for a completion request."""

    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CompletionRespChoice(BaseModel):
    """Represents a single choice in a completion response."""

    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: str
    logprobs: Optional[CompletionLogProbs] = None
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
