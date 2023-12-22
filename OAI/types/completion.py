""" Completion API protocols """
from time import time
from typing import List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from OAI.types.common import CommonCompletionRequest, LogProbs, UsageStats


class CompletionRespChoice(BaseModel):
    """Represents a single choice in a completion response."""

    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: str
    logprobs: Optional[LogProbs] = None
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
