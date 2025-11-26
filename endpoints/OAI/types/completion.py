"""Completion API protocols"""

from pydantic import BaseModel, Field
from time import time
from uuid import uuid4

from endpoints.OAI.types.common import CommonCompletionRequest, UsageStats


class CompletionLogProbs(BaseModel):
    """Represents log probabilities for a completion request."""

    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[float | None] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[dict[str, float] | None] = Field(default_factory=list)


class CompletionRespChoice(BaseModel):
    """Represents a single choice in a completion response."""

    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: str | None = None
    logprobs: CompletionLogProbs | None = None
    text: str


# Inherited from common request
class CompletionRequest(CommonCompletionRequest):
    """Represents a completion request."""

    # Prompt can also contain token ids, but that's out of scope
    # for this project.
    prompt: str | list[str]


class CompletionResponse(BaseModel):
    """Represents a completion response."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid4().hex}")
    choices: list[CompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "text_completion"
    usage: UsageStats | None = None
