""" Common types for OAI. """
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from common.sampling import BaseSamplerRequest


class LogProbs(BaseModel):
    """Represents log probabilities."""

    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class UsageStats(BaseModel):
    """Represents usage stats."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CommonCompletionRequest(BaseSamplerRequest):
    """Represents a common completion request."""

    # Model information
    # This parameter is not used, the loaded model is used instead
    model: Optional[str] = None

    # Extra OAI request stuff
    best_of: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    echo: Optional[bool] = Field(
        description="Not parsed. Only used for OAI compliance.", default=False
    )
    logprobs: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    n: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.", default=1
    )
    suffix: Optional[str] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    user: Optional[str] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )

    # Generation info (remainder is in BaseSamplerRequest superclass)
    stream: Optional[bool] = False
