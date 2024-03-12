""" Common types for OAI. """
from pydantic import BaseModel, Field
from typing import Optional

from common.sampling import BaseSamplerRequest


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

    # Generation info (remainder is in BaseSamplerRequest superclass)
    stream: Optional[bool] = False
    logprobs: Optional[int] = 0

    # Extra OAI request stuff
    best_of: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    echo: Optional[bool] = Field(
        description="Not parsed. Only used for OAI compliance.", default=False
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

    def to_gen_params(self):
        extra_gen_params = {"logprobs": self.logprobs}

        return super().to_gen_params(**extra_gen_params)
