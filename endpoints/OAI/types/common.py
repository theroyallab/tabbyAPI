"""Common types for OAI."""

from pydantic import BaseModel, Field
from typing import Optional

from common.sampling import BaseSamplerRequest, get_default_sampler_value


class UsageStats(BaseModel):
    """Represents usage stats."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponseFormat(BaseModel):
    type: str = "text"


class CommonCompletionRequest(BaseSamplerRequest):
    """Represents a common completion request."""

    # Model information
    # This parameter is not used, the loaded model is used instead
    model: Optional[str] = None

    # Generation info (remainder is in BaseSamplerRequest superclass)
    stream: Optional[bool] = False
    logprobs: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("logprobs", 0)
    )
    response_format: Optional[CompletionResponseFormat] = Field(
        default_factory=CompletionResponseFormat
    )
    n: Optional[int] = Field(default_factory=lambda: get_default_sampler_value("n", 1))

    # Extra OAI request stuff
    best_of: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    echo: Optional[bool] = Field(
        description="Not parsed. Only used for OAI compliance.", default=False
    )
    suffix: Optional[str] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    user: Optional[str] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )

    def validate_params(self):
        # Temperature
        if self.n < 1:
            raise ValueError(f"n must be greater than or equal to 1. Got {self.n}")

        return super().validate_params()

    def to_gen_params(self):
        extra_gen_params = {
            "stream": self.stream,
            "logprobs": self.logprobs,
        }

        return super().to_gen_params(**extra_gen_params)
