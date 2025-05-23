"""Common types for OAI."""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Union

from common.sampling import BaseSamplerRequest, get_default_sampler_value


class UsageStats(BaseModel):
    """Represents usage stats."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponseFormat(BaseModel):
    type: str = "text"


class ChatCompletionStreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class CommonCompletionRequest(BaseSamplerRequest):
    """Represents a common completion request."""

    # Model information
    # This parameter is not used, the loaded model is used instead
    model: Optional[str] = None

    # Generation info (remainder is in BaseSamplerRequest superclass)
    stream: Optional[bool] = False
    stream_options: Optional[ChatCompletionStreamOptions] = None
    response_format: Optional[CompletionResponseFormat] = Field(
        default_factory=CompletionResponseFormat
    )
    n: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("n", 1),
        ge=1,
    )
    max_tokens: Optional[int] = Field(
        default=16,
        ge=0,
        description="The maximum number of tokens to generate. Defaults to 16.",
    )
    logprobs: Optional[Union[bool, int]] = Field(
        default=None,
        description="Whether to include log probabilities. Can be boolean or integer (1-5 for compatibility)"
    )
    top_logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Number of most likely tokens to return per position (0-5)"
    )

    # Extra OAI request stuff
    best_of: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.",
        default=None,
        ge=1,
    )
    echo: Optional[bool] = Field(
        description="Echo the prompt in the response.", default=False
    )
    suffix: Optional[str] = Field(
        description=(
            "Optional text to append after the completion. Currently ignored as "
            "insertion is not supported."
        ),
        default=None,
    )
    user: Optional[str] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )

    @model_validator(mode="after")
    def check_best_of(cls, data):
        """Validate unsupported best_of combinations."""
        best_of = data.best_of
        n = data.n or 1
        if best_of is not None:
            if data.stream:
                raise ValueError("best_of cannot be used with streaming")
            if best_of > 1:
                raise ValueError("best_of greater than 1 is not supported")
            if best_of < n:
                raise ValueError("best_of must be greater than or equal to n")
        return data