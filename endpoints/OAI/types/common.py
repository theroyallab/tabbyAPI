"""Common types for OAI."""

from pydantic import BaseModel, Field

from common.sampling import BaseSamplerRequest, get_default_sampler_value


class UsageStats(BaseModel):
    """Represents usage stats."""

    prompt_tokens: int
    prompt_time: float | None = None
    prompt_tokens_per_sec: float | str | None = None
    completion_tokens: int
    completion_time: float | None = None
    completion_tokens_per_sec: float | str | None = None
    total_tokens: int
    total_time: float | None = None


class CompletionResponseFormat(BaseModel):
    type: str = "text"


class ChatCompletionStreamOptions(BaseModel):
    include_usage: bool | None = False


class CommonCompletionRequest(BaseSamplerRequest):
    """Represents a common completion request."""

    # Model information
    # This parameter is not used, the loaded model is used instead
    model: str | None = None

    # Generation info (remainder is in BaseSamplerRequest superclass)
    stream: bool | None = False
    stream_options: ChatCompletionStreamOptions | None = None
    response_format: CompletionResponseFormat | None = Field(
        default_factory=CompletionResponseFormat
    )
    n: int | None = Field(
        default_factory=lambda: get_default_sampler_value("n", 1),
        ge=1,
    )

    # Extra OAI request stuff
    best_of: int | None = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    echo: bool | None = Field(
        description="Not parsed. Only used for OAI compliance.", default=False
    )
    suffix: str | None = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    user: str | None = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
