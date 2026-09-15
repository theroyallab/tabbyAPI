"""Common types for OAI."""

from pydantic import BaseModel, Field, model_serializer
from typing import Optional, Union

from common.sampling import BaseSamplerRequest, get_default_sampler_value


class PromptTokensDetails(BaseModel):
    """OpenAI-style prompt token details: how much of the prompt came from the cache."""

    cached_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    """
    OpenAI-style completion token details. Speculative decoding is reported in the
    Predicted Outputs fields: draft tokens the model confirmed or discarded. Both are
    0 without a draft model. completion_tokens counts only emitted tokens, so rejected
    drafts are never double counted.
    """

    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class UsageStats(BaseModel):
    """Represents usage stats."""

    prompt_tokens: int
    # Always present, like OpenAI's, so clients never see null here
    prompt_tokens_details: PromptTokensDetails = Field(default_factory=PromptTokensDetails)
    prompt_time: Optional[float] = None
    prompt_tokens_per_sec: Optional[Union[float, str]] = None
    completion_tokens: int
    completion_tokens_details: CompletionTokensDetails = Field(
        default_factory=CompletionTokensDetails
    )
    completion_time: Optional[float] = None
    completion_tokens_per_sec: Optional[Union[float, str]] = None
    total_tokens: int
    total_time: Optional[float] = None


class Timings(BaseModel):
    """
    llama-server compatible generation timings (llama.cpp server_slot_stats::to_json).
    The draft keys are only present when a draft model ran, like llama.cpp, which
    sets them for drafted generations only.
    """

    cache_n: int
    prompt_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float
    draft_n: Optional[int] = None
    draft_n_accepted: Optional[int] = None

    # Absent, not null: draft_n=None means no draft ran, and the JSON carries no key
    @model_serializer(mode="wrap")
    def _drop_unset_draft_keys(self, handler):
        return {key: value for key, value in handler(self).items() if value is not None}


class CompletionResponseFormat(BaseModel):
    type: str = "text"
    json_schema: Optional[object] = None


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
