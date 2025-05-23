from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Union, Literal
from time import time
from uuid import uuid4

from endpoints.OAI.types.common import CommonCompletionRequest, UsageStats


class LogProbRequest(CommonCompletionRequest):
    """Request for log probability calculation."""

    prompt: str

    # Override and constrain max_tokens (0 by default for logprob-only)
    max_tokens: Optional[int] = Field(
        default=0,
        ge=0,
        le=0,
        description="Must be 0. This endpoint only calculates logprobs for the prompt tokens.",
    )

    # `n` is kept for compatibility with the OpenAI API, but logprob
    # calculation is deterministic and only a single choice is useful.
    # Constrain the field so a request can't ask for more than one.
    n: Optional[int] = Field(
        default=1,
        ge=1,
        le=1,
        description="Number of choices to return. Only a value of 1 is supported.",
    )

    # This will be handled internally, not exposed as a user parameter
    # but added to the sampler
    stream: Optional[bool] = Field(
        default=False,
        description="Not currently supported for logprob endpoint",
    )


class TokenLogProbs(BaseModel):
    """Log probabilities for tokens in the prompt."""

    tokens: List[str]
    token_logprobs: List[Optional[float]]
    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None
    text_offset: Optional[List[int]] = None
    sum: float

    @field_validator("sum")
    def validate_sum(cls, v, values):
        """Validate that sum matches the sum of token_logprobs."""
        if "token_logprobs" in values.data:
            # Calculate sum ignoring None values (first token has no prior context)
            calculated_sum = sum(
                lp for lp in values.data["token_logprobs"] if lp is not None
            )
            if abs(v - calculated_sum) > 1e-6:
                return calculated_sum
        return v

    @model_validator(mode="after")
    def validate_lengths(cls, values):
        """Ensure all arrays are the same length."""
        tokens_len = len(values.tokens)
        if not (
            tokens_len
            == len(values.token_logprobs)
            == len(values.top_logprobs or [])
            == len(values.text_offset or [])
        ):
            raise ValueError(
                "tokens, token_logprobs, top_logprobs and text_offset must have the same length"
            )
        return values


class LogProbChoice(BaseModel):
    """A single logprob choice."""

    logprobs: TokenLogProbs
    index: int = 0


class LogProbResponse(BaseModel):
    """Response for log probability calculation."""

    id: str = Field(default_factory=lambda: f"logprob-{uuid4().hex}")
    choices: List[LogProbChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: Literal["logprob"] = Field(default="logprob")
    usage: Optional[UsageStats] = None
