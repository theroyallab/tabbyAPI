"""Completion API protocols"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Union, Literal
from time import time
from uuid import uuid4

from endpoints.OAI.types.common import CommonCompletionRequest, UsageStats


class CompletionLogProbs(BaseModel):
    """Represents log probabilities for a completion request."""

    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lengths(cls, values):
        """Ensure all arrays are the same length."""
        tokens_len = len(values.tokens)
        if not (
            tokens_len
            == len(values.token_logprobs)
            == len(values.top_logprobs)
            == len(values.text_offset)
        ):
            raise ValueError(
                "tokens, token_logprobs, top_logprobs and text_offset must have the same length"
            )
        return values


class CompletionRespChoice(BaseModel):
    """Represents a single choice in a completion response."""

    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None
    logprobs: Optional[CompletionLogProbs] = None
    text: str


# Inherited from common request
class CompletionRequest(CommonCompletionRequest):
    """Represents a completion request."""

    # Prompt may contain token IDs as well as raw strings. Accept a nested
    # list of token IDs for compatibility with clients that wrap the token
    # array in an additional list (e.g. ``[[1, 2, 3]]``).
    prompt: Union[str, List[str], List[int], List[List[int]]]


class CompletionResponse(BaseModel):
    """Represents a completion response."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid4().hex}")
    choices: List[CompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: Literal["text_completion"] = Field(default="text_completion")
    usage: Optional[UsageStats] = None
