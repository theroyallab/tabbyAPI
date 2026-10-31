"""Completion API protocols"""

from pydantic import BaseModel, Field
from time import time
from typing import Dict, List, Optional, Union
from uuid import uuid4

from common.sampling import BaseSamplerRequest
from endpoints.OAI.types.chat_completion import ChatCompletionLogprobs
from endpoints.OAI.types.common import CommonCompletionRequest, Timings, UsageStats


class CompletionLogprobs(BaseModel):
    tokens: List[str] = Field(default_factory=list)
    token_logprobs: List[float] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)
    text_offset: List[int] = Field(default_factory=list)


def chat_logprobs_to_completion_logprobs(
    chat_logprobs: ChatCompletionLogprobs,
) -> CompletionLogprobs:
    tokens = []
    token_logprobs = []
    top_logprobs = []
    text_offset = []
    offset = 0

    for entry in chat_logprobs.content:
        tokens.append(entry.token)
        token_logprobs.append(entry.logprob)
        top_logprobs.append(
            {tp.token: tp.logprob for tp in entry.top_logprobs} if entry.top_logprobs else None
        )
        text_offset.append(offset)
        offset += len(entry.token)

    return CompletionLogprobs(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        text_offset=text_offset,
    )


class CompletionRespChoice(BaseModel):
    """Represents a single choice in a completion response."""

    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str] = None
    eos_reason: Optional[str] = None
    stop_str: Optional[str] = None
    logprobs: Optional[CompletionLogprobs] = None
    text: str


# Inherited from common request
class CompletionRequest(CommonCompletionRequest):
    """Represents a completion request."""

    # Prompt can also contain token ids, but that's out of scope
    # for this project.
    prompt: Union[str, List[str]]


class ContinuationsRequest(BaseSamplerRequest):
    """Asks which ways a prompt can go on under the given sampler settings."""

    prompt: str
    max_options: int = Field(
        default=10, ge=1, le=64, description="The most likely next tokens to follow, at most"
    )
    max_branch_tokens: int = Field(
        default=20,
        ge=1,
        le=256,
        description=(
            "Length to follow each one for, as long as a single token keeps passing "
            "truncation. An incomplete character can take a few tokens more."
        ),
    )


class ContinuationOption(BaseModel):
    """One way a prompt can go on, up to its next branch point."""

    text: str
    first: str = Field(description="The part of text that came from the first token")
    token_ids: List[int]
    healed: bool = Field(
        default=False,
        description=(
            "Whether the first token takes the place of the prompt's last token, by token "
            "healing. The text only has what it adds."
        ),
    )
    prob: float = Field(
        description="Probability of a generation starting this way under the sampler settings"
    )
    end: str = Field(
        description=(
            "Why the path ends: branch (several tokens pass truncation next), stop (a "
            "stop token, included in the text), limit (max_branch_tokens or the context)"
        )
    )


class ContinuationsResponse(BaseModel):
    """The ways a prompt can go on, likeliest first."""

    options: List[ContinuationOption]
    candidates: int = Field(description="Tokens that passed truncation, before max_options")
    truncated: bool = Field(
        description="Whether the sampler settings truncate at all. Otherwise every token branches."
    )
    prompt_tokens: int
    cached_tokens: int
    generated_tokens: int
    time: float = Field(description="Seconds taken")


class CompletionResponse(BaseModel):
    """Represents a completion response."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid4().hex}")
    choices: List[CompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "text_completion"
    usage: Optional[UsageStats] = None
    timings: Optional[Timings] = None
