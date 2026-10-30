from typing import List, Optional

from pydantic import BaseModel, Field


class StatusJob(BaseModel):
    """A generation job, in progress or recently finished."""

    request_id: str
    label: str
    stage: str = Field(
        description=(
            "queued, prefill, generating, waiting (requeued, until there is room in "
            "the cache), resuming (taken back after a requeue) or finished"
        )
    )
    prompt_tokens: int
    cached_tokens: int
    prefill_tokens: int = Field(description="Prompt tokens processed so far, cached ones included")
    gen_tokens: int
    resume_tokens: int
    resume_total: int
    tokens_per_second: Optional[float] = None
    duration: float = Field(
        description="Seconds since the request was queued, or until it finished"
    )
    waiting_for: Optional[float] = Field(
        default=None, description="Seconds since the last token, for a waiting job"
    )
    text_length: int = Field(description="Characters generated so far")
    finish_reason: Optional[str] = None
    can_cancel: bool = Field(
        description="Whether the job is in progress and the key asking can cancel it (admin only)"
    )


class StatusResponse(BaseModel):
    """Live server status: generation jobs, cache statistics and token counters."""

    jobs: List[StatusJob]
    recent: List[StatusJob] = Field(description="Recently finished jobs, newest first")
    completed: int = Field(description="Jobs finished since startup")
    generated_tokens: int = Field(description="Tokens generated since startup")
    prefilled_tokens: int = Field(description="Prompt tokens processed since startup, not cached")
    cache: Optional[dict] = Field(
        default=None, description="The engine's cache statistics, when a model is loaded"
    )
    backend: Optional[dict] = Field(
        default=None,
        description=(
            "The inference backend: name, version, and the git commit when it runs "
            "from a source checkout"
        ),
    )


class StatusJobText(BaseModel):
    """The text a job has generated, from an offset."""

    request_id: str
    offset: int
    text: str
    text_length: int
    finished: bool
    finish_reason: Optional[str] = None
    prompt: Optional[str] = Field(default=None, description="The prompt, when asked for")


class LoadOptionField(BaseModel):
    """One load option, for building a form."""

    name: str
    type: str = Field(
        description=(
            "bool, int, float, string, choice, list (of numbers), number_or_auto, "
            "text (free-form, e.g. layer_map) or json (a mapping)"
        )
    )
    default: Optional[object] = Field(
        default=None, description="What the backend uses when the option is left out"
    )
    description: Optional[str] = None
    choices: Optional[List[str]] = None
    free: bool = Field(default=False, description="Whether values outside choices are accepted")
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None


class LoadOptionGroup(BaseModel):
    name: str
    section: str = Field(description='"model", or "draft_model" for the draft_model block')
    fields: List[LoadOptionField]


class LoadOptionsSchemaResponse(BaseModel):
    """Every option a model load request can set, grouped."""

    groups: List[LoadOptionGroup]
