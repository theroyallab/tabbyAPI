"""Types for the /v1/decisions endpoint (SGLang-compatible shape).

A decision endpoint answers typed questions with probabilities read from the
model's next-token distribution at the answer position. No text is generated.
"""

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

PROMPT_FORMAT_VERSION = 1

MIN_OPTIONS = 2
MAX_OPTIONS = 26
MIN_LEVELS = 2
MAX_LEVELS = 10


class DecisionOption(BaseModel):
    name: str
    description: Optional[str] = None


class ChoiceQuestion(BaseModel):
    """Pick one of 2..26 named options; labels A..Z in list order."""

    id: str
    type: Literal["choice"] = "choice"
    question: str
    options: List[DecisionOption]


class ScoreQuestion(BaseModel):
    """Rate on 2..10 ordered levels; labels 0..9, lowest level first."""

    id: str
    type: Literal["score"] = "score"
    question: str
    levels: List[str]


class YesNoQuestion(BaseModel):
    """Answer yes or no, with optional descriptions of each."""

    id: str
    type: Literal["yes_no"] = "yes_no"
    question: str
    yes_description: Optional[str] = None
    no_description: Optional[str] = None


class DecisionsRequest(BaseModel):
    # "input" is a Python keyword; keep the wire name
    input: Union[str, dict, list]
    questions: List[
        Annotated[
            Union[ChoiceQuestion, ScoreQuestion, YesNoQuestion],
            Field(discriminator="type"),
        ]
    ] = Field(
        ...,
        min_length=1,
    )
    # Divides the label logits before the softmax over labels. Does not change
    # label_mass. Sampling itself is unaffected: no text is generated.
    temperature: float = Field(default=1.0, gt=0)
    # Passthrough for the model's chat template, same semantics as on
    # /v1/chat/completions. Merged over the model's template_vars_default, so
    # a model that thinks by default still answers at the right position. The
    # answer is read directly after the rendered prompt; a template that
    # leaves a reasoning block open shifts the position and fails the label
    # token check with a 400.
    template_vars: Optional[dict] = None
    enable_thinking: Optional[bool] = None
    # Accepted for client convenience; selects an inline model like elsewhere.
    model: Optional[str] = None

    @field_validator("input", mode="after")
    @classmethod
    def input_not_blank(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("input must not be blank")
        return value


class AnswerChoice(BaseModel):
    type: Literal["choice"] = "choice"
    probabilities: dict[str, float]
    label_mass: float
    choice: str


class AnswerScore(BaseModel):
    type: Literal["score"] = "score"
    probabilities: dict[str, float]
    label_mass: float
    score: float


class AnswerYesNo(BaseModel):
    type: Literal["yes_no"] = "yes_no"
    probabilities: dict[str, float]
    label_mass: float


DecisionsAnswer = Union[AnswerChoice, AnswerScore, AnswerYesNo]


class DecisionsResponse(BaseModel):
    object: Literal["decisions"] = "decisions"
    model: str
    prompt_format_version: int = PROMPT_FORMAT_VERSION
    answers: dict[str, DecisionsAnswer]
    usage: dict[str, int]
