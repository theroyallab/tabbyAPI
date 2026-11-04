"""
Types for the decision endpoints.

/v1/decisions follows OpenAI's Decisions API (POST /v1/decisions, `client.decisions.create`
in the OpenAI SDK): shared `input`, a list of typed `questions`, and an `answers` list in
question order. /v1/systemone follows TypeSafe's System One API as llama.cpp and vLLM
serve it: a `state`, a name-keyed `questions` map, and a name-keyed `answers` map. Both
are answered by the same label read (endpoints/OAI/utils/decisions.py).
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, model_validator

# Labels are single letters, so a choice is limited to 26 options
MAX_CHOICES = 26
MIN_CHOICES = 2
MIN_LEVELS = 2
MAX_LEVELS = 10
MAX_QUESTIONS = 64

ChoiceValue = Union[StrictStr, StrictBool]


class DecisionModel(BaseModel):
    """Unknown fields are rejected, as the OpenAI API does."""

    model_config = ConfigDict(extra="forbid")


# -- OpenAI Decisions API: request


class DecisionInputText(DecisionModel):
    type: Literal["input_text"]
    text: StrictStr


class DecisionInputMessage(DecisionModel):
    role: Literal["user"]
    content: Union[StrictStr, List[DecisionInputText]]
    type: Literal["message"] = "message"

    @model_validator(mode="before")
    @classmethod
    def reject_images(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and isinstance(data.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "input_image"
                for part in data["content"]
            )
        ):
            raise ValueError("Decisions supports text input only; images are not supported")
        return data


class QuestionBase(DecisionModel):
    instructions: StrictStr
    name: Optional[StrictStr] = None

    @model_validator(mode="after")
    def validate_name(self):
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name must be a string when provided")
        return self


class PredicateQuestion(QuestionBase):
    type: Literal["predicate"]


class ChoiceOption(DecisionModel):
    value: ChoiceValue
    description: StrictStr = ""


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"]
    choices: List[ChoiceOption] = Field(min_length=MIN_CHOICES, max_length=MAX_CHOICES)

    @model_validator(mode="after")
    def validate_choices(self):
        # Values are typed: the string "true" and the boolean true are distinct
        values = [(type(c.value), c.value) for c in self.choices]
        if len(set(values)) != len(values):
            raise ValueError("choice values must be distinct")
        return self


class ScoreLevel(DecisionModel):
    label: StrictStr
    description: StrictStr = ""


class ScoreQuestion(QuestionBase):
    type: Literal["score"]
    levels: List[ScoreLevel] = Field(min_length=MIN_LEVELS, max_length=MAX_LEVELS)


DecisionQuestion = Annotated[
    Union[PredicateQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")
]


class DecisionRequest(DecisionModel):
    # OpenAI requires the model; here it selects an inline load like the other
    # routes and may be left out for the loaded model
    model: Optional[StrictStr] = None
    input: Union[StrictStr, List[DecisionInputMessage]]
    questions: List[DecisionQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)
    # Caller metadata in the OpenAI API; accepted and ignored
    safety_identifier: Optional[Annotated[StrictStr, Field(max_length=128)]] = None


# -- OpenAI Decisions API: response


class PredicateAnswer(DecisionModel):
    type: Literal["predicate"] = "predicate"
    name: Optional[str]
    probability: float


class ChoiceProbability(DecisionModel):
    value: ChoiceValue
    probability: float


class ChoiceAnswer(DecisionModel):
    type: Literal["choice"] = "choice"
    name: Optional[str]
    choice: ChoiceValue
    probabilities: List[ChoiceProbability]
    confidence: float


class ScoreProbability(DecisionModel):
    value: int
    label: str
    probability: float


class ScoreAnswer(DecisionModel):
    type: Literal["score"] = "score"
    name: Optional[str]
    score: float
    probabilities: List[ScoreProbability]
    confidence: float


DecisionAnswer = Annotated[
    Union[PredicateAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")
]


class InputTokensDetails(DecisionModel):
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class OutputTokensDetails(DecisionModel):
    reasoning_tokens: int = 0


class DecisionUsage(DecisionModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens: int
    output_tokens_details: OutputTokensDetails = Field(default_factory=OutputTokensDetails)
    total_tokens: int


class DecisionResponse(DecisionModel):
    model: str
    answers: List[DecisionAnswer]
    usage: DecisionUsage


# -- TypeSafe System One API (/v1/systemone)


class SystemOneQuestion(BaseModel):
    """A question as System One takes it: `type` choice, score or noul, free-form
    `instructions`, and `criteria` whose shape depends on the type (an option-to-
    description map, an ordered list of level descriptions, or the descriptions of
    true and false)."""

    type: str
    instructions: Any = ""
    criteria: Any = None


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    state: Any = Field(..., description="What the questions are about: text or JSON.")
    questions: Dict[str, SystemOneQuestion] = Field(
        ..., min_length=1, description="Question id to question, asked in this order."
    )
    instructions: Optional[str] = Field(
        default=None, description="Context placed ahead of the questions."
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = None


class SystemOneUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class QuestionDiagnostics(BaseModel):
    label_mass: float = Field(
        description="Probability the model put on the labels, over the full vocabulary. "
        "A low value means most of the probability went to tokens that are not labels."
    )
    argmax_is_label: bool


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, Dict[str, Any]]
    usage: SystemOneUsage
    diagnostics: Dict[str, QuestionDiagnostics]
