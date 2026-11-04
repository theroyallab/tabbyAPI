"""
Typed questions answered from the model's next-token distribution, serving
/v1/decisions (OpenAI's Decisions API) and /v1/systemone (TypeSafe's System One
API). No text is kept; one token is generated per question and discarded.

How a question is read, following vLLM's label read for generation models:

1. The question is rendered as one user message: the state, a blank line, then
   "Question: <instructions>", one line per option with its label
   ("A: value - description"), and an instruction to answer with one label.
   Labels are letters A..Z for a choice (and for the OpenAI route's predicate
   and score, which are choices over "false"/"true" and the level labels), the
   digits 0..9 for a System One score, and "yes"/"no" for a System One noul.
2. The prompt goes through the model's chat template with the generation prompt
   added and thinking turned off, and the model generates exactly one token.
3. The full-vocabulary logits at that position give each label's probability.
   A label counts every vocabulary token that spells it, with or without
   leading whitespace (" A" as well as "A", "Yes" as well as "yes"), so a model
   that puts its mass on the spaced variant is read correctly. The label sets
   come from one scan of the tokenizer's vocabulary, cached per model.
4. Softmax over the label log-probabilities gives the reported distribution;
   `label_mass` is the labels' total probability over the vocabulary, and
   `confidence` is the winning label's share of it.

The questions of a request are independent one-token jobs that run
concurrently; the engine batches their prefills and the page cache shares the
state they all start with.
"""

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from common import model
from common.networking import request_tag
from common.sampling import BaseSamplerRequest
from endpoints.OAI.types.chat_completion import ChatCompletionMessage
from endpoints.OAI.types.decisions import (
    MAX_CHOICES,
    MAX_LEVELS,
    ChoiceAnswer,
    ChoiceProbability,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    InputTokensDetails,
    PredicateAnswer,
    PredicateQuestion,
    QuestionDiagnostics,
    ScoreAnswer,
    ScoreProbability,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneUsage,
)
from endpoints.OAI.utils.chat_completion import format_messages_with_template

#: A choice labels its options A to Z in order
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGIT_LABELS = tuple("0123456789")
NOUL_LABELS = ("yes", "no")

#: Labels read case-insensitively: "Yes" and "YES" count for "yes"
_CASELESS_LABELS = frozenset(NOUL_LABELS)

#: Whitespace a label token may carry in front of (or after) the label itself
_LABEL_WHITESPACE = " \t"


class DecisionValidationError(ValueError):
    """A request that can never be answered (bad counts, unusable labels...)."""


# -- questions


@dataclass(frozen=True)
class Option:
    name: str
    description: Optional[str] = None


@dataclass(frozen=True)
class Question:
    """One rendered question: its options in order, the label each one answers
    with, and the type that writes the prompt."""

    id: str
    type: "QuestionType"
    instructions: str
    options: Tuple[Option, ...]
    labels: Tuple[str, ...]


class QuestionType:
    """How a question is written for the model: one labeled line per option
    and a reply instruction. The prompt wording is vLLM's, so a model behaves
    the same behind either server."""

    name: str = "choice"
    label_set: Tuple[str, ...] = LABELS
    reply_instruction: str = "Answer with the letter of one option only."

    def parse_options(self, qid: str, criteria: Any) -> List[Option]:
        if not isinstance(criteria, dict) or not criteria:
            raise DecisionValidationError(
                f"question {qid!r}: choice criteria must map option names to a description or null"
            )
        return [
            Option(str(name), None if desc is None else str(desc))
            for name, desc in criteria.items()
        ]

    def reply_line(self, label: str, option: Option) -> str:
        if option.description:
            return f"{label}: {option.name} - {option.description}"
        return f"{label}: {option.name}"

    def body(self, question: Question) -> str:
        """The options and the reply instruction, after the question's heading."""

        lines = [
            self.reply_line(label, option)
            for label, option in zip(question.labels, question.options, strict=True)
        ]
        lines.append(self.reply_instruction)
        return "\n".join(lines)

    def prompt(self, question: Question) -> str:
        """The question as the model reads it, after the state."""

        body = self.body(question)
        if not question.instructions:
            return body
        return f"Question: {question.instructions}\n{body}"

    def answer(self, question: Question, probs: List[float], label_mass: float) -> dict:
        """A System One answer. probs[i] is the probability of labels[i] among
        the labels and sums to 1; label_mass is their total over the vocabulary."""

        top = argmax(probs)
        return {
            "type": self.name,
            "choice": question.options[top].name,
            "probabilities": {o.name: p for o, p in zip(question.options, probs, strict=True)},
            # The chosen label's probability over the whole vocabulary, so a
            # read that mostly wanted a non-label token reports low confidence
            "confidence": probs[top] * label_mass,
        }


class NoulQuestionType(QuestionType):
    """Yes or no. criteria may describe what true and false mean."""

    name = "noul"
    label_set = NOUL_LABELS
    reply_instruction = "Answer with yes or no only."

    def parse_options(self, qid: str, criteria: Any) -> List[Option]:
        if criteria is not None and (
            not isinstance(criteria, dict) or not criteria.keys() <= {"true", "false"}
        ):
            raise DecisionValidationError(
                f"question {qid!r}: noul criteria must be an object with true and false"
            )
        criteria = criteria or {}
        true, false = criteria.get("true"), criteria.get("false")
        return [
            Option("yes", None if true is None else str(true)),
            Option("no", None if false is None else str(false)),
        ]

    def reply_line(self, label: str, option: Option) -> str:
        # A side without a description still shows its label; without it in the
        # prompt, models answer "Yes" or "No" instead
        return f"{label}: {option.description}" if option.description else label

    def answer(self, question: Question, probs: List[float], label_mass: float) -> dict:
        return {
            "type": self.name,
            "noul": probs[0],
            "probabilities": {"yes": probs[0], "no": probs[1]},
            "confidence": max(probs) * label_mass,
        }


class ScoreQuestionType(QuestionType):
    """Rate the state on an ordered scale. criteria is the ordered list of
    level descriptions, lowest first; a level's label is its index."""

    name = "score"
    label_set = DIGIT_LABELS
    reply_instruction = "Answer with the number of one level only."

    def parse_options(self, qid: str, criteria: Any) -> List[Option]:
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise DecisionValidationError(
                f"question {qid!r}: score criteria must be an ordered list of levels"
            )
        return [Option(str(level)) for level in criteria]

    def answer(self, question: Question, probs: List[float], label_mass: float) -> dict:
        top = argmax(probs)
        return {
            "type": self.name,
            # The expected level, 0-indexed, so a level's index in the request's
            # criteria is its value
            "score": sum(i * p for i, p in enumerate(probs)),
            "legend": {str(i): o.name for i, o in enumerate(question.options)},
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "confidence": probs[top] * label_mass,
        }


QUESTION_TYPES: Dict[str, QuestionType] = {
    "choice": QuestionType(),
    "noul": NoulQuestionType(),
    "score": ScoreQuestionType(),
}


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def label_softmax(logprobs: Sequence[float]) -> List[float]:
    """Softmax of the label logprobs."""

    top = max(logprobs)
    weights = [math.exp(lp - top) for lp in logprobs]
    total = sum(weights)
    return [w / total for w in weights]


def build_question(
    qid: str,
    type_name: str,
    instructions: Any,
    criteria: Any,
    max_options: int = MAX_CHOICES,
) -> Question:
    """A System One question: options parsed from `criteria` by its type and
    labeled with the type's label set."""

    if not qid:
        raise DecisionValidationError("question ids must be non-empty")
    qtype = QUESTION_TYPES.get(type_name)
    if qtype is None:
        raise DecisionValidationError(f"question {qid!r}: unknown question type {type_name!r}")
    options = qtype.parse_options(qid, criteria)
    limit = min(max_options, len(qtype.label_set))
    if len(options) > limit:
        raise DecisionValidationError(
            f"question {qid!r}: at most {limit} options for a {type_name} question"
        )
    return Question(
        id=qid,
        type=qtype,
        instructions=render_state(instructions) if instructions else "",
        options=tuple(options),
        labels=qtype.label_set[: len(options)],
    )


def render_state(value: Any) -> str:
    """A state or instructions as prompt text: a string as is, JSON otherwise."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# -- label tokens


def label_token_ids(pieces: Sequence[str], labels: Sequence[str]) -> Dict[str, List[int]]:
    """
    Every token of the vocabulary that spells one of `labels`, as the model may
    write it at the start of its reply: the label itself and the label with
    leading or trailing blanks (" A", "A "), and for case-insensitive labels
    ("yes", "no") any capitalization. One scan of `pieces`, the vocabulary as
    decoded strings indexed by token id.
    """

    wanted = {}
    for label in labels:
        key = label.casefold() if label in _CASELESS_LABELS else label
        wanted[key] = label
    found: Dict[str, List[int]] = {label: [] for label in labels}
    for token_id, piece in enumerate(pieces):
        core = piece.strip(_LABEL_WHITESPACE)
        if not core or len(core) > max(len(label) for label in labels):
            continue
        label = wanted.get(core)
        if (
            label is None
            and core.casefold() in wanted
            and wanted[core.casefold()] in (_CASELESS_LABELS)
        ):
            label = wanted[core.casefold()]
        if label is not None:
            found[label].append(token_id)
    return found


def _label_cache(container) -> Dict[str, List[int]]:
    """The label token ids of the loaded model, scanned once per tokenizer."""

    tokenizer = container.tokenizer
    cache = getattr(container, "_decision_label_ids", None)
    if cache is not None and cache[0] is tokenizer:
        return cache[1]
    pieces = tokenizer.get_id_to_piece_list(include_special_tokens=False)
    ids = label_token_ids(pieces, LABELS + DIGIT_LABELS + NOUL_LABELS)
    container._decision_label_ids = (tokenizer, ids)
    return ids


def label_groups(container, question: Question) -> List[List[int]]:
    """The token ids that count for each label of `question`."""

    ids = _label_cache(container)
    groups = []
    for label in question.labels:
        group = ids.get(label) or []
        if not group:
            raise DecisionValidationError(
                f"label {label!r} is not a token of this model's vocabulary, so the "
                f"question cannot be read"
            )
        groups.append(group)
    return groups


# -- the read


@dataclass
class QuestionRead:
    """What one question's read produced."""

    probs: List[float]
    label_mass: float
    argmax_is_label: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


def read_logits(logits: torch.Tensor, groups: List[List[int]]) -> Tuple[List[float], float, bool]:
    """
    The label distribution at the answer position. `logits` is the model's
    output for that position (any leading singleton dimensions). A label's
    probability is the sum over its tokens; the softmax over labels gives the
    reported probabilities, label_mass their total over the vocabulary.
    """

    row = logits.detach().to(torch.float32).reshape(-1, logits.shape[-1])[0]
    logprobs = torch.log_softmax(row, dim=-1)
    label_logprobs = [torch.logsumexp(logprobs[g], dim=0).item() for g in groups]
    label_mass = sum(math.exp(lp) for lp in label_logprobs)
    argmax_is_label = any(int(row.argmax().item()) in g for g in groups)
    return label_softmax(label_logprobs), label_mass, argmax_is_label


def _decision_template_vars(container, chat_template_kwargs: Optional[dict]) -> dict:
    """Template variables for a read: the model's defaults, thinking off unless
    the request turns it on, the model's forced values on top."""

    return {
        "add_generation_prompt": True,
        **container.template_vars_default,
        "enable_thinking": False,
        **(chat_template_kwargs or {}),
        **container.template_vars_force,
    }


async def _render_prompt(container, content: str, template_vars: dict) -> str:
    prompt, _, rendered_vars = await format_messages_with_template(
        [ChatCompletionMessage(role="user", content=content)],
        existing_template_vars=dict(template_vars),
    )
    # The template renders the BOS token and the backend adds one when the
    # tokenizer config says so; strip the rendered one as the chat route does
    bos_token = rendered_vars.get("bos_token")
    if bos_token and container.hf_model.add_bos_token() and prompt.startswith(bos_token):
        prompt = prompt.removeprefix(bos_token)
    return prompt


async def _read_question(
    container,
    question: Question,
    content: str,
    template_vars: dict,
    request,
    request_id: str,
    disconnect_handler,
) -> QuestionRead:
    """Render one question, generate one token, and read the label distribution."""

    groups = label_groups(container, question)
    prompt = await _render_prompt(container, content, template_vars)

    # Greedy, one token; min_tokens keeps a stop token at the answer position
    # from ending the job before its logits are reported
    params = BaseSamplerRequest(max_tokens=1, min_tokens=1, temperature=0)

    logits = None
    prompt_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    async for generation in container.generate_gen(
        request_id,
        prompt,
        params,
        disconnect_handler=disconnect_handler,
        label=f"{request_tag(request)} decisions/{question.id}",
        return_logits=True,
    ):
        if generation.get("logits") is not None:
            logits = generation["logits"]
        if generation.get("prompt_tokens") is not None:
            prompt_tokens = generation["prompt_tokens"]
        if generation.get("generated_tokens") is not None:
            output_tokens = generation["generated_tokens"]
        if generation.get("cached_tokens") is not None:
            cached_tokens = generation["cached_tokens"]

    if logits is None:
        raise DecisionValidationError(
            "the model produced no next-token distribution at the answer position; "
            "the prompt may not fit or the template may have shifted the answer position"
        )

    probs, label_mass, argmax_is_label = read_logits(logits, groups)
    return QuestionRead(
        probs=probs,
        label_mass=label_mass,
        argmax_is_label=argmax_is_label,
        input_tokens=prompt_tokens,
        output_tokens=max(output_tokens, 1),
        cached_tokens=cached_tokens,
    )


def question_content(state: str, question: Question, instructions: Optional[str] = None) -> str:
    """The user turn: the state, then the question (with any shared instructions ahead)."""

    parts = [state, question.type.prompt(question)]
    if instructions:
        parts.insert(1, instructions)
    return "\n\n".join(parts)


async def read_questions(
    questions: List[Question],
    state: str,
    request,
    disconnect_handler,
    chat_template_kwargs: Optional[dict] = None,
    instructions: Optional[str] = None,
) -> List[QuestionRead]:
    """Read every question concurrently; waits for all of them even when one fails."""

    container = model.container
    template_vars = _decision_template_vars(container, chat_template_kwargs)
    tasks = [
        asyncio.create_task(
            _read_question(
                container,
                question,
                question_content(state, question, instructions),
                template_vars,
                request,
                f"{request.state.id}-{idx}",
                disconnect_handler,
            )
        )
        for idx, question in enumerate(questions)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    first_error = next((r for r in results if isinstance(r, BaseException)), None)
    if first_error is not None:
        raise first_error
    return list(results)


# -- OpenAI Decisions API


def make_read_question(index: int, question, max_options: int = MAX_CHOICES) -> Question:
    """The OpenAI question as a letter-labeled choice: a predicate chooses between
    "false" and "true", a choice between its values written as JSON, a score
    between its level labels."""

    if isinstance(question, PredicateQuestion):
        options = [Option("false"), Option("true")]
    elif isinstance(question, ChoiceQuestion):
        options = [
            Option(json.dumps(c.value, ensure_ascii=False), c.description or None)
            for c in question.choices
        ]
    else:
        options = [Option(level.label, level.description or None) for level in question.levels]
    limit = min(max_options, len(LABELS))
    if len(options) > limit:
        raise DecisionValidationError(f"This model supports at most {limit} choices per question")
    return Question(
        id=str(index),
        type=QUESTION_TYPES["choice"],
        instructions=question.instructions,
        options=tuple(options),
        labels=LABELS[: len(options)],
    )


def make_answer(question, probs: List[float], label_mass: float) -> DecisionAnswer:
    if isinstance(question, PredicateQuestion):
        return PredicateAnswer(name=question.name, probability=probs[1])
    top = argmax(probs)
    confidence = probs[top] * label_mass
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            name=question.name,
            choice=question.choices[top].value,
            probabilities=[
                ChoiceProbability(value=choice.value, probability=probability)
                for choice, probability in zip(question.choices, probs, strict=True)
            ],
            confidence=confidence,
        )
    return ScoreAnswer(
        name=question.name,
        score=sum(i * probability for i, probability in enumerate(probs)),
        probabilities=[
            ScoreProbability(value=i, label=level.label, probability=probability)
            for i, (level, probability) in enumerate(zip(question.levels, probs, strict=True))
        ],
        confidence=confidence,
    )


def input_text(value) -> str:
    """The shared input as prompt text: messages joined by blank lines, parts by newlines."""

    if isinstance(value, str):
        return value
    return "\n\n".join(
        message.content
        if isinstance(message.content, str)
        else "\n".join(part.text for part in message.content)
        for message in value
    )


async def generate_decisions(
    data: DecisionRequest, request, disconnect_handler
) -> DecisionResponse:
    """Answer an OpenAI Decisions request."""

    questions = [make_read_question(i, q) for i, q in enumerate(data.questions)]
    reads = await read_questions(questions, input_text(data.input), request, disconnect_handler)

    input_tokens = sum(read.input_tokens for read in reads)
    output_tokens = sum(read.output_tokens for read in reads)
    return DecisionResponse(
        model=model.container.model_dir.name,
        answers=[
            make_answer(question, read.probs, read.label_mass)
            for question, read in zip(data.questions, reads, strict=True)
        ],
        usage=DecisionUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=sum(read.cached_tokens for read in reads)
            ),
        ),
    )


# -- TypeSafe System One API


def build_systemone_questions(questions: Dict[str, SystemOneQuestion]) -> List[Question]:
    return [
        build_question(
            qid,
            q.type,
            q.instructions,
            q.criteria,
            MAX_LEVELS if q.type == "score" else MAX_CHOICES,
        )
        for qid, q in questions.items()
    ]


async def generate_systemone(
    data: SystemOneRequest, request, disconnect_handler
) -> SystemOneResponse:
    """Answer a System One request."""

    questions = build_systemone_questions(data.questions)
    reads = await read_questions(
        questions,
        render_state(data.state) if data.state is not None else "",
        request,
        disconnect_handler,
        chat_template_kwargs=data.chat_template_kwargs,
        instructions=data.instructions,
    )
    return SystemOneResponse(
        model=model.container.model_dir.name,
        answers={
            q.id: q.type.answer(q, read.probs, read.label_mass)
            for q, read in zip(questions, reads, strict=True)
        },
        usage=SystemOneUsage(
            input_tokens=sum(read.input_tokens for read in reads),
            output_tokens=sum(read.output_tokens for read in reads),
        ),
        diagnostics={
            q.id: QuestionDiagnostics(
                label_mass=read.label_mass, argmax_is_label=read.argmax_is_label
            )
            for q, read in zip(questions, reads, strict=True)
        },
    )
