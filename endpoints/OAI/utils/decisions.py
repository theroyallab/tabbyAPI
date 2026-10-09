"""Implementation of /v1/decisions: typed questions answered with label
probabilities read from the model's next-token distribution at the answer
position. No text is generated.

The prompt wording is versioned (PROMPT_FORMAT_VERSION); a change to it must
bump the version so clients can pin and detect drift.
"""

import asyncio
import json
from typing import List, Tuple

import torch

from common import model
from common.networking import request_tag
from common.sampling import BaseSamplerRequest
from endpoints.OAI.types.chat_completion import ChatCompletionMessage
from endpoints.OAI.types.decisions import (
    AnswerChoice,
    AnswerScore,
    AnswerYesNo,
    ChoiceQuestion,
    DecisionsRequest,
    DecisionsResponse,
    ScoreQuestion,
)
from endpoints.OAI.utils.chat_completion import format_messages_with_template

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
YES_NO_LABELS = ("yes", "no")


class DecisionValidationError(ValueError):
    """A request that can never be answered (bad counts, unusable labels...)."""


def _input_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def render_question_message(request_input, question) -> str:
    """The versioned user message for one question."""

    lines = [_input_text(request_input), "", question.question]

    if isinstance(question, ChoiceQuestion):
        for idx, option in enumerate(question.options):
            label = CHOICE_LABELS[idx]
            if option.description:
                lines.append(f"{label}: {option.name} - {option.description}")
            else:
                lines.append(f"{label}: {option.name}")
        lines.append("Answer with only the letter of your choice.")
    elif isinstance(question, ScoreQuestion):
        for idx, level in enumerate(question.levels):
            lines.append(f"{idx}: {level}")
        lines.append("Answer with only the number that best matches.")
    else:  # YesNoQuestion
        if question.yes_description:
            lines.append(f"yes: {question.yes_description}")
        if question.no_description:
            lines.append(f"no: {question.no_description}")
        lines.append("Answer with yes or no, all lowercase.")

    return "\n".join(lines)


def question_labels(question) -> Tuple[List[str], List[str]]:
    """Answer labels in option order, plus human-readable names for each."""

    if isinstance(question, ChoiceQuestion):
        names = [option.name for option in question.options]
        return list(CHOICE_LABELS[: len(names)]), names
    if isinstance(question, ScoreQuestion):
        return [str(idx) for idx in range(len(question.levels))], list(question.levels)
    return list(YES_NO_LABELS), list(YES_NO_LABELS)


def validate_decisions_request(data: DecisionsRequest) -> None:
    """Raise DecisionValidationError on anything that can't be answered."""

    def bad(message: str):
        raise DecisionValidationError(message)

    seen_ids = set()
    for position, question in enumerate(data.questions, start=1):
        name = question.id if question.id else f"#{position}"
        if not question.id.strip():
            bad(f"question id must not be blank (question {name})")
        if question.id in seen_ids:
            bad(f"repeated question id: {question.id}")
        seen_ids.add(question.id)

        if isinstance(question, ChoiceQuestion):
            if not (2 <= len(question.options) <= 26):
                bad(f"question {name}: choice needs 2 to 26 options")
            names = [option.name.strip().casefold() for option in question.options]
            if any(not n for n in names):
                bad(f"question {name}: option names must not be blank")
            if len(set(names)) != len(names):
                bad(f"question {name}: option names must not repeat")
            for option in question.options:
                if any(ord(c) < 32 or ord(c) == 127 for c in option.name):
                    bad(f"question {name}: control characters or line breaks in option names")
        elif isinstance(question, ScoreQuestion):
            if not (2 <= len(question.levels) <= 10):
                bad(f"question {name}: score needs 2 to 10 levels")
            if any(not level.strip() for level in question.levels):
                bad(f"question {name}: levels must not be blank")


def answer_label_token_ids(prompt: str, labels: List[str], tokenizer) -> List[int]:
    """Token id for each label, requiring it to be one distinct token at the
    answer position (the token that directly follows the rendered prompt).

    The relative comparison (prompt+label vs prompt) cancels BOS handling and
    matches how the generation pass encodes the prompt. Labels are appended
    with no leading space; a tokenizer that merges the preceding character
    into the label fails the length check and is rejected.
    """

    base_ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
    base_len = base_ids.shape[-1]

    token_ids = []
    for label in labels:
        with_label = tokenizer.encode(prompt + label, add_bos=True, encode_special_tokens=True)
        if with_label.shape[-1] != base_len + 1:
            raise DecisionValidationError(
                f"label {label!r} is not one distinct token at the answer position "
                "for this model's tokenizer. If the chat template opens a "
                "reasoning block by default, pass template_vars to turn it off "
                '(e.g. {"enable_thinking": false}).'
            )
        token_ids.append(with_label[0, -1].item())
    return token_ids


def label_distribution(
    logits: torch.Tensor, label_ids: List[int], temperature: float
) -> Tuple[List[float], float]:
    """Probability of each label at the answer position, plus label_mass: the
    full-vocabulary probability the model puts on the offered labels. Neither
    is a calibrated probability that the decision is correct.
    """

    row = logits.detach().to(torch.float32).reshape(-1)
    logprobs = torch.log_softmax(row, dim=-1)
    selected = logprobs[label_ids]

    label_mass = selected.exp().sum().item()
    scaled = selected / temperature
    probs = torch.softmax(scaled, dim=-1).tolist()
    return probs, label_mass


def compose_answer(question, labels, names, probs, label_mass):
    probabilities = dict(zip(labels, probs, strict=True))

    if isinstance(question, ChoiceQuestion):
        best = max(range(len(probs)), key=probs.__getitem__)
        return AnswerChoice(probabilities=probabilities, label_mass=label_mass, choice=names[best])
    if isinstance(question, ScoreQuestion):
        weighted = sum(idx * p for idx, p in enumerate(probs))
        return AnswerScore(probabilities=probabilities, label_mass=label_mass, score=weighted)
    return AnswerYesNo(probabilities=probabilities, label_mass=label_mass)


async def _answer_one_question(
    question,
    request_input,
    data: DecisionsRequest,
    request,
    request_id: str,
    disconnect_handler,
):
    """Render, generate one position, and read the label distribution."""

    message_text = render_question_message(request_input, question)
    # Same merge order as chat completions; without the model defaults the
    # answer position can land inside a reasoning block.
    request_vars = {}
    if data.enable_thinking is not None:
        request_vars["enable_thinking"] = data.enable_thinking
    request_vars.update(data.template_vars or {})
    template_vars = {
        "add_generation_prompt": True,
        **model.container.template_vars_default,
        **request_vars,
        **model.container.template_vars_force,
    }
    prompt, _, _ = await format_messages_with_template(
        [ChatCompletionMessage(role="user", content=message_text)],
        existing_template_vars=template_vars,
    )

    tokenizer = model.container.tokenizer
    labels, names = question_labels(question)
    label_ids = answer_label_token_ids(prompt, labels, tokenizer)

    # The sampled token is discarded; greedy keeps the sampler stack trivial.
    # min_tokens=1 masks the stop list at the single answer position, so a
    # model that would end its turn exactly here still yields its logits.
    params = BaseSamplerRequest(max_tokens=1, min_tokens=1, temperature=0)
    params._return_logits = True

    logits = None
    prompt_tokens = 0
    async for generation in model.container.generate_gen(
        request_id,
        prompt,
        params,
        disconnect_handler=disconnect_handler,
        label=f"{request_tag(request)} decisions/{question.id}",
    ):
        if generation.get("logits") is not None:
            logits = generation["logits"]
        if generation.get("prompt_tokens") is not None:
            prompt_tokens = generation["prompt_tokens"]

    if logits is None:
        raise DecisionValidationError(
            "the model produced no next-token distribution at the answer "
            "position; the prompt may not fit or the template may have "
            "shifted the answer position"
        )

    probs, label_mass = label_distribution(logits, label_ids, data.temperature)
    answer = compose_answer(question, labels, names, probs, label_mass)
    return question.id, answer, prompt_tokens


async def generate_decisions(
    data: DecisionsRequest,
    request,
    disconnect_handler,
) -> DecisionsResponse:
    """Run every question and compose the response."""

    validate_decisions_request(data)

    # Each question is an independent one-token job; the backend batches
    # their prefills. Wait for all of them even on failure so no task is
    # orphaned, then re-raise the first error for the router's error mapping.
    tasks = [
        asyncio.create_task(
            _answer_one_question(
                question, data.input, data, request, f"{request.state.id}-{idx}", disconnect_handler
            )
        )
        for idx, question in enumerate(data.questions)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    first_error = next((r for r in results if isinstance(r, BaseException)), None)
    if first_error is not None:
        raise first_error
    answers = {}
    prompt_tokens = 0
    for question_id, answer, tokens in results:
        answers[question_id] = answer
        prompt_tokens += tokens

    return DecisionsResponse(
        model=model.container.model_dir.name,
        answers=answers,
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": 0},
    )
