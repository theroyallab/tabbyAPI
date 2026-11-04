"""
Tests for the decision endpoints: the OpenAI Decisions schema and adapter, the
System One question types, the vocabulary-scan label tokens, the logits read
and the request wiring. The protocol and prompt cases mirror vLLM's test suite
for the same two APIs, so a model behaves the same behind either server.
"""

import asyncio
import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from endpoints.OAI.types.decisions import DecisionRequest, SystemOneRequest
from endpoints.OAI.utils import decisions as dec
from endpoints.OAI.utils.decisions import (
    LABELS,
    DecisionValidationError,
    Question,
    build_question,
    input_text,
    label_softmax,
    label_token_ids,
    make_answer,
    make_read_question,
    read_logits,
)


def request(**kwargs):
    return DecisionRequest.model_validate(
        {
            "model": "test-model",
            "input": "The package is broken.",
            "questions": [{"type": "predicate", "instructions": "Is it damaged?"}],
            **kwargs,
        }
    )


# -- OpenAI schema and adapter (vLLM's test_decisions.py)


def test_predicate_returns_probability_of_true_and_null_name():
    question = request().questions[0]
    read = make_read_question(0, question)
    assert [option.name for option in read.options] == ["false", "true"]
    assert read.labels == ("A", "B")
    assert make_answer(question, [0.1, 0.9], 0.5).model_dump() == {
        "type": "predicate",
        "name": None,
        "probability": 0.9,
    }


def test_choice_preserves_boolean_and_string_values():
    question = request(
        questions=[
            {
                "type": "choice",
                "name": "decision",
                "instructions": "Select a value.",
                "choices": [{"value": True}, {"value": "true"}, {"value": False}],
            }
        ]
    ).questions[0]
    read = make_read_question(0, question)
    assert [option.name for option in read.options] == ["true", '"true"', "false"]
    answer = make_answer(question, [0.7, 0.2, 0.1], 0.5).model_dump()
    assert answer["name"] == "decision"
    assert answer["choice"] is True
    assert answer["probabilities"] == [
        {"value": True, "probability": 0.7},
        {"value": "true", "probability": 0.2},
        {"value": False, "probability": 0.1},
    ]
    assert answer["confidence"] == pytest.approx(0.35)


def test_score_is_weighted_average_of_zero_based_level_indices():
    question = request(
        questions=[
            {
                "type": "score",
                "instructions": "Rate severity.",
                "levels": [{"label": label} for label in ("low", "medium", "high")],
            }
        ]
    ).questions[0]
    answer = make_answer(question, [0.1, 0.7, 0.2], 0.8).model_dump()
    assert answer["score"] == pytest.approx(1.1)
    assert answer["probabilities"] == [
        {"value": 0, "label": "low", "probability": 0.1},
        {"value": 1, "label": "medium", "probability": 0.7},
        {"value": 2, "label": "high", "probability": 0.2},
    ]
    assert answer["confidence"] == pytest.approx(0.56)


def test_text_messages_preserve_evidence_order():
    parsed = request(
        input=[
            {"role": "user", "content": "first"},
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "second"},
                    {"type": "input_text", "text": "third"},
                ],
            },
        ]
    )
    assert input_text(parsed.input) == "first\n\nsecond\nthird"
    assert input_text([]) == ""


@pytest.mark.parametrize("evidence", ["", [], [{"role": "user", "content": []}]])
def test_empty_input_is_valid_under_the_openai_schema(evidence):
    assert input_text(request(input=evidence).input) == ""


def test_score_levels_with_the_same_label_keep_their_positions():
    question = request(
        questions=[
            {
                "type": "score",
                "instructions": "Rate severity.",
                "levels": [
                    {"label": "damage", "description": "cosmetic"},
                    {"label": "damage", "description": "unusable"},
                ],
            }
        ]
    ).questions[0]
    read = make_read_question(0, question)
    assert read.labels == ("A", "B")
    assert read.type.prompt(read) == (
        "Question: Rate severity.\n"
        "A: damage - cosmetic\n"
        "B: damage - unusable\n"
        "Answer with the letter of one option only."
    )
    answer = make_answer(question, [0.1, 0.9], 0.8).model_dump()
    assert answer["score"] == pytest.approx(0.9)
    assert [p["value"] for p in answer["probabilities"]] == [0, 1]


@pytest.mark.parametrize(
    "override,match",
    [
        ({"input": [{"role": "assistant", "content": "x"}]}, "user"),
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
                        ],
                    }
                ]
            },
            "text input only",
        ),
        ({"input": {"text": "x"}}, "Input should"),
        ({"questions": []}, "at least 1"),
        (
            {"questions": [{"type": "predicate", "instructions": "x", "name": None}]},
            "name must be a string",
        ),
        ({"questions": [{"type": "predicate", "instructions": 1}]}, "valid string"),
        (
            {"questions": [{"type": "predicate", "instructions": "x", "criteria": {}}]},
            "Extra inputs",
        ),
        ({"safety_identifier": "x" * 129}, "at most 128"),
        ({"stream": True}, "Extra inputs"),
    ],
)
def test_rejects_invalid_or_unsupported_inputs(override, match):
    with pytest.raises(ValidationError, match=match):
        request(**override)


@pytest.mark.parametrize("values", [[True, True], ["a", "a"], [1, "a"], [None, "a"]])
def test_choice_rejects_duplicates_and_non_string_non_boolean_values(values):
    with pytest.raises(ValidationError):
        request(
            questions=[
                {"type": "choice", "instructions": "x", "choices": [{"value": v} for v in values]}
            ]
        )


@pytest.mark.parametrize("count,valid", [(1, False), (2, True), (26, True), (27, False)])
def test_choice_limits(count, valid):
    data = {
        "questions": [
            {
                "type": "choice",
                "instructions": "x",
                "choices": [{"value": str(i)} for i in range(count)],
            }
        ]
    }
    if valid:
        assert len(request(**data).questions[0].choices) == count
    else:
        with pytest.raises(ValidationError):
            request(**data)


@pytest.mark.parametrize("count,valid", [(1, False), (2, True), (10, True), (11, False)])
def test_score_limits(count, valid):
    data = {
        "questions": [
            {
                "type": "score",
                "instructions": "x",
                "levels": [{"label": str(i)} for i in range(count)],
            }
        ]
    }
    if valid:
        request(**data)
    else:
        with pytest.raises(ValidationError):
            request(**data)


def test_question_limit_and_optional_fields():
    questions = [{"type": "predicate", "instructions": "x"}] * 64
    parsed = request(questions=questions, safety_identifier=None)
    assert len(parsed.questions) == 64
    with pytest.raises(ValidationError, match="at most 64"):
        request(questions=[*questions, questions[0]])
    # The model is optional here: it selects an inline load like the other routes
    assert (
        DecisionRequest.model_validate(
            {"input": "x", "questions": [{"type": "predicate", "instructions": "x"}]}
        ).model
        is None
    )


# -- System One question types (vLLM's test_question_types.py)


def choice(criteria=None, instructions="Which team?", max_options=26):
    criteria = criteria or {"billing": "money", "outage": None, "other": None}
    return build_question("q", "choice", instructions, criteria, max_options)


def noul(criteria=None, instructions="Reply within the hour?"):
    return build_question("q", "noul", instructions, criteria)


def score(criteria=None, instructions="How angry?"):
    return build_question(
        "q", "score", instructions, criteria or ["calm", "annoyed", "furious"], 10
    )


def test_choice_labels_follow_option_order():
    q = choice()
    assert q.labels == ("A", "B", "C")
    assert [a.name for a in q.options] == ["billing", "outage", "other"]
    assert q.options[0].description == "money"
    wide = choice(criteria={str(i): None for i in range(26)})
    assert wide.labels[-1] == "Z"


def test_choice_prompt():
    q = choice()
    assert q.type.prompt(q) == (
        "Question: Which team?\n"
        "A: billing - money\n"
        "B: outage\n"
        "C: other\n"
        "Answer with the letter of one option only."
    )


@pytest.mark.parametrize(
    "q,body",
    [
        (choice(), "A: billing - money\nB: outage\nC: other\n"),
        (noul(criteria={"true": "t", "false": "f"}), "yes: t\nno: f\n"),
        (score(), "0: calm\n1: annoyed\n2: furious\n"),
        # Instructions are kept as given, trailing newlines included
        (noul(instructions="Reply within the hour?\n\n"), "yes\nno\n"),
    ],
)
def test_prompt_is_the_heading_then_the_body(q, body):
    body += q.type.reply_instruction
    assert q.type.body(q) == body
    assert q.type.prompt(q) == f"Question: {q.instructions}\n{body}"
    bare = replace(q, instructions="")
    assert bare.type.prompt(bare) == body


def test_choice_answer_shape():
    q = choice()
    answer = q.type.answer(q, [0.2, 0.7, 0.1], 0.5)
    assert answer == {
        "type": "choice",
        "choice": "outage",
        "probabilities": {"billing": 0.2, "outage": 0.7, "other": 0.1},
        "confidence": 0.35,
    }


@pytest.mark.parametrize(
    "qid,type_name,criteria,match",
    [
        ("", "choice", {"x": None, "y": None}, "non-empty"),
        ("q", "nope", {"x": None, "y": None}, "unknown question type"),
        ("q", "choice", ["x", "y"], "must map option names"),
        ("q", "noul", ["yes", "no"], "must be an object"),
        ("q", "noul", {"maybe": "same day"}, "must be an object"),
        ("q", "score", {"0": "calm", "1": "furious"}, "ordered list of levels"),
        ("q", "score", ["calm"], "ordered list of levels"),
    ],
)
def test_build_question_rejects(qid, type_name, criteria, match):
    with pytest.raises(DecisionValidationError, match=match):
        build_question(qid, type_name, "", criteria, 128)


def test_one_option_choice():
    q = choice(criteria={"only": None})
    assert q.labels == ("A",)
    assert q.type.answer(q, [1.0], 0.4)["confidence"] == 0.4


def test_option_limit():
    with pytest.raises(DecisionValidationError, match="at most 2 options"):
        choice(criteria={"x": None, "y": None, "z": None}, max_options=2)
    with pytest.raises(DecisionValidationError, match="at most 26 options"):
        choice(criteria={str(i): None for i in range(27)})


def test_noul_labels_and_prompt():
    q = noul()
    assert q.labels == ("yes", "no")
    assert [a.name for a in q.options] == ["yes", "no"]
    assert (
        q.type.prompt(q) == "Question: Reply within the hour?\nyes\nno\nAnswer with yes or no only."
    )
    described = noul(criteria={"true": "same day", "false": "later"})
    assert described.type.prompt(described) == (
        "Question: Reply within the hour?\nyes: same day\nno: later\nAnswer with yes or no only."
    )


def test_noul_answer_shape():
    q = noul()
    assert q.type.answer(q, [0.8, 0.2], 0.5) == {
        "type": "noul",
        "noul": 0.8,
        "probabilities": {"yes": 0.8, "no": 0.2},
        "confidence": 0.4,
    }


def test_score_labels_are_levels():
    assert score().labels == ("0", "1", "2")
    with pytest.raises(DecisionValidationError, match="at most 10"):
        score(criteria=[str(i) for i in range(11)])


def test_score_prompt_and_answer():
    q = score()
    assert q.type.prompt(q) == (
        "Question: How angry?\n0: calm\n1: annoyed\n2: furious\nAnswer with the number of one level only."
    )
    assert q.type.answer(q, [0.5, 0.25, 0.25], 0.4) == {
        "type": "score",
        "score": 0.75,
        "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
        "probabilities": {"0": 0.5, "1": 0.25, "2": 0.25},
        "confidence": 0.2,
    }


def test_systemone_request_shape():
    parsed = SystemOneRequest.model_validate(
        {
            "state": {"ticket": 12, "text": "charged twice"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": None},
                },
                "angry": {"type": "noul"},
            },
        }
    )
    assert list(parsed.questions) == ["route", "angry"]
    assert dec.render_state(parsed.state) == '{"ticket": 12, "text": "charged twice"}'
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "x", "questions": {}})
    with pytest.raises(ValidationError, match="Extra inputs"):
        SystemOneRequest.model_validate(
            {"state": "x", "questions": {"q": {"type": "noul"}}, "stream": True}
        )


# -- label tokens from the vocabulary


PIECES = [
    "<bos>",
    "A",
    " A",
    "B",
    " B",
    "\tA",
    "A.",
    "AB",
    "yes",
    "Yes",
    " yes",
    " YES",
    "no",
    " No",
    "nope",
    "0",
    "1",
    " 1",
    "10",
    "",
    " ",
]


def test_label_tokens_include_whitespace_and_case_variants():
    ids = label_token_ids(PIECES, ("A", "B", "yes", "no", "0", "1"))
    piece = lambda i: PIECES[i]  # noqa: E731
    assert [piece(i) for i in ids["A"]] == ["A", " A", "\tA"]
    assert [piece(i) for i in ids["B"]] == ["B", " B"]
    assert [piece(i) for i in ids["yes"]] == ["yes", "Yes", " yes", " YES"]
    assert [piece(i) for i in ids["no"]] == ["no", " No"]
    assert [piece(i) for i in ids["0"]] == ["0"]
    assert [piece(i) for i in ids["1"]] == ["1", " 1"]
    # Letters are case-sensitive labels: "a" would be a different token set
    assert all(set(a).isdisjoint(b) for a in ids.values() for b in ids.values() if a is not b)


def test_label_tokens_are_scanned_once_per_tokenizer():
    tokenizer = SimpleNamespace(get_id_to_piece_list=lambda include_special_tokens=False: PIECES)
    container = SimpleNamespace(tokenizer=tokenizer)
    q = choice(criteria={"x": None, "y": None})
    groups = dec.label_groups(container, q)
    assert groups == [[1, 2, 5], [3, 4]]
    container.tokenizer = SimpleNamespace(get_id_to_piece_list=lambda **k: ["A", "B"])
    assert dec.label_groups(container, q) == [[0], [1]]


def test_missing_label_token_is_a_400_not_a_crash():
    tokenizer = SimpleNamespace(get_id_to_piece_list=lambda **k: ["A", "B"])
    container = SimpleNamespace(tokenizer=tokenizer)
    with pytest.raises(DecisionValidationError, match="not a token"):
        dec.label_groups(container, choice())


# -- the read


def test_read_logits_sums_label_variants_and_reports_mass():
    vocab = 8
    logits = torch.full((1, 1, vocab), -20.0)
    # A: tokens 1 and 2 (A and " A"); B: token 3; the rest is noise
    logits[0, 0, 1] = math.log(0.3)
    logits[0, 0, 2] = math.log(0.3)
    logits[0, 0, 3] = math.log(0.2)
    logits[0, 0, 7] = math.log(0.2)
    probs, label_mass, argmax_is_label = read_logits(logits, [[1, 2], [3]])
    assert probs == pytest.approx([0.75, 0.25], abs=1e-6)
    assert label_mass == pytest.approx(0.8, abs=1e-6)
    assert argmax_is_label is True
    # The reported probabilities and the diagnostics agree with vLLM's shape
    assert label_softmax([math.log(0.6), math.log(0.2)]) == pytest.approx([0.75, 0.25])


def test_read_logits_argmax_outside_labels():
    logits = torch.zeros((1, 1, 5))
    logits[0, 0, 4] = 5.0
    probs, label_mass, argmax_is_label = read_logits(logits, [[0], [1]])
    assert probs == pytest.approx([0.5, 0.5])
    assert label_mass < 0.05
    assert argmax_is_label is False


# -- wiring through the container


def make_container(logits_for_prompt, bos="<bos>", add_bos=True):
    """A container whose stream_generate yields the logits chosen by the rendered prompt."""

    prompts = []
    params_seen = []

    async def stream_generate(request_id, prompt, params, **kwargs):
        # The registry-owning wrapper, not generate_gen: the read must go through it
        prompts.append(prompt)
        params_seen.append(params)
        assert kwargs["return_logits"] is True
        yield {"prompt_tokens": 10, "generated_tokens": 1, "logits": logits_for_prompt(prompt)}
        yield {
            "prompt_tokens": 10,
            "generated_tokens": 1,
            "cached_tokens": 4,
            "finish_reason": "length",
        }

    async def generate_gen(*args, **kwargs):
        raise AssertionError("reads must use stream_generate so the job registry is kept")
        yield  # pragma: no cover

    return SimpleNamespace(
        tokenizer=SimpleNamespace(get_id_to_piece_list=lambda **k: PIECES),
        template_vars_default={"enable_thinking": True},
        template_vars_force={},
        hf_model=SimpleNamespace(add_bos_token=lambda: add_bos),
        model_dir=SimpleNamespace(name="fake-model"),
        stream_generate=stream_generate,
        generate_gen=generate_gen,
        prompts=prompts,
        params_seen=params_seen,
    )


async def fake_format(messages, existing_template_vars=None):
    vars_ = existing_template_vars or {}
    thinking = "<think>" if vars_.get("enable_thinking") else ""
    prompt = f"<bos><user>{messages[0].content}<model>{thinking}"
    return prompt, None, {**vars_, "bos_token": "<bos>"}


def run(coro):
    return asyncio.run(coro)


def test_openai_request_end_to_end():
    def logits_for(prompt):
        # The predicate's "true" is B; put mass on " B" (id 4) and a little on "A" (id 1)
        row = torch.full((1, 1, len(PIECES)), -30.0)
        row[0, 0, 4] = math.log(0.6)
        row[0, 0, 1] = math.log(0.2)
        row[0, 0, 7] = math.log(0.2)  # "AB": not a label, so label_mass is 0.8
        return row

    container = make_container(logits_for)
    req = SimpleNamespace(state=SimpleNamespace(id="r1"), headers={}, client=None)
    data = request(
        input=[{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        questions=[
            {"type": "predicate", "name": "damaged", "instructions": "Is it damaged?"},
            {
                "type": "choice",
                "instructions": "Pick.",
                "choices": [{"value": "x"}, {"value": "y"}],
            },
        ],
    )
    with (
        patch.object(dec, "model", SimpleNamespace(container=container)),
        patch.object(dec, "format_messages_with_template", fake_format),
        patch.object(dec, "request_tag", lambda r: "req"),
    ):
        response = run(dec.generate_decisions(data, req, None)).model_dump()

    assert set(response) == {"model", "answers", "usage"}
    predicate, choice_answer = response["answers"]
    assert predicate == {"type": "predicate", "name": "damaged", "probability": pytest.approx(0.75)}
    assert choice_answer["name"] is None
    assert choice_answer["choice"] == "y"
    assert [p["value"] for p in choice_answer["probabilities"]] == ["x", "y"]
    assert choice_answer["confidence"] == pytest.approx(0.75 * 0.8, abs=1e-6)
    assert response["usage"] == {
        "input_tokens": 20,
        "input_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 22,
    }
    # The prompt: evidence joined by blank lines, question heading, labeled options,
    # thinking off, and the template's BOS stripped because the backend adds one
    assert container.prompts[0] == (
        "<user>first\n\nsecond\n\n"
        "Question: Is it damaged?\nA: false\nB: true\n"
        "Answer with the letter of one option only.<model>"
    )


def test_systemone_request_end_to_end():
    def logits_for(prompt):
        row = torch.full((1, 1, len(PIECES)), -30.0)
        if "yes\nno" in prompt:  # the noul: "Yes" (9) and " yes" (10) vs "no" (12)
            row[0, 0, 9] = math.log(0.5)
            row[0, 0, 10] = math.log(0.3)
            row[0, 0, 12] = math.log(0.2)
        else:  # the score over 0/1: " 1" (17) wins
            row[0, 0, 15] = math.log(0.25)
            row[0, 0, 17] = math.log(0.75)
        return row

    container = make_container(logits_for, add_bos=False)
    req = SimpleNamespace(state=SimpleNamespace(id="r2"))
    data = SystemOneRequest.model_validate(
        {
            "state": "Customer: charged twice.",
            "instructions": "Support triage.",
            "questions": {
                "angry": {"type": "noul", "instructions": "Is the customer angry?"},
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["can wait", "today"],
                },
            },
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    with (
        patch.object(dec, "model", SimpleNamespace(container=container)),
        patch.object(dec, "format_messages_with_template", fake_format),
        patch.object(dec, "request_tag", lambda r: "req"),
    ):
        response = run(dec.generate_systemone(data, req, None)).model_dump()

    assert response["answers"]["angry"] == {
        "type": "noul",
        "noul": pytest.approx(0.8),
        "probabilities": {"yes": pytest.approx(0.8), "no": pytest.approx(0.2)},
        "confidence": pytest.approx(0.8),
    }
    assert response["answers"]["urgency"]["score"] == pytest.approx(0.75)
    assert response["answers"]["urgency"]["legend"] == {"0": "can wait", "1": "today"}
    assert response["usage"] == {"input_tokens": 20, "output_tokens": 2}
    assert response["diagnostics"]["angry"]["argmax_is_label"] is True
    # Shared instructions sit between the state and the question; the request
    # turned thinking back on, and the BOS stays when the backend doesn't add one
    assert container.prompts[0].startswith(
        "<bos><user>Customer: charged twice.\n\nSupport triage.\n\nQuestion: Is the customer angry?\nyes\nno\n"
    )
    assert container.prompts[0].endswith("<model><think>")


def test_validation_errors_surface_before_any_read():
    container = make_container(lambda prompt: torch.zeros((1, 1, len(PIECES))))
    req = SimpleNamespace(state=SimpleNamespace(id="r3"))
    data = SystemOneRequest.model_validate(
        {"state": "x", "questions": {"q": {"type": "score", "criteria": ["only one"]}}}
    )
    with (
        patch.object(dec, "model", SimpleNamespace(container=container)),
        patch.object(dec, "format_messages_with_template", fake_format),
        patch.object(dec, "request_tag", lambda r: "req"),
    ):
        with pytest.raises(DecisionValidationError, match="ordered list of levels"):
            run(dec.generate_systemone(data, req, None))
    assert container.prompts == []


def test_forced_sampler_overrides_do_not_reach_the_read(monkeypatch):
    """A chat preset forcing token limits, healing or a grammar must not move the
    answer position or constrain the first token."""

    from common import sampling

    forced = {
        "max_tokens": {"override": 200, "force": True},
        "min_tokens": {"override": 50, "force": True},
        "token_healing": {"override": True, "force": True},
        "temperature": {"override": 0.8, "force": True},
        "regex_pattern": {"override": "[a-z]+", "force": True},
        "top_logprobs": {"override": 5, "force": True},
    }
    monkeypatch.setattr(sampling.SamplerOverridesContainer, "effective", lambda self: forced)
    chat_params = sampling.BaseSamplerRequest()
    assert chat_params.max_tokens == 200 and chat_params.token_healing is True

    params = dec.read_params()
    assert params.max_tokens == 1 and params.min_tokens == 1
    assert params.temperature == 0 and params.token_healing is False
    assert params.regex_pattern is None and params.top_logprobs == 0

    container = make_container(lambda prompt: torch.zeros((1, 1, len(PIECES))))
    req = SimpleNamespace(state=SimpleNamespace(id="r4"))
    with (
        patch.object(dec, "model", SimpleNamespace(container=container)),
        patch.object(dec, "format_messages_with_template", fake_format),
        patch.object(dec, "request_tag", lambda r: "req"),
    ):
        run(dec.generate_decisions(request(), req, None))
    (seen,) = container.params_seen
    assert seen.max_tokens == 1 and seen.token_healing is False


def test_forced_template_variables_do_not_turn_thinking_back_on():
    """A model config forcing enable_thinking for chat still reads at the answer
    position; only the request's own template arguments can turn thinking on."""

    container = make_container(lambda prompt: torch.zeros((1, 1, len(PIECES))))
    container.template_vars_force = {"enable_thinking": True}
    req = SimpleNamespace(state=SimpleNamespace(id="r5"))
    data = SystemOneRequest.model_validate(
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    )
    with (
        patch.object(dec, "model", SimpleNamespace(container=container)),
        patch.object(dec, "format_messages_with_template", fake_format),
        patch.object(dec, "request_tag", lambda r: "req"),
    ):
        run(dec.generate_systemone(data, req, None))
        assert container.prompts[-1].endswith("<model>")

        data.chat_template_kwargs = {"enable_thinking": True}
        run(dec.generate_systemone(data, req, None))
        assert container.prompts[-1].endswith("<model><think>")


def test_question_dataclass_is_immutable():
    q = choice()
    assert isinstance(q, Question)
    with pytest.raises(Exception):
        q.labels = ()
