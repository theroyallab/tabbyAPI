"""Tests for Qwen3 reasoning parser parity with modern Qwen3.5 behavior."""

from endpoints.OAI.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser


class _FakeTokenizer:
    def get_vocab(self):
        return {
            "<think>": 101,
            "</think>": 102,
        }


def _parser(enable_thinking=None) -> Qwen3ReasoningParser:
    kwargs = {}
    if enable_thinking is not None:
        kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return Qwen3ReasoningParser(_FakeTokenizer(), **kwargs)


def test_non_stream_extract_thinking_mode_with_prefilled_start_token():
    parser = _parser(enable_thinking=True)

    reasoning, content = parser.extract_reasoning("reasoning</think>answer", request=None)
    assert reasoning == "reasoning"
    assert content == "answer"


def test_non_stream_extract_without_end_token_treated_as_content():
    parser = _parser(enable_thinking=True)

    reasoning, content = parser.extract_reasoning("reasoning only", request=None)
    assert reasoning is None
    assert content == "reasoning only"


def test_non_stream_extract_non_thinking_mode_content_only():
    parser = _parser(enable_thinking=False)

    reasoning, content = parser.extract_reasoning("<think>hidden</think>visible", request=None)
    assert reasoning is None
    assert content == "hiddenvisible"


def test_streaming_prefilled_think_mode_splits_reasoning_and_content():
    parser = _parser(enable_thinking=True)

    first = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="analysis ",
        delta_text="analysis ",
        previous_token_ids=[],
        current_token_ids=[11],
        delta_token_ids=[11],
    )
    assert first is not None
    assert first.reasoning == "analysis "
    assert first.content is None

    second = parser.extract_reasoning_streaming(
        previous_text="analysis ",
        current_text="analysis step</think>final ",
        delta_text="step</think>final ",
        previous_token_ids=[11],
        current_token_ids=[11, 12, 102, 13],
        delta_token_ids=[12, 102, 13],
    )
    assert second is not None
    assert second.reasoning == "step"
    assert second.content == "final "

    third = parser.extract_reasoning_streaming(
        previous_text="analysis step</think>final ",
        current_text="analysis step</think>final answer",
        delta_text="answer",
        previous_token_ids=[11, 12, 102, 13],
        current_token_ids=[11, 12, 102, 13, 14],
        delta_token_ids=[14],
    )
    assert third is not None
    assert third.reasoning is None
    assert third.content == "answer"


def test_streaming_non_thinking_mode_emits_content_only():
    parser = _parser(enable_thinking=False)

    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="plain output",
        delta_text="plain output",
        previous_token_ids=[],
        current_token_ids=[11, 12],
        delta_token_ids=[11, 12],
    )
    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == "plain output"


def test_streaming_strips_generated_start_token_when_present():
    parser = _parser(enable_thinking=True)

    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>reason",
        delta_text="<think>reason",
        previous_token_ids=[],
        current_token_ids=[101, 11],
        delta_token_ids=[101, 11],
    )
    assert delta is not None
    assert delta.reasoning == "reason"
    assert delta.content is None
