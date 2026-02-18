"""Tests for Exaone4 reasoning parser behavior."""

from endpoints.OAI.reasoning.exaone4_reasoning_parser import Exaone4ReasoningParser


class _FakeTokenizer:
    def get_vocab(self):
        return {
            "<think>": 101,
            "</think>": 102,
        }


def _parser(enable_thinking: bool) -> Exaone4ReasoningParser:
    return Exaone4ReasoningParser(
        _FakeTokenizer(),
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


def test_non_thinking_mode_emits_content_only():
    parser = _parser(enable_thinking=False)

    reasoning, content = parser.extract_reasoning("hello", request=None)
    assert reasoning is None
    assert content == "hello"

    reasoning, content = parser.extract_reasoning("</think>hello", request=None)
    assert reasoning is None
    assert content == "hello"

    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="hello",
        delta_text="hello",
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
    )
    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == "hello"


def test_thinking_mode_extract_reasoning_and_content_non_streaming():
    parser = _parser(enable_thinking=True)

    reasoning, content = parser.extract_reasoning(
        "<think>reason</think>answer", request=None
    )
    assert reasoning == "reason"
    assert content == "answer"

    reasoning, content = parser.extract_reasoning("reason</think>answer", request=None)
    assert reasoning == "reason"
    assert content == "answer"


def test_thinking_mode_without_end_token_is_reasoning_only():
    parser = _parser(enable_thinking=True)

    reasoning, content = parser.extract_reasoning("reasoning only", request=None)
    assert reasoning == "reasoning only"
    assert content is None


def test_thinking_streaming_prefill_flow_without_start_token():
    parser = _parser(enable_thinking=True)

    first = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="reason ",
        delta_text="reason ",
        previous_token_ids=[],
        current_token_ids=[11],
        delta_token_ids=[11],
    )
    assert first is not None
    assert first.reasoning == "reason "
    assert first.content is None

    second = parser.extract_reasoning_streaming(
        previous_text="reason ",
        current_text="reason more</think>final",
        delta_text="more</think>final",
        previous_token_ids=[11],
        current_token_ids=[11, 12, 102, 13],
        delta_token_ids=[12, 102, 13],
    )
    assert second is not None
    assert second.reasoning == "more"
    assert second.content == "final"

    third = parser.extract_reasoning_streaming(
        previous_text="reason more</think>final",
        current_text="reason more</think>final!",
        delta_text="!",
        previous_token_ids=[11, 12, 102, 13],
        current_token_ids=[11, 12, 102, 13, 14],
        delta_token_ids=[14],
    )
    assert third is not None
    assert third.reasoning is None
    assert third.content == "!"


def test_thinking_streaming_handles_split_end_token_boundary():
    parser = _parser(enable_thinking=True)

    first = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="analysis </thi",
        delta_text="analysis </thi",
        previous_token_ids=[],
        current_token_ids=[11],
        delta_token_ids=[11],
    )
    assert first is not None
    assert first.reasoning == "analysis "
    assert first.content is None

    second = parser.extract_reasoning_streaming(
        previous_text="analysis </thi",
        current_text="analysis </think>answer",
        delta_text="nk>answer",
        previous_token_ids=[11],
        current_token_ids=[11, 102, 12],
        delta_token_ids=[12],
    )
    assert second is not None
    assert second.reasoning is None
    assert second.content == "answer"


def test_thinking_streaming_handles_split_tool_call_boundary_without_end_token():
    parser = _parser(enable_thinking=True)

    first = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="analysis <tool_c",
        delta_text="analysis <tool_c",
        previous_token_ids=[],
        current_token_ids=[11],
        delta_token_ids=[11],
    )
    assert first is not None
    assert first.reasoning == "analysis "
    assert first.content is None

    second = parser.extract_reasoning_streaming(
        previous_text="analysis <tool_c",
        current_text='analysis <tool_call>{"name":"lookup","arguments":{}}',
        delta_text='all>{"name":"lookup","arguments":{}}',
        previous_token_ids=[11],
        current_token_ids=[11, 12],
        delta_token_ids=[12],
    )
    assert second is not None
    assert second.reasoning is None
    assert second.content == '<tool_call>{"name":"lookup","arguments":{}}'


def test_thinking_streaming_handles_split_deepseek_tool_boundary_without_end_token():
    parser = _parser(enable_thinking=True)

    first = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="analysis <｜tool▁call▁b",
        delta_text="analysis <｜tool▁call▁b",
        previous_token_ids=[],
        current_token_ids=[11],
        delta_token_ids=[11],
    )
    assert first is not None
    assert first.reasoning == "analysis "
    assert first.content is None

    second = parser.extract_reasoning_streaming(
        previous_text="analysis <｜tool▁call▁b",
        current_text=(
            "analysis <｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"q\":\"tabby\"}"
            "<｜tool▁call▁end｜>"
        ),
        delta_text='egin｜>lookup<｜tool▁sep｜>{"q":"tabby"}<｜tool▁call▁end｜>',
        previous_token_ids=[11],
        current_token_ids=[11, 12],
        delta_token_ids=[12],
    )
    assert second is not None
    assert second.reasoning is None
    assert second.content == (
        '<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{"q":"tabby"}<｜tool▁call▁end｜>'
    )


def test_thinking_mode_content_ids_and_end_detection():
    parser = _parser(enable_thinking=True)

    assert parser.is_reasoning_end([1, 2, 102]) is True
    assert parser.is_reasoning_end([1, 2, 3]) is False

    assert parser.extract_content_ids([10, 101, 20, 102, 30, 31]) == [30, 31]
    assert parser.extract_content_ids([10, 101, 20]) == []
