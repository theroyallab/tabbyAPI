"""Tests for Mistral reasoning parser parity with vLLM behavior."""

from endpoints.OAI.reasoning.mistral_reasoning_parser import MistralReasoningParser


class _FakeTokenizer:
    def get_vocab(self):
        return {
            "[THINK]": 301,
            "[/THINK]": 302,
        }


def _parser() -> MistralReasoningParser:
    return MistralReasoningParser(_FakeTokenizer())


def test_extract_reasoning_with_valid_think_section():
    parser = _parser()

    reasoning, content = parser.extract_reasoning(
        "[THINK]This is a reasoning section[/THINK]This is the rest",
        request=None,
    )

    assert reasoning == "This is a reasoning section"
    assert content == "This is the rest"


def test_extract_reasoning_with_invalid_end_token_only():
    parser = _parser()

    reasoning, content = parser.extract_reasoning(
        "This is a reasoning section[/THINK]This is the rest",
        request=None,
    )

    assert reasoning is None
    assert content == "This is a reasoning sectionThis is the rest"


def test_extract_reasoning_with_begin_token_only():
    parser = _parser()

    reasoning, content = parser.extract_reasoning(
        "[THINK]This is a reasoning section",
        request=None,
    )

    assert reasoning == "This is a reasoning section"
    assert content is None


def test_extract_reasoning_without_think_tokens():
    parser = _parser()

    reasoning, content = parser.extract_reasoning("This is content", request=None)

    assert reasoning is None
    assert content == "This is content"


def test_is_reasoning_end_and_extract_content_ids():
    parser = _parser()

    assert parser.is_reasoning_end([1, parser.start_token_id, parser.end_token_id]) is True
    assert parser.is_reasoning_end([1, 2, 3]) is False

    assert parser.extract_content_ids([7, parser.start_token_id, 9, parser.end_token_id, 10]) == [7, 10]
    assert parser.extract_content_ids([7, parser.start_token_id, 9]) == [7]
