"""Tests for tool call parsing helpers."""

import json

from endpoints.OAI.utils.tools import ToolCallProcessor


def _arguments_dict(tool_call):
    return json.loads(tool_call.function.arguments)


def test_from_json_handles_markdown_fences_and_flat_shape():
    payload = """```json
[{"name": "get_weather", "arguments": {"city": "Seoul"}}]
```"""

    parsed = ToolCallProcessor.from_json(payload)

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul"}


def test_from_xml_parses_qwen3_coder_style_blocks():
    payload = (
        "<think>internal reasoning</think>"
        "<tool_call><function=get_weather>"
        "<parameter=city>\nSeoul\n</parameter>"
        "<parameter=days>\n3\n</parameter>"
        "</function></tool_call>"
    )

    parsed = ToolCallProcessor.from_xml(payload)

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul", "days": 3}


def test_from_auto_parses_json_inside_tool_call_wrapper():
    payload = (
        "<tool_call>"
        '{"name": "search", "arguments": {"query": "tabbyapi"}}'
        "</tool_call>"
    )

    parsed = ToolCallProcessor.from_auto(payload)

    assert len(parsed) == 1
    assert parsed[0].function.name == "search"
    assert _arguments_dict(parsed[0]) == {"query": "tabbyapi"}


def test_extract_content_and_tools_splits_content_from_xml_calls():
    payload = (
        "I will call a tool now. "
        "<tool_call><function=search><parameter=q>\ntabby\n</parameter>"
        "</function></tool_call>"
        " Done."
    )

    content, parsed = ToolCallProcessor.extract_content_and_tools(payload)

    assert "I will call a tool now." in content
    assert "Done." in content
    assert len(parsed) == 1
    assert parsed[0].function.name == "search"


def test_filter_by_name_keeps_only_requested_function():
    payload = (
        "["
        '{"name": "a", "arguments": {}},'
        '{"name": "b", "arguments": {}}'
        "]"
    )
    parsed = ToolCallProcessor.from_json(payload)

    filtered = ToolCallProcessor.filter_by_name(parsed, "b")

    assert len(filtered) == 1
    assert filtered[0].function.name == "b"
