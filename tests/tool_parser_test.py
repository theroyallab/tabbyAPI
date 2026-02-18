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


def test_parse_with_hermes_parser_handles_wrapped_json():
    payload = (
        "<tool_call>"
        '{"name":"weather","arguments":{"city":"Seoul"}}'
        "</tool_call>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="hermes")

    assert len(parsed) == 1
    assert parsed[0].function.name == "weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul"}


def test_parse_with_llama_parser_handles_sequential_json():
    payload = (
        "<|python_tag|>"
        '{"name":"a","arguments":{"x":1}};'
        '{"name":"b","arguments":{"y":2}}'
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="llama")

    assert len(parsed) == 2
    assert parsed[0].function.name == "a"
    assert _arguments_dict(parsed[0]) == {"x": 1}
    assert parsed[1].function.name == "b"
    assert _arguments_dict(parsed[1]) == {"y": 2}


def test_parse_with_pythonic_parser_extracts_function_calls():
    payload = "[get_weather(city='San Francisco', days=3)]"

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="pythonic")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "San Francisco", "days": 3}


def test_parse_with_deepseek_v31_parser():
    payload = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="deepseek_v31")

    assert len(parsed) == 1
    assert parsed[0].function.name == "foo"
    assert _arguments_dict(parsed[0]) == {"x": 1}


def test_parse_with_deepseek_v3_parser():
    payload = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>lookup\n"
        "```json\n"
        '{"q":"tabbyapi"}'
        "\n```\n"
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="deepseek_v3")

    assert len(parsed) == 1
    assert parsed[0].function.name == "lookup"
    assert _arguments_dict(parsed[0]) == {"q": "tabbyapi"}


def test_parse_with_deepseek_v32_parser():
    payload = (
        "<｜DSML｜function_calls>"
        '<｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="location" string="true">Seoul</｜DSML｜parameter>'
        '<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜function_calls>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="deepseek_v32")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"location": "Seoul", "days": 3}


def test_parse_with_openai_parser_handles_functions_recipient():
    payload = (
        '[{"recipient":"functions.get_weather","content":"{\\"city\\":\\"Seoul\\"}"}]'
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="openai")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul"}


def test_parser_key_dispatch_overrides_format_for_qwen3_xml():
    payload = (
        "<tool_call><function=search>"
        "<parameter=q>\ntabby\n</parameter>"
        "</function></tool_call>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="qwen3_xml")

    assert len(parsed) == 1
    assert parsed[0].function.name == "search"
    assert _arguments_dict(parsed[0]) == {"q": "tabby"}


def test_parser_failure_falls_back_to_format_parser():
    payload = (
        "<tool_call><function=lookup>"
        "<parameter=id>\n42\n</parameter>"
        "</function></tool_call>"
    )

    parsed = ToolCallProcessor.parse(payload, format="xml", parser_key="openai")

    assert len(parsed) == 1
    assert parsed[0].function.name == "lookup"
    assert _arguments_dict(parsed[0]) == {"id": 42}
