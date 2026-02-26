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


def test_from_xml_supports_single_quote_object_parameter():
    payload = (
        "<tool_call><function=test_types>"
        "<parameter=obj_param>\n{'key': 'value'}\n</parameter>"
        "</function></tool_call>"
    )

    parsed = ToolCallProcessor.from_xml(payload)

    assert len(parsed) == 1
    assert parsed[0].function.name == "test_types"
    assert _arguments_dict(parsed[0]) == {"obj_param": {"key": "value"}}


def test_from_xml_parses_incomplete_function_block_at_generation_cutoff():
    payload = (
        "I'll call a tool. "
        "<tool_call><function=get_weather>"
        "<parameter=city>\nSeoul\n</parameter>"
        "<parameter=days>\n3\n</parameter>"
        # Missing </function></tool_call> on purpose
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


def test_parse_with_ernie45_parser_handles_tool_call_json():
    payload = (
        "<tool_call>"
        '{"name":"get_weather","arguments":{"city":"Seoul"}}'
        "</tool_call>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="ernie45")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul"}


def test_parse_with_jamba_parser_handles_tool_calls_tag_array():
    payload = (
        "<tool_calls>"
        '[{"name":"get_weather","arguments":{"city":"Seoul","days":2}}]'
        "</tool_calls>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="jamba")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul", "days": 2}


def test_parse_with_minimax_parser_handles_line_delimited_json():
    payload = (
        "<tool_calls>\n"
        '{"name":"foo","arguments":{"x":1}}\n'
        '{"name":"bar","arguments":{"y":2}}\n'
        "</tool_calls>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="minimax")

    assert len(parsed) == 2
    assert parsed[0].function.name == "foo"
    assert _arguments_dict(parsed[0]) == {"x": 1}
    assert parsed[1].function.name == "bar"
    assert _arguments_dict(parsed[1]) == {"y": 2}


def test_parse_with_glm45_parser_handles_name_and_json_body():
    payload = '<tool_call>lookup\n{"id":42,"q":"tabbyapi"}</tool_call>'

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="glm45")

    assert len(parsed) == 1
    assert parsed[0].function.name == "lookup"
    assert _arguments_dict(parsed[0]) == {"id": 42, "q": "tabbyapi"}


def test_parse_with_minimax_m2_parser_handles_invoke_parameters():
    payload = (
        '<minimax:tool_call><invoke name="lookup">'
        '<parameter name="id">42</parameter>'
        '<parameter name="query">tabbyapi</parameter>'
        "</invoke></minimax:tool_call>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="minimax_m2")

    assert len(parsed) == 1
    assert parsed[0].function.name == "lookup"
    assert _arguments_dict(parsed[0]) == {"id": 42, "query": "tabbyapi"}


def test_parse_with_seed_oss_parser_handles_seed_xml():
    payload = (
        "<seed:tool_call><function=get_weather>"
        "<parameter=city>\nSeoul\n</parameter>"
        "<parameter=days>\n2\n</parameter>"
        "</function></seed:tool_call>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="seed_oss")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul", "days": 2}


def test_parse_with_olmo3_parser_handles_function_calls_wrapper():
    payload = "<function_calls>\nget_weather(city='Seoul', days=2)\n</function_calls>"

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="olmo3")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul", "days": 2}


def test_parse_with_step3p5_parser_handles_qwen3_xml_shape():
    payload = (
        "<tool_call><function=search>"
        "<parameter=q>\ntabbyapi\n</parameter>"
        "</function></tool_call>"
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="step3p5")

    assert len(parsed) == 1
    assert parsed[0].function.name == "search"
    assert _arguments_dict(parsed[0]) == {"q": "tabbyapi"}


def test_parse_with_openai_parser_handles_functions_recipient():
    payload = (
        '[{"recipient":"functions.get_weather","content":"{\\"city\\":\\"Seoul\\"}"}]'
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="openai")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul"}


def test_parse_with_mistral_parser_handles_pre_v11_json():
    payload = (
        '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Seoul","days":2}}]'
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="mistral")

    assert len(parsed) == 1
    assert parsed[0].function.name == "get_weather"
    assert _arguments_dict(parsed[0]) == {"city": "Seoul", "days": 2}
    assert parsed[0].id.isalnum()
    assert len(parsed[0].id) == 9


def test_parse_with_mistral_parser_handles_v11_style_segments():
    payload = (
        '[TOOL_CALLS]search{"q":"tabbyapi"}'
        '[TOOL_CALLS]lookup{"id":42}'
    )

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="mistral")

    assert len(parsed) == 2
    assert parsed[0].function.name == "search"
    assert _arguments_dict(parsed[0]) == {"q": "tabbyapi"}
    assert parsed[1].function.name == "lookup"
    assert _arguments_dict(parsed[1]) == {"id": 42}
    assert parsed[1].id.isalnum()
    assert len(parsed[1].id) == 9


def test_parse_with_mistral_parser_falls_back_to_standard_json():
    payload = '[{"name":"lookup","arguments":{"id":42}}]'

    parsed = ToolCallProcessor.parse(payload, format="json", parser_key="mistral")

    assert len(parsed) == 1
    assert parsed[0].function.name == "lookup"
    assert _arguments_dict(parsed[0]) == {"id": 42}


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
