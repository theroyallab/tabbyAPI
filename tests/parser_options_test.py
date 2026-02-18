"""Tests for vLLM-compatible parser option mapping."""

from endpoints.OAI.utils.parser_options import (
    list_tool_call_parsers,
    resolve_tool_call_format,
)


def test_parser_key_registry_contains_core_vllm_keys():
    parser_keys = list_tool_call_parsers()

    assert "openai" in parser_keys
    assert "qwen3_coder" in parser_keys
    assert "qwen3_xml" in parser_keys
    assert "mistral" in parser_keys
    assert "deepseek_v3" in parser_keys


def test_resolve_tool_call_format_uses_vllm_mapping():
    assert resolve_tool_call_format("openai", "json") == "json"
    assert resolve_tool_call_format("qwen3_coder", "json") == "xml"
    assert resolve_tool_call_format("auto", "json") == "auto"


def test_resolve_tool_call_format_falls_back_and_rejects_unknown():
    assert resolve_tool_call_format(None, "json") == "json"
    assert resolve_tool_call_format("unknown_parser", "json") == ""
