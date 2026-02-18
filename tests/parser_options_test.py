"""Tests for vLLM-compatible parser option mapping."""

from endpoints.OAI.utils.parser_options import (
    list_tool_call_parsers,
    parser_uses_native_tool_generation,
    resolve_tool_call_parser_key,
    resolve_tool_call_format,
)


def test_parser_key_registry_contains_core_vllm_keys():
    parser_keys = list_tool_call_parsers()

    assert "openai" in parser_keys
    assert "qwen3_coder" in parser_keys
    assert "qwen3_xml" in parser_keys
    assert "mistral" in parser_keys
    assert "deepseek_v3" in parser_keys
    assert "llama" in parser_keys


def test_resolve_tool_call_format_uses_vllm_mapping():
    assert resolve_tool_call_format("openai", "json") == "json"
    assert resolve_tool_call_format("qwen3_coder", "json") == "xml"
    assert resolve_tool_call_format("auto", "json") == "auto"
    assert resolve_tool_call_format("llama", "json") == "json"
    assert resolve_tool_call_parser_key("llama") == "llama3_json"


def test_resolve_tool_call_format_falls_back_and_rejects_unknown():
    assert resolve_tool_call_format(None, "json") == "json"
    assert resolve_tool_call_format("unknown_parser", "json") == ""


def test_native_generation_flags_cover_native_syntax_parsers():
    assert parser_uses_native_tool_generation("qwen3_coder", "json") is True
    assert parser_uses_native_tool_generation("deepseek_v31", "json") is True
    assert parser_uses_native_tool_generation("pythonic", "json") is True
    assert parser_uses_native_tool_generation("hermes", "json") is False
