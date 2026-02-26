"""Tests for vLLM-compatible parser option mapping."""

from endpoints.OAI.utils.parser_options import (
    TOOL_CALL_PARSER_FORMATS,
    list_tool_call_parsers,
    parser_uses_native_tool_generation,
    resolve_tool_call_parser_key,
    resolve_tool_call_format,
)
from endpoints.OAI.utils.tools import ToolCallProcessor


VLLM_CANONICAL_TOOL_PARSERS = {
    "deepseek_v3",
    "deepseek_v31",
    "deepseek_v32",
    "ernie45",
    "glm45",
    "glm47",
    "granite",
    "granite-20b-fc",
    "hermes",
    "hunyuan_a13b",
    "internlm",
    "jamba",
    "kimi_k2",
    "llama3_json",
    "llama4_json",
    "llama4_pythonic",
    "longcat",
    "minimax",
    "minimax_m2",
    "mistral",
    "olmo3",
    "openai",
    "phi4_mini_json",
    "pythonic",
    "qwen3_coder",
    "qwen3_xml",
    "seed_oss",
    "step3",
    "step3p5",
    "xlam",
    "gigachat3",
    "functiongemma",
}


def test_parser_key_registry_contains_core_vllm_keys():
    parser_keys = list_tool_call_parsers()

    assert "openai" in parser_keys
    assert "qwen3_coder" in parser_keys
    assert "qwen3_xml" in parser_keys
    assert "mistral" in parser_keys
    assert "deepseek_v3" in parser_keys
    assert "llama" in parser_keys


def test_parser_key_registry_matches_vllm_set_plus_local_aliases():
    parser_keys = list_tool_call_parsers()

    # Canonical set should match current vLLM registry.
    assert VLLM_CANONICAL_TOOL_PARSERS.issubset(parser_keys)
    assert set(TOOL_CALL_PARSER_FORMATS.keys()) - {"auto"} == VLLM_CANONICAL_TOOL_PARSERS

    # Local compatibility alias.
    assert "llama" in parser_keys


def test_every_configured_canonical_parser_has_dispatch_handler():
    dispatcher = ToolCallProcessor._parser_dispatcher()
    canonical = set(TOOL_CALL_PARSER_FORMATS.keys()) - {"auto"}

    missing = sorted(canonical - set(dispatcher.keys()))
    assert missing == []


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
    assert parser_uses_native_tool_generation("mistral", "json") is True
    assert parser_uses_native_tool_generation("hermes", "json") is False
