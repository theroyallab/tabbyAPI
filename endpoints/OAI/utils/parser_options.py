"""Parser option helpers for vLLM-compatible chat settings."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Dict, Set


# Mirrors vLLM parser keys to keep CLI/config ergonomics familiar.
# Source of truth: vllm/tool_parsers/__init__.py::_TOOL_PARSERS_TO_REGISTER
# Format is the fallback parsing mode supported by ToolCallProcessor.
TOOL_CALL_PARSER_FORMATS: Dict[str, str] = {
    "deepseek_v3": "json",
    "deepseek_v31": "json",
    "deepseek_v32": "json",
    "ernie45": "json",
    "glm45": "json",
    "glm47": "json",
    "granite-20b-fc": "json",
    "granite": "json",
    "hermes": "json",
    "hunyuan_a13b": "json",
    "internlm": "json",
    "jamba": "json",
    "kimi_k2": "json",
    "llama3_json": "json",
    "llama4_json": "json",
    "llama4_pythonic": "json",
    "longcat": "json",
    "minimax_m2": "json",
    "minimax": "json",
    "mistral": "json",
    "olmo3": "json",
    "openai": "json",
    "phi4_mini_json": "json",
    "pythonic": "json",
    "qwen3_coder": "xml",
    "qwen3_xml": "xml",
    "seed_oss": "json",
    "step3": "json",
    "step3p5": "json",
    "xlam": "json",
    "gigachat3": "json",
    "functiongemma": "json",
    # Convenience alias for mixed/inferred content
    "auto": "auto",
}

# Compatibility aliases accepted by this server.
# Keys are user-facing parser names, values are canonical parser keys.
TOOL_CALL_PARSER_ALIASES: Dict[str, str] = {
    "llama": "llama3_json",
}

# Parsers that should generate tool calls in their native syntax on tool pass
# (no JSON schema constraint). Most JSON-style parsers should stay constrained.
NATIVE_TOOL_GENERATION_PARSERS: Set[str] = {
    "auto",
    "deepseek_v3",
    "deepseek_v31",
    "deepseek_v32",
    "llama4_pythonic",
    "pythonic",
    "qwen3_coder",
    "qwen3_xml",
}


def resolve_tool_call_parser_key(tool_call_parser: str | None) -> str | None:
    """Normalize a user parser key to its canonical key."""
    if not tool_call_parser:
        return None
    return TOOL_CALL_PARSER_ALIASES.get(tool_call_parser, tool_call_parser)


def list_tool_call_parsers() -> Set[str]:
    return set(TOOL_CALL_PARSER_FORMATS.keys()).union(TOOL_CALL_PARSER_ALIASES.keys())


def resolve_tool_call_format(
    tool_call_parser: str | None, fallback_format: str
) -> str:
    """Resolve effective parser format from configured parser key."""
    if not tool_call_parser:
        return fallback_format
    parser_key = resolve_tool_call_parser_key(tool_call_parser)
    return TOOL_CALL_PARSER_FORMATS.get(parser_key, "")


def parser_uses_native_tool_generation(
    tool_call_parser: str | None, fallback_format: str
) -> bool:
    """Whether tool pass should use native model format (unconstrained)."""
    if not tool_call_parser:
        return fallback_format in ("xml", "auto")
    parser_key = resolve_tool_call_parser_key(tool_call_parser)
    if parser_key in NATIVE_TOOL_GENERATION_PARSERS:
        return True
    return resolve_tool_call_format(parser_key, fallback_format) in ("xml", "auto")
