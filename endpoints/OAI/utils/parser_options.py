"""Parser option helpers for vLLM-compatible chat settings."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Dict, Set


# Mirrors vLLM parser keys to keep CLI/config ergonomics familiar.
# Source of truth: vllm/tool_parsers/__init__.py::_TOOL_PARSERS_TO_REGISTER
# Format is the parsing mode supported by ToolCallProcessor.
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


def list_tool_call_parsers() -> Set[str]:
    return set(TOOL_CALL_PARSER_FORMATS.keys())


def resolve_tool_call_format(
    tool_call_parser: str | None, fallback_format: str
) -> str:
    """Resolve effective parser format from configured parser key."""
    if not tool_call_parser:
        return fallback_format
    return TOOL_CALL_PARSER_FORMATS.get(tool_call_parser, "")
