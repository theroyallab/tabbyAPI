"""Tool call processing utilities for OAI server."""

from common.logger import xlogger
from typing import List

from endpoints.OAI.types.tools import ToolCall
from endpoints.OAI.utils.toolcall_formats import (
    qwen3_coder,
    minimax_m2,
    glm4_5,
    mistral_old,
    mistral,
)

ALL_TOOLCALL_FORMATS = {
    "glm4_5": glm4_5,
    "glm4_6": glm4_5,
    "glm4_7": glm4_5,
    "minimax_m2": minimax_m2,
    "minimax_m2_1": minimax_m2,
    "minimax_m2_5": minimax_m2,
    "mistral_old": mistral_old,
    "mistral": mistral,
    "qwen3_coder": qwen3_coder,
    "qwen3_5": qwen3_coder,
}


def _get_parser(tool_format: str):
    if not tool_format:
        return None
    parser = ALL_TOOLCALL_FORMATS.get(tool_format)
    if not parser:
        xlogger.error(f"Unknown tool format given: {tool_format}")
    return parser


def get_toolcall_tags(tool_format: str):
    parser = _get_parser(tool_format)
    if not parser:
        return None, None
    return parser.TOOLCALL_START, parser.TOOLCALL_END


def is_supported_format(tool_format: str) -> bool:
    return tool_format in ALL_TOOLCALL_FORMATS


def parse_toolcalls(tool_calls_str: str, tool_format: str) -> List[ToolCall]:
    """
    Dispatch tool call parsing to the appropriate format handler.

    Args:
        tool_calls_str: Raw tool call text from model generation.
        tool_format: See below

    Returns:
        List of parsed ToolCall objects. Empty list on parse failure (never raises).
    """

    try:
        parser = _get_parser(tool_format)
        if not parser:
            return []

        return parser.parse_toolcalls(tool_calls_str)

    except Exception as e:
        xlogger.error(
            "ToolCallProcessor.parse: Failed to parse tool calls",
            {"tool_format": tool_format, "e": str(e)},
            details=f"(format={tool_format}): {e}",
        )
        return []
