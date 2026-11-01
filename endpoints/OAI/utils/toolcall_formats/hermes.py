import json
import re

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import (
    FormatSignature,
    unwrap_tool_args,
    parse_json_tool_call,
)

"""
Hermes-style JSON tool calls: Qwen2.5 / Qwen3 (original), SmolLM3, Nous Hermes
and most fine-tunes trained on the same template.

Raw format:
    <tool_call>
    {"name": "__FUNCTION_NAME__", "arguments": {"__PARAMETER_NAME_1__": ...}}
    </tool_call>

One JSON object per <tool_call> block; parallel calls are consecutive blocks.
"""

TOOLCALL_START = "<tool_call>"
TOOLCALL_END = "</tool_call>"

DETECT = FormatSignature(
    # The template's tool instructions spell out the JSON shape. The other
    # <tool_call> formats carry <function= or <arg_key> instead
    template_markers=("<tool_call>", ('{"name":', '{\\"name\\":')),
    template_exclude=("<function=", "<arg_key>"),
    special_tokens=("<tool_call>", "</tool_call>"),
    architectures=("Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "SmolLM3"),
    reasoning_tags=("<think>", "</think>"),
)

_OUTER = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _parse_block(block: str):
    """Parse one block's JSON object, tolerating text around it."""

    block = block.strip()
    try:
        return json.loads(block)
    except (json.JSONDecodeError, ValueError):
        pass

    # The model sometimes wraps the object in a code fence or adds a trailing
    # note; take the outermost braces
    start, end = block.find("{"), block.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(block[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []
    for match in _OUTER.finditer(text):
        inner = match.group(1)
        call = _parse_block(inner)
        if not isinstance(call, dict) or not call.get("name"):
            # Fallback: check if the block contains pseudo-XML function calls (<function=...>)
            if "<function=" in inner:
                from endpoints.OAI.utils.toolcall_formats import qwen3_coder

                parsed_xml = qwen3_coder.parse_toolcalls(match.group(0))
                results.extend(parsed_xml)
            continue

        func_name = str(call["name"])
        raw_args = call.get("arguments", call.get("parameters", {}))
        args = unwrap_tool_args(raw_args, func_name)

        results.append(
            ToolCall(
                function=Tool(
                    name=func_name, arguments=json.dumps(args, ensure_ascii=False)
                )
            )
        )

    # If no outer <tool_call> tags were matched, also check if the entire text is a JSON tool call
    if not results and "{" in text and "}" in text:
        json_call = parse_json_tool_call(text)
        if json_call:
            func_name, args = json_call
            results.append(
                ToolCall(
                    function=Tool(
                        name=func_name, arguments=json.dumps(args, ensure_ascii=False)
                    )
                )
            )

    xlogger.debug(
        f"hermes: Parsed {len(results)} tool calls", {"raw_text": text, "results": results}
    )
    return results
