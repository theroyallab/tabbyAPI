import json
import re

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import FormatSignature

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
        call = _parse_block(match.group(1))
        if not isinstance(call, dict) or not call.get("name"):
            continue

        args = call.get("arguments", call.get("parameters", {}))
        if isinstance(args, str):
            # Arguments already serialized; keep them if they're valid JSON
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"input": args}
        if args is None:
            args = {}

        results.append(
            ToolCall(
                function=Tool(
                    name=str(call["name"]), arguments=json.dumps(args, ensure_ascii=False)
                )
            )
        )

    xlogger.debug(
        f"hermes: Parsed {len(results)} tool calls", {"raw_text": text, "results": results}
    )
    return results
