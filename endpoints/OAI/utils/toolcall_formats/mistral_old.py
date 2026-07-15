import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool

"""
Mistral family, v2-v7 - JSON list

Raw format:
    [TOOL_CALLS]
    [
        {"name": "__FUNCTION_NAME_1__", "arguments": {"key": "value"}, "id": "__CALL_ID_1__"},
        {"name": "__FUNCTION_NAME_2__", "arguments": {"key": "value"}, "id": "__CALL_ID_2__"}
    ]

The model emits the [TOOL_CALLS] control token followed by a JSON array of
tool call objects. Each object contains "name", "arguments" (already a dict),
and optionally "id" (v3+). Multiple tool calls for parallel invocation appear as
multiple entries in the array.

There is no end token; tool calls simply appear at the end of the response stream.
"""

TOOLCALL_START = "[TOOL_CALLS]"
TOOLCALL_END = None

# Match [TOOL_CALLS] followed by a JSON array
_TOOLCALL_BLOCK = re.compile(r"\[TOOL_CALLS]\s*(\[.*])", re.DOTALL)


def parse_toolcalls(text: str) -> list[ToolCall]:
    match = _TOOLCALL_BLOCK.search(text)
    if not match:
        return []

    raw_json = match.group(1)

    try:
        calls = json.loads(raw_json)
    except json.JSONDecodeError as e:
        xlogger.warning(
            "mistral_old: Failed to parse tool call JSON",
            {"exception": str(e), "raw_text": text, "raw_json": raw_json},
        )
        return []

    if not isinstance(calls, list):
        calls = [calls]

    results = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        func_name = call.get("name")
        if not func_name:
            continue
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        args_json = json.dumps(arguments, ensure_ascii=False)
        func = Tool(name=func_name, arguments=args_json)
        if "id" in call:
            results.append(ToolCall(id=call["id"], function=func))
        else:
            results.append(ToolCall(function=func))

    xlogger.debug(
        f"mistral_old: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
