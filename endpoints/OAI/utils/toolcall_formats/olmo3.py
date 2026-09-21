import json
import re

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import FormatSignature

"""
OLMo 3 - function-call lines with JSON values

Raw format:
    <function_calls>
    __FUNCTION_NAME__(__PARAMETER_NAME_1__=__JSON_VALUE_1__, __PARAMETER_NAME_2__=__JSON_VALUE_2__)
    __FUNCTION_NAME__(...)
    </function_calls>

One call per line inside a single block. Values are JSON literals (the template
renders them with tojson), so true/false/null and nested objects are JSON, not
Python. Keys are identifiers.
"""

TOOLCALL_START = "<function_calls>"
TOOLCALL_END = "</function_calls>"

DETECT = FormatSignature(
    template_markers=("<function_calls>", "</function_calls>"),
    special_tokens=("<function_calls>", "</function_calls>"),
    architectures=("Olmo3", "Olmo2"),
    reasoning_tags=("<think>", "</think>"),
)

_OUTER = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)
_CALL = re.compile(r"([A-Za-z_][\w.\-]*)\s*\((.*)\)\s*$", re.DOTALL)
_KEY = re.compile(r"\s*([A-Za-z_][\w\-]*)\s*=\s*", re.DOTALL)


def _split_args(text: str) -> list[str]:
    """Split a call's argument text on top-level commas, respecting strings and nesting."""

    parts, depth, quote, escape, start = [], 0, None, False, 0
    for i, ch in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        parts.append(tail)
    return parts


def _parse_value(raw: str):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    # Python-style literals the model may fall back to
    literal = {"True": True, "False": False, "None": None}
    if raw in literal:
        return literal[raw]
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1]
    return raw


def _parse_call(line: str) -> ToolCall | None:
    match = _CALL.match(line.strip())
    if not match:
        return None
    name, arg_text = match.group(1), match.group(2)

    args = {}
    for part in _split_args(arg_text):
        key_match = _KEY.match(part)
        if not key_match:
            continue
        args[key_match.group(1)] = _parse_value(part[key_match.end() :])

    return ToolCall(function=Tool(name=name, arguments=json.dumps(args, ensure_ascii=False)))


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []
    blocks = [m.group(1) for m in _OUTER.finditer(text)]
    if not blocks and TOOLCALL_START in text:
        # Unterminated block at the end of the stream
        blocks = [text[text.rfind(TOOLCALL_START) + len(TOOLCALL_START) :]]

    for block in blocks:
        # Calls are one per line, but a multi-line JSON value can span lines;
        # re-join continuation lines until the parentheses balance
        pending = ""
        for line in block.splitlines():
            pending = f"{pending}\n{line}" if pending else line
            if pending.count("(") <= pending.count(")"):
                call = _parse_call(pending)
                if call is not None:
                    results.append(call)
                pending = ""
        if pending.strip():
            call = _parse_call(pending)
            if call is not None:
                results.append(call)

    xlogger.debug(
        f"olmo3: Parsed {len(results)} tool calls", {"raw_text": text, "results": results}
    )
    return results
