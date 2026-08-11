import json
import re

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool

"""
Muse Glimmer - structural message format with ATEM tool call syntax

Tool calls are assistant messages addressed to a tool recipient. The Glimmer
stream parser emits each one on the tool channel as:

    <|start|>assistant to=__NAME__<|message|><atem:function_calls>
    <atem:invoke name="__NAME__">
    <atem:parameter name="__PARAMETER_NAME__">__PARAMETER_VALUE__</atem:parameter>
    ...
    </atem:invoke>
    </atem:function_calls><|eom|>

The invoke name inside the body is authoritative; the recipient duplicates
it. Scalar parameter values are written as-is, lists and objects as JSON.
Per the format's own system prompt, the output is not expected to be valid
XML and is parsed with regular expressions.

This format requires the Glimmer stream parser and is selected automatically
for Muse Glimmer models; TOOLCALL_START/END are None since there are no tags
for TagStreamParser to scan for.
"""

TOOLCALL_START = None
TOOLCALL_END = None

_INVOKE_OPEN = re.compile(r'<atem:invoke name="([^"]+)">')
_PARAM = re.compile(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', re.DOTALL)


def _coerce_param_value(raw: str) -> any:
    """
    JSON-decode lists, objects, numbers, booleans and null; keep anything
    else as a string. String values are not stripped: the template renders
    them verbatim, with no whitespace around the parameter tags.
    """

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def parse_toolcalls(text: str) -> list[ToolCall]:
    # Scan for invoke blocks directly: every message on the tool channel is a
    # tool call, and each may hold one or more invokes. A block runs until
    # its closing tag or, if the model omitted it, the next invoke or the end
    # of the text.
    opens = list(_INVOKE_OPEN.finditer(text))

    results = []
    for idx, open_match in enumerate(opens):
        func_name = open_match.group(1)
        body_start = open_match.end()
        body_end = opens[idx + 1].start() if idx + 1 < len(opens) else len(text)
        close = text.find("</atem:invoke>", body_start, body_end)
        body = text[body_start : close if close != -1 else body_end]

        args = {}
        for pm in _PARAM.finditer(body):
            args[pm.group(1)] = _coerce_param_value(pm.group(2))

        args_json = json.dumps(args, ensure_ascii=False)
        results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    if not opens and "<|message|>" in text:
        xlogger.warning(
            "muse_glimmer: Tool message contains no parseable <atem:invoke> block",
            {"raw_text": text},
        )

    xlogger.debug(
        f"muse_glimmer: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
