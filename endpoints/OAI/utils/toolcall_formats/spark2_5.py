import json
import re

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

"""
Spark X2.5 - bar-pipe tagged, arg-key/arg-value XML syntax

Raw format:

     <|tool_call|>
        add<arg_key>a</arg_key><arg_value>2</arg_value><arg_key>b</arg_key><arg_value>3</arg_value>
     <|/tool_call|>

Notes:
  * The function name sits directly between the opening tag and the first
    <arg_key> — there is no <function=...> element (unlike qwen3_coder).
  * Arguments are consecutive <arg_key>/<arg_value> pairs; one call carries
    as many pairs as the model emits.
  * Some runs (observed on the small 4B variant) drop the pipe characters
    and emit a bare <tool_call>/</tool_call> instead. The stream router
    keys off the tagged form, so the bare form passes through as content;
    this module therefore recognizes the pipe form only, consistent with
    the other format parsers in this directory.
"""

TOOLCALL_START = "<|tool_call|>"
TOOLCALL_END = "<|/tool_call|>"

# Function name: first run of word/dash characters after the opening tag
# (tolerates whitespace between the tag and the name).
_NAME = re.compile(r"^\s*([\w-]+)")

# <arg_key>name</arg_key> <arg_value>value</arg_value> pairs, with
# order-insensitive whitespace between the elements.
_ARG = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


def parse_toolcalls(text: str) -> list[ToolCall]:
    """
    Parse one or more Spark X2.5 tool calls out of raw generated text.

    A call is delimited by TOOLCALL_START / TOOLCALL_END. An unterminated
    trailing block (stream end) is parsed from whatever follows the tag.
    Never raises; returns an empty list when nothing matches.
    """
    results: list[ToolCall] = []
    pos = 0
    n = len(text)

    while pos < n:
        start = text.find(TOOLCALL_START, pos)
        if start == -1:
            break
        end = text.find(TOOLCALL_END, start)
        if end == -1:
            end = n  # unterminated block at end of stream
        pos = end + len(TOOLCALL_END)

        body = text[start + len(TOOLCALL_START):end].strip()

        name_m = _NAME.match(body)
        name = name_m.group(1) if name_m else ""
        arg_region = body[name_m.end():] if name_m else body

        args: dict[str, any] = {}
        for am in _ARG.finditer(arg_region):
            key = am.group(1).strip()
            if key:
                args[key] = coerce_param_value(am.group(2))

        if name:
            results.append(
                ToolCall(
                    function=Tool(
                        name=name,
                        arguments=json.dumps(args, ensure_ascii=False),
                    )
                )
            )

    if results:
        xlogger.debug(
            f"spark2_5: Parsed {len(results)} tool calls",
            {"raw_text": text, "results": results},
        )

    return results
