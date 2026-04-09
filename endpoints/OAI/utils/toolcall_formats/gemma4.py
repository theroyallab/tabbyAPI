import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

"""
Gemma 4 family - special-token tool call protocol

Raw format:
    <|tool_call>call:__FUNCTION_NAME__{__KEY_1__:__VAL_1__,__KEY_2__:__VAL_2__,...}<tool_call|>

String values are wrapped in <|"|>...<|"|> escape tokens; other values
(numbers, bools) appear bare. Multiple <|tool_call>...<tool_call|> blocks
may appear for parallel tool calls.

Note the asymmetric outer delimiters: opening is <|tool_call> (pipe on the
left only), closing is <tool_call|> (pipe on the right only). These are
literal Gemma 4 special tokens, not balanced XML.
"""

TOOLCALL_START = "<|tool_call>"
TOOLCALL_END = "<tool_call|>"

_OUTER = re.compile(r"<\|tool_call>(.*?)<tool_call\|>", re.DOTALL)
_HEAD = re.compile(r"^\s*call:([^\s{]+)\s*\{(.*)\}\s*$", re.DOTALL)

# One pass over the args body. Each match is a single key/value pair where
# the value is EITHER a <|"|>...<|"|> string (captured in `strval`) OR a
# bare token up to the next comma or end-of-body (captured in `bareval`).
_ARG = re.compile(
    r"""
    \s* ([^\s:,{}]+) \s* :       \s*        # key
    (?:
        <\|"\|> (?P<strval> .*?) <\|"\|>    # quoted string value
      |
        (?P<bareval> [^,]*?)                # bare value (lazy, up to comma/end)
    )
    \s* (?: , | \Z )
    """,
    re.DOTALL | re.VERBOSE,
)


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []
    for om in _OUTER.finditer(text):
        head = _HEAD.match(om.group(1))
        if not head:
            continue
        func_name = head.group(1).strip()
        if not func_name:
            continue

        args: dict[str, any] = {}
        for am in _ARG.finditer(head.group(2)):
            key = am.group(1)
            if am.group("strval") is not None:
                args[key] = am.group("strval")  # already a str
            else:
                args[key] = coerce_param_value(am.group("bareval").strip())

        results.append(
            ToolCall(function=Tool(name=func_name, arguments=json.dumps(args, ensure_ascii=False)))
        )

    xlogger.debug(
        f"gemma4: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results