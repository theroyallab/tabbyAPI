import re
import json
from itertools import zip_longest
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

"""
Hy3 (Tencent Hunyuan) - tokenized XML with interleaved key/value pairs

Raw format:
    <tool_calls:opensource>
    <tool_call:opensource>__FUNCTION_NAME__<tool_sep:opensource>
    <arg_key:opensource>__PARAMETER_NAME_1__</arg_key:opensource>
    <arg_value:opensource>__PARAMETER_VALUE_1__</arg_value:opensource>
    <arg_key:opensource>__PARAMETER_NAME_2__</arg_key:opensource>
    <arg_value:opensource>__PARAMETER_VALUE_2__</arg_value:opensource>
    ...
    </tool_call:opensource>
    ...
    </tool_calls:opensource>

Every tag is a single added token; the ':opensource' suffix is part of the
token string. Parallel calls appear as multiple <tool_call:opensource> blocks
inside a single <tool_calls:opensource> wrapper, which the model closes before
emitting EOS. String argument values are rendered raw; other types are
rendered as JSON.
"""

TOOLCALL_START = "<tool_calls:opensource>"
TOOLCALL_END = "</tool_calls:opensource>"

_OUTER = re.compile(r"<tool_call:opensource>(.*?)</tool_call:opensource>", re.DOTALL)
_FUNC_NAME = re.compile(r"^(.*?)(?=<tool_sep:opensource>|<arg_key:opensource>|$)", re.DOTALL)
_ARG_KEY = re.compile(r"<arg_key:opensource>(.*?)</arg_key:opensource>", re.DOTALL)
_ARG_VALUE = re.compile(r"<arg_value:opensource>(.*?)</arg_value:opensource>", re.DOTALL)


def parse_toolcalls(text: str) -> list[ToolCall]:
    outer_matches = list(_OUTER.finditer(text))

    results = []
    for om in outer_matches:
        inner = om.group(1)

        # Extract function name: everything before <tool_sep:opensource>
        name_match = _FUNC_NAME.match(inner)
        if not name_match:
            continue
        func_name = name_match.group(1).strip()
        if not func_name:
            continue

        # Extract interleaved key/value pairs
        keys = [m.group(1).strip() for m in _ARG_KEY.finditer(inner)]
        values = [m.group(1).strip() for m in _ARG_VALUE.finditer(inner)]

        args: dict[str, any] = {}
        for key, val in zip_longest(keys, values):
            args[key] = coerce_param_value(val)

        args_json = json.dumps(args, ensure_ascii=False)
        results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"hy3: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
