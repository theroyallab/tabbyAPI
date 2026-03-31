import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

"""
MiniMax M2 family (M2, M2.1, M2.5) - structured XML syntax

Raw format:
    <minimax:tool_call>
        <invoke name="__FUNCTION_NAME__">
            <parameter name="__PARAMETER_NAME_1__">__PARAMETER_1__</parameter>
            <parameter name="__PARAMETER_NAME_2__">__PARAMETER_2__</parameter>
            ...
        </invoke>
    </minimax:tool_call>

Multiple <invoke> blocks may appear within a single <minimax:tool_call> wrapper
for parallel tool calls. The input text may contain multiple <minimax:tool_call>
blocks.

Note: This format does NOT apply to MiniMax-M1, which uses a different
JSON-based tool call format.
"""

TOOLCALL_START = "<minimax:tool_call>"
TOOLCALL_END = "</minimax:tool_call>"

_OUTER = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
_INVOKE = re.compile(r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>', re.DOTALL)
_PARAM = re.compile(r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>', re.DOTALL)


def parse_toolcalls(text: str) -> list[ToolCall]:

    outer_matches = list(_OUTER.finditer(text))

    results = []
    for om in outer_matches:
        inner = om.group(1)
        for im in _INVOKE.finditer(inner):
            func_name = im.group(1)
            func_body = im.group(2)
            args: dict[str, any] = {}
            for pm in _PARAM.finditer(func_body):
                key = pm.group(1).strip()
                val = pm.group(2).strip()
                val = coerce_param_value(val)
                args[key] = val

            args_json = json.dumps(args, ensure_ascii=False)
            results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"minimax_m2: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results