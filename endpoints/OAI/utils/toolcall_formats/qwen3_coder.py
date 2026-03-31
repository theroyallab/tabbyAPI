import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

"""
Qwen3.5 / Qwen3-Coder - pseudo-XML syntax

Raw format:
    <tool_call>
        <function=__FUNCTION_NAME__}> 
            <parameter=__PARAMETER_NAME_1__>
                __PARAMETER_1__
            </parameter>
            <parameter=__PARAMETER_NAME_2__>
                __PARAMETER_2__
            </parameter>
            ...
        </function>
    </tool_call>
"""

# TODO: the outer <tool_call> wrapper is supposedly optional in some deployments; the parser
#   handles both, but detecting tool calls in the stream currently relies on <tool_call> being
#   emitted by the model.

TOOLCALL_START = "<tool_call>"
TOOLCALL_END = "</tool_call>"

_OUTER = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNC = re.compile(r"<function=([^>\s]+)[^>]*>(.*?)</function>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>\s]+)[^>]*>(.*?)</parameter>", re.DOTALL)


def parse_toolcalls(text: str) -> list[ToolCall]:

    # If there are outer <tool_call> wrappers, unwrap them; otherwise use the whole text
    segments: list[tuple[str, str]] = []  # (raw, inner)
    outer_matches = list(_OUTER.finditer(text))
    if outer_matches:
        is_wrapped = False
        for m in outer_matches:
            segments.append((m.group(0), m.group(1)))
    else:
        # No outer wrapper — look for bare <function=...> blocks
        is_wrapped = True
        segments = [(text, text)]

    results = []
    for raw_outer, inner in segments:
        for fm in _FUNC.finditer(inner):
            func_name = fm.group(1)
            func_body = fm.group(2)
            args: dict[str, any] = {}
            for pm in _PARAM.finditer(func_body):
                key = pm.group(1).strip()
                val = pm.group(2).strip()
                val = coerce_param_value(val)
                args[key] = val

            args_json = json.dumps(args, ensure_ascii = False)
            results.append(ToolCall(function = Tool(name = func_name, arguments = args_json)))

    xlogger.debug(
        f"qwen3_coder: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results, "is_wrapped": is_wrapped}
    )
    return results