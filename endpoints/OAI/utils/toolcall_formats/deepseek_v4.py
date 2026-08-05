import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool

"""
DeepSeek-V4 - DSML invoke blocks with explicitly typed parameters

Raw format:
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="__FUNCTION_NAME__">
    <｜DSML｜parameter name="__PARAM_1__" string="true|false">__VALUE_1__</｜DSML｜parameter>
    <｜DSML｜parameter name="__PARAM_2__" string="true|false">__VALUE_2__</｜DSML｜parameter>
    </｜DSML｜invoke>
    <｜DSML｜invoke name="__FUNCTION_NAME_2__">
    ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

The string attribute makes parameter typing explicit: string="true" values
are taken verbatim, string="false" values are parsed as JSON. ｜DSML｜ is a
single added token; the rest of each tag is ordinary text.

Note that the model's chat template requires tool results to appear in the
same order as the corresponding tool calls; the server sorts tool messages
before templating to guarantee this.
"""

TOOLCALL_START = "<｜DSML｜tool_calls>"
TOOLCALL_END = "</｜DSML｜tool_calls>"

_INVOKE = re.compile(r'<｜DSML｜invoke name="(.*?)">(.*?)</｜DSML｜invoke>', re.DOTALL)
_PARAM = re.compile(
    r'<｜DSML｜parameter name="(.*?)" string="(true|false)">(.*?)</｜DSML｜parameter>',
    re.DOTALL,
)


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []
    for im in _INVOKE.finditer(text):
        func_name = im.group(1).strip()
        if not func_name:
            continue

        args = {}
        for pm in _PARAM.finditer(im.group(2)):
            param_name, is_string, value = pm.groups()
            if is_string == "true":
                args[param_name] = value
            else:
                try:
                    args[param_name] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[param_name] = value

        args_json = json.dumps(args, ensure_ascii=False)
        results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"deepseek_v4: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
