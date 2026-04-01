import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool

"""
Mistral family, v11+ - Name/Args tokens

Raw format (single call):
    [TOOL_CALLS]get_weather[ARGS]{"location": "Paris", "format": "celsius"}

Raw format (parallel calls):
    [TOOL_CALLS]read[ARGS]{"filePath": "/path/x.jpg"}
    [TOOL_CALLS]read[ARGS]{"filePath": "/path/y.jpg"}

The model emits [TOOL_CALLS] followed by the function name as plain text,
then [ARGS] followed by a JSON object of arguments. For parallel calls the
pattern repeats with no separator. There is no id field in the raw output;
IDs should be assigned by the API server.

There is no end token; the sequence ends at EOS.
"""

TOOLCALL_START = "[TOOL_CALLS]"
TOOLCALL_END = None

_TOOLCALL_PAIR = re.compile(
    r"\[TOOL_CALLS]\s*(\S+?)\s*\[ARGS]\s*(\{.*?)(?=\[TOOL_CALLS]|$)", re.DOTALL
)


def parse_toolcalls(text: str) -> list[ToolCall]:
    matches = _TOOLCALL_PAIR.findall(text)
    if not matches:
        return []

    results = []
    for func_name, raw_args in matches:
        func_name = func_name.strip()
        raw_args = raw_args.strip()
        if not func_name:
            continue

        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as e:
            xlogger.warning(
                "mistral: Failed to parse tool call arguments",
                {
                    "exception": str(e),
                    "function": func_name,
                    "raw_args": raw_args,
                },
            )
            continue

        if not isinstance(arguments, dict):
            xlogger.warning(
                "mistral: Arguments is not a dict",
                {"function": func_name, "arguments": arguments},
            )
            continue

        args_json = json.dumps(arguments, ensure_ascii=False)
        results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"mistral: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
