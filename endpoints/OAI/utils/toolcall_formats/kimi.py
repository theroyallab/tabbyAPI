import json
import re

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import FormatSignature

"""
Kimi K2 / Kimi Linear - special-token delimited calls with JSON arguments

Raw format:
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.__FUNCTION_NAME__:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
    <|tool_call_begin|>functions.__FUNCTION_NAME__:1<|tool_call_argument_begin|>{...}<|tool_call_end|>
    <|tool_calls_section_end|>

The argument text after <|tool_call_argument_begin|> is a JSON object.

Each call carries an id of the form functions.<name>:<index>; the function name
is the part between the namespace and the index. The id is passed through as
the tool call id so tool results can be matched back the way the template
expects ("## Return of <id>").
"""

TOOLCALL_START = "<|tool_calls_section_begin|>"
TOOLCALL_END = "<|tool_calls_section_end|>"

DETECT = FormatSignature(
    template_markers=("<|tool_call_begin|>", "<|tool_call_argument_begin|>"),
    special_tokens=("<|tool_call_begin|>", "<|tool_call_argument_begin|>", "<|tool_call_end|>"),
    architectures=("KimiLinear", "DeepseekV3ForCausalLM"),
    reasoning_tags=("<think>", "</think>"),
)

_CALL = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<id>[^<\s]+)\s*<\|tool_call_argument_begin\|>"
    r"\s*(?P<args>.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)
_ID = re.compile(r"^(?:functions\.)?(?P<name>.+?)(?::\d+)?$")


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []
    for match in _CALL.finditer(text):
        call_id = match.group("id")
        name = _ID.match(call_id).group("name")

        raw_args = match.group("args")
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, ValueError):
            args = {"input": raw_args}
        if not isinstance(args, dict):
            args = {"input": args}

        results.append(
            ToolCall(
                id=call_id,
                function=Tool(name=name, arguments=json.dumps(args, ensure_ascii=False)),
            )
        )

    xlogger.debug(f"kimi: Parsed {len(results)} tool calls", {"raw_text": text, "results": results})
    return results
