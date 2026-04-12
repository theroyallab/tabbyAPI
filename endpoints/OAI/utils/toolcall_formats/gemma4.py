import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool

"""
Gemma-4 - pseudo-JSON syntax

Raw format:
    <|tool_call>call:function_name{
        key1:<|"|>value1<|"|>,
        key2:true,
        nested:{inner:<|"|>value<|"|>}
    }<tool_call|>
"""

TOOLCALL_START = "<|tool_call>"
TOOLCALL_END = "<tool_call|>"

_CALL_PATTERN = re.compile(
    r"<\|tool_call>call:\s*([a-zA-Z0-9_.-]+)\s*\{(.*?)\}<tool_call\|>", re.DOTALL
)
_STRING_PATTERN = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"|<\|"\|>(.*?)<\|"\|>', re.DOTALL)
_KEY_PATTERN = re.compile(r"([a-zA-Z0-9_]+)\s*:")


def _gemma_to_json(raw_args: str) -> dict:
    if not raw_args or not raw_args.strip():
        return {}

    strings = []

    def repl_string(match):
        s = match.group(0)
        # If it's a Gemma custom string <|"|>...<|"|>
        if s.startswith('<|"|>'):
            # Dump the inner content using C-optimized dumps to handle JSON escaping natively
            s = json.dumps(match.group(2))

        strings.append(s)
        # Use a token containing symbols (@) so _KEY_PATTERN won't accidentally match it
        return f"@STR_{len(strings) - 1}@"

    # 1. Protect all strings (standard and Gemma)
    text = _STRING_PATTERN.sub(repl_string, raw_args)

    # 2. Quote bare keys (e.g., key: -> "key":)
    text = _KEY_PATTERN.sub(r'"\1":', text)

    # 3. Restore strings directly into the raw JSON text
    text = re.sub(r"@STR_(\d+)@", lambda m: strings[int(m.group(1))], text)

    # 4. Ensure wrapped in braces
    text = text.strip()
    if not text.startswith("{"):
        text = f"{{{text}}}"

    # 5. Native parse (acts as structural validation + converts true/false/null safely)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        xlogger.debug("gemma4: JSON decoding failed for raw args", {"text": text, "error": str(e)})
        return {}


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []

    for m in _CALL_PATTERN.finditer(text):
        func_name = m.group(1)
        raw_args = m.group(2)

        args_dict = _gemma_to_json(raw_args)

        # Standardize strictly to JSON for endpoints wrapper
        args_json = json.dumps(args_dict, ensure_ascii=False)
        results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"gemma4: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
