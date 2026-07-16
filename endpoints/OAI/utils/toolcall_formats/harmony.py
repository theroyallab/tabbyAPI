import json
import re

from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool

"""
Harmony (gpt-oss) - structural message format

Tool calls are not delimited by tags in plain text; they are commentary
messages addressed to a recipient. The Harmony stream parser emits each one
on the tool channel as:

    <|channel|>commentary to=functions.__NAME__ <|constrain|>json<|message|>__JSON_ARGS__<|call|>

The recipient may also precede the channel (`assistant
to=functions.x<|channel|>commentary json<|message|>...`), as the chat
template renders it. One call per message; the model stops at <|call|>, so
in practice there is at most one call per generation.

This format requires the Harmony stream parser and is selected automatically
for Harmony models; TOOLCALL_START/END are None since there are no tags for
TagStreamParser to scan for.
"""

TOOLCALL_START = None
TOOLCALL_END = None

_RECIPIENT = re.compile(r"\bto=([^\s<]+)")


def parse_toolcalls(text: str) -> list[ToolCall]:
    results = []
    for segment in text.split("<|call|>"):
        if "<|message|>" not in segment:
            continue
        header, body = segment.split("<|message|>", 1)

        recipient = _RECIPIENT.search(header)
        if not recipient:
            continue
        func_name = recipient.group(1).removeprefix("functions.")

        args_json = body.strip()
        try:
            # Normalize, and validate that the arguments are a JSON object
            args_json = json.dumps(json.loads(args_json), ensure_ascii=False)
        except json.JSONDecodeError:
            xlogger.warning(
                "harmony: Tool call arguments are not valid JSON, passing through as-is",
                {"func_name": func_name, "args": args_json},
            )

        results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"harmony: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
