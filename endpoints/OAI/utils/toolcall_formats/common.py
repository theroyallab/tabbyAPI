import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FormatSignature:
    """
    Evidence that identifies a tool call format, used to auto-select a parser
    for a model. Each field is one kind of evidence; the detector in
    endpoints/OAI/utils/tools.py scores every registered format against the
    model's chat template, tokenizer and architecture and picks the best match.
    """

    # Markers that must all appear in the chat template (the literal strings
    # the template emits around a tool call in assistant history). An entry
    # may itself be a tuple of alternatives, any one of which satisfies it
    template_markers: tuple = ()
    # Strings whose presence in the template rules this format out, for
    # separating formats that share an outer tag
    template_exclude: tuple = ()
    # Tokens that must all be single tokens in the model's tokenizer
    special_tokens: tuple = ()
    # Substrings matched case-insensitively against the model's architecture
    # name (config.json "architectures"), a weak tie-breaker only
    architectures: tuple = ()
    # Reasoning tags the models using this format are trained with, or None
    reasoning_tags: Optional[tuple] = None
    # Tag-less formats (Harmony, Glimmer) that take over the whole stream
    structured: bool = False


# Markdown code fence patterns
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
CODE_FENCE_END_RE = re.compile(r"\s*```\s*$", re.MULTILINE)


def coerce_param_value(raw: str) -> any:
    """Coerce a raw parameter value string to the appropriate Python type.

    Strategy (safe, no eval()):
      1. Strip leading/trailing newlines (official template emits \\n
         after opening tag and before closing tag).
      2. Try json.loads — handles objects, arrays, numbers, bools, null.
      3. Fall back to plain string.
    """
    # Strip template-inserted newlines around values
    stripped = raw.strip()

    # Empty string
    if not stripped:
        return ""

    # Try JSON parse (handles objects, arrays, numbers, booleans, null)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to string — never eval()
    return stripped


def strip_code_fences(text: str) -> str:
    """Strip markdown code fence blocks if present."""
    text = text.strip()
    return CODE_FENCE_END_RE.sub("", CODE_FENCE_RE.sub("", text)).strip()


def unwrap_tool_args(args: any, func_name: Optional[str] = None) -> dict:
    """
    Unwrap nested tool call argument dictionaries and remove redundant function name keys.

    Handles models emitting pseudo-XML or JSON with nested arguments envelopes, e.g.:
      - {"arguments": {"url": "..."}}
      - {"name": "func", "arguments": {"url": "..."}}
      - {"parameters": "{\"url\": \"...\"}"}
    """
    if isinstance(args, str):
        cleaned = strip_code_fences(args)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                args = parsed
            else:
                return {"input": args}
        except (json.JSONDecodeError, ValueError):
            return {"input": args}

    if not isinstance(args, dict):
        return {}

    # Copy to avoid mutating original in caller
    result = dict(args)

    # Strip redundant function name/id if present alongside arguments
    redundant_keys = {"name", "function", "tool_name"}
    for rk in redundant_keys:
        if rk in result and (result[rk] == func_name or ("arguments" in result or "parameters" in result)):
            del result[rk]

    # Check for wrapped argument envelope: {"arguments": {...}} or {"parameters": {...}}
    wrapper_keys = ("arguments", "parameters")
    for wk in wrapper_keys:
        if wk in result and (len(result) == 1 or all(k in {"type", "id", "index", wk} for k in result)):
            inner = result[wk]
            if isinstance(inner, str):
                inner_cleaned = strip_code_fences(inner)
                try:
                    inner_parsed = json.loads(inner_cleaned)
                    if isinstance(inner_parsed, dict):
                        inner = inner_parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            if isinstance(inner, dict):
                return unwrap_tool_args(inner, func_name)

    return result


def parse_json_tool_call(block: str) -> Optional[tuple[str, dict]]:
    """
    Parse a JSON tool call object, tolerating code fences and surrounding text.
    Returns (func_name, args_dict) or None.
    """
    cleaned = strip_code_fences(block)
    call = None
    try:
        call = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                call = json.loads(cleaned[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    if isinstance(call, dict) and call.get("name"):
        func_name = str(call["name"])
        raw_args = call.get("arguments", call.get("parameters", {}))
        args = unwrap_tool_args(raw_args, func_name)
        return func_name, args

    return None

