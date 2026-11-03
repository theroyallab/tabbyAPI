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


# {function name: {parameter name: JSON schema type}} taken from a request's tool
# definitions; None where the schema gives a parameter no single type
ParamTypes = dict[str, dict[str, Optional[str]]]


def param_types_from_tools(tools) -> ParamTypes:
    """
    Declared parameter types per function from the request's tool definitions
    (ToolSpec objects or the equivalent dicts). A parameter without a single
    `type` in its schema (anyOf, missing) maps to None, which keeps the
    type-guessing coercion for it.
    """

    types: ParamTypes = {}
    for tool in tools or []:
        function = getattr(tool, "function", None)
        if function is None and isinstance(tool, dict):
            function = tool.get("function")
        if function is None:
            continue
        if isinstance(function, dict):
            name, parameters = function.get("name"), function.get("parameters")
        else:
            name = getattr(function, "name", None)
            parameters = getattr(function, "parameters", None)
        if not name:
            continue
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        declared = {}
        for key, schema in (properties or {}).items():
            schema_type = schema.get("type") if isinstance(schema, dict) else None
            declared[key] = schema_type if isinstance(schema_type, str) else None
        types[name] = declared
    return types


def coerce_param_value(raw: str, declared: Optional[str] = None) -> any:
    """Coerce a raw parameter value string to the appropriate Python type.

    Strategy (safe, no eval()):
      1. Strip leading/trailing newlines (official template emits \\n
         after opening tag and before closing tag).
      2. A parameter the tool schema declares as `string` is returned as is:
         "123" stays a string, and a streaming parser can emit it as it is
         generated without waiting to learn its type.
      3. Otherwise try json.loads (objects, arrays, numbers, bools, null)
         and fall back to the plain string.
    """
    # Strip template-inserted newlines around values
    stripped = raw.strip()

    if declared == "string":
        return stripped

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
