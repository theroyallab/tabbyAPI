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
