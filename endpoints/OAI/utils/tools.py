"""Tool call processing utilities for OAI server."""

import json
import re
from loguru import logger
from typing import Any, List, Tuple

from endpoints.OAI.types.tools import ToolCall, Tool


TOOL_CALL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "function": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {
                        # Converted to OAI's string in post process
                        "type": "object"
                    },
                },
                "required": ["name", "arguments"],
            },
        },
        "required": ["function"],
    },
}

# ---------------------------------------------------------------------------
# XML parsing regex patterns
# Derived from vLLM's Qwen3CoderToolParser and the official Qwen parser.
# These handle both complete and partially-closed tags.
# ---------------------------------------------------------------------------

# Matches complete <tool_call>...</tool_call> blocks
TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    re.DOTALL,
)

# Matches <function=NAME>BODY</function> blocks
FUNCTION_RE = re.compile(
    r"<function=(.*?)>(.*?)</function>",
    re.DOTALL,
)

# Matches <parameter=KEY>VALUE</terminator>
# Terminates on: </parameter>, next <parameter=, </function>, or <tool_call>
PARAMETER_RE = re.compile(
    r"<parameter=(.*?)>(.*?)"
    r"(?:</parameter>|(?=<parameter=)|(?=</function>)|(?=<tool_call>))",
    re.DOTALL,
)

# Think block patterns
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_UNCLOSED_RE = re.compile(r"<think>(?!.*</think>).*$", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """
    Strip <think>...</think> blocks from text.
    Handles both complete and unclosed blocks (quantization can cause
    the model to never close a think tag).
    """
    original = text

    # Complete blocks first
    text = THINK_BLOCK_RE.sub("", text)

    # Unclosed block (think started but never closed — strip to end)
    text = THINK_UNCLOSED_RE.sub("", text)

    if text != original:
        if THINK_UNCLOSED_RE.search(original):
            logger.warning(
                "XML Parser: Stripped unclosed <think> block "
                "(possible quantization degradation)"
            )
        else:
            logger.debug("XML Parser: Stripped <think> block(s) from output")

    return text


def _coerce_param_value(raw: str) -> Any:
    """
    Coerce a raw parameter value string to the appropriate Python type.

    Strategy (safe, no eval()):
      1. Strip leading/trailing newlines (official template emits \\n
         after opening tag and before closing tag).
      2. Try json.loads — handles objects, arrays, numbers, bools, null.
      3. Fall back to plain string.
    """
    # Strip template-inserted newlines around values
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]

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


class ToolCallProcessor:

    # ------------------------------------------------------------------
    # JSON parsing (existing behavior, unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def from_json(tool_calls_str: str) -> List[ToolCall]:
        """Postprocess tool call JSON to a parseable class."""

        logger.debug(
            f"JSON Parser: Parsing tool calls from JSON "
            f"({len(tool_calls_str)} chars)"
        )

        tool_calls = json.loads(tool_calls_str)
        for tool_call in tool_calls:
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"]
            )

        result = [ToolCall(**tool_call) for tool_call in tool_calls]
        logger.debug(f"JSON Parser: Successfully parsed {len(result)} tool call(s)")
        return result

    # ------------------------------------------------------------------
    # XML parsing (Qwen3-Coder / GLM-4.5 style)
    # ------------------------------------------------------------------

    @staticmethod
    def from_xml(raw_text: str) -> List[ToolCall]:
        """
        Parse Qwen3-Coder XML-format tool calls into ToolCall objects.

        Handles:
          - Wrapped: <tool_call><function=name>...</function></tool_call>
          - Bare: <function=name>...</function> (missing wrapper)
          - Multiple sequential tool call blocks
          - <think> blocks (stripped)
          - Multi-line parameter values
          - Missing </parameter> closing tags
        """
        logger.debug(
            f"XML Parser: Parsing tool calls from XML " f"({len(raw_text)} chars)"
        )
        logger.debug(f"XML Parser: Raw input: {raw_text[:500]}...")

        # Stage 1: Strip think blocks
        text = _strip_think_blocks(raw_text)

        # Stage 2: Check for incomplete XML at end (generation cutoff)
        stripped_end = text.rstrip()
        if stripped_end.endswith(("<", "</", "<parameter", "<function")):
            logger.warning(
                f"XML Parser: Detected incomplete XML tag at end: "
                f"...{stripped_end[-80:]}"
            )
            text = re.sub(r"<[^>]*$", "", text)

        # Stage 3: Extract function blocks
        # First, find all wrapped <tool_call>...</tool_call> blocks
        wrapped_positions = [
            (m.start(), m.end()) for m in TOOL_CALL_BLOCK_RE.finditer(text)
        ]

        # Collect function blocks from inside wrapped regions
        function_blocks = []
        for match in TOOL_CALL_BLOCK_RE.finditer(text):
            inner = match.group(1)
            for func_match in FUNCTION_RE.finditer(inner):
                function_blocks.append((func_match.group(1), func_match.group(2)))

        # Find bare <function> blocks NOT inside any wrapped region
        for func_match in FUNCTION_RE.finditer(text):
            pos = func_match.start()
            is_wrapped = any(start <= pos < end for start, end in wrapped_positions)
            if not is_wrapped:
                logger.debug(
                    "XML Parser: Found bare <function> block without "
                    "<tool_call> wrapper (common Qwen3-Coder behavior)"
                )
                function_blocks.append((func_match.group(1), func_match.group(2)))

        if not function_blocks:
            logger.warning(
                f"XML Parser: No <function=...> blocks found in text: "
                f"{text[:200]}..."
            )
            return []

        # Stage 4: Parse each function block into a ToolCall
        tool_calls = []
        for func_name_raw, func_body in function_blocks:
            func_name = func_name_raw.strip()

            # Extract parameters
            params = {}
            for param_match in PARAMETER_RE.finditer(func_body):
                key = param_match.group(1).strip()
                value_raw = param_match.group(2)
                value = _coerce_param_value(value_raw)
                params[key] = value
                logger.debug(f"XML Parser:   param '{key}' = {repr(value)[:100]}")

            arguments_json = json.dumps(params, ensure_ascii=False)

            tool_call = ToolCall(
                function=Tool(name=func_name, arguments=arguments_json)
            )
            tool_calls.append(tool_call)
            logger.debug(
                f"XML Parser: Parsed tool call: {func_name}"
                f"({', '.join(params.keys())})"
            )

        logger.debug(f"XML Parser: Successfully parsed {len(tool_calls)} tool call(s)")
        return tool_calls

    # ------------------------------------------------------------------
    # Auto-detect parsing (JSON → JSON-in-tool_call → XML)
    # ------------------------------------------------------------------

    @staticmethod
    def from_auto(raw_text: str) -> List[ToolCall]:
        """
        Auto-detect format and parse.

        Tries in order:
          1. Pure JSON (standard TabbyAPI / Llama)
          2. JSON inside <tool_call> wrapper (Qwen3-Instruct style)
          3. XML with <function=...> tags (Qwen3-Coder style)
        """
        logger.debug("Auto Parser: Attempting format auto-detection")

        # Attempt 1: Pure JSON array
        try:
            result = ToolCallProcessor.from_json(raw_text)
            logger.debug("Auto Parser: Detected JSON format")
            return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Auto Parser: Not JSON ({e}), trying next format")

        # Attempt 2: JSON inside <tool_call> wrapper (Qwen3-Instruct)
        try:
            for match in TOOL_CALL_BLOCK_RE.finditer(raw_text):
                inner = match.group(1).strip()
                if inner.startswith("{"):
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    tool_calls = []
                    for tc in parsed:
                        name = tc.get("name", "")
                        arguments = tc.get("arguments", {})
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments)
                        elif not isinstance(arguments, str):
                            arguments = json.dumps(arguments)
                        tool_calls.append(
                            ToolCall(function=Tool(name=name, arguments=arguments))
                        )
                    if tool_calls:
                        logger.debug(
                            "Auto Parser: Detected JSON-inside-tool_call "
                            "format (Qwen3-Instruct style)"
                        )
                        return tool_calls
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Auto Parser: Not JSON-in-tool_call ({e}), trying XML")

        # Attempt 3: XML format (Qwen3-Coder style)
        result = ToolCallProcessor.from_xml(raw_text)
        if result:
            logger.debug("Auto Parser: Detected XML format")
        else:
            logger.warning("Auto Parser: All format detection attempts failed")
        return result

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    @staticmethod
    def parse(tool_calls_str: str, format: str = "json") -> List[ToolCall]:
        """
        Dispatch tool call parsing to the appropriate format handler.

        Args:
            tool_calls_str: Raw tool call text from model generation.
            format: One of ``"json"``, ``"xml"``, ``"auto"``.

        Returns:
            List of parsed ToolCall objects.  Empty list on parse failure
            (never raises).
        """
        logger.debug(f"ToolCallProcessor.parse: format={format}")

        try:
            if format == "xml":
                return ToolCallProcessor.from_xml(tool_calls_str)
            elif format == "auto":
                return ToolCallProcessor.from_auto(tool_calls_str)
            else:
                return ToolCallProcessor.from_json(tool_calls_str)
        except Exception as e:
            logger.error(
                f"ToolCallProcessor.parse: Failed to parse tool calls "
                f"(format={format}): {e}"
            )
            logger.debug(
                f"ToolCallProcessor.parse: Raw text was: " f"{tool_calls_str[:500]}..."
            )
            return []

    # ------------------------------------------------------------------
    # Content / tool-call separation
    # ------------------------------------------------------------------

    @staticmethod
    def extract_content_and_tools(
        raw_text: str,
    ) -> Tuple[str, List[ToolCall]]:
        """
        Separate plain text content from XML tool call blocks.

        Used when the model mixes reasoning text with tool calls, e.g.:
        ``"I'll help with that: <tool_call><function=...>...``

        Returns:
            Tuple of (remaining_content, tool_calls).
        """
        logger.debug("extract_content_and_tools: Separating content and tools")

        text = _strip_think_blocks(raw_text)

        # Collect all XML regions to exclude from content
        xml_regions = []

        # Wrapped tool call blocks
        for match in TOOL_CALL_BLOCK_RE.finditer(text):
            xml_regions.append((match.start(), match.end()))

        # Bare function blocks not inside wrappers
        for match in FUNCTION_RE.finditer(text):
            pos = match.start()
            is_wrapped = any(start <= pos < end for start, end in xml_regions)
            if not is_wrapped:
                xml_regions.append((match.start(), match.end()))

        # Sort and extract content (everything outside XML regions)
        xml_regions.sort()
        content_parts = []
        last_end = 0
        for start, end in xml_regions:
            if start > last_end:
                part = text[last_end:start].strip()
                if part:
                    content_parts.append(part)
            last_end = end
        if last_end < len(text):
            part = text[last_end:].strip()
            if part:
                content_parts.append(part)

        content = " ".join(content_parts).strip()

        # Parse tool calls from the full text
        tool_calls = ToolCallProcessor.from_xml(text)

        logger.debug(
            f"extract_content_and_tools: Found {len(tool_calls)} tool "
            f"call(s), content={'yes' if content else 'no'} "
            f"({len(content)} chars)"
        )

        return content, tool_calls

    # ------------------------------------------------------------------
    # Serialisation helpers (unchanged from original)
    # ------------------------------------------------------------------

    @staticmethod
    def dump(tool_calls: List[ToolCall]) -> List[dict]:
        """
        Convert ToolCall objects to a list of dictionaries.

        Args:
            tool_calls (List[ToolCall]): List of ToolCall objects to convert

        Returns:
            List[dict]: List of dictionaries representing the tool calls
        """

        # Don't use list comprehension here
        # as that will fail rather than warn
        dumped_tool_calls = []
        for tool_call_obj in tool_calls:
            try:
                dumped_tool_calls.append(tool_call_obj.model_dump())
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Error processing tool call: {e}")
        return dumped_tool_calls

    @staticmethod
    def to_json(tool_calls: List[ToolCall]) -> str:
        """
        Convert ToolCall objects to JSON string representation.

        Args:
            tool_calls (List[ToolCall]): List of ToolCall objects to convert

        Returns:
            str: JSON representation of the tool calls
        """

        if not tool_calls:
            return ""

        # Use the dump method to get the list of dictionaries
        dumped_tool_calls = ToolCallProcessor.dump(tool_calls)

        # Serialize the dumped array
        return json.dumps(dumped_tool_calls, indent=2)
