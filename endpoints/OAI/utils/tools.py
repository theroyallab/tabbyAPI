"""Tool call processing utilities for OAI server."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
import re
from loguru import logger
from typing import Any, Callable, Dict, List, Tuple

from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.parser_options import resolve_tool_call_parser_key


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

# Markdown code fence patterns
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
CODE_FENCE_END_RE = re.compile(r"\s*```\s*$", re.MULTILINE)

# DeepSeek family patterns
DEEPSEEK_V31_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?P<name>.*?)<｜tool▁sep｜>(?P<args>.*?)<｜tool▁call▁end｜>",
    re.DOTALL,
)
DEEPSEEK_V3_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?P<type>.*?)<｜tool▁sep｜>(?P<name>.*?)\n```json\n(?P<args>.*?)\n```(?:\s*)<｜tool▁call▁end｜>",  # noqa: E501
    re.DOTALL,
)
DEEPSEEK_V32_INVOKE_RE = re.compile(
    r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</｜DSML｜invoke>',
    re.DOTALL,
)
DEEPSEEK_V32_PARAM_RE = re.compile(
    r'<｜DSML｜parameter\s+name="(?P<name>[^"]+)"(?:\s+string="(?P<string>true|false)")?\s*>(?P<value>.*?)</｜DSML｜parameter>',  # noqa: E501
    re.DOTALL,
)


def _strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> blocks from text.

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
    """Coerce a raw parameter value string to the appropriate Python type.

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
    _PARSER_DISPATCHER: Dict[str, Callable[[str], List[ToolCall]]] = {}

    # ------------------------------------------------------------------
    # JSON normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tool_calls(raw) -> list:
        """Normalize model-emitted tool call payloads into OAI-like objects.

        Accepted forms:
        - [{"type":"function","function":{"name":...,"arguments":{...}}}]
        - [{"name":...,"arguments":{...}}]
        - {"name":...,"arguments":{...}}
        """
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raise ValueError("tool_calls payload is not list/dict")

        normalized: list = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            if "function" in item and isinstance(item["function"], dict):
                fn = item["function"]
                name = fn.get("name")
                arguments = fn.get("arguments", {})
            else:
                name = item.get("name")
                arguments = item.get("arguments", {})

            if name is None:
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}

            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    },
                }
            )
        return normalized

    @staticmethod
    def _safe_json_loads(payload: str) -> list:
        """Best-effort JSON parse for model-emitted tool payloads.

        Handles: clean JSON, markdown-fenced JSON, JSON substrings in
        surrounding text, flat {name, arguments} dicts, and single objects.
        """
        # Direct parse
        try:
            return ToolCallProcessor._normalize_tool_calls(json.loads(payload))
        except (json.JSONDecodeError, ValueError):
            pass

        # Clean up common model artifacts (markdown fences, whitespace)
        cleaned = payload.strip()
        cleaned = CODE_FENCE_RE.sub("", cleaned)
        cleaned = CODE_FENCE_END_RE.sub("", cleaned)
        cleaned = cleaned.strip()

        # Try cleaned
        try:
            return ToolCallProcessor._normalize_tool_calls(json.loads(cleaned))
        except (json.JSONDecodeError, ValueError):
            pass

        # Find JSON array substring
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return ToolCallProcessor._normalize_tool_calls(
                    json.loads(cleaned[start : end + 1])
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Find JSON object substring
        obj_start = cleaned.find("{")
        obj_end = cleaned.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            try:
                return ToolCallProcessor._normalize_tool_calls(
                    json.loads(cleaned[obj_start : obj_end + 1])
                )
            except (json.JSONDecodeError, ValueError):
                pass

        raise json.JSONDecodeError(
            "Could not extract valid JSON from payload", payload, 0
        )

    @staticmethod
    def _build_tool_calls_from_normalized(raw: Any) -> List[ToolCall]:
        """Normalize dict/list payload and build ToolCall models."""
        normalized = ToolCallProcessor._normalize_tool_calls(raw)
        for tool_call in normalized:
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"], ensure_ascii=False
            )
        return [ToolCall(**tool_call) for tool_call in normalized]

    @staticmethod
    def _decode_json_sequence(text: str) -> List[Any]:
        """Decode multiple JSON values from a single string."""
        decoder = json.JSONDecoder()
        values: List[Any] = []
        idx = 0
        while idx < len(text):
            while idx < len(text) and text[idx] in " \t\r\n,;":
                idx += 1
            if idx >= len(text):
                break
            if text.startswith("<|python_tag|>", idx):
                idx += len("<|python_tag|>")
                continue
            try:
                value, end = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                break
            values.append(value)
            idx += end
        return values

    @staticmethod
    def _coerce_argument_payload(arguments_raw: str) -> str:
        """Normalize raw argument payload to a JSON string where possible."""
        payload = arguments_raw.strip()
        if not payload:
            return "{}"
        try:
            return json.dumps(json.loads(payload), ensure_ascii=False)
        except (json.JSONDecodeError, ValueError, TypeError):
            return payload

    @staticmethod
    def _ast_to_literal(node: ast.AST) -> Any:
        """Safely convert AST literal nodes to Python primitives."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [ToolCallProcessor._ast_to_literal(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return [ToolCallProcessor._ast_to_literal(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values):
                literal_key = ToolCallProcessor._ast_to_literal(key)  # type: ignore[arg-type]
                if not isinstance(literal_key, str):
                    raise ValueError("pythonic parser requires string dict keys")
                result[literal_key] = ToolCallProcessor._ast_to_literal(value)
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ToolCallProcessor._ast_to_literal(node.operand)
        raise ValueError(f"unsupported pythonic AST node: {type(node).__name__}")

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def from_hermes(raw_text: str) -> List[ToolCall]:
        """Parse Hermes-style JSON tool calls (often wrapped in <tool_call>)."""
        text = _strip_think_blocks(raw_text)
        wrapped_calls = []
        for match in TOOL_CALL_BLOCK_RE.finditer(text):
            inner = match.group(1).strip()
            if not inner:
                continue
            try:
                parsed = json.loads(inner)
            except (json.JSONDecodeError, ValueError):
                continue
            wrapped_calls.extend(ToolCallProcessor._build_tool_calls_from_normalized(parsed))

        if wrapped_calls:
            return wrapped_calls

        return ToolCallProcessor.from_json(text)

    @staticmethod
    def from_llama(raw_text: str) -> List[ToolCall]:
        """Parse Llama JSON tool calls (single/multiple JSON objects)."""
        text = _strip_think_blocks(raw_text).strip()
        if text.startswith("<|python_tag|>"):
            text = text[len("<|python_tag|>") :].lstrip()

        try:
            parsed = ToolCallProcessor.from_json(text)
            if parsed:
                return parsed
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        decoded = ToolCallProcessor._decode_json_sequence(text)
        if not decoded:
            return []

        flattened = []
        for item in decoded:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)

        return ToolCallProcessor._build_tool_calls_from_normalized(flattened)

    @staticmethod
    def from_openai(raw_text: str) -> List[ToolCall]:
        """Best-effort parser for OpenAI/Harmony-style text payloads."""
        text = _strip_think_blocks(raw_text).strip()
        try:
            parsed = ToolCallProcessor.from_json(text)
            if parsed:
                return parsed
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        decoded = ToolCallProcessor._decode_json_sequence(text)
        tool_calls: List[ToolCall] = []
        normalized_items = []
        for value in decoded:
            candidates = value if isinstance(value, list) else [value]
            for item in candidates:
                if not isinstance(item, dict):
                    continue

                nested = item.get("tool_calls")
                if nested:
                    try:
                        tool_calls.extend(
                            ToolCallProcessor._build_tool_calls_from_normalized(nested)
                        )
                    except (ValueError, KeyError, TypeError):
                        pass

                recipient = item.get("recipient")
                content = item.get("content")
                if isinstance(recipient, str) and recipient.startswith("functions."):
                    fn_name = recipient.split("functions.", 1)[1]
                    if isinstance(content, str):
                        payload = ToolCallProcessor._coerce_argument_payload(content)
                    elif content is None:
                        payload = "{}"
                    else:
                        payload = json.dumps(content, ensure_ascii=False)
                    tool_calls.append(
                        ToolCall(function=Tool(name=fn_name, arguments=payload))
                    )
                    continue

                if "name" in item:
                    normalized_items.append(item)

        if normalized_items:
            tool_calls.extend(
                ToolCallProcessor._build_tool_calls_from_normalized(normalized_items)
            )

        return tool_calls

    @staticmethod
    def from_pythonic(raw_text: str) -> List[ToolCall]:
        """Parse Pythonic list-of-calls tool syntax."""
        text = _strip_think_blocks(raw_text).strip()
        if text.startswith("<|python_tag|>"):
            text = text[len("<|python_tag|>") :].lstrip()
        if not text:
            return []

        if not text.startswith("[") and re.match(r"^[A-Za-z_]\w*\s*\(", text):
            text = f"[{text}]"

        expression = ast.parse(text, mode="eval").body
        call_nodes = expression.elts if isinstance(expression, ast.List) else [expression]

        tool_calls = []
        for node in call_nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            args_dict: Dict[str, Any] = {}
            if node.args:
                args_dict["_args"] = [
                    ToolCallProcessor._ast_to_literal(argument)
                    for argument in node.args
                ]
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                args_dict[keyword.arg] = ToolCallProcessor._ast_to_literal(keyword.value)

            tool_calls.append(
                ToolCall(
                    function=Tool(
                        name=node.func.id,
                        arguments=json.dumps(args_dict, ensure_ascii=False),
                    )
                )
            )

        return tool_calls

    @staticmethod
    def from_deepseek_v31(raw_text: str) -> List[ToolCall]:
        """Parse DeepSeek v3.1 tool call syntax."""
        tool_calls = []
        for match in DEEPSEEK_V31_CALL_RE.finditer(raw_text):
            name = match.group("name").strip()
            if not name:
                continue
            arguments = ToolCallProcessor._coerce_argument_payload(match.group("args"))
            tool_calls.append(ToolCall(function=Tool(name=name, arguments=arguments)))
        return tool_calls

    @staticmethod
    def from_deepseek_v3(raw_text: str) -> List[ToolCall]:
        """Parse DeepSeek v3 tool call syntax."""
        tool_calls = []
        for match in DEEPSEEK_V3_CALL_RE.finditer(raw_text):
            name = match.group("name").strip()
            if not name:
                continue
            arguments = ToolCallProcessor._coerce_argument_payload(match.group("args"))
            tool_calls.append(ToolCall(function=Tool(name=name, arguments=arguments)))

        if tool_calls:
            return tool_calls

        return ToolCallProcessor.from_deepseek_v31(raw_text)

    @staticmethod
    def from_deepseek_v32(raw_text: str) -> List[ToolCall]:
        """Parse DeepSeek v3.2 DSML tool call syntax."""
        tool_calls = []
        for invoke in DEEPSEEK_V32_INVOKE_RE.finditer(raw_text):
            function_name = invoke.group("name").strip()
            if not function_name:
                continue

            params: Dict[str, Any] = {}
            body = invoke.group("body")
            for param in DEEPSEEK_V32_PARAM_RE.finditer(body):
                key = param.group("name").strip()
                value_raw = param.group("value")
                is_string = param.group("string") == "true"
                if is_string:
                    value = value_raw.strip("\n")
                else:
                    value = _coerce_param_value(value_raw)
                params[key] = value

            tool_calls.append(
                ToolCall(
                    function=Tool(
                        name=function_name,
                        arguments=json.dumps(params, ensure_ascii=False),
                    )
                )
            )

        if tool_calls:
            return tool_calls

        return ToolCallProcessor.from_deepseek_v31(raw_text)

    @staticmethod
    def from_json(tool_calls_str: str) -> List[ToolCall]:
        """Postprocess tool call JSON to a parseable class.

        Handles clean JSON arrays, markdown-fenced output, flat dicts,
        and other common model output variations via _safe_json_loads.
        """
        logger.debug(f"JSON Parser: Parsing tool calls ({len(tool_calls_str)} chars)")

        tool_calls = ToolCallProcessor._safe_json_loads(tool_calls_str)
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
        """Parse Qwen3-Coder XML-format tool calls into ToolCall objects.

        Handles:
          - Wrapped: <tool_call><function=name>...</function></tool_call>
          - Bare: <function=name>...</function> (missing wrapper)
          - Multiple sequential tool call blocks
          - <think> blocks (stripped)
          - Multi-line parameter values
          - Missing </parameter> closing tags
        """
        logger.debug(f"XML Parser: Parsing tool calls ({len(raw_text)} chars)")

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
                    "<tool_call> wrapper"
                )
                function_blocks.append((func_match.group(1), func_match.group(2)))

        if not function_blocks:
            logger.warning("XML Parser: No <function=...> blocks found")
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

            arguments_json = json.dumps(params, ensure_ascii=False)

            tool_call = ToolCall(
                function=Tool(name=func_name, arguments=arguments_json)
            )
            tool_calls.append(tool_call)

        logger.debug(f"XML Parser: Successfully parsed {len(tool_calls)} tool call(s)")
        return tool_calls

    # ------------------------------------------------------------------
    # Auto-detect parsing (JSON → JSON-in-tool_call → XML)
    # ------------------------------------------------------------------

    @staticmethod
    def from_auto(raw_text: str) -> List[ToolCall]:
        """Auto-detect format and parse.

        Tries in order:
          1. Pure JSON (standard TabbyAPI / Llama)
          2. JSON inside <tool_call> wrappers (Qwen3-Instruct style)
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

        # Attempt 2: JSON inside <tool_call> wrappers (Qwen3-Instruct)
        try:
            all_tool_calls = []
            for match in TOOL_CALL_BLOCK_RE.finditer(raw_text):
                inner = match.group(1).strip()
                if inner.startswith("{") or inner.startswith("["):
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    if isinstance(parsed, list):
                        for tc in parsed:
                            name = tc.get("name", "")
                            arguments = tc.get("arguments", {})
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments)
                            elif not isinstance(arguments, str):
                                arguments = json.dumps(arguments)
                            all_tool_calls.append(
                                ToolCall(function=Tool(name=name, arguments=arguments))
                            )
            if all_tool_calls:
                logger.debug(
                    "Auto Parser: Detected JSON-inside-tool_call "
                    f"format ({len(all_tool_calls)} call(s))"
                )
                return all_tool_calls
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
    def _parser_dispatcher() -> Dict[str, Callable[[str], List[ToolCall]]]:
        """Registry for parser-key-specific handlers."""
        if not ToolCallProcessor._PARSER_DISPATCHER:
            ToolCallProcessor._PARSER_DISPATCHER = {
                "deepseek_v3": ToolCallProcessor.from_deepseek_v3,
                "deepseek_v31": ToolCallProcessor.from_deepseek_v31,
                "deepseek_v32": ToolCallProcessor.from_deepseek_v32,
                "hermes": ToolCallProcessor.from_hermes,
                "llama": ToolCallProcessor.from_llama,
                "llama3_json": ToolCallProcessor.from_llama,
                "llama4_json": ToolCallProcessor.from_llama,
                "openai": ToolCallProcessor.from_openai,
                "pythonic": ToolCallProcessor.from_pythonic,
                "qwen3_coder": ToolCallProcessor.from_xml,
                "qwen3_xml": ToolCallProcessor.from_xml,
            }
        return ToolCallProcessor._PARSER_DISPATCHER

    @staticmethod
    def parse(
        tool_calls_str: str, format: str = "json", parser_key: str | None = None
    ) -> List[ToolCall]:
        """Dispatch tool call parsing to the appropriate format handler.

        Args:
            tool_calls_str: Raw tool call text from model generation.
            format: One of ``"json"``, ``"xml"``, ``"auto"``.
            parser_key: Optional vLLM-compatible parser key.

        Returns:
            List of parsed ToolCall objects.  Empty list on parse failure
            (never raises).
        """
        try:
            if parser_key:
                canonical_key = resolve_tool_call_parser_key(parser_key) or parser_key
                parser = ToolCallProcessor._parser_dispatcher().get(canonical_key)
                if parser:
                    try:
                        parsed = parser(tool_calls_str)
                    except Exception as exc:
                        logger.warning(
                            "Parser '{}' failed: {}. Falling back to format '{}'.",
                            canonical_key,
                            str(exc),
                            format,
                        )
                    else:
                        if parsed:
                            return parsed

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
            return []

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_name(
        tool_calls: List[ToolCall], function_name: str
    ) -> List[ToolCall]:
        """Filter parsed tool calls to only those matching a function name."""
        filtered = [tc for tc in tool_calls if tc.function.name == function_name]
        if not filtered:
            logger.warning(
                f"filter_by_name: No tool calls matched '{function_name}' "
                f"(had {len(tool_calls)} call(s))"
            )
        return filtered

    # ------------------------------------------------------------------
    # Content / tool-call separation
    # ------------------------------------------------------------------

    @staticmethod
    def extract_content_and_tools(
        raw_text: str,
    ) -> Tuple[str, List[ToolCall]]:
        """Separate plain text content from XML tool call blocks.

        Used when the model mixes reasoning text with tool calls, e.g.:
        ``"I'll help with that: <tool_call><function=...>...``

        Returns:
            Tuple of (remaining_content, tool_calls).
        """
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
        """Convert ToolCall objects to a list of dictionaries.

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
        """Convert ToolCall objects to JSON string representation.

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
