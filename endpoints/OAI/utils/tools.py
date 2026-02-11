import json
import re
from loguru import logger
from typing import List

from endpoints.OAI.types.tools import ToolCall


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


class ToolCallProcessor:
    @staticmethod
    def _normalize_tool_calls(raw) -> list[dict]:
        """
        Normalize model-emitted tool call payloads into OAI-like objects.

        Accepted forms:
        - [{"type":"function","function":{"name":...,"arguments":{...}}}]
        - [{"name":...,"arguments":{...}}]
        - {"name":...,"arguments":{...}}
        """
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raise json.JSONDecodeError("tool_calls payload is not list/dict", str(raw), 0)

        normalized: list[dict] = []
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
    def _safe_loads(payload: str):
        """Best-effort JSON parse for model-emitted tool payloads."""
        try:
            return ToolCallProcessor._normalize_tool_calls(json.loads(payload))
        except json.JSONDecodeError:
            # Common model artifacts: fenced code blocks or leading text.
            cleaned = payload.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            # EXAONE-like tool tags: <tool_call>{...}</tool_call>
            tag_matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", cleaned, re.DOTALL)
            if tag_matches:
                parsed = []
                for candidate in tag_matches:
                    parsed.append(json.loads(candidate))
                return ToolCallProcessor._normalize_tool_calls(parsed)

            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]
                return ToolCallProcessor._normalize_tool_calls(json.loads(cleaned))

            # Fallback: single JSON object
            obj_start = cleaned.find("{")
            obj_end = cleaned.rfind("}")
            if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                cleaned = cleaned[obj_start : obj_end + 1]
            return ToolCallProcessor._normalize_tool_calls(json.loads(cleaned))

    @staticmethod
    def from_json(tool_calls_str: str) -> List[ToolCall]:
        """Postprocess tool call JSON to a parseable class"""

        tool_calls = ToolCallProcessor._safe_loads(tool_calls_str)
        for tool_call in tool_calls:
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"]
            )

        return [ToolCall(**tool_call) for tool_call in tool_calls]

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

    @staticmethod
    def filter_by_name(tool_calls_str: str, function_name: str) -> str:
        tool_calls = ToolCallProcessor._safe_loads(tool_calls_str)
        filtered = [
            item
            for item in tool_calls
            if item.get("function", {}).get("name") == function_name
        ]
        return json.dumps(filtered)
