import json
from loguru import logger

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
    def from_json(tool_calls_str: str) -> list[ToolCall]:
        """Postprocess tool call JSON to a parseable class"""

        tool_calls = json.loads(tool_calls_str)
        for tool_call in tool_calls:
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"]
            )

        return [ToolCall(**tool_call) for tool_call in tool_calls]

    @staticmethod
    def dump(tool_calls: list[ToolCall]) -> list[dict]:
        """
        Convert ToolCall objects to a list of dictionaries.

        Args:
            tool_calls (list[ToolCall]): list of ToolCall objects to convert

        Returns:
            list[dict]: list of dictionaries representing the tool calls
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
    def to_json(tool_calls: list[ToolCall]) -> str:
        """
        Convert ToolCall objects to JSON string representation.

        Args:
            tool_calls (list[ToolCall]): list of ToolCall objects to convert

        Returns:
            str: JSON representation of the tool calls
        """

        if not tool_calls:
            return ""

        # Use the dump method to get the list of dictionaries
        dumped_tool_calls = ToolCallProcessor.dump(tool_calls)

        # Serialize the dumped array
        return json.dumps(dumped_tool_calls, indent=2)
