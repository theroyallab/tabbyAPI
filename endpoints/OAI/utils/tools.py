import json
from loguru import logger
from typing import List

from endpoints.OAI.types.tools import ToolCall


class ToolCallProcessor:
    @staticmethod
    def from_json(tool_calls_str: str) -> List[ToolCall]:
        """Postprocess tool call JSON to a parseable class"""

        tool_calls = json.loads(tool_calls_str)
        for tool_call in tool_calls:
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"]
            )

        return [ToolCall(**tool_call) for tool_call in tool_calls]

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

        # Don't use list comprehension here
        # as that will fail rather than warn
        dumped_tool_calls = []
        for tool_call_obj in tool_calls:
            try:
                dumped_tool_calls.append(tool_call_obj.model_dump())
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Error processing tool call: {e}")

        # Serialize the dumped array
        return json.dumps(dumped_tool_calls, indent=2)
