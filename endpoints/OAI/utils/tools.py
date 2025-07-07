import json
from loguru import logger
from typing import List

from endpoints.OAI.types.tools import ToolCall
import secrets
import string


TOOL_CALL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
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
            "type": {"type": "string", "enum": ["function"]},
        },
        "required": ["id", "function", "type"],
    },
}


class ToolCallProcessor:
    @staticmethod
    def from_json(tool_calls_str: str) -> List[ToolCall]:
        """Postprocess tool call JSON to a parseable class"""

        def generate_random_id(prefix="call_", length=22):
                alphabet = string.ascii_letters + string.digits
                random_part = ''.join(secrets.choice(alphabet) for _ in range(length))
                return f"{prefix}{random_part}"
        
        loaded = json.loads(tool_calls_str)
        tool_calls = loaded if isinstance(loaded, list) else [loaded]
        updated_tool_calls = []
        for tool_call in tool_calls:
            # Generate id and static type
            call = {
                "id": generate_random_id(),
                "type": "function",
                "function": {}
            }
            # Handle both cases: function attribute or direct name/arguments
            if "function" in tool_call:
                call["function"] = {
                    "name": tool_call["function"].pop("name"),
                    "arguments": json.dumps(tool_call["function"].pop("arguments"))
                }
            else:
                call["function"] = {
                    "name": tool_call.pop("name"),
                    "arguments": json.dumps(tool_call.pop("arguments"))
                }
            updated_tool_calls.append(call)
        tool_calls = updated_tool_calls

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