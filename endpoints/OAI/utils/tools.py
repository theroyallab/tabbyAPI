import json
from loguru import logger
from typing import List, Optional

from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.xml_tool_processors import XMLToolCallProcessorFactory


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
    def from_json(tool_calls_str: str) -> List[ToolCall]:
        """Postprocess tool call JSON to a parseable class"""

        tool_calls = json.loads(tool_calls_str)
        for tool_call in tool_calls:
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"]
            )

        return [ToolCall(**tool_call) for tool_call in tool_calls]

    @staticmethod
    def from_xml(
        tool_calls_text: str, tools: List[Tool], xml_processor_type: str = "glm45"
    ) -> List[ToolCall]:
        """Process XML tool calls and convert to ToolCall objects"""
        try:
            processor = XMLToolCallProcessorFactory.create_processor(xml_processor_type)
            return processor.parse_xml_to_json(tool_calls_text, tools)
        except Exception as e:
            logger.error(f"Error processing XML tool calls: {e}")
            return []

    @staticmethod
    def from_text(
        tool_calls_text: str,
        tools: List[Tool],
        tool_call_format: str = "json",
        xml_processor_type: Optional[str] = None,
    ) -> List[ToolCall]:
        """
        Process tool calls from text, detecting format and routing appropriately.

        Args:
            tool_calls_text: Raw text containing tool calls
            tools: Available tools for validation
            tool_call_format: Format type ("json" or "xml")
            xml_processor_type: Type of XML processor to use if format is XML

        Returns:
            List of parsed ToolCall objects
        """
        if tool_call_format.lower() == "xml":
            if not xml_processor_type:
                logger.warning(
                    "XML format specified but no xml_processor_type provided, "
                    "using glm45"
                )
                xml_processor_type = "glm45"
            return ToolCallProcessor.from_xml(
                tool_calls_text, tools, xml_processor_type
            )
        else:
            return ToolCallProcessor.from_json(tool_calls_text)

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
