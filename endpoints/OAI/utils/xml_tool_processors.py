"""XML tool call processors for converting XML-based tool calls to OpenAI format."""

import ast
import json
import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from endpoints.OAI.types.tools import ToolCall, Tool, ToolSpec

logger = logging.getLogger(__name__)


class BaseXMLToolCallProcessor(ABC):
    """Base class for XML-based tool call processors."""

    def __init__(self):
        self.tool_start_pattern: str = ""
        self.tool_end_pattern: str = ""

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains XML format tool calls."""
        pass

    @abstractmethod
    def parse_xml_to_json(self, text: str, tools: List[ToolSpec]) -> List[ToolCall]:
        """Parse XML tool calls from text and convert to OpenAI JSON format."""
        pass

    def _parse_arguments(self, json_value: str) -> Tuple[Any, bool]:
        """Parse argument value, trying JSON first, then literal_eval."""
        try:
            try:
                parsed_value = json.loads(json_value)
            except (json.JSONDecodeError, ValueError):
                parsed_value = ast.literal_eval(json_value)
            return parsed_value, True
        except (json.JSONDecodeError, ValueError, SyntaxError):
            return json_value, False

    def _get_argument_type(
        self, func_name: str, arg_key: str, tools: List[ToolSpec]
    ) -> Optional[str]:
        """Get the expected type of an argument based on tool definition."""
        name_to_tool = {tool.function.name: tool for tool in tools}
        if func_name not in name_to_tool:
            return None
        tool = name_to_tool[func_name]
        if arg_key not in tool.function.parameters["properties"]:
            return None
        return tool.function.parameters["properties"][arg_key].get("type", None)

    def _create_tool_call(
        self, name: str, arguments: Dict[str, Any], call_id: Optional[str] = None
    ) -> ToolCall:
        """Create a ToolCall object from parsed data."""
        return ToolCall(
            id=call_id
            or f"call_{hash(f'{name}_{json.dumps(arguments, sort_keys=True)}')}",
            type="function",
            function=Tool(name=name, arguments=json.dumps(arguments)),
        )


class GLM45ToolCallProcessor(BaseXMLToolCallProcessor):
    """
    Tool call processor for GLM-4.5 models.

    Handles XML format like:
    <tool_call>function_name
    <arg_key>parameter1</arg_key>
    <arg_value>value1</arg_value>
    <arg_key>parameter2</arg_key>
    <arg_value>value2</arg_value>
    </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_start_pattern = "<tool_call>"
        self.tool_end_pattern = "</tool_call>"
        self.func_call_regex = r"<tool_call>.*?</tool_call>"
        self.func_detail_regex = r"<tool_call>([^\n]*)\n(.*)</tool_call>"
        self.func_arg_regex = r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains GLM-4.5 format tool calls."""
        return self.tool_start_pattern in text

    def parse_xml_to_json(self, text: str, tools: List[ToolSpec]) -> List[ToolCall]:
        """Parse GLM-4.5 XML tool calls and convert to OpenAI JSON format."""
        if not self.has_tool_call(text):
            return []

        # Find all tool call matches
        match_results = re.findall(self.func_call_regex, text, re.DOTALL)
        tool_calls = []

        try:
            for match_result in match_results:
                # Extract function name and arguments section
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                if not func_detail:
                    logger.warning(f"Could not parse tool call: {match_result}")
                    continue

                func_name = func_detail.group(1).strip()
                func_args_section = func_detail.group(2).strip()

                # Extract argument key-value pairs
                arg_pairs = re.findall(
                    self.func_arg_regex, func_args_section, re.DOTALL
                )
                arguments = {}

                for arg_key, arg_value in arg_pairs:
                    arg_key = arg_key.strip()
                    arg_value = arg_value.strip()

                    # Get expected argument type from tool definition
                    arg_type = self._get_argument_type(func_name, arg_key, tools)

                    # Parse non-string arguments
                    if arg_type != "string":
                        arg_value, _ = self._parse_arguments(arg_value)

                    arguments[arg_key] = arg_value

                # Create ToolCall object
                tool_call = self._create_tool_call(func_name, arguments)
                tool_calls.append(tool_call)

            return tool_calls

        except Exception as e:
            logger.error(f"Error parsing GLM-4.5 XML tool calls: {e}")
            return []


class XMLToolCallProcessorFactory:
    """Factory for creating appropriate XML tool call processors."""

    _processors = {
        "glm45": GLM45ToolCallProcessor,
        "glm-4.5": GLM45ToolCallProcessor,
        "glm4": GLM45ToolCallProcessor,
    }

    @classmethod
    def create_processor(cls, processor_type: str) -> BaseXMLToolCallProcessor:
        """Create an XML tool call processor of the specified type."""
        processor_class = cls._processors.get(processor_type.lower())
        if not processor_class:
            raise ValueError(f"Unknown XML tool call processor type: {processor_type}")
        return processor_class()

    @classmethod
    def register_processor(cls, name: str, processor_class: type):
        """Register a new XML tool call processor type."""
        cls._processors[name.lower()] = processor_class

    @classmethod
    def get_available_processors(cls) -> List[str]:
        """Get list of available processor types."""
        return list(cls._processors.keys())
