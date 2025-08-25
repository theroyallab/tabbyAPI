"""Unit tests for XML tool call processing functionality."""

import pytest
import json
from endpoints.OAI.utils.xml_tool_processors import (
    GLM45ToolCallProcessor,
    XMLToolCallProcessorFactory,
)
from endpoints.OAI.types.tools import ToolCall, ToolSpec, Function


class TestGLM45ToolCallProcessor:
    """Test GLM-4.5 XML tool call processor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = GLM45ToolCallProcessor()
        self.sample_tools = [
            ToolSpec(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information for a city",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format",
                            },
                            "units": {
                                "type": "string",
                                "description": "Temperature units",
                            },
                        },
                    },
                ),
            ),
            ToolSpec(
                type="function",
                function=Function(
                    name="calculate_sum",
                    description="Calculate the sum of numbers",
                    parameters={
                        "type": "object",
                        "properties": {
                            "numbers": {
                                "type": "array",
                                "description": "List of numbers",
                            },
                            "precision": {
                                "type": "integer",
                                "description": "Decimal precision",
                            },
                        },
                    },
                ),
            ),
        ]

    def test_has_tool_call_positive(self):
        """Test detection of XML tool calls."""
        text_with_tool = """Some text before
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>
Some text after"""

        assert self.processor.has_tool_call(text_with_tool) is True

    def test_has_tool_call_negative(self):
        """Test when no tool calls are present."""
        text_without_tool = "This is just regular text with no tool calls."

        assert self.processor.has_tool_call(text_without_tool) is False

    def test_parse_single_tool_call(self):
        """Test parsing a single XML tool call."""
        xml_text = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2024-06-27</arg_value>
</tool_call>"""

        result = self.processor.parse_xml_to_json(xml_text, self.sample_tools)

        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].function.name == "get_weather"

        arguments = json.loads(result[0].function.arguments)
        assert arguments["city"] == "Beijing"
        assert arguments["date"] == "2024-06-27"

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple XML tool calls."""
        xml_text = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2024-06-27</arg_value>
</tool_call>

<tool_call>calculate_sum
<arg_key>numbers</arg_key>
<arg_value>[1, 2, 3, 4, 5]</arg_value>
<arg_key>precision</arg_key>
<arg_value>2</arg_value>
</tool_call>"""

        result = self.processor.parse_xml_to_json(xml_text, self.sample_tools)

        assert len(result) == 2

        # First tool call
        assert result[0].function.name == "get_weather"
        args1 = json.loads(result[0].function.arguments)
        assert args1["city"] == "Beijing"
        assert args1["date"] == "2024-06-27"

        # Second tool call
        assert result[1].function.name == "calculate_sum"
        args2 = json.loads(result[1].function.arguments)
        assert args2["numbers"] == [1, 2, 3, 4, 5]
        assert args2["precision"] == 2

    def test_parse_with_json_values(self):
        """Test parsing XML tool calls with JSON-formatted argument values."""
        xml_text = """<tool_call>calculate_sum
<arg_key>numbers</arg_key>
<arg_value>[10, 20, 30]</arg_value>
<arg_key>precision</arg_key>
<arg_value>3</arg_value>
</tool_call>"""

        result = self.processor.parse_xml_to_json(xml_text, self.sample_tools)

        assert len(result) == 1
        arguments = json.loads(result[0].function.arguments)
        assert arguments["numbers"] == [10, 20, 30]
        assert arguments["precision"] == 3

    def test_parse_with_surrounding_text(self):
        """Test parsing XML tool calls with surrounding text."""
        xml_text = """I need to check the weather and do some calculations.

<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Shanghai</arg_value>
<arg_key>units</arg_key>
<arg_value>metric</arg_value>
</tool_call>

Let me also calculate something:

<tool_call>calculate_sum
<arg_key>numbers</arg_key>
<arg_value>[5, 10, 15]</arg_value>
</tool_call>

That should do it."""

        result = self.processor.parse_xml_to_json(xml_text, self.sample_tools)

        assert len(result) == 2
        assert result[0].function.name == "get_weather"
        assert result[1].function.name == "calculate_sum"

    def test_parse_malformed_xml(self):
        """Test handling of malformed XML."""
        malformed_xml = """<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing
</tool_call>"""  # Missing closing tag for arg_value

        result = self.processor.parse_xml_to_json(malformed_xml, self.sample_tools)

        # Should create tool call but with empty arguments due to malformed arg_value
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        arguments = json.loads(result[0].function.arguments)
        assert arguments == {}  # Empty arguments due to malformed XML

    def test_empty_input(self):
        """Test parsing empty input."""
        result = self.processor.parse_xml_to_json("", self.sample_tools)
        assert len(result) == 0

    def test_no_matching_tools(self):
        """Test parsing with no matching tools in the tool list."""
        xml_text = """<tool_call>unknown_function
<arg_key>param</arg_key>
<arg_value>value</arg_value>
</tool_call>"""

        result = self.processor.parse_xml_to_json(xml_text, self.sample_tools)

        # Should still parse but with no type validation
        assert len(result) == 1
        assert result[0].function.name == "unknown_function"


class TestXMLToolCallProcessorFactory:
    """Test XML tool call processor factory."""

    def test_create_glm45_processor(self):
        """Test creating GLM-4.5 processor."""
        processor = XMLToolCallProcessorFactory.create_processor("glm45")
        assert isinstance(processor, GLM45ToolCallProcessor)

    def test_create_glm45_processor_variations(self):
        """Test creating GLM-4.5 processor with different name variations."""
        for name in ["glm45", "glm-4.5", "GLM45", "GLM-4.5"]:
            processor = XMLToolCallProcessorFactory.create_processor(name)
            assert isinstance(processor, GLM45ToolCallProcessor)

    def test_create_unknown_processor(self):
        """Test error handling for unknown processor type."""
        with pytest.raises(ValueError, match="Unknown XML tool call processor type"):
            XMLToolCallProcessorFactory.create_processor("unknown_processor")

    def test_get_available_processors(self):
        """Test getting list of available processors."""
        processors = XMLToolCallProcessorFactory.get_available_processors()
        assert "glm45" in processors
        assert "glm-4.5" in processors
        assert "glm4" in processors


class TestBaseXMLToolCallProcessor:
    """Test base XML tool call processor functionality."""

    def test_parse_arguments_json(self):
        """Test parsing JSON-formatted argument values."""
        processor = GLM45ToolCallProcessor()  # Use concrete implementation

        # Test JSON parsing
        result, success = processor._parse_arguments('{"key": "value"}')
        assert success is True
        assert result == {"key": "value"}

        # Test array parsing
        result, success = processor._parse_arguments("[1, 2, 3]")
        assert success is True
        assert result == [1, 2, 3]

        # Test number parsing
        result, success = processor._parse_arguments("42")
        assert success is True
        assert result == 42

    def test_parse_arguments_literal(self):
        """Test parsing literal argument values."""
        processor = GLM45ToolCallProcessor()

        # Test string that can't be parsed as JSON
        result, success = processor._parse_arguments("simple_string")
        assert success is False
        assert result == "simple_string"

    def test_get_argument_type(self):
        """Test getting argument type from tool definition."""
        processor = GLM45ToolCallProcessor()
        tools = [
            ToolSpec(
                type="function",
                function=Function(
                    name="test_func",
                    description="Test function",
                    parameters={
                        "type": "object",
                        "properties": {
                            "str_param": {"type": "string"},
                            "int_param": {"type": "integer"},
                        },
                    },
                ),
            )
        ]

        assert processor._get_argument_type("test_func", "str_param", tools) == "string"
        assert (
            processor._get_argument_type("test_func", "int_param", tools) == "integer"
        )
        assert processor._get_argument_type("test_func", "unknown_param", tools) is None
        assert processor._get_argument_type("unknown_func", "param", tools) is None


if __name__ == "__main__":
    pytest.main([__file__])
