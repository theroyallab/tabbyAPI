# XML Tool Calling Implementation for TabbyAPI

This document describes the XML-based tool calling support implemented for GLM-4.5 and Qwen3-coder models in TabbyAPI.

## Overview

Some models (GLM-4.5, Qwen3-coder) generate tool calls in XML format, which differs from the OpenAI JSON format that TabbyAPI expects. This implementation provides a generic XML tool call processor that converts various XML tool call formats to OpenAI-compatible JSON format.

## Architecture

### Components

1. **BaseXMLToolCallProcessor** (`endpoints/OAI/utils/xml_tool_processors.py`)
   - Abstract base class for XML tool call processors
   - Provides common functionality for parsing and converting tool calls
   - Extensible design allows support for other XML-based models

2. **GLM45ToolCallProcessor** (`endpoints/OAI/utils/xml_tool_processors.py`)
   - Concrete implementation for GLM-4.5 specific XML format
   - Handles the `<tool_call>` and `<arg_key>/<arg_value>` structure
   - Converts XML to OpenAI JSON format

3. **Qwen3CoderToolCallProcessor** (`endpoints/OAI/utils/xml_tool_processors.py`)
   - Concrete implementation for Qwen3-coder specific XML format
   - Handles nested `<tool_call><function=name><parameter=name>value</parameter></function></tool_call>` structure
   - Supports multi-line parameter values
   - Converts XML to OpenAI JSON format

4. **XMLToolCallProcessorFactory** (`endpoints/OAI/utils/xml_tool_processors.py`)
   - Factory class for creating appropriate XML processors
   - Supports GLM-4.5 ("glm45", "glm-4.5", "glm4") and Qwen3-coder ("qwen3-coder", "qwen3") processors
   - Supports extensibility by allowing registration of new processor types

5. **Enhanced TemplateMetadata** (`common/templating.py`)
   - Extended to support XML tool call configuration
   - New fields: `tool_call_format`, `xml_processor_type`, `tool_end`

6. **Enhanced ToolCallProcessor** (`endpoints/OAI/utils/tools.py`)
   - Added `from_text()` method that routes to appropriate processor
   - Added `from_xml()` method for XML-specific processing
   - Maintains backward compatibility with JSON processing

### Supported XML Formats

#### GLM-4.5 XML Format

The GLM-4.5 model generates tool calls in this format:

```xml
<tool_call>function_name
<arg_key>parameter1</arg_key>
<arg_value>value1</arg_value>
<arg_key>parameter2</arg_key>
<arg_value>value2</arg_value>
</tool_call>
```

#### Qwen3-coder XML Format

The Qwen3-coder model generates tool calls in this nested format:

```xml
<tool_call>
<function=function_name>
<parameter=parameter1>
value1
</parameter>
<parameter=parameter2>
This is a multi-line
parameter value that spans
multiple lines
</parameter>
</function>
</tool_call>
```

Both formats get converted to OpenAI JSON format:

```json
{
  "id": "call_12345",
  "type": "function",
  "function": {
    "name": "function_name",
    "arguments": "{\"parameter1\": \"value1\", \"parameter2\": \"value2\"}"
  }
}
```

## Usage

### Template Configuration

#### GLM-4.5 Template

The GLM-4.5 template (`templates/tool_calls/glm-4p5-chat-template-tabbyapi.jinja`) includes:

```jinja
{# Metadata #}
{%- set stop_strings = ["<|user|>", "<|assistant|>", "<|observation|>", "<|system|>"] -%}
{%- set tool_start = "<tool_call>" -%}
{%- set tool_end = "</tool_call>" -%}
{%- set tool_call_format = "xml" -%}
{%- set xml_processor_type = "glm45" -%}
```

#### Qwen3-coder Template

The Qwen3-coder template (`templates/tool_calls/qwen3-coder-tabbyapi.jinja`) includes:

```jinja
{# XML Tool Call Processing Configuration #}
{%- set tool_call_format = "xml" -%}
{%- set xml_processor_type = "qwen3-coder" -%}
```

### Loading Models

#### GLM-4.5 Models

When loading a GLM-4.5 model, specify the tool-calling template:

```yaml
# config.yml
model:
  model_name: "path/to/glm-4.5-model"
  prompt_template: "tool_calls/glm-4p5-chat-template-tabbyapi"
```

#### Qwen3-coder Models

When loading a Qwen3-coder model, specify the tool-calling template:

```yaml
# config.yml
model:
  model_name: "path/to/qwen3-coder-model"
  prompt_template: "tool_calls/qwen3-coder-tabbyapi"
```

Or via API:

```bash
# GLM-4.5
curl -X POST "http://localhost:5000/v1/model/load" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "path/to/glm-4.5-model",
    "prompt_template": "tool_calls/glm-4p5-chat-template-tabbyapi"
  }'

# Qwen3-coder
curl -X POST "http://localhost:5000/v1/model/load" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "path/to/qwen3-coder-model",
    "prompt_template": "tool_calls/qwen3-coder-tabbyapi"
  }'
```

### Tool Call Request

Standard OpenAI-compatible tool calling request:

```json
{
  "model": "glm-4.5",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Beijing?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "City name"
            },
            "date": {
              "type": "string",
              "description": "Date in YYYY-MM-DD format"
            }
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

## Integration Flow

1. **Template Processing**: The template metadata indicates XML format tool calls
2. **Model Generation**: GLM-4.5 generates XML tool calls when `<tool_call>` trigger is detected
3. **XML Parsing**: `GLM45ToolCallProcessor` parses the XML structure
4. **JSON Conversion**: XML is converted to OpenAI-compatible JSON format
5. **Standard Pipeline**: Converted tool calls flow through normal TabbyAPI processing

## Extensibility

### Adding New XML Processors

To support other XML-based models:

1. Create a new processor class extending `BaseXMLToolCallProcessor`
2. Implement the required methods for the specific XML format
3. Register the processor with the factory:

```python
# Custom processor
class CustomXMLProcessor(BaseXMLToolCallProcessor):
    def has_tool_call(self, text: str) -> bool:
        return "<custom_tool>" in text
    
    def parse_xml_to_json(self, text: str, tools: List[Tool]) -> List[ToolCall]:
        # Custom parsing logic
        pass

# Register processor
XMLToolCallProcessorFactory.register_processor("custom", CustomXMLProcessor)
```

### Template Configuration

Create a template with appropriate metadata:

```jinja
{%- set tool_call_format = "xml" -%}
{%- set xml_processor_type = "custom" -%}
{%- set tool_start = "<custom_tool>" -%}
{%- set tool_end = "</custom_tool>" -%}
```

## Testing

Unit tests are provided in `tests/test_xml_tool_calls.py` covering:

- XML parsing functionality
- Multiple tool call handling  
- JSON conversion accuracy
- Error handling for malformed XML
- Factory pattern functionality
- Argument type processing

Run tests with:

```bash
python -m pytest tests/test_xml_tool_calls.py -v
```

## Error Handling

The implementation includes robust error handling:

- **Malformed XML**: Returns empty tool call list, logs error
- **Unknown Functions**: Still processes but without type validation
- **Parsing Failures**: Falls back gracefully, maintains system stability
- **Missing Dependencies**: Graceful degradation to JSON processing

## Performance Considerations

- **Regex-based Parsing**: Efficient for typical tool call volumes
- **Lazy Evaluation**: Processors created only when needed
- **Memory Efficient**: Processes tool calls incrementally
- **Caching**: Template metadata cached after first extraction

## Compatibility

- **Backward Compatible**: Existing JSON tool calling continues to work
- **OpenAI Standard**: Output format matches OpenAI API specification
- **Streaming Support**: Works with both streaming and non-streaming responses
- **Multi-tool**: Supports multiple tool calls in single response

## Troubleshooting

### Common Issues

1. **Tool calls not detected**
   - Verify template has `tool_call_format = "xml"`
   - Check `tool_start` matches model output
   - Ensure `xml_processor_type` is correct

2. **Parsing errors**
   - Validate XML format matches expected structure
   - Check for missing closing tags
   - Verify argument key/value pairing

3. **JSON conversion failures**
   - Check argument types in tool definitions
   - Validate JSON-formatted argument values
   - Review error logs for specific parsing issues

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger("endpoints.OAI.utils.xml_tool_processors").setLevel(logging.DEBUG)
```

This implementation provides a robust, extensible foundation for XML-based tool calling in TabbyAPI while maintaining full compatibility with existing JSON-based tool calling functionality.