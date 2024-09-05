"""Support functions to enable tool calling"""

from typing import List, Dict
import json

from endpoints.OAI.types.tools import ToolCall
from endpoints.OAI.types.chat_completion import ChatCompletionRequest

def postprocess_tool_call(call_str: str) -> List[ToolCall]:
    print(call_str)
    tool_calls = json.loads(call_str)
    print(tool_calls)
    for tool_call in tool_calls:
        tool_call["function"]["arguments"] = json.dumps(
            tool_call["function"]["arguments"]
        )
    return [ToolCall(**tool_call) for tool_call in tool_calls]

def generate_strict_schemas(data: ChatCompletionRequest) -> Dict:
    # Generate the $defs section
    defs = generate_defs(data.tools)
    
    # Generate the root structure (now an array)
    root_structure = {
        "type": "array",
        "items": {"$ref": "#/$defs/ModelItem"}
    }
    
    # Combine the $defs and root structure
    full_schema = {
        "$defs": defs,
        **root_structure
    }
    
    return full_schema

def generate_defs(tools: List) -> Dict:
    defs = {}
    
    for i, tool in enumerate(tools):
        function_name = f"Function{i + 1}" if i > 0 else "Function"
        arguments_name = f"Arguments{i + 1}" if i > 0 else "Arguments"
        name_const = f"Name{i + 1}" if i > 0 else "Name"
        
        # Generate Arguments schema
        defs[arguments_name] = generate_arguments_schema(tool.function.parameters)
        
        # Generate Name schema
        defs[name_const] = {
            "const": tool.function.name,
            "title": name_const,
            "type": "string"
        }
        
        # Generate Function schema
        defs[function_name] = {
            "type": "object",
            "properties": {
                "name": {"$ref": f"#/$defs/{name_const}"},
                "arguments": {"$ref": f"#/$defs/{arguments_name}"}
            },
            "required": ["name", "arguments"]
        }
    
    # Add ModelItem and Type schemas
    defs["ModelItem"] = generate_model_item_schema(len(tools))
    defs["Type"] = {
        "const": "function",
        "type": "string"
    }
    
    return defs

def generate_arguments_schema(parameters: Dict) -> Dict:
    properties = {}
    required = parameters.get('required', [])
    
    for name, details in parameters.get('properties', {}).items():
        properties[name] = {"type": details['type']}
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }

def generate_model_item_schema(num_functions: int) -> Dict:
    function_refs = [{"$ref": f"#/$defs/Function{i + 1}" if i > 0 else "#/$defs/Function"} for i in range(num_functions)]
    
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "function": {
                "anyOf": function_refs
            },
            "type": {"$ref": "#/$defs/Type"}
        },
        "required": ["id", "function", "type"]
    }