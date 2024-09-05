"""Support functions to enable tool calling"""

from typing import List, Dict
from copy import deepcopy
import json

from endpoints.OAI.types.tools import ToolCall
from endpoints.OAI.types.chat_completion import ChatCompletionRequest

def postprocess_tool_call(call_str: str) -> List[ToolCall]:
    print(call_str)
    tool_calls = json.loads(call_str)
    for tool_call in tool_calls:
        tool_call["function"]["arguments"] = json.dumps(
            tool_call["function"]["arguments"]
        )
    return [ToolCall(**tool_call) for tool_call in tool_calls]


def generate_strict_schemas(data: ChatCompletionRequest):
    base_schema = {
        "$defs": {},
        "properties": {
            "id": {"title": "Id", "type": "string"},
            "function": {"title": "Function"},
            "type": {"$ref": "#/$defs/Type"}
        },
        "required": ["id", "function", "type"],
        "title": "ModelItem",
        "type": "object"
    }
    
    function_schemas = []
    argument_schemas = {}
    
    for i, tool in enumerate(data.tools):
        function_name = f"Function{i+1}" if i > 0 else "Function"
        argument_name = f"Arguments{i+1}" if i > 0 else "Arguments"
        name_def = f"Name{i+1}" if i > 0 else "Name"
        
        # Create Name definition
        base_schema["$defs"][name_def] = {
            "const": tool.function.name,
            "enum": [tool.function.name],
            "title": name_def,
            "type": "string"
        }
        
        # Create Arguments definition
        arg_properties = {}
        required_params = tool.function.parameters.get('required', [])
        for arg_name, arg_info in tool.function.parameters.get('properties', {}).items():
            arg_properties[arg_name] = {
                "title": arg_name.capitalize(),
                "type": arg_info['type']
            }
        
        argument_schemas[argument_name] = {
            "properties": arg_properties,
            "required": required_params,
            "title": argument_name,
            "type": "object"
        }
        
        # Create Function definition
        function_schema = {
            "properties": {
                "name": {"$ref": f"#/$defs/{name_def}"},
                "arguments": {"$ref": f"#/$defs/{argument_name}"}
            },
            "required": ["name", "arguments"],
            "title": function_name,
            "type": "object"
        }
        
        function_schemas.append({"$ref": f"#/$defs/{function_name}"})
        base_schema["$defs"][function_name] = function_schema
    
    # Add argument schemas to $defs
    base_schema["$defs"].update(argument_schemas)
    
    # Add Type definition
    base_schema["$defs"]["Type"] = {
        "const": "function",
        "enum": ["function"],
        "title": "Type",
        "type": "string"
    }
    
    # Set up the function property
    base_schema["properties"]["function"]["anyOf"] = function_schemas
    
    return base_schema


# def generate_strict_schemas(data: ChatCompletionRequest):
#     schema = {
#         "type": "object",
#         "properties": {
#             "name": {"type": "string"},
#             "arguments": {
#                 "type": "object",
#                 "properties": {},
#                 "required": []
#             }
#         },
#         "required": ["name", "arguments"]
#     }

#     function_schemas = []
    
#     for tool in data.tools:
#         func_schema = deepcopy(schema)
#         func_schema["properties"]["name"]["enum"] = [tool.function.name]
#         raw_params = tool.function.parameters.get('properties', {})
#         required_params = tool.function.parameters.get('required', [])
        
#         # Add argument properties and required fields
#         arg_properties = {}
#         for arg_name, arg_type in raw_params.items():
#             arg_properties[arg_name] = {"type": arg_type['type']}
        
#         func_schema["properties"]["arguments"]["properties"] = arg_properties
#         func_schema["properties"]["arguments"]["required"] = required_params

#         function_schemas.append(func_schema)
    
#     return _create_full_schema(function_schemas)

# def _create_full_schema(function_schemas: List) -> Dict:
#     # Define the master schema structure with placeholders for function schemas
#     tool_call_schema = {
#         "$schema": "http://json-schema.org/draft-07/schema#",
#         "type": "array",
#         "items": {
#             "type": "object",
#             "properties": {
#                 "id": {"type": "string"},
#                 "function": {
#                     "type": "object",  # Add this line
#                     "oneOf": function_schemas
#                 },
#                 "type": {"type": "string", "enum": ["function"]}
#             },
#             "required": ["id", "function", "type"]
#         }
#     }
   
#     print(json.dumps(tool_call_schema, indent=2))
#     return tool_call_schema