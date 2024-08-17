from pydantic import BaseModel
from typing import Dict, Literal

tool_call_schema = {
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
                        "type": "object"
                        # ^ Converted to string in post processing, format enforced while inf
                    },
                },
                "required": ["name", "arguments"],
            },
            "type": {"type": "string", "enum": ["function"]},
        },
        "required": ["id", "function", "type"],
    },
}


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, object]


class ToolSpec(BaseModel):
    function: Function
    type: Literal["function"]


class Tool(BaseModel):
    name: str
    arguments: str
    #  ^ Seems illogical (we'd imagine this is Dict[str, object]) but
    # OAI lib types actually specifies this as a string, handled by post
    # processing tool call


class ToolCall(BaseModel):
    id: str
    function: Tool
    type: Literal["function"]
