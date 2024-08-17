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


class Function(BaseModel):
    """Represents a description of a tool function."""

    name: str
    description: str
    parameters: Dict[str, object]


class ToolSpec(BaseModel):
    """Wrapper for an inner tool function."""

    function: Function
    type: Literal["function"]


class Tool(BaseModel):
    """Represents an OAI tool description."""

    name: str

    # Makes more sense to be a dict, but OAI knows best
    arguments: str


class ToolCall(BaseModel):
    """Represents an OAI tool description."""

    id: str
    function: Tool
    type: Literal["function"]
