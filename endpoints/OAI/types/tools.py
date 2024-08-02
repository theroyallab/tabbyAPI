from pydantic import BaseModel
from typing import Dict, Literal

import string
import random

def build_tool_id():
    alphabet = string.ascii_lowercase + string.digits
    return random.choices(alphabet, k=8)


openai_tool_call_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "function": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "arguments": {
            "type": "object",
            "additionalProperties": True
          }
        },
        "required": ["name", "arguments"]
      },
      "type": {
        "type": "string",
        "enum": ["function"]
      }
    },
    "required": ["id", "function", "type"]
  }
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


class ToolCall(BaseModel):
    id: str
    function: Tool
    type: Literal["function"]