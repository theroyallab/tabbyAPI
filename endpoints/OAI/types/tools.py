from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union
from uuid import uuid4

import string
import random

def build_tool_id():
    alphabet = string.ascii_lowercase + string.digits
    return random.choices(alphabet, k=8)

default_tool_call_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Generated schema for Root",
  "type": "array",
  "items": {
    "type": "object"
  }
}

class Function(BaseModel):
    name: str = Field(..., description="The name of the function")
    description: str = Field(..., description="Description of the function")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the function")

class Tool(BaseModel):
    type: Literal["function"]
    function: Function

class FunctionCall(BaseModel):
    name: str = Field(..., description="The name of the function")
    arguments: List[Dict]