from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union
from uuid import uuid4

import string
import random

def build_tool_id():
    alphabet = string.ascii_lowercase + string.digits
    return random.choices(alphabet, k=8)

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

class ToolCall(BaseModel):
    id: str = Field(
        default_factory=build_tool_id,
        description="The unique identifier for the tool call",
    )
    type: Literal["function"]
    function: FunctionCall