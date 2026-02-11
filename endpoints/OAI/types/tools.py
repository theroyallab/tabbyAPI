from pydantic import BaseModel, Field
from typing import Dict, Literal
from uuid import uuid4


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

    id: str = Field(default_factory=lambda: str(uuid4()).replace("-", "")[:9])
    function: Tool
    type: Literal["function"] = "function"


class NamedToolFunction(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    function: NamedToolFunction
    type: Literal["function"] = "function"
