from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional
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
    """Represents an OAI tool call.

    The ``index`` field is optional so it can be omitted in non-streaming
    responses (where OpenAI does not include it) via ``exclude_none=True``,
    while being set explicitly for streaming deltas where it is required
    by strict validators like the Vercel AI SDK.
    """

    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:24]}")
    function: Tool
    type: Literal["function"] = "function"
    index: Optional[int] = None


class NamedToolFunction(BaseModel):
    """Represents a named function reference for tool_choice."""

    name: str


class NamedToolChoice(BaseModel):
    """Represents a named tool choice (forces a specific function call)."""

    function: NamedToolFunction
    type: Literal["function"] = "function"
