"""Types for the Anthropic Messages API."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4


# Request content blocks
#
# Blocks carry per-type fields, so they're modelled as a discriminated union
# with a permissive fallback. The fallback lets the converter report an
# unsupported block by name instead of returning a pydantic validation dump.
# Fields the server ignores (cache_control, citations) are dropped silently,
# which is what pydantic does with extra keys by default.


class TextBlock(BaseModel):
    """A text block."""

    type: Literal["text"]
    text: str


class ThinkingBlock(BaseModel):
    """A thinking block replayed by the client from a previous turn."""

    type: Literal["thinking"]
    thinking: str = ""
    signature: Optional[str] = None


class RedactedThinkingBlock(BaseModel):
    """An encrypted thinking block. Carries nothing this server can replay."""

    type: Literal["redacted_thinking"]
    data: Optional[str] = None


class ImageSource(BaseModel):
    """Where an image block's data comes from."""

    model_config = ConfigDict(extra="allow")

    # Not a Literal so an unsupported source is reported by name
    type: str
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None
    file_id: Optional[str] = None


class ImageBlock(BaseModel):
    """An image, either inline base64 or a URL to fetch."""

    type: Literal["image"]
    source: ImageSource


class ToolUseBlock(BaseModel):
    """A tool call the model made on a previous turn, replayed by the client."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """The result of a tool call, sent back in a user turn."""

    type: Literal["tool_result"]
    tool_use_id: str

    # Anthropic allows a bare string or a list of blocks here
    content: Optional[Union[str, List["RequestContentBlock"]]] = None
    is_error: Optional[bool] = False


class UnsupportedBlock(BaseModel):
    """Any block type this server does not handle yet."""

    model_config = ConfigDict(extra="allow")

    type: str


KnownRequestBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        ThinkingBlock,
        RedactedThinkingBlock,
        ToolUseBlock,
        ToolResultBlock,
    ],
    Field(discriminator="type"),
]
RequestContentBlock = Union[KnownRequestBlock, UnsupportedBlock]

ToolResultBlock.model_rebuild()


class ToolDefinition(BaseModel):
    """
    A tool the model may call.

    Anthropic's server-side tools (web search, code execution) arrive in the
    same list distinguished by a `type`, so the field is modelled here in
    order to reject them by name rather than fail schema validation.
    """

    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    type: Optional[str] = None


class ToolChoice(BaseModel):
    """How the model should choose among the available tools."""

    model_config = ConfigDict(extra="allow")

    # Not a Literal so an unrecognized mode is reported by name
    type: str
    name: Optional[str] = None
    disable_parallel_tool_use: Optional[bool] = None


class AnthropicMessage(BaseModel):
    """
    A single turn of the conversation.

    A system role is accepted mid-list: recent Anthropic models take operator
    instructions that way rather than by editing the top-level system prompt,
    and Claude Code sends them on every request.
    """

    role: Literal["user", "assistant", "system"]
    content: Union[str, List[RequestContentBlock]]


class ThinkingConfig(BaseModel):
    """
    Reasoning configuration. Only the on/off distinction is used; a token
    budget has no equivalent in a local chat template.
    """

    model_config = ConfigDict(extra="allow")

    # Not a Literal: the set of thinking types grows over time and an
    # unrecognized one should not fail the request
    type: str = "enabled"
    budget_tokens: Optional[int] = None


class Metadata(BaseModel):
    """Request metadata. Only user_id is carried over."""

    model_config = ConfigDict(extra="allow")

    user_id: Optional[str] = None


class MessagesRequest(BaseModel):
    """Represents an Anthropic Messages request."""

    messages: List[AnthropicMessage]

    # Required by the Anthropic API, unlike OAI where it's a sampler default
    max_tokens: int = Field(..., ge=1)

    # Optional here, unlike the Anthropic API: TabbyAPI serves the loaded
    # model when a request doesn't name one
    model: Optional[str] = None

    system: Optional[Union[str, List[TextBlock]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False

    # Bounds are validated against the sampler request these map onto, so a
    # local model can be driven outside the ranges the Anthropic API accepts
    temperature: Optional[float] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)

    thinking: Optional[ThinkingConfig] = None
    metadata: Optional[Metadata] = None

    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None


class CountTokensRequest(BaseModel):
    """Represents an Anthropic token counting request."""

    messages: List[AnthropicMessage]
    model: Optional[str] = None
    system: Optional[Union[str, List[TextBlock]]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None


# Response types


class ResponseTextBlock(BaseModel):
    """A text block in a response."""

    type: Literal["text"] = "text"
    text: str


class ResponseThinkingBlock(BaseModel):
    """
    A thinking block in a response.

    The signature is always empty: it authenticates thinking replayed to the
    Anthropic API, and there is nothing to authenticate against locally. The
    field is emitted anyway because SDK response models require it.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class ResponseToolUseBlock(BaseModel):
    """A tool call in a response."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


ResponseContentBlock = Union[ResponseThinkingBlock, ResponseTextBlock, ResponseToolUseBlock]


class Usage(BaseModel):
    """Token usage for a response."""

    input_tokens: int
    output_tokens: int

    # Always zero. Emitted because SDK response models expect the fields and
    # clients divide by them when reporting cache efficiency.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Represents an Anthropic Messages response."""

    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ResponseContentBlock]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage


class CountTokensResponse(BaseModel):
    """Represents an Anthropic token counting response."""

    input_tokens: int
