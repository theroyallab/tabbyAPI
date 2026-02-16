from pydantic import BaseModel


class TokenCountRequest(BaseModel):
    """Represents a KAI tokenization request."""

    prompt: str


class TokenCountResponse(BaseModel):
    """Represents a KAI tokenization response."""

    value: int
    ids: list[int]
