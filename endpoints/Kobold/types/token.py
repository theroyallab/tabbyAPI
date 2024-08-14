from pydantic import BaseModel
from typing import List


class TokenCountRequest(BaseModel):
    """Represents a KAI tokenization request."""

    prompt: str


class TokenCountResponse(BaseModel):
    """Represents a KAI tokenization response."""

    value: int
    ids: List[int]
