"""Lora types"""

from pydantic import BaseModel, Field
from time import time
from typing import Optional, List


class LoraCard(BaseModel):
    """Represents a single Lora card."""

    id: str = "test"
    object: str = "lora"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    scaling: Optional[float] = None


class LoraList(BaseModel):
    """Represents a list of Lora cards."""

    object: str = "list"
    data: List[LoraCard] = Field(default_factory=list)


class LoraLoadInfo(BaseModel):
    """Represents a single Lora load info."""

    name: str
    scaling: Optional[float] = 1.0


class LoraLoadRequest(BaseModel):
    """Represents a Lora load request."""

    loras: List[LoraLoadInfo]
    skip_queue: bool = False


class LoraLoadResponse(BaseModel):
    """Represents a Lora load response."""

    success: List[str] = Field(default_factory=list)
    failure: List[str] = Field(default_factory=list)
