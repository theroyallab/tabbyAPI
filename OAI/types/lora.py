from pydantic import BaseModel, Field
from time import time
from typing import Optional, List

class LoraCard(BaseModel):
    id: str = "test"
    object: str = "lora"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    scaling: Optional[float] = None

class LoraList(BaseModel):
    object: str = "list"
    data: List[LoraCard] = Field(default_factory=list)

class LoraLoadInfo(BaseModel):
    name: str
    scaling: Optional[float] = 1.0

class LoraLoadRequest(BaseModel):
    loras: List[LoraLoadInfo]

class LoraLoadResponse(BaseModel):
    success: List[str] = Field(default_factory=list)
    failure: List[str] = Field(default_factory=list)
