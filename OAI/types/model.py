from pydantic import BaseModel, Field
from time import time
from typing import List, Optional

class ModelCard(BaseModel):
    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"

class LoraCard(BaseModel):
    id: str = "test"
    object: str = "lora"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    scaling: Optional[float] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)

class LoraList(BaseModel):
    object: str = "list"
    data: List[LoraCard] = Field(default_factory=list)

class DraftModelLoadRequest(BaseModel):
    draft_model_name: str
    draft_rope_alpha: float = 1.0
    draft_rope_scale: float = 1.0

class ModelLoadRequest(BaseModel):
    name: str
    max_seq_len: Optional[int] = 4096
    gpu_split_auto: Optional[bool] = True
    gpu_split: Optional[List[float]] = Field(default_factory=list)
    rope_scale: Optional[float] = 1.0
    rope_alpha: Optional[float] = 1.0
    no_flash_attention: Optional[bool] = False
    low_mem: Optional[bool] = False
    draft: Optional[DraftModelLoadRequest] = None

class LoraLoadRequest(BaseModel):
    loras: List[dict]

class ModelLoadResponse(BaseModel):
    model_type: str = "model"
    module: int
    modules: int
    status: str
