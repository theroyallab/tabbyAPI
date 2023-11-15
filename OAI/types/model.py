from pydantic import BaseModel, Field
from time import time
from typing import List, Optional

class ModelCard(BaseModel):
    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)

class ModelLoadRequest(BaseModel):
    name: str
    max_seq_len: Optional[int] = 4096
    gpu_split: Optional[str] = "auto"
    rope_scale: Optional[float] = 1.0
    rope_alpha: Optional[float] = 1.0
    no_flash_attention: Optional[bool] = False
    low_mem: Optional[bool] = False

class ModelLoadResponse(BaseModel):
    module: int
    modules: int
    status: str
