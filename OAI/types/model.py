from pydantic import BaseModel, Field
from time import time
from typing import List, Optional

class ModelCardParameters(BaseModel):
    max_seq_len: Optional[int] = 4096
    rope_scale: Optional[float] = 1.0
    rope_alpha: Optional[float] = 1.0
    draft: Optional['ModelCard'] = None

class ModelCard(BaseModel):
    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    parameters: Optional[ModelCardParameters] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)

class DraftModelLoadRequest(BaseModel):
    draft_model_name: str
    draft_rope_alpha: float = 1.0
    draft_rope_scale: float = 1.0

# TODO: Unify this with ModelCardParams
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

class ModelLoadResponse(BaseModel):
    model_type: str = "model"
    module: int
    modules: int
    status: str
