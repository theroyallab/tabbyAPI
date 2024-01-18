""" Contains model card types. """
from pydantic import BaseModel, Field, ConfigDict
from time import time
from typing import List, Optional

from common.gen_logging import LogPreferences


class ModelCardParameters(BaseModel):
    """Represents model card parameters."""

    # Safe to do this since it's guaranteed to fetch a max seq len
    # from model_container
    max_seq_len: Optional[int] = None
    rope_scale: Optional[float] = 1.0
    rope_alpha: Optional[float] = 1.0
    cache_mode: Optional[str] = "FP16"
    prompt_template: Optional[str] = None
    num_experts_per_token: Optional[int] = None
    use_cfg: Optional[bool] = None
    draft: Optional["ModelCard"] = None


class ModelCard(BaseModel):
    """Represents a single model card."""

    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    logging: Optional[LogPreferences] = None
    parameters: Optional[ModelCardParameters] = None


class ModelList(BaseModel):
    """Represents a list of model cards."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class DraftModelLoadRequest(BaseModel):
    """Represents a draft model load request."""

    draft_model_name: str
    draft_rope_scale: Optional[float] = 1.0
    draft_rope_alpha: Optional[float] = Field(
        description="Automatically calculated if not present",
        default=None,
        examples=[1.0],
    )


# TODO: Unify this with ModelCardParams
class ModelLoadRequest(BaseModel):
    """Represents a model load request."""

    name: str

    # Max seq len is fetched from config.json of the model by default
    max_seq_len: Optional[int] = Field(
        description="Leave this blank to use the model's base sequence length",
        default=None,
        examples=[4096],
    )
    override_base_seq_len: Optional[int] = Field(
        description=(
            "Overrides the model's base sequence length. " "Leave blank if unsure"
        ),
        default=None,
        examples=[4096],
    )
    gpu_split_auto: Optional[bool] = True
    gpu_split: Optional[List[float]] = Field(
        default_factory=list, examples=[[24.0, 20.0]]
    )
    rope_scale: Optional[float] = Field(
        description="Automatically pulled from the model's config if not present",
        default=None,
        examples=[1.0],
    )
    rope_alpha: Optional[float] = Field(
        description="Automatically calculated if not present",
        default=None,
        examples=[1.0],
    )
    no_flash_attention: Optional[bool] = False
    # low_mem: Optional[bool] = False
    cache_mode: Optional[str] = "FP16"
    prompt_template: Optional[str] = None
    num_experts_per_token: Optional[int] = None
    use_cfg: Optional[bool] = None
    draft: Optional[DraftModelLoadRequest] = None


class ModelLoadResponse(BaseModel):
    """Represents a model load response."""

    # Avoids pydantic namespace warning
    model_config = ConfigDict(protected_namespaces=[])

    model_type: str = "model"
    module: int
    modules: int
    status: str
