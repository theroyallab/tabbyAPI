"""Contains model card types."""

from pydantic import BaseModel, Field, ConfigDict
from time import time
from typing import List, Optional

from common.gen_logging import GenLogPreferences
from common.model import get_config_default


class ModelCardParameters(BaseModel):
    """Represents model card parameters."""

    # Safe to do this since it's guaranteed to fetch a max seq len
    # from model_container
    max_seq_len: Optional[int] = None
    rope_scale: Optional[float] = 1.0
    rope_alpha: Optional[float] = 1.0
    cache_size: Optional[int] = None
    cache_mode: Optional[str] = "FP16"
    chunk_size: Optional[int] = 2048
    prompt_template: Optional[str] = None
    num_experts_per_token: Optional[int] = None

    # Draft is another model, so include it in the card params
    draft: Optional["ModelCard"] = None


class ModelCard(BaseModel):
    """Represents a single model card."""

    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    logging: Optional[GenLogPreferences] = None
    parameters: Optional[ModelCardParameters] = None


class ModelList(BaseModel):
    """Represents a list of model cards."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class DraftModelLoadRequest(BaseModel):
    """Represents a draft model load request."""

    # Required
    draft_model_name: str

    # Config arguments
    draft_rope_scale: Optional[float] = Field(
        default_factory=lambda: get_config_default(
            "draft_rope_scale", 1.0, model_type="draft"
        )
    )
    draft_rope_alpha: Optional[float] = Field(
        description="Automatically calculated if not present",
        default_factory=lambda: get_config_default(
            "draft_rope_alpha", None, model_type="draft"
        ),
        examples=[1.0],
    )
    draft_cache_mode: Optional[str] = Field(
        default_factory=lambda: get_config_default(
            "draft_cache_mode", "FP16", model_type="draft"
        )
    )


class ModelLoadRequest(BaseModel):
    """Represents a model load request."""

    # Required
    name: str

    # Config arguments

    # Max seq len is fetched from config.json of the model by default
    max_seq_len: Optional[int] = Field(
        description="Leave this blank to use the model's base sequence length",
        default_factory=lambda: get_config_default("max_seq_len"),
        examples=[4096],
    )
    override_base_seq_len: Optional[int] = Field(
        description=(
            "Overrides the model's base sequence length. " "Leave blank if unsure"
        ),
        default_factory=lambda: get_config_default("override_base_seq_len"),
        examples=[4096],
    )
    cache_size: Optional[int] = Field(
        description=("Number in tokens, must be greater than or equal to max_seq_len"),
        default_factory=lambda: get_config_default("cache_size"),
        examples=[4096],
    )
    gpu_split_auto: Optional[bool] = Field(
        default_factory=lambda: get_config_default("gpu_split_auto", True)
    )
    autosplit_reserve: Optional[List[float]] = Field(
        default_factory=lambda: get_config_default("autosplit_reserve", [96])
    )
    gpu_split: Optional[List[float]] = Field(
        default_factory=lambda: get_config_default("gpu_split", []),
        examples=[[24.0, 20.0]],
    )
    rope_scale: Optional[float] = Field(
        description="Automatically pulled from the model's config if not present",
        default_factory=lambda: get_config_default("rope_scale"),
        examples=[1.0],
    )
    rope_alpha: Optional[float] = Field(
        description="Automatically calculated if not present",
        default_factory=lambda: get_config_default("rope_alpha"),
        examples=[1.0],
    )
    cache_mode: Optional[str] = Field(
        default_factory=lambda: get_config_default("cache_mode", "FP16")
    )
    chunk_size: Optional[int] = Field(
        default_factory=lambda: get_config_default("chunk_size", 2048)
    )
    prompt_template: Optional[str] = Field(
        default_factory=lambda: get_config_default("prompt_template")
    )
    num_experts_per_token: Optional[int] = Field(
        default_factory=lambda: get_config_default("num_experts_per_token")
    )
    fasttensors: Optional[bool] = Field(
        default_factory=lambda: get_config_default("fasttensors", False)
    )

    # Non-config arguments
    draft: Optional[DraftModelLoadRequest] = None
    skip_queue: Optional[bool] = False


class EmbeddingModelLoadRequest(BaseModel):
    name: str
    embeddings_device: Optional[str] = Field(
        default_factory=lambda: get_config_default(
            "embeddings_device", model_type="embedding"
        )
    )


class ModelLoadResponse(BaseModel):
    """Represents a model load response."""

    # Avoids pydantic namespace warning
    model_config = ConfigDict(protected_namespaces=[])

    model_type: str = "model"
    module: int
    modules: int
    status: str
