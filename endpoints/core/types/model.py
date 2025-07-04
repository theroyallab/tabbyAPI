"""Contains model card types."""

from pydantic import BaseModel, Field, ConfigDict
from time import time
from typing import List, Literal, Optional, Union

from common.config_models import LoggingConfig
from common.tabby_config import config


class ModelCardParameters(BaseModel):
    """Represents model card parameters."""

    # Safe to do this since it's guaranteed to fetch a max seq len
    # from model_container
    max_seq_len: Optional[int] = None
    rope_scale: Optional[float] = 1.0
    rope_alpha: Optional[float] = 1.0
    max_batch_size: Optional[int] = 1
    cache_size: Optional[int] = None
    cache_mode: Optional[str] = "FP16"
    chunk_size: Optional[int] = 2048
    prompt_template: Optional[str] = None
    prompt_template_content: Optional[str] = None
    use_vision: Optional[bool] = False

    # Draft is another model, so include it in the card params
    draft: Optional["ModelCard"] = None


class ModelCard(BaseModel):
    """Represents a single model card."""

    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    logging: Optional[LoggingConfig] = None
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
    draft_rope_scale: Optional[float] = None
    draft_rope_alpha: Optional[Union[float, Literal["auto"]]] = Field(
        description='Automatically calculated if set to "auto"',
        default=None,
        examples=[1.0],
    )
    draft_gpu_split: Optional[List[float]] = Field(
        default_factory=list,
        examples=[[24.0, 20.0]],
    )


class ModelLoadRequest(BaseModel):
    """Represents a model load request."""

    # Avoids pydantic namespace warning
    model_config = ConfigDict(protected_namespaces=[])

    # Required
    model_name: str

    # Config arguments
    backend: Optional[str] = Field(
        description="Backend to use",
        default=None,
    )
    max_seq_len: Optional[int] = Field(
        description="Leave this blank to use the model's base sequence length",
        default=None,
        examples=[4096],
    )
    cache_size: Optional[int] = Field(
        description=("Number in tokens, must be greater than or equal to max_seq_len"),
        default=None,
        examples=[4096],
    )
    tensor_parallel: Optional[bool] = None
    gpu_split_auto: Optional[bool] = None
    autosplit_reserve: Optional[List[float]] = None
    gpu_split: Optional[List[float]] = Field(
        default_factory=list,
        examples=[[24.0, 20.0]],
    )
    rope_scale: Optional[float] = Field(
        description="Automatically pulled from the model's config if not present",
        default=None,
        examples=[1.0],
    )
    rope_alpha: Optional[Union[float, Literal["auto"]]] = Field(
        description='Automatically calculated if set to "auto"',
        default=None,
        examples=[1.0],
    )
    cache_mode: Optional[str] = None
    chunk_size: Optional[int] = None
    prompt_template: Optional[str] = None
    vision: Optional[bool] = None

    # Non-config arguments
    draft_model: Optional[DraftModelLoadRequest] = None
    skip_queue: Optional[bool] = False


class EmbeddingModelLoadRequest(BaseModel):
    embedding_model_name: str

    # Set default from the config
    embeddings_device: Optional[str] = Field(config.embeddings.embeddings_device)


class ModelLoadResponse(BaseModel):
    """Represents a model load response."""

    # Avoids pydantic namespace warning
    model_config = ConfigDict(protected_namespaces=[])

    model_type: str = "model"
    module: int
    modules: int
    status: str


class ModelDefaultGenerationSettings(BaseModel):
    """Contains default generation settings for model props."""

    n_ctx: int


class ModelPropsResponse(BaseModel):
    """Represents a model props response."""

    total_slots: int = 1
    chat_template: str = ""
    default_generation_settings: ModelDefaultGenerationSettings
