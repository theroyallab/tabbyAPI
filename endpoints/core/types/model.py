"""Contains model card types."""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from time import time
from typing import Literal

from common.config_models import LoggingConfig
from common.tabby_config import config


class ModelCardParameters(BaseModel):
    """Represents model card parameters."""

    # Safe to do this since it's guaranteed to fetch a max seq len
    # from model_container
    max_seq_len: int | None = None
    cache_size: int | None = None
    cache_mode: str | None = "FP16"
    rope_scale: float | None = 1.0
    rope_alpha: float | None = 1.0
    max_batch_size: int | None = 1
    chunk_size: int | None = 2048
    prompt_template: str | None = None
    prompt_template_content: str | None = None
    use_vision: bool | None = False

    # Draft is another model, so include it in the card params
    draft: ModelCard | None = None


class ModelCard(BaseModel):
    """Represents a single model card."""

    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    logging: LoggingConfig | None = None
    parameters: ModelCardParameters | None = None


class ModelList(BaseModel):
    """Represents a list of model cards."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class DraftModelLoadRequest(BaseModel):
    """Represents a draft model load request."""

    # Required
    draft_model_name: str

    # Config arguments
    draft_rope_scale: float | None = None
    draft_rope_alpha: float | Literal["auto"] | None = Field(
        description='Automatically calculated if set to "auto"',
        default=None,
        examples=[1.0],
    )
    draft_gpu_split: list[float] | None = Field(
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
    backend: str | None = Field(
        description="Backend to use",
        default=None,
    )
    max_seq_len: int | None = Field(
        description="Leave this blank to use the model's base sequence length",
        default=None,
        examples=[4096],
    )
    cache_size: int | None = Field(
        description="Number in tokens, must be multiple of 256",
        default=None,
        examples=[4096],
    )
    cache_mode: str | None = None
    tensor_parallel: bool | None = None
    tensor_parallel_backend: str | None = "native"
    gpu_split_auto: bool | None = None
    autosplit_reserve: list[float] | None = None
    gpu_split: list[float] | None = Field(
        default_factory=list,
        examples=[[24.0, 20.0]],
    )
    rope_scale: float | None = Field(
        description="Automatically pulled from the model's config if not present",
        default=None,
        examples=[1.0],
    )
    rope_alpha: float | Literal["auto"] | None = Field(
        description='Automatically calculated if set to "auto"',
        default=None,
        examples=[1.0],
    )
    chunk_size: int | None = None
    output_chunking: bool | None = True
    prompt_template: str | None = None
    vision: bool | None = None

    # Non-config arguments
    draft_model: DraftModelLoadRequest | None = None
    skip_queue: bool | None = False


class EmbeddingModelLoadRequest(BaseModel):
    embedding_model_name: str

    # Set default from the config
    embeddings_device: str | None = Field(config.embeddings.embeddings_device)


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
