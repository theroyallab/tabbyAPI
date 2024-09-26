"""Contains model card types."""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from time import time
from typing import List, Literal, Optional, Union

from backends.exllamav2.types import DraftModelInstanceConfig, ModelInstanceConfig
from common.config_models import LoggingConfig
from common.tabby_config import config


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
    logging: Optional[LoggingConfig] = None
    parameters: Optional[ModelCardParameters] = None


class ModelList(BaseModel):
    """Represents a list of model cards."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class ModelLoadRequest(ModelInstanceConfig):
    """Represents a model load request."""

    # These Fields only exist to stop a breaking change
    name: Optional[str] = Field(
        None, description="model name to load", deprecated="Use model_name instead"
    )
    fasttensors: Optional[bool] = Field(
        None,
        description="ignored, set globally from config.yml",
        deprecated="Use model config instead",
    )

    # Non-config arguments
    draft: Optional[DraftModelInstanceConfig] = None
    skip_queue: Optional[bool] = False

    # for the name value
    @model_validator(mode="after")
    def set_model_name(self):
        """Sets the model name."""
        if self.name and self.model_name is None:
            self.model_name = self.name
        return self


class EmbeddingModelLoadRequest(BaseModel):
    name: str

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
