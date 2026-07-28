"""Contains model card types."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, computed_field
from time import time
from typing import List, Literal, Optional, Union

from common.config_models import LoggingConfig
from common.tabby_config import config


class ModelCardParameters(BaseModel):
    """Represents model card parameters."""

    # Safe to do this since it's guaranteed to fetch a max seq len
    # from model_container
    max_seq_len: Optional[int] = None
    cache_size: Optional[int] = None
    cache_mode: Optional[str] = "FP16"
    rope_scale: Optional[float] = 1.0
    rope_alpha: Optional[float] = 1.0
    max_batch_size: Optional[int] = 1
    chunk_size: Optional[int] = 2048
    prompt_template: Optional[str] = None
    prompt_template_content: Optional[str] = None
    use_vision: Optional[bool] = False

    # Draft is another model, so include it in the card params
    draft: Optional["ModelCard"] = None


class ModelCard(BaseModel):
    """
    Represents a single model card.

    Carries the OpenAI fields (object, created) alongside the Anthropic ones
    (type, display_name, created_at), which name the same things differently.
    Anthropic SDK model types require theirs, and serving both keeps one
    listing usable by either client.
    """

    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    logging: Optional[LoggingConfig] = None
    parameters: Optional[ModelCardParameters] = None

    # Anthropic aliases, filled from the fields above when not set
    type: str = "model"
    display_name: Optional[str] = None
    created_at: Optional[str] = None

    def model_post_init(self, __context):
        if self.display_name is None:
            self.display_name = self.id

        if self.created_at is None:
            self.created_at = (
                datetime.fromtimestamp(self.created, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

    @computed_field
    @property
    def max_input_tokens(self) -> Optional[int]:
        """The context window under the name the Anthropic API gives it."""

        return self.parameters.max_seq_len if self.parameters else None

    @computed_field
    @property
    def max_tokens(self) -> Optional[int]:
        """
        The output ceiling, which is the context window here.

        TabbyAPI caps a generation by the context minus the prompt rather than
        by a separate output limit, so the context length is the honest upper
        bound. The field is served because Anthropic's model schema carries it
        beside max_input_tokens and a client reading one expects the other.
        """

        return self.parameters.max_seq_len if self.parameters else None


class ModelList(BaseModel):
    """Represents a list of model cards."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)

    # Anthropic pagination fields. TabbyAPI serves the whole list at once, so
    # there is never another page. The ids are computed rather than stored
    # because callers build the list empty and append to data afterwards.
    has_more: bool = False

    @computed_field
    @property
    def first_id(self) -> Optional[str]:
        return self.data[0].id if self.data else None

    @computed_field
    @property
    def last_id(self) -> Optional[str]:
        return self.data[-1].id if self.data else None


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
        description="Number in tokens, must be multiple of 256",
        default=None,
        examples=[4096],
    )
    cache_mode: Optional[str] = None
    tensor_parallel: Optional[bool] = None
    tensor_parallel_backend: Optional[str] = "native"
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
    chunk_size: Optional[int] = None
    output_chunking: Optional[bool] = True
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
