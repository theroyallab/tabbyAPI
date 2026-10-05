"""Contains model card types."""

import difflib
from pydantic import BaseModel, Field, ConfigDict, model_validator
from time import time
from typing import ClassVar, List, Literal, Optional, Union

from common.config_models import LoggingConfig
from common.logger import xlogger
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


class ModelCardMeta(BaseModel):
    """
    Model metadata in the shape llama-server attaches to /v1/models entries, which
    local-model clients read to size their context window. Only fields TabbyAPI can
    determine are included.
    """

    n_ctx_train: int = Field(0, description="Context length the model was trained with")
    n_ctx: Optional[int] = Field(None, description="Loaded context length (max_seq_len)")
    n_vocab: int = 0
    n_embd: int = 0
    size: int = Field(0, description="Size of the weight files in bytes (loaded model only)")


class ModelCard(BaseModel):
    """Represents a single model card."""

    id: str = "test"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "tabbyAPI"
    logging: Optional[LoggingConfig] = None
    parameters: Optional[ModelCardParameters] = None
    meta: Optional[ModelCardMeta] = None


class ModelList(BaseModel):
    """Represents a list of model cards."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class _WarnOnUnknownFields(BaseModel):
    """
    Load requests ignore keys they don't know, so existing clients keep working,
    but a silently dropped option is hard to notice. Log each unknown key once
    per request, with the closest known option when it looks like a typo.
    """

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_fields(cls, values):
        if not isinstance(values, dict):
            return values

        known = set(cls.model_fields)
        for name in values:
            if name in known:
                continue
            close = difflib.get_close_matches(name, known, n=1, cutoff=0.75)
            hint = f' Did you mean "{close[0]}"?' if close else ""
            xlogger.warning(
                f'Ignoring unknown option "{name}" in {cls._request_label} request.{hint}'
            )

        return values


class DraftModelLoadRequest(_WarnOnUnknownFields):
    """Represents a draft model load request."""

    _request_label: ClassVar[str] = "the draft_model block of a model load"

    # Not needed for the mtp and ngram draft modes
    draft_model_name: Optional[str] = None

    # Config arguments
    draft_mode: Optional[Literal["model", "disabled", "mtp", "ngram"]] = None
    draft_cache_mode: Optional[str] = None
    draft_num_tokens: Optional[int] = None
    dynamic_draft: Optional[bool] = None
    ngram_match_min: Optional[int] = None
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


class ModelLoadRequest(_WarnOnUnknownFields):
    """
    Represents a model load request. Options left out fall back to the model
    folder's tabby_config.yml, then the config's use_as_default keys, then the
    backend defaults.
    """

    _request_label: ClassVar[str] = "a model load"

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
    recurrent_checkpoint_interval: Optional[int] = Field(
        description="Recurrent checkpoint interval during generation, multiple of 256",
        default=None,
        examples=[2048],
        multiple_of=256,
        gt=0,
    )
    recurrent_checkpoint_interval_pp: Optional[int] = Field(
        description="Recurrent checkpoint interval during prompt ingestion, multiple of 256",
        default=None,
        examples=[2048],
        multiple_of=256,
        gt=0,
    )
    max_batch_size: Optional[int] = None
    prompt_template: Optional[str] = None
    vision: Optional[bool] = None
    vision_offload: Optional[bool] = None
    sampling: Optional[dict] = None
    warmup: Optional[bool] = None

    # Memory and offload
    ngram_ram: Optional[bool] = None
    embed_stream_from_disk: Optional[bool] = None
    cpu_moe_offload_layers: Optional[int] = None
    cpu_moe_split_experts: Optional[int] = None
    cpu_moe_threads: Optional[int] = None

    # Template variables
    template_vars_default: Optional[dict] = None
    template_vars_force: Optional[dict] = None
    force_enable_thinking: Optional[bool] = None

    # Reasoning and tool call parsing
    reasoning: Optional[bool] = None
    reasoning_start_token: Optional[str] = None
    reasoning_end_token: Optional[str] = None
    start_in_reasoning: Optional[str] = None
    tool_calls_in_reasoning: Optional[bool] = None
    reasoning_budget_tokens: Optional[int] = None
    reasoning_budget_message: Optional[str] = None
    tool_format: Optional[str] = Field(
        default=None,
        description='Format name, "auto", or an empty string to disable tool call parsing',
    )
    harmony: Optional[bool] = None
    muse_glimmer: Optional[bool] = None

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


class ModelPropsModalities(BaseModel):
    """Input modalities of the loaded model."""

    vision: bool = False


class ModelPropsResponse(BaseModel):
    """
    Represents a model props response, in the shape of llama-server's /props so
    clients written for it can discover the context size and modalities.
    """

    total_slots: int = 1
    model_path: str = ""
    chat_template: str = ""
    default_generation_settings: ModelDefaultGenerationSettings
    modalities: ModelPropsModalities = Field(default_factory=ModelPropsModalities)
