from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr
from typing import List, Literal, Optional, Union
from pathlib import Path

from pydantic_core import PydanticUndefined

CACHE_SIZES = Literal["FP16", "Q8", "Q6", "Q4"]


class Metadata(BaseModel):
    """metadata model for config options"""

    include_in_config: Optional[bool] = Field(True)


class BaseConfigModel(BaseModel):
    """Base model for config models with added metadata"""

    _metadata: Metadata = PrivateAttr(Metadata())


class ConfigOverrideConfig(BaseConfigModel):
    """Model for overriding a provided config file."""

    # TODO: convert this to a pathlib.path?
    config: Optional[str] = Field(
        None, description=("Path to an overriding config.yml file")
    )

    _metadata: Metadata = PrivateAttr(Metadata(include_in_config=False))


class UtilityActions(BaseConfigModel):
    """Model used for arg actions."""

    # YAML export options
    export_config: Optional[str] = Field(
        None, description="generate a template config file"
    )
    config_export_path: Optional[Path] = Field(
        "config_sample.yml", description="path to export configuration file to"
    )

    # OpenAPI JSON export options
    export_openapi: Optional[bool] = Field(
        False, description="export openapi schema files"
    )
    openapi_export_path: Optional[Path] = Field(
        "openapi.json", description="path to export openapi schema to"
    )

    _metadata: Metadata = PrivateAttr(Metadata(include_in_config=False))


class NetworkConfig(BaseConfigModel):
    """Model for network configuration."""

    host: Optional[str] = Field("127.0.0.1", description=("The IP to host on"))
    port: Optional[int] = Field(5000, description=("The port to host on"))
    disable_auth: Optional[bool] = Field(
        False, description=("Disable HTTP token authentication with requests")
    )
    send_tracebacks: Optional[bool] = Field(
        False,
        description=("Decide whether to send error tracebacks over the API"),
    )
    api_servers: Optional[List[Literal["OAI", "Kobold"]]] = Field(
        default_factory=list,
        description=("API servers to enable. Options: (OAI, Kobold)"),
    )


# TODO: Migrate config.yml to have the log_ prefix
# This is a breaking change.
class LoggingConfig(BaseConfigModel):
    """Model for logging configuration."""

    log_prompt: Optional[bool] = Field(
        False,
        description=("Enable prompt logging"),
        validation_alias=AliasChoices("log_prompt", "prompt"),
    )
    log_generation_params: Optional[bool] = Field(
        False,
        description=("Enable generation parameter logging"),
        validation_alias=AliasChoices("log_generation_params", "generation_params"),
    )
    log_requests: Optional[bool] = Field(
        False,
        description=("Enable request logging"),
        validation_alias=AliasChoices("log_requests", "requests"),
    )


class ModelConfig(BaseConfigModel):
    """Model for LLM configuration."""

    # TODO: convert this to a pathlib.path?
    model_dir: str = Field(
        "models",
        description=(
            "Overrides the directory to look for models (default: models). Windows "
            "users, do NOT put this path in quotes."
        ),
    )
    use_dummy_models: Optional[bool] = Field(
        False,
        description=(
            "Sends dummy model names when the models endpoint is queried. Enable this "
            "if looking for specific OAI models."
        ),
    )
    model_name: Optional[str] = Field(
        None,
        description=(
            "An initial model to load. Make sure the model is located in the model "
            "directory! REQUIRED: This must be filled out to load a model on startup."
        ),
    )
    use_as_default: List[str] = Field(
        default_factory=list,
        description=(
            "Names of args to use as a default fallback for API load requests "
            "(default: []). Example: ['max_seq_len', 'cache_mode']"
        ),
    )
    max_seq_len: Optional[int] = Field(
        None,
        description=(
            "Max sequence length. Fetched from the model's base sequence length in "
            "config.json by default."
        ),
        ge=0,
    )
    override_base_seq_len: Optional[int] = Field(
        None,
        description=(
            "Overrides base model context length. WARNING: Only use this if the "
            "model's base sequence length is incorrect."
        ),
        ge=0,
    )
    tensor_parallel: Optional[bool] = Field(
        False,
        description=(
            "Load model with tensor parallelism. Fallback to autosplit if GPU split "
            "isn't provided."
        ),
    )
    gpu_split_auto: Optional[bool] = Field(
        True,
        description=(
            "Automatically allocate resources to GPUs (default: True). Not parsed for "
            "single GPU users."
        ),
    )
    autosplit_reserve: List[int] = Field(
        [96],
        description=(
            "Reserve VRAM used for autosplit loading (default: 96 MB on GPU 0). "
            "Represented as an array of MB per GPU."
        ),
    )
    gpu_split: List[float] = Field(
        default_factory=list,
        description=(
            "An integer array of GBs of VRAM to split between GPUs (default: []). "
            "Used with tensor parallelism."
        ),
    )
    rope_scale: Optional[float] = Field(
        1.0,
        description=(
            "Rope scale (default: 1.0). Same as compress_pos_emb. Only use if the "
            "model was trained on long context with rope."
        ),
    )
    rope_alpha: Optional[Union[float, Literal["auto"]]] = Field(
        1.0,
        description=(
            "Rope alpha (default: 1.0). Same as alpha_value. Set to 'auto' to auto- "
            "calculate."
        ),
    )
    cache_mode: Optional[CACHE_SIZES] = Field(
        "FP16",
        description=(
            "Enable different cache modes for VRAM savings (default: FP16). Possible "
            f"values: {str(CACHE_SIZES)[15:-1]}"
        ),
    )
    cache_size: Optional[int] = Field(
        None,
        description=(
            "Size of the prompt cache to allocate (default: max_seq_len). Must be a "
            "multiple of 256."
        ),
        multiple_of=256,
        gt=0,
    )
    chunk_size: Optional[int] = Field(
        2048,
        description=(
            "Chunk size for prompt ingestion (default: 2048). A lower value reduces "
            "VRAM usage but decreases ingestion speed."
        ),
        gt=0,
    )
    max_batch_size: Optional[int] = Field(
        None,
        description=(
            "Set the maximum number of prompts to process at one time (default: "
            "None/Automatic). Automatically calculated if left blank."
        ),
        ge=1,
    )
    prompt_template: Optional[str] = Field(
        None,
        description=(
            "Set the prompt template for this model. If empty, attempts to look for "
            "the model's chat template."
        ),
    )
    num_experts_per_token: Optional[int] = Field(
        None,
        description=(
            "Number of experts to use per token. Fetched from the model's "
            "config.json. For MoE models only."
        ),
        ge=1,
    )
    fasttensors: Optional[bool] = Field(
        False,
        description=(
            "Enables fasttensors to possibly increase model loading speeds (default: "
            "False)."
        ),
    )

    _metadata: Metadata = PrivateAttr(Metadata())
    model_config = ConfigDict(protected_namespaces=())


class DraftModelConfig(BaseConfigModel):
    """Model for draft LLM model configuration."""

    # TODO: convert this to a pathlib.path?
    draft_model_dir: Optional[str] = Field(
        "models",
        description=(
            "Overrides the directory to look for draft models (default: models)"
        ),
    )
    draft_model_name: Optional[str] = Field(
        None,
        description=(
            "An initial draft model to load. Ensure the model is in the model"
            "directory."
        ),
    )
    draft_rope_scale: Optional[float] = Field(
        1.0,
        description=(
            "Rope scale for draft models (default: 1.0). Same as compress_pos_emb. "
            "Use if the draft model was trained on long context with rope."
        ),
    )
    draft_rope_alpha: Optional[float] = Field(
        None,
        description=(
            "Rope alpha for draft models (default: None). Same as alpha_value. Leave "
            "blank to auto-calculate the alpha value."
        ),
    )
    draft_cache_mode: Optional[CACHE_SIZES] = Field(
        "FP16",
        description=(
            "Cache mode for draft models to save VRAM (default: FP16). Possible "
            f"values: {str(CACHE_SIZES)[15:-1]}"
        ),
    )


class LoraInstanceModel(BaseConfigModel):
    """Model representing an instance of a Lora."""

    name: str = Field(..., description=("Name of the LoRA model"))
    scaling: float = Field(
        1.0,
        description=("Scaling factor for the LoRA model (default: 1.0)"),
        ge=0,
    )


class LoraConfig(BaseConfigModel):
    """Model for lora configuration."""

    # TODO: convert this to a pathlib.path?
    lora_dir: Optional[str] = Field(
        "loras", description=("Directory to look for LoRAs (default: 'loras')")
    )
    loras: Optional[List[LoraInstanceModel]] = Field(
        None,
        description=(
            "List of LoRAs to load and associated scaling factors (default scaling: "
            "1.0)"
        ),
    )


class SamplingConfig(BaseConfigModel):
    """Model for sampling (overrides) config."""

    override_preset: Optional[str] = Field(
        None, description=("Select a sampler override preset")
    )


class DeveloperConfig(BaseConfigModel):
    """Model for developer settings configuration."""

    unsafe_launch: Optional[bool] = Field(
        False, description=("Skip Exllamav2 version check")
    )
    disable_request_streaming: Optional[bool] = Field(
        False, description=("Disables API request streaming")
    )
    cuda_malloc_backend: Optional[bool] = Field(
        False, description=("Runs with the pytorch CUDA malloc backend")
    )
    uvloop: Optional[bool] = Field(
        False, description=("Run asyncio using Uvloop or Winloop")
    )
    realtime_process_priority: Optional[bool] = Field(
        False,
        description=(
            "Set process to use a higher priority For realtime process priority, run "
            "as administrator or sudo Otherwise, the priority will be set to high"
        ),
    )


class EmbeddingsConfig(BaseConfigModel):
    """Model for embeddings configuration."""

    # TODO: convert this to a pathlib.path?
    embedding_model_dir: Optional[str] = Field(
        "models",
        description=(
            "Overrides directory to look for embedding models (default: models)"
        ),
    )
    embeddings_device: Optional[Literal["cpu", "auto", "cuda"]] = Field(
        "cpu",
        description=(
            "Device to load embedding models on (default: cpu). Possible values: cpu, "
            "auto, cuda. If using an AMD GPU, set this value to 'cuda'."
        ),
    )
    embedding_model_name: Optional[str] = Field(
        None, description=("The embeddings model to load")
    )


class TabbyConfigModel(BaseModel):
    """Base model for a TabbyConfig."""

    config: ConfigOverrideConfig = Field(
        default_factory=ConfigOverrideConfig.model_construct
    )
    network: NetworkConfig = Field(default_factory=NetworkConfig.model_construct)
    logging: LoggingConfig = Field(default_factory=LoggingConfig.model_construct)
    model: ModelConfig = Field(default_factory=ModelConfig.model_construct)
    draft_model: DraftModelConfig = Field(
        default_factory=DraftModelConfig.model_construct
    )
    lora: LoraConfig = Field(default_factory=LoraConfig.model_construct)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig.model_construct)
    developer: DeveloperConfig = Field(default_factory=DeveloperConfig.model_construct)
    embeddings: EmbeddingsConfig = Field(
        default_factory=EmbeddingsConfig.model_construct
    )
    actions: UtilityActions = Field(default_factory=UtilityActions.model_construct)

    model_config = ConfigDict(validate_assignment=True, protected_namespaces=())


def generate_config_file(
    model: BaseConfigModel = None,
    filename: str = "config_sample.yml",
    indentation: int = 2,
) -> None:
    """Creates a config.yml file from Pydantic models."""

    schema = model if model else TabbyConfigModel()
    yaml = ""

    for field, field_data in schema.model_fields.items():
        subfield_model = field_data.default_factory()
        if not subfield_model._metadata.include_in_config:
            continue

        yaml += f"# {subfield_model.__doc__}\n"
        yaml += f"{field}:\n"
        for subfield, subfield_data in subfield_model.model_fields.items():
            value = subfield_data.default
            value = value if value is not None else ""
            value = value if value is not PydanticUndefined else ""
            yaml += f"{' ' * indentation}# {subfield_data.description}\n"
            yaml += f"{' ' * indentation}{subfield}: {value}\n"
        yaml += "\n"

    with open(filename, "w") as f:
        f.write(yaml)
