from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    PrivateAttr,
    field_validator,
)
from typing import List, Literal, Optional
from pathlib import Path

from backends.exllamav2.types import DraftModelInstanceConfig, ModelInstanceConfig

class Metadata(BaseModel):
    """metadata model for config options"""

    include_in_config: bool = Field(
        True, description="if the model is included by the config file generator"
    )


class BaseConfigModel(BaseModel):
    """Base model for config models with added metadata"""

    _metadata: Metadata = PrivateAttr(Metadata())


class ConfigOverrideConfig(BaseConfigModel):
    """Model for overriding a provided config file."""

    config: Optional[FilePath] = Field(
        None, description=("Path to an overriding config.yml file")
    )

    _metadata: Metadata = PrivateAttr(Metadata(include_in_config=False))


class UtilityActions(BaseConfigModel):
    """Model used for arg actions."""

    # YAML export options
    export_config: bool = Field(False, description="generate a template config file")
    config_export_path: Path = Field(
        "config_sample.yml", description="path to export configuration file to"
    )

    # OpenAPI JSON export options
    export_openapi: bool = Field(False, description="export openapi schema files")
    openapi_export_path: Path = Field(
        "openapi.json", description="path to export openapi schema to"
    )

    _metadata: Metadata = PrivateAttr(Metadata(include_in_config=False))


class NetworkConfig(BaseConfigModel):
    """Options for networking"""

    # TODO: convert to IPvAnyAddress?
    host: str = Field(
        "127.0.0.1",
        description=(
            "The IP to host on (default: 127.0.0.1).\n"
            "Use 0.0.0.0 to expose on all network adapters."
        ),
    )
    port: int = Field(5000, description=("The port to host on (default: 5000)."))
    disable_auth: bool = Field(
        False,
        description=(
            "Disable HTTP token authentication with requests.\n"
            "WARNING: This will make your instance vulnerable!\n"
            "Turn on this option if you are ONLY connecting from localhost."
        ),
    )
    send_tracebacks: bool = Field(
        False,
        description=(
            "Send tracebacks over the API (default: False).\n"
            "NOTE: Only enable this for debug purposes."
        ),
    )
    api_servers: List[Literal["oai", "kobold"]] = Field(
        ["OAI"],
        description=(
            'Select API servers to enable (default: ["OAI"]).\n'
            "Possible values: OAI, Kobold."
        ),
    )

    # Converts all strings in the api_servers list to lowercase
    # NOTE: Expand if more models need this validator
    @field_validator("api_servers", mode="before")
    def api_server_validator(cls, api_servers):
        return [server_name.lower() for server_name in api_servers]


# TODO: Migrate config.yml to have the log_ prefix
# This is a breaking change.
class LoggingConfig(BaseConfigModel):
    """Options for logging"""

    log_prompt: bool = Field(
        False,
        description=("Enable prompt logging (default: False)."),
    )
    log_generation_params: bool = Field(
        False,
        description=("Enable generation parameter logging (default: False)."),
    )
    log_requests: bool = Field(
        False,
        description=(
            "Enable request logging (default: False).\n"
            "NOTE: Only use this for debugging!"
        ),
    )


class ModelConfig(BaseConfigModel, ModelInstanceConfig):
    """
    Options for model overrides and loading
    Please read the comments to understand how arguments are handled
    between initial and API loads
    """

    model_dir: DirectoryPath = Field(
        "models",
        description=(
            "Directory to look for models (default: models).\n"
            "Windows users, do NOT put this path in quotes!"
        ),
    )
    inline_model_loading: bool = Field(
        False,
        description=(
            "Allow direct loading of models "
            "from a completion or chat completion request (default: False)."
        ),
    )
    use_dummy_models: bool = Field(
        False,
        description=(
            "Sends dummy model names when the models endpoint is queried.\n"
            "Enable this if the client is looking for specific OAI models."
        ),
    )
    use_as_default: List[str] = Field(
        default_factory=list,
        description=(
            "Names of args to use as a fallback for API load requests (default: []).\n"
            "For example, if you always want cache_mode to be Q4 "
            'instead of on the inital model load, add "cache_mode" to this array.\n'
            "Example: ['max_seq_len', 'cache_mode']."
        ),
    )
    fasttensors: bool = Field(
        False,
        description=(
            "Enables fasttensors to possibly increase model loading speeds "
            "(default: False)."
        ),
    )

    _metadata: Metadata = PrivateAttr(Metadata())
    model_config = ConfigDict(protected_namespaces=())


class DraftModelConfig(BaseConfigModel, DraftModelInstanceConfig):
    """
    Options for draft models (speculative decoding)
    This will use more VRAM!
    """

    draft_model_dir: DirectoryPath = Field(
        "models",
        description=("Directory to look for draft models (default: models)"),
    )


class LoraInstanceModel(BaseConfigModel):
    """Model representing an instance of a Lora."""

    name: Optional[str] = None
    scaling: float = Field(1.0, ge=0)


class LoraConfig(BaseConfigModel):
    """Options for Loras"""

    # TODO: convert this to a pathlib.path?
    lora_dir: DirectoryPath = Field(
        "loras", description=("Directory to look for LoRAs (default: loras).")
    )
    loras: Optional[List[LoraInstanceModel]] = Field(
        None,
        description=(
            "List of LoRAs to load and associated scaling factors "
            "(default scale: 1.0).\n"
            "For the YAML file, add each entry as a YAML list:\n"
            "- name: lora1\n"
            "  scaling: 1.0"
        ),
    )


class EmbeddingsConfig(BaseConfigModel):
    """
    Options for embedding models and loading.
    NOTE: Embeddings requires the "extras" feature to be installed
    Install it via "pip install .[extras]"
    """

    embedding_model_dir: DirectoryPath = Field(
        "models",
        description=("Directory to look for embedding models (default: models)."),
    )
    embeddings_device: Literal["cpu", "auto", "cuda"] = Field(
        "cpu",
        description=(
            "Device to load embedding models on (default: cpu).\n"
            "Possible values: cpu, auto, cuda.\n"
            "NOTE: It's recommended to load embedding models on the CPU.\n"
            "If using an AMD GPU, set this value to 'cuda'."
        ),
    )
    embedding_model_name: Optional[str] = Field(
        None,
        description=("An initial embedding model to load on the infinity backend."),
    )


class SamplingConfig(BaseConfigModel):
    """Options for Sampling"""

    override_preset: Optional[str] = Field(
        None,
        description=(
            "Select a sampler override preset (default: None).\n"
            "Find this in the sampler-overrides folder.\n"
            "This overrides default fallbacks for sampler values "
            "that are passed to the API."
        ),
    )


class DeveloperConfig(BaseConfigModel):
    """Options for development and experimentation"""

    unsafe_launch: bool = Field(
        False,
        description=(
            "Skip Exllamav2 version check (default: False).\n"
            "WARNING: It's highly recommended to update your dependencies rather "
            "than enabling this flag."
        ),
    )
    disable_request_streaming: bool = Field(
        False, description=("Disable API request streaming (default: False).")
    )
    cuda_malloc_backend: bool = Field(
        False, description=("Enable the torch CUDA malloc backend (default: False).")
    )
    uvloop: bool = Field(
        False,
        description=(
            "Run asyncio using Uvloop or Winloop which can improve performance.\n"
            "NOTE: It's recommended to enable this, but if something breaks "
            "turn this off."
        ),
    )
    realtime_process_priority: bool = Field(
        False,
        description=(
            "Set process to use a higher priority.\n"
            "For realtime process priority, run as administrator or sudo.\n"
            "Otherwise, the priority will be set to high."
        ),
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
    embeddings: EmbeddingsConfig = Field(
        default_factory=EmbeddingsConfig.model_construct
    )
    sampling: SamplingConfig = Field(default_factory=SamplingConfig.model_construct)
    developer: DeveloperConfig = Field(default_factory=DeveloperConfig.model_construct)
    actions: UtilityActions = Field(default_factory=UtilityActions.model_construct)

    model_config = ConfigDict(validate_assignment=True, protected_namespaces=())
