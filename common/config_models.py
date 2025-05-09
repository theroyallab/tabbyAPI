from pydantic import (
    BaseModel,
    ConfigDict,
    constr,
    Field,
    PrivateAttr,
    field_validator,
)
from typing import List, Literal, Optional, Union


CACHE_SIZES = Literal["FP16", "Q8", "Q6", "Q4"]
CACHE_TYPE = Union[CACHE_SIZES, constr(pattern=r"^[2-8]\s*,\s*[2-8]$")]


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


class NetworkConfig(BaseConfigModel):
    """Options for networking"""

    host: Optional[str] = Field(
        "127.0.0.1",
        description=(
            "The IP to host on (default: 127.0.0.1).\n"
            "Use 0.0.0.0 to expose on all network adapters."
        ),
    )
    port: Optional[int] = Field(
        5000, description=("The port to host on (default: 5000).")
    )
    disable_auth: Optional[bool] = Field(
        False,
        description=(
            "Disable HTTP token authentication with requests.\n"
            "WARNING: This will make your instance vulnerable!\n"
            "Turn on this option if you are ONLY connecting from localhost."
        ),
    )
    disable_fetch_requests: Optional[bool] = Field(
        False,
        description=(
            "Disable fetching external content in response to requests,"
            "such as images from URLs."
        ),
    )
    send_tracebacks: Optional[bool] = Field(
        False,
        description=(
            "Send tracebacks over the API (default: False).\n"
            "NOTE: Only enable this for debug purposes."
        ),
    )
    api_servers: Optional[List[Literal["oai", "kobold"]]] = Field(
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

    log_prompt: Optional[bool] = Field(
        False,
        description=("Enable prompt logging (default: False)."),
    )
    log_generation_params: Optional[bool] = Field(
        False,
        description=("Enable generation parameter logging (default: False)."),
    )
    log_requests: Optional[bool] = Field(
        False,
        description=(
            "Enable request logging (default: False).\n"
            "NOTE: Only use this for debugging!"
        ),
    )


class ModelConfig(BaseConfigModel):
    """
    Options for model overrides and loading
    Please read the comments to understand how arguments are handled
    between initial and API loads
    """

    # TODO: convert this to a pathlib.path?
    model_dir: str = Field(
        "models",
        description=(
            "Directory to look for models (default: models).\n"
            "Windows users, do NOT put this path in quotes!"
        ),
    )
    inline_model_loading: Optional[bool] = Field(
        False,
        description=(
            "Allow direct loading of models "
            "from a completion or chat completion request (default: False).\n"
            "This method of loading is strict by default.\n"
            "Enable dummy models to add exceptions for invalid model names."
        ),
    )
    use_dummy_models: Optional[bool] = Field(
        False,
        description=(
            "Sends dummy model names when the models endpoint is queried. "
            "(default: False)\n"
            "Enable this if the client is looking for specific OAI models.\n"
        ),
    )
    dummy_model_names: List[str] = Field(
        default=["gpt-3.5-turbo"],
        description=(
            "A list of fake model names that are sent via the /v1/models endpoint. "
            '(default: ["gpt-3.5-turbo"])\n'
            "Also used as bypasses for strict mode if inline_model_loading is true."
        ),
    )
    model_name: Optional[str] = Field(
        None,
        description=(
            "An initial model to load.\n"
            "Make sure the model is located in the model directory!\n"
            "REQUIRED: This must be filled out to load a model on startup."
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
    backend: Optional[str] = Field(
        None,
        description=(
            "Backend to use for this model (auto-detect if not specified)\n"
            "Options: exllamav2, exllamav3"
        ),
    )
    max_seq_len: Optional[int] = Field(
        None,
        description=(
            "Max sequence length (default: Empty).\n"
            "Fetched from the model's base sequence length in config.json by default."
        ),
        ge=0,
    )
    tensor_parallel: Optional[bool] = Field(
        False,
        description=(
            "Load model with tensor parallelism.\n"
            "Falls back to autosplit if GPU split isn't provided.\n"
            "This ignores the gpu_split_auto value."
        ),
    )
    gpu_split_auto: Optional[bool] = Field(
        True,
        description=(
            "Automatically allocate resources to GPUs (default: True).\n"
            "Not parsed for single GPU users."
        ),
    )
    autosplit_reserve: List[float] = Field(
        [96],
        description=(
            "Reserve VRAM used for autosplit loading (default: 96 MB on GPU 0).\n"
            "Represented as an array of MB per GPU."
        ),
    )
    gpu_split: List[float] = Field(
        default_factory=list,
        description=(
            "An integer array of GBs of VRAM to split between GPUs (default: []).\n"
            "Used with tensor parallelism."
        ),
    )
    rope_scale: Optional[float] = Field(
        1.0,
        description=(
            "Rope scale (default: 1.0).\n"
            "Same as compress_pos_emb.\n"
            "Use if the model was trained on long context with rope.\n"
            "Leave blank to pull the value from the model."
        ),
    )
    rope_alpha: Optional[Union[float, Literal["auto"]]] = Field(
        None,
        description=(
            "Rope alpha (default: None).\n"
            'Same as alpha_value. Set to "auto" to auto-calculate.\n'
            "Leaving this value blank will either pull from the model "
            "or auto-calculate."
        ),
    )
    cache_mode: Optional[CACHE_TYPE] = Field(
        "FP16",
        description=(
            "Enable different cache modes for VRAM savings (default: FP16).\n"
            f"Possible values for exllamav2: {str(CACHE_SIZES)[15:-1]}.\n"
            "For exllamav3, specify the pair k_bits,v_bits where k_bits and v_bits "
            "are integers from 2-8 (i.e. 8,8)."
        ),
    )
    cache_size: Optional[int] = Field(
        None,
        description=(
            "Size of the prompt cache to allocate (default: max_seq_len).\n"
            "Must be a multiple of 256 and can't be less than max_seq_len.\n"
            "For CFG, set this to 2 * max_seq_len."
        ),
        multiple_of=256,
        gt=0,
    )
    chunk_size: Optional[int] = Field(
        2048,
        description=(
            "Chunk size for prompt ingestion (default: 2048).\n"
            "A lower value reduces VRAM usage but decreases ingestion speed.\n"
            "NOTE: Effects vary depending on the model.\n"
            "An ideal value is between 512 and 4096."
        ),
        gt=0,
    )
    max_batch_size: Optional[int] = Field(
        None,
        description=(
            "Set the maximum number of prompts to process at one time "
            "(default: None/Automatic).\n"
            "Automatically calculated if left blank.\n"
            "NOTE: Only available for Nvidia ampere (30 series) and above GPUs."
        ),
        ge=1,
    )
    prompt_template: Optional[str] = Field(
        None,
        description=(
            "Set the prompt template for this model. (default: None)\n"
            "If empty, attempts to look for the model's chat template.\n"
            "If a model contains multiple templates in its tokenizer_config.json,\n"
            "set prompt_template to the name of the template you want to use.\n"
            "NOTE: Only works with chat completion message lists!"
        ),
    )
    vision: Optional[bool] = Field(
        False,
        description=(
            "Enables vision support if the model supports it. (default: False)"
        ),
    )

    _metadata: Metadata = PrivateAttr(Metadata())
    model_config = ConfigDict(protected_namespaces=())


class DraftModelConfig(BaseConfigModel):
    """
    Options for draft models (speculative decoding)
    This will use more VRAM!
    """

    # TODO: convert this to a pathlib.path?
    draft_model_dir: Optional[str] = Field(
        "models",
        description=("Directory to look for draft models (default: models)"),
    )
    draft_model_name: Optional[str] = Field(
        None,
        description=(
            "An initial draft model to load.\n"
            "Ensure the model is in the model directory."
        ),
    )
    draft_rope_scale: Optional[float] = Field(
        1.0,
        description=(
            "Rope scale for draft models (default: 1.0).\n"
            "Same as compress_pos_emb.\n"
            "Use if the draft model was trained on long context with rope."
        ),
    )
    draft_rope_alpha: Optional[float] = Field(
        None,
        description=(
            "Rope alpha for draft models (default: None).\n"
            'Same as alpha_value. Set to "auto" to auto-calculate.\n'
            "Leaving this value blank will either pull from the model "
            "or auto-calculate."
        ),
    )
    draft_cache_mode: Optional[CACHE_SIZES] = Field(
        "FP16",
        description=(
            "Cache mode for draft models to save VRAM (default: FP16).\n"
            f"Possible values: {str(CACHE_SIZES)[15:-1]}."
        ),
    )
    draft_gpu_split: List[float] = Field(
        default_factory=list,
        description=(
            "An integer array of GBs of VRAM to split between GPUs (default: []).\n"
            "If this isn't filled in, the draft model is autosplit."
        ),
    )


class LoraInstanceModel(BaseConfigModel):
    """Model representing an instance of a Lora."""

    name: Optional[str] = None
    scaling: float = Field(1.0, ge=0)


class LoraConfig(BaseConfigModel):
    """Options for Loras"""

    # TODO: convert this to a pathlib.path?
    lora_dir: Optional[str] = Field(
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

    # TODO: convert this to a pathlib.path?
    embedding_model_dir: Optional[str] = Field(
        "models",
        description=("Directory to look for embedding models (default: models)."),
    )
    embeddings_device: Optional[Literal["cpu", "auto", "cuda"]] = Field(
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

    unsafe_launch: Optional[bool] = Field(
        False,
        description=(
            "Skip Exllamav2 version check (default: False).\n"
            "WARNING: It's highly recommended to update your dependencies rather "
            "than enabling this flag."
        ),
    )
    disable_request_streaming: Optional[bool] = Field(
        False, description=("Disable API request streaming (default: False).")
    )
    cuda_malloc_backend: Optional[bool] = Field(
        False, description=("Enable the torch CUDA malloc backend (default: False).")
    )
    realtime_process_priority: Optional[bool] = Field(
        False,
        description=(
            "Set process to use a higher priority.\n"
            "For realtime process priority, run as administrator or sudo.\n"
            "Otherwise, the priority will be set to high."
        ),
    )


class TabbyConfigModel(BaseModel):
    """Base model for a TabbyConfig."""

    config: Optional[ConfigOverrideConfig] = Field(
        default_factory=ConfigOverrideConfig.model_construct
    )
    network: Optional[NetworkConfig] = Field(
        default_factory=NetworkConfig.model_construct
    )
    logging: Optional[LoggingConfig] = Field(
        default_factory=LoggingConfig.model_construct
    )
    model: Optional[ModelConfig] = Field(default_factory=ModelConfig.model_construct)
    draft_model: Optional[DraftModelConfig] = Field(
        default_factory=DraftModelConfig.model_construct
    )
    lora: Optional[LoraConfig] = Field(default_factory=LoraConfig.model_construct)
    embeddings: Optional[EmbeddingsConfig] = Field(
        default_factory=EmbeddingsConfig.model_construct
    )
    sampling: Optional[SamplingConfig] = Field(
        default_factory=SamplingConfig.model_construct
    )
    developer: Optional[DeveloperConfig] = Field(
        default_factory=DeveloperConfig.model_construct
    )

    model_config = ConfigDict(validate_assignment=True, protected_namespaces=())
