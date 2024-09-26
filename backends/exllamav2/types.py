from typing import List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

CACHE_SIZES = Literal["FP16", "Q8", "Q6", "Q4"]


class DraftModelInstanceConfig(BaseModel):
    draft_model_name: Optional[str] = Field(
        None,
        description=(
            "An initial draft model to load.\n"
            "Ensure the model is in the model directory."
        ),
    )
    draft_rope_scale: float = Field(
        1.0,
        description=(
            "Rope scale for draft models (default: 1.0).\n"
            "Same as compress_pos_emb.\n"
            "Use if the draft model was trained on long context with rope."
        ),
    )
    draft_rope_alpha: Optional[Union[float, Literal["auto"]]] = Field(
        None,
        description=(
            "Rope alpha for draft models (default: None).\n"
            'Same as alpha_value. Set to "auto" to auto-calculate.\n'
            "Leaving this value blank will either pull from the model "
            "or auto-calculate."
        ),
        examples=[1.0],
    )
    draft_cache_mode: CACHE_SIZES = Field(
        "FP16",
        description=(
            "Cache mode for draft models to save VRAM (default: FP16).\n"
            f"Possible values: {str(CACHE_SIZES)[15:-1]}."
        ),
    )


class ModelInstanceConfig(BaseModel):
    """
    Options for model overrides and loading
    Please read the comments to understand how arguments are handled
    between initial and API loads
    """

    model_name: Optional[str] = Field(
        None,
        description=(
            "An initial model to load.\n"
            "Make sure the model is located in the model directory!\n"
            "REQUIRED: This must be filled out to load a model on startup."
        ),
    )
    max_seq_len: Optional[int] = Field(
        None,
        description=(
            "Max sequence length (default: Empty).\n"
            "Fetched from the model's base sequence length in config.json by default."
        ),
        ge=0,
        examples=[16384, 4096, 2048],
    )
    override_base_seq_len: Optional[int] = Field(
        None,
        description=(
            "Overrides base model context length (default: Empty).\n"
            "WARNING: Don't set this unless you know what you're doing!\n"
            "Again, do NOT use this for configuring context length, "
            "use max_seq_len above ^"
        ),
        ge=0,
        examples=[4096],
    )
    tensor_parallel: bool = Field(
        False,
        description=(
            "Load model with tensor parallelism.\n"
            "Falls back to autosplit if GPU split isn't provided.\n"
            "This ignores the gpu_split_auto value."
        ),
    )
    gpu_split_auto: bool = Field(
        True,
        description=(
            "Automatically allocate resources to GPUs (default: True).\n"
            "Not parsed for single GPU users."
        ),
    )
    autosplit_reserve: List[int] = Field(
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
    rope_scale: float = Field(
        1.0,
        description=(
            "Rope scale (default: 1.0).\n"
            "Same as compress_pos_emb.\n"
            "Use if the model was trained on long context with rope.\n"
            "Leave blank to pull the value from the model."
        ),
        examples=[1.0],
    )
    rope_alpha: Optional[Union[float, Literal["auto"]]] = Field(
        "auto",
        description=(
            "Rope alpha (default: None).\n"
            'Same as alpha_value. Set to "auto" to auto-calculate.\n'
            "Leaving this value blank will either pull from the model "
            "or auto-calculate."
        ),
        examples=["auto", 1.0],
    )
    cache_mode: CACHE_SIZES = Field(
        "FP16",
        description=(
            "Enable different cache modes for VRAM savings (default: FP16).\n"
            f"Possible values: {str(CACHE_SIZES)[15:-1]}."
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
        examples=[4096],
    )
    chunk_size: int = Field(
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
    num_experts_per_token: Optional[int] = Field(
        None,
        description=(
            "Number of experts to use per token.\n"
            "Fetched from the model's config.json if empty.\n"
            "NOTE: For MoE models only.\n"
            "WARNING: Don't set this unless you know what you're doing!"
        ),
        ge=1,
    )

    model_config = ConfigDict(protected_namespaces=())
