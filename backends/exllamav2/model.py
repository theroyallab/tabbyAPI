"""The model container class for ExLlamaV2 models."""

import gc
import math
import pathlib
import threading
import time
import traceback

import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
from itertools import zip_longest
from loguru import logger
from typing import List, Optional, Union

from backends.exllamav2.grammar import ExLlamaV2Grammar
from common.concurrency import iterate_in_threadpool
from common.gen_logging import (
    log_generation_params,
    log_metrics,
    log_prompt,
    log_response,
)
from common.templating import (
    PromptTemplate,
    TemplateLoadError,
    find_template_from_model,
)
from common.transformers_utils import GenerationConfig
from common.utils import coalesce, unwrap


class ExllamaV2Container:
    """The model container class for ExLlamaV2 models."""

    # Exl2 vars
    config: Optional[ExLlamaV2Config] = None
    draft_config: Optional[ExLlamaV2Config] = None
    model: Optional[ExLlamaV2] = None
    draft_model: Optional[ExLlamaV2] = None
    cache: Optional[ExLlamaV2Cache] = None
    draft_cache: Optional[ExLlamaV2Cache] = None
    tokenizer: Optional[ExLlamaV2Tokenizer] = None
    generator: Optional[ExLlamaV2StreamingGenerator] = None
    prompt_template: Optional[PromptTemplate] = None
    active_loras: List[ExLlamaV2Lora] = []

    # Internal config vars
    cache_mode: str = "FP16"
    use_cfg: bool = False
    generation_config: Optional[GenerationConfig] = None

    # GPU split vars
    gpu_split: Optional[list] = None
    gpu_split_auto: bool = True
    autosplit_reserve: List[float] = [96 * 1024**2]

    # Load state
    model_is_loading: bool = False
    model_loaded: bool = False

    def __init__(self, model_directory: pathlib.Path, quiet=False, **kwargs):
        """
        Create model container

        Args:
            model_dir (int): Model directory containing config.json,
                tokenizer.model etc.
            quiet (bool): Suppress console output
            load_progress_callback (function, optional): A function to call for
                each module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int,
                             loading_draft: bool)
            **kwargs:
                `cache_mode` (str): Sets cache mode, "FP16" or "FP8"
                    (defaulf: "FP16")
                'max_seq_len' (int): Override model's default max sequence
                    length (default: 4096)
                'rope_scale' (float): Set RoPE scaling factor for model
                    (default: 1.0)
                'rope_alpha' (float): Set RoPE alpha (NTK) factor for model
                    (default: 1.0)
                'prompt_template' (str): Manually sets the prompt template for
                    this model (default: None)
                'chunk_size' (int): Sets the maximum chunk size for the model
                    (default: 2048)
                    Inferencing in chunks reduces overall VRAM overhead by
                    processing very long sequences in smaller batches. This
                    limits the size of temporary buffers needed for the hidden
                    state and attention weights.
                'draft_model_dir' (str): Draft model directory
                'draft_rope_scale' (float): Set RoPE scaling factor for draft
                    model (default: 1.0)
                'draft_rope_alpha' (float): RoPE alpha (NTK) factor for draft
                    model. By default, the draft model's alpha value is
                    calculated automatically to scale to the size of the
                    full model.
                'lora_dir' (str): LoRA directory
                'loras' (list[dict]): List of loras to be loaded, consisting of
                    'name' and 'scaling'
                'gpu_split_auto' (bool): Automatically split model across
                    available devices (default: True)
                'gpu_split' (list[float]): Allocation for weights and (some)
                    tensors, per device
                'no_flash_attn' (bool): Turns off flash attention
                    (increases vram usage) (default: False)
                'use_cfg" (bool): Enables CFG support. Disables flash attention
                    (default: False)
        """

        self.quiet = quiet
        self.cache_mode = unwrap(kwargs.get("cache_mode"), "FP16")

        # Turn off GPU split if the user is using 1 GPU
        gpu_count = torch.cuda.device_count()
        gpu_split_auto = unwrap(kwargs.get("gpu_split_auto"), True)

        if gpu_count > 1 and gpu_split_auto:
            # Auto GPU split parameters
            self.gpu_split_auto = gpu_split_auto

            autosplit_reserve_megabytes = unwrap(kwargs.get("autosplit_reserve"), [96])
            self.autosplit_reserve = list(
                map(
                    lambda value: int(math.ceil(value * 1024**2)),
                    autosplit_reserve_megabytes,
                )
            )
        elif gpu_count > 1:
            # Manual GPU split
            self.gpu_split = kwargs.get("gpu_split")
            self.gpu_split_auto = False
        else:
            # One GPU setup
            self.gpu_split_auto = False
            logger.info("Disabling GPU split because one GPU is in use.")

        self.config = ExLlamaV2Config()
        self.config.model_dir = str(model_directory.resolve())

        # Make the max seq len 4096 before preparing the config
        # This is a better default than 2048
        self.config.max_seq_len = 4096

        # Hardcode max output length to 16
        self.config.max_output_len = 16

        self.config.prepare()

        # Then override the base_seq_len if present
        override_base_seq_len = kwargs.get("override_base_seq_len")
        if override_base_seq_len:
            self.config.max_seq_len = override_base_seq_len

        # Grab the base model's sequence length before overrides for
        # rope calculations
        base_seq_len = self.config.max_seq_len

        # Set the target seq len if present
        target_max_seq_len = kwargs.get("max_seq_len")
        if target_max_seq_len:
            self.config.max_seq_len = target_max_seq_len

        # Set the rope scale
        self.config.scale_pos_emb = unwrap(
            kwargs.get("rope_scale"), self.config.scale_pos_emb
        )

        # Automatically calculate rope alpha
        self.config.scale_alpha_value = unwrap(
            kwargs.get("rope_alpha"), self.calculate_rope_alpha(base_seq_len)
        )

        # Enable CFG if present
        self.use_cfg = unwrap(kwargs.get("use_cfg"), False)

        # Enable fasttensors loading if present
        self.config.fasttensors = unwrap(kwargs.get("fasttensors"), False)

        # Turn off flash attention if CFG is on
        # Workaround until batched FA2 is fixed in exllamav2 upstream
        self.config.no_flash_attn = (
            True if self.use_cfg else unwrap(kwargs.get("no_flash_attention"), False)
        )

        # Try to set prompt template
        self.prompt_template = self.find_prompt_template(
            kwargs.get("prompt_template"), model_directory
        )

        # Load generation config overrides
        generation_config_path = (
            pathlib.Path(self.config.model_dir) / "generation_config.json"
        )
        if generation_config_path.exists():
            try:
                self.generation_config = GenerationConfig.from_file(
                    generation_config_path.parent
                )
            except Exception:
                logger.error(traceback.format_exc())
                logger.warning(
                    "Skipping generation config load because of an unexpected error."
                )

        # Catch all for template lookup errors
        if self.prompt_template:
            logger.info(
                f'Using template "{self.prompt_template.name}" for chat completions.'
            )
        else:
            logger.warning(
                "Chat completions are disabled because a prompt "
                "template wasn't provided or auto-detected."
            )

        # Set num of experts per token if provided
        num_experts_override = kwargs.get("num_experts_per_token")
        if num_experts_override:
            self.config.num_experts_per_token = kwargs.get("num_experts_per_token")

        # Make sure chunk size is >= 16 and <= max seq length
        user_chunk_size = unwrap(kwargs.get("chunk_size"), 2048)
        chunk_size = sorted((16, user_chunk_size, self.config.max_seq_len))[1]
        self.config.max_input_len = chunk_size
        self.config.max_attention_size = chunk_size**2

        draft_args = unwrap(kwargs.get("draft"), {})
        draft_model_name = draft_args.get("draft_model_name")
        enable_draft = draft_args and draft_model_name

        # Always disable draft if params are incorrectly configured
        if draft_args and draft_model_name is None:
            logger.warning(
                "Draft model is disabled because a model name "
                "wasn't provided. Please check your config.yml!"
            )
            enable_draft = False

        if enable_draft:
            self.draft_config = ExLlamaV2Config()
            draft_model_path = pathlib.Path(
                unwrap(draft_args.get("draft_model_dir"), "models")
            )
            draft_model_path = draft_model_path / draft_model_name

            self.draft_config.model_dir = str(draft_model_path.resolve())
            self.draft_config.prepare()

            self.draft_config.scale_pos_emb = unwrap(
                draft_args.get("draft_rope_scale"), 1.0
            )

            # Automatically calculate draft rope alpha
            self.draft_config.scale_alpha_value = unwrap(
                draft_args.get("draft_rope_alpha"),
                self.calculate_rope_alpha(self.draft_config.max_seq_len),
            )
            self.draft_config.max_seq_len = self.config.max_seq_len

            if chunk_size:
                self.draft_config.max_input_len = chunk_size
                self.draft_config.max_attention_size = chunk_size**2

    def find_prompt_template(self, prompt_template_name, model_directory):
        """Tries to find a prompt template using various methods"""

        logger.info("Attempting to load a prompt template if present.")

        find_template_functions = [
            lambda: PromptTemplate.from_model_json(
                pathlib.Path(self.config.model_dir) / "tokenizer_config.json",
                "chat_template",
            ),
            lambda: PromptTemplate.from_file(find_template_from_model(model_directory)),
        ]

        # Add lookup from prompt template name if provided
        if prompt_template_name:
            find_template_functions[:0] = [
                lambda: PromptTemplate.from_file(prompt_template_name),
                lambda: PromptTemplate.from_model_json(
                    pathlib.Path(self.config.model_dir) / "tokenizer_config.json",
                    "chat_template",
                    prompt_template_name,
                ),
            ]

        # Continue on exception since functions are tried as they fail
        for template_func in find_template_functions:
            try:
                prompt_template = template_func()
                if prompt_template is not None:
                    return prompt_template
            except TemplateLoadError as e:
                logger.warning(f"TemplateLoadError: {str(e)}")
                continue
            except Exception:
                logger.error(traceback.format_exc())
                logger.warning(
                    "An unexpected error happened when trying to load the template. "
                    "Trying other methods."
                )
                continue

    def calculate_rope_alpha(self, base_seq_len):
        """Calculate the rope alpha value for a given sequence length."""
        ratio = self.config.max_seq_len / base_seq_len

        # Default to a 1 alpha if the sequence length is ever less
        # than or equal to 1
        if ratio <= 1.0:
            alpha = 1
        else:
            alpha = -0.13436 + 0.80541 * ratio + 0.28833 * ratio**2
        return alpha

    def get_model_path(self, is_draft: bool = False):
        """Get the path for this model."""

        model_path = pathlib.Path(
            self.draft_config.model_dir if is_draft else self.config.model_dir
        )
        return model_path

    def get_model_parameters(self):
        model_params = {
            "name": self.get_model_path().name,
            "rope_scale": self.config.scale_pos_emb,
            "rope_alpha": self.config.scale_alpha_value,
            "max_seq_len": self.config.max_seq_len,
            "cache_mode": self.cache_mode,
            "chunk_size": self.config.max_input_len,
            "num_experts_per_token": self.config.num_experts_per_token,
            "use_cfg": self.use_cfg,
            "prompt_template": self.prompt_template.name
            if self.prompt_template
            else None,
        }

        if self.draft_config:
            draft_model_params = {
                "name": self.get_model_path(is_draft=True).name,
                "rope_scale": self.draft_config.scale_pos_emb,
                "rope_alpha": self.draft_config.scale_alpha_value,
                "max_seq_len": self.draft_config.max_seq_len,
            }

            model_params["draft"] = draft_model_params

        return model_params

    async def load(self, progress_callback=None):
        """
        Load model

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int)
        """

        async for _ in self.load_gen(progress_callback):
            pass

    async def load_loras(self, lora_directory: pathlib.Path, **kwargs):
        """
        Load loras
        """

        loras = unwrap(kwargs.get("loras"), [])
        success: List[str] = []
        failure: List[str] = []

        for lora in loras:
            lora_name = lora.get("name")
            lora_scaling = unwrap(lora.get("scaling"), 1.0)

            if lora_name is None:
                logger.warning(
                    "One of your loras does not have a name. Please check your "
                    "config.yml! Skipping lora load."
                )
                failure.append(lora_name)
                continue

            logger.info(f"Loading lora: {lora_name} at scaling {lora_scaling}")
            lora_path = lora_directory / lora_name

            self.active_loras.append(
                ExLlamaV2Lora.from_directory(self.model, lora_path, lora_scaling)
            )
            logger.info(f"Lora successfully loaded: {lora_name}")
            success.append(lora_name)

        # Return success and failure names
        return {"success": success, "failure": failure}

    async def load_gen(self, progress_callback=None):
        """Basic async wrapper around the loading generator"""

        load_generator = self.load_gen_sync(progress_callback)
        async for value in iterate_in_threadpool(load_generator):
            yield value

    @torch.inference_mode()
    def load_gen_sync(self, progress_callback=None):
        """
        Load model, generator function

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int)

        Runs under a shared inference mode context.
        """

        # Notify that the model is being loaded
        self.model_is_loading = True

        # Reset tokenizer namespace vars and create a tokenizer
        ExLlamaV2Tokenizer.unspecial_piece_to_id = {}
        ExLlamaV2Tokenizer.unspecial_id_to_piece = {}
        ExLlamaV2Tokenizer.extended_id_to_piece = {}
        ExLlamaV2Tokenizer.extended_piece_to_id = {}

        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        # Calculate autosplit reserve for all GPUs
        gpu_count = torch.cuda.device_count()
        autosplit_reserve = self.autosplit_reserve + [0] * (
            gpu_count - len(self.autosplit_reserve)
        )

        # Load draft model if a config is present
        if self.draft_config:
            self.draft_model = ExLlamaV2(self.draft_config)
            if not self.quiet:
                logger.info("Loading draft model: " + self.draft_config.model_dir)

            self.draft_cache = ExLlamaV2Cache(self.draft_model, lazy=True)
            for value in self.draft_model.load_autosplit_gen(
                self.draft_cache,
                reserve_vram=autosplit_reserve,
                last_id_only=True,
                callback_gen=progress_callback,
            ):
                if value:
                    yield value

            # Test VRAM allocation with a full-length forward pass
            input_ids = torch.zeros((1, self.config.max_input_len), dtype=torch.long)
            self.draft_model.forward(input_ids, cache=self.cache, preprocess_only=True)

        self.model = ExLlamaV2(self.config)
        if not self.quiet:
            logger.info("Loading model: " + self.config.model_dir)

        # Load model with manual split
        # Entrypoint for single GPU users
        if not self.gpu_split_auto:
            logger.info("Loading with a manual GPU split (or a one GPU setup)")

            for value in self.model.load_gen(
                self.gpu_split,
                callback_gen=progress_callback,
            ):
                if value:
                    yield value

        batch_size = 2 if self.use_cfg else 1

        if self.cache_mode == "Q4":
            self.cache = ExLlamaV2Cache_Q4(
                self.model, lazy=self.gpu_split_auto, batch_size=batch_size
            )
        elif self.cache_mode == "FP8":
            self.cache = ExLlamaV2Cache_8bit(
                self.model, lazy=self.gpu_split_auto, batch_size=batch_size
            )
        else:
            self.cache = ExLlamaV2Cache(
                self.model, lazy=self.gpu_split_auto, batch_size=batch_size
            )

        # Load model with autosplit
        if self.gpu_split_auto:
            logger.info("Loading with autosplit")

            for value in self.model.load_autosplit_gen(
                self.cache,
                reserve_vram=autosplit_reserve,
                last_id_only=True,
                callback_gen=progress_callback,
            ):
                if value:
                    yield value

        # Test VRAM allocation with a full-length forward pass
        input_ids = torch.zeros((1, self.config.max_input_len), dtype=torch.long)
        self.model.forward(input_ids, cache=self.cache, preprocess_only=True)

        # Create generator
        self.generator = ExLlamaV2StreamingGenerator(
            self.model,
            self.cache,
            self.tokenizer,
            self.draft_model,
            self.draft_cache,
        )

        # Clean up any extra vram usage from torch and cuda
        # (Helps reduce VRAM bottlenecking on Windows)
        gc.collect()
        torch.cuda.empty_cache()

        # Update model load state
        self.model_is_loading = False
        self.model_loaded = True
        logger.info("Model successfully loaded.")

    def unload(self, loras_only: bool = False):
        """
        Free all VRAM resources used by this model
        """

        for lora in self.active_loras:
            lora.unload()

        self.active_loras = []

        # Unload the entire model if not just unloading loras
        if not loras_only:
            if self.model:
                self.model.unload()
            self.model = None

            if self.draft_model:
                self.draft_model.unload()
            self.draft_model = None

            self.config = None
            self.cache = None
            self.tokenizer = None
            self.generator = None

            # Set all model state variables to False
            self.model_is_loading = False
            self.model_loaded = False

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Loras unloaded." if loras_only else "Model unloaded.")

    def encode_tokens(self, text: str, **kwargs):
        """Wrapper to encode tokens from a text string"""

        return (
            self.tokenizer.encode(
                text,
                add_bos=unwrap(kwargs.get("add_bos_token"), True),
                encode_special_tokens=unwrap(kwargs.get("encode_special_tokens"), True),
            )
            .flatten()
            .tolist()
        )

    def decode_tokens(self, ids: List[int], **kwargs):
        """Wrapper to decode tokens from a list of IDs"""

        ids = torch.tensor([ids])
        return self.tokenizer.decode(
            ids,
            decode_special_tokens=unwrap(kwargs.get("decode_special_tokens"), True),
        )[0]

    # TODO: Maybe support generation_config for eos_token
    def get_special_tokens(
        self, add_bos_token: bool = True, ban_eos_token: bool = False
    ):
        return {
            "bos_token": self.tokenizer.bos_token if add_bos_token else "",
            "eos_token": self.tokenizer.eos_token if not ban_eos_token else "",
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
        }

    def get_logprobs(self, token_ids: torch.Tensor, token_probs: torch.Tensor):
        top_tokens = list(
            map(
                lambda index: self.tokenizer.extended_id_to_piece.get(
                    index, self.tokenizer.id_to_piece[index]
                ),
                token_ids.flatten().tolist(),
            )
        )

        top_values = torch.log(token_probs).flatten().tolist()

        # Cannot return -inf in JSON
        cleaned_values = list(
            map(lambda value: -1000 if value == float("-inf") else value, top_values)
        )

        return dict(zip_longest(top_tokens, cleaned_values))

    async def generate(self, prompt: str, **kwargs):
        """Generate a response to a prompt"""
        generations = []
        async for generation in self.generate_gen(prompt, **kwargs):
            generations.append(generation)

        joined_generation = {
            "text": "",
            "prompt_tokens": 0,
            "generation_tokens": 0,
            "offset": [],
            "token_probs": {},
            "logprobs": [],
        }

        if generations:
            # Get finish_reason first and then shift where -1 points to
            if "finish_reason" in generations[-1]:
                finish_reason_gen = generations.pop()
                joined_generation["finish_reason"] = finish_reason_gen.get(
                    "finish_reason"
                )
            else:
                joined_generation["finish_reason"] = "stop"

        if len(generations) > 0:
            for generation in generations:
                joined_generation["text"] += unwrap(generation.get("text"), "")
                joined_generation["offset"].append(unwrap(generation.get("offset"), -1))
                joined_generation["token_probs"].update(
                    unwrap(generation.get("token_probs"), {})
                )

                # Include empty logprob dicts for index preservation
                joined_generation["logprobs"].append(
                    unwrap(generation.get("logprobs"), {})
                )

            joined_generation["prompt_tokens"] = unwrap(
                generations[-1].get("prompt_tokens"), 0
            )
            joined_generation["generation_tokens"] = unwrap(
                generations[-1].get("generated_tokens"), 0
            )

        return joined_generation

    def check_unsupported_settings(self, **kwargs):
        """Check and warn the user if a sampler is unsupported. Meant for dev wheels!"""

        return kwargs

    async def generate_gen(
        self, prompt: str, abort_event: Optional[threading.Event] = None, **kwargs
    ):
        """Basic async wrapper for completion generator"""

        sync_generator = self.generate_gen_sync(prompt, abort_event, **kwargs)
        async for value in iterate_in_threadpool(sync_generator):
            yield value

    @torch.inference_mode()
    def generate_gen_sync(
        self, prompt: str, abort_event: Optional[threading.Event] = None, **kwargs
    ):
        """
        Create generator function for prompt completion.

        for kwargs, check common/sampling.py
        """

        token_healing = unwrap(kwargs.get("token_healing"), False)
        stream_interval = unwrap(kwargs.get("stream_interval"), 0)
        generate_window = max(
            unwrap(kwargs.get("generate_window"), 512), self.config.max_seq_len // 8
        )

        # Sampler settings
        gen_settings = ExLlamaV2Sampler.Settings()

        # Check unsupported settings for dev wheels
        kwargs = self.check_unsupported_settings(**kwargs)

        # Apply settings
        gen_settings.temperature = unwrap(kwargs.get("temperature"), 1.0)
        gen_settings.temperature_last = unwrap(kwargs.get("temperature_last"), False)
        gen_settings.smoothing_factor = unwrap(kwargs.get("smoothing_factor"), 0.0)
        gen_settings.top_k = unwrap(kwargs.get("top_k"), 0)
        gen_settings.top_p = unwrap(kwargs.get("top_p"), 1.0)
        gen_settings.top_a = unwrap(kwargs.get("top_a"), 0.0)
        gen_settings.min_p = unwrap(kwargs.get("min_p"), 0.0)
        gen_settings.tfs = unwrap(kwargs.get("tfs"), 1.0)
        gen_settings.typical = unwrap(kwargs.get("typical"), 1.0)
        gen_settings.mirostat = unwrap(kwargs.get("mirostat"), False)

        # DynaTemp settings
        max_temp = unwrap(kwargs.get("max_temp"), 1.0)
        min_temp = unwrap(kwargs.get("min_temp"), 1.0)

        if max_temp > min_temp:
            gen_settings.max_temp = max_temp
            gen_settings.min_temp = min_temp
            gen_settings.temp_exponent = unwrap(kwargs.get("temp_exponent"), 1.0)
        else:
            # Force to default values
            gen_settings.max_temp = 1.0
            gen_settings.min_temp = 1.0
            gen_settings.temp_exponent = 1.0

        # Warn if max/min temp values are > 0
        # and if they're less than or equal to each other
        if max_temp < min_temp or (
            1 not in {min_temp, max_temp} and max_temp == min_temp
        ):
            logger.warning(
                "Max temp is less than or equal to min temp, skipping DynaTemp."
            )

        # Default tau and eta fallbacks don't matter if mirostat is off
        gen_settings.mirostat_tau = unwrap(kwargs.get("mirostat_tau"), 1.5)
        gen_settings.mirostat_eta = unwrap(kwargs.get("mirostat_eta"), 0.1)

        # Set CFG scale and negative prompt
        cfg_scale = unwrap(kwargs.get("cfg_scale"), 1.0)
        negative_prompt = None
        if cfg_scale not in [None, 1.0]:
            if self.use_cfg:
                gen_settings.cfg_scale = cfg_scale

                # If the negative prompt is empty, use the BOS token
                negative_prompt = unwrap(
                    kwargs.get("negative_prompt"), self.tokenizer.bos_token
                )
            else:
                logger.warning(
                    "CFG is currently disabled. "
                    "Please reload your model with use_cfg = True.",
                )

        gen_settings.token_repetition_penalty = unwrap(
            kwargs.get("repetition_penalty"), 1.0
        )
        gen_settings.token_frequency_penalty = unwrap(
            kwargs.get("frequency_penalty"), 0.0
        )
        gen_settings.token_presence_penalty = unwrap(
            kwargs.get("presence_penalty"), 0.0
        )

        # Applies for all penalties despite being called token_repetition_range
        gen_settings.token_repetition_range = unwrap(
            kwargs.get("penalty_range"), self.config.max_seq_len
        )

        # Dynamically scale penalty range to output tokens
        # Only do this if freq/pres pen is enabled
        # and the repetition range is -1
        auto_scale_penalty_range = (
            gen_settings.token_frequency_penalty != 0
            or gen_settings.token_presence_penalty != 0
        ) and gen_settings.token_repetition_range == -1

        # Always make sure the fallback is 0 if range < 0
        # It's technically fine to use -1, but this just validates the passed
        # fallback
        # Always default to 0 if something goes wrong
        if gen_settings.token_repetition_range < 0:
            fallback_decay = 0
        else:
            fallback_decay = gen_settings.token_repetition_range
        gen_settings.token_repetition_decay = coalesce(
            kwargs.get("repetition_decay"), fallback_decay, 0
        )

        stop_conditions: List[Union[str, int]] = unwrap(kwargs.get("stop"), [])
        banned_strings: List[str] = unwrap(kwargs.get("banned_strings"), [])
        add_bos_token = unwrap(kwargs.get("add_bos_token"), True)
        ban_eos_token = unwrap(kwargs.get("ban_eos_token"), False)
        logit_bias = kwargs.get("logit_bias")

        # Logprobs
        request_logprobs = unwrap(kwargs.get("logprobs"), 0)

        # Speculative Ngram
        self.generator.speculative_ngram = unwrap(
            kwargs.get("speculative_ngram"), False
        )

        # Override sampler settings for temp = 0
        if gen_settings.temperature == 0:
            gen_settings.temperature = 1.0
            gen_settings.top_k = 1
            gen_settings.top_p = 0
            gen_settings.typical = 0

        # Store the gen settings for logging purposes
        gen_settings_log_dict = vars(gen_settings)

        # Set banned tokens
        banned_tokens = unwrap(kwargs.get("banned_tokens"), [])
        if banned_tokens:
            gen_settings.disallow_tokens(self.tokenizer, banned_tokens)

        # Set logit bias
        if logit_bias:
            # Create a vocab tensor if it doesn't exist for token biasing
            if gen_settings.token_bias is None:
                padding = -self.tokenizer.config.vocab_size % 32
                gen_settings.token_bias = torch.zeros(
                    (self.tokenizer.config.vocab_size + padding,),
                    dtype=torch.float,
                )

            # Map logits to the tensor with their biases
            for token_id, bias in logit_bias.items():
                if 0 <= token_id < len(self.tokenizer.id_to_piece):
                    gen_settings.token_bias[token_id] = bias
                else:
                    logger.warning(
                        f"Logit bias: Token {token_id} not present "
                        "in the model's vocab. Skipping."
                    )

        # Initialize grammar handler
        grammar_handler = ExLlamaV2Grammar()
        gen_settings.filters = []

        # Add JSON schema filter if it exists
        json_schema = unwrap(kwargs.get("json_schema"))
        if json_schema:
            grammar_handler.add_json_schema_filter(
                json_schema, gen_settings, self.model, self.tokenizer
            )

        # Add EBNF filter if it exists
        grammar_string = unwrap(kwargs.get("grammar_string"))
        if grammar_string:
            grammar_handler.add_ebnf_filter(
                grammar_string, gen_settings, self.model, self.tokenizer
            )

        # Fetch EOS tokens from generation_config if they exist
        eos_tokens = (
            self.generation_config.eos_tokens()
            if self.generation_config
            else [self.tokenizer.eos_token_id]
        )

        # Ban the EOS token if specified. If not, append to stop conditions
        # as well.
        # Set this below logging to avoid polluting the stop strings array
        if ban_eos_token:
            gen_settings.disallow_tokens(self.tokenizer, eos_tokens)
        else:
            stop_conditions += eos_tokens

        # Stop conditions
        self.generator.set_stop_conditions(stop_conditions)

        # Tokenized context
        ids, offsets = self.tokenizer.encode(
            [prompt, negative_prompt]
            if negative_prompt and gen_settings.cfg_scale not in [None, 1.0]
            else prompt,
            add_bos=add_bos_token,
            encode_special_tokens=True,
            return_offsets=True,
        )
        mask = (
            self.tokenizer.padding_mask(ids)
            if self.use_cfg and gen_settings.cfg_scale not in [None, 1.0]
            else None
        )
        context_len = len(ids[0])

        if context_len > self.config.max_seq_len:
            logger.warning(
                f"Context length {context_len} is greater than max_seq_len "
                f"{self.config.max_seq_len}. Generation is truncated and "
                "metrics may not be accurate."
            )

        prompt_tokens = ids.shape[-1]

        # Automatically set max_tokens to fill up the context
        # This should be an OK default, but may be changed in the future
        max_tokens = unwrap(
            kwargs.get("max_tokens"), self.config.max_seq_len - prompt_tokens
        )

        # Set min_tokens to generate while keeping EOS banned
        min_tokens = unwrap(kwargs.get("min_tokens"), 0)

        # This is an inverse of skip_special_tokens
        decode_special_tokens = unwrap(not kwargs.get("skip_special_tokens"), False)

        begin_stream_args = {
            "token_healing": token_healing,
            "loras": self.active_loras,
            "return_probabilities": request_logprobs > 0,
            "return_top_tokens": request_logprobs,
            "return_logits": request_logprobs > 0,
            "abort_event": abort_event,
        }

        if self.use_cfg:
            begin_stream_args.update(
                {
                    "input_mask": mask,
                    "position_offsets": offsets,
                }
            )

        # Log generation options to console
        # Some options are too large, so log the args instead
        log_generation_params(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            stream=kwargs.get("stream"),
            **gen_settings_log_dict,
            token_healing=token_healing,
            auto_scale_penalty_range=auto_scale_penalty_range,
            generate_window=generate_window,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=eos_tokens,
            add_bos_token=add_bos_token,
            ban_eos_token=ban_eos_token,
            skip_special_tokens=not decode_special_tokens,
            speculative_ngram=self.generator.speculative_ngram,
            logprobs=request_logprobs,
            stop_conditions=stop_conditions,
            banned_tokens=banned_tokens,
            banned_strings=banned_strings,
            logit_bias=logit_bias,
        )

        # Log prompt to console
        log_prompt(prompt, negative_prompt)

        # Begin
        generated_tokens = 0
        full_response = ""
        start_time = time.time()
        last_chunk_time = start_time

        save_tokens = torch.empty((ids.shape[0], 0), dtype=torch.bool)
        chunk_buffer = ""
        chunk_tokens = 0

        while True:
            # Ingest prompt
            if chunk_tokens == 0:
                ids = torch.cat((ids, save_tokens), dim=-1)
                save_tokens = torch.empty((ids.shape[0], 0), dtype=torch.bool)
                overflow = ids.shape[-1] + generate_window - self.config.max_seq_len
                active_ids = ids[:, max(0, overflow) :]
                chunk_tokens = self.config.max_seq_len - active_ids.shape[-1]

                # Kick off the streaming generation
                self.generator.begin_stream_ex(
                    active_ids, gen_settings, **begin_stream_args
                )

                # Reset offsets for subsequent passes if the context is truncated
                offsets = None

            if auto_scale_penalty_range:
                gen_settings.token_repetition_range = generated_tokens

            # Run dict generation
            # Guarantees return of chunk, eos, and chunk_token_ids
            if generated_tokens < min_tokens:
                raw_generation = self.generator.stream_ex(ban_tokens=eos_tokens)
            else:
                raw_generation = self.generator.stream_ex()

            if token_healing:
                # Extract healed token
                ids[:, -1] = self.generator.sequence_ids[:, -2]
                token_healing = False

            # Get parameters that will always exist
            chunk = raw_generation["chunk"]
            eos = raw_generation["eos"]
            tokens = raw_generation["chunk_token_ids"]

            save_tokens = torch.cat(
                (save_tokens, tokens.expand(save_tokens.shape[0], -1)), dim=-1
            )
            chunk_buffer += chunk

            generated_tokens += 1
            chunk_tokens -= 1

            # Yield output
            now = time.time()
            elapsed = now - last_chunk_time

            if chunk_buffer != "" and (
                elapsed > stream_interval or eos or generated_tokens == max_tokens
            ):
                generation = {
                    "text": chunk_buffer,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "offset": len(full_response),
                }

                if request_logprobs > 0:
                    # Get top tokens and probs
                    top_tokens = unwrap(
                        raw_generation.get("top_tokens"),
                        torch.empty((1, 0, 1), dtype=torch.long),
                    )

                    top_probs = unwrap(
                        raw_generation.get("top_probs"),
                        torch.empty((1, 0, 1), dtype=torch.float),
                    )

                    if top_tokens.numel() > 0 and top_probs.numel() > 0:
                        logprobs = self.get_logprobs(top_tokens, top_probs)
                        generation["logprobs"] = logprobs

                        # The first logprob is the selected token prob
                        generation["token_probs"] = {
                            token: logprobs[token]
                            for token in list(logprobs.keys())[:1]
                        }

                yield generation
                full_response += chunk_buffer
                chunk_buffer = ""
                last_chunk_time = now

            if eos or generated_tokens == max_tokens:
                # Print response
                log_response(full_response)

                # Print metrics
                elapsed_time = last_chunk_time - start_time
                context_len = None if ids is None else context_len

                log_metrics(
                    generated_tokens, elapsed_time, context_len, self.config.max_seq_len
                )

                finish_reason = "length" if generated_tokens == max_tokens else "stop"
                generation = {"finish_reason": finish_reason}
                yield generation

                break
