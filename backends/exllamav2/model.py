"""The model container class for ExLlamaV2 models."""

import asyncio
import gc
import math
import pathlib
import traceback
import torch
import uuid
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2DynamicGeneratorAsync,
    ExLlamaV2DynamicJobAsync,
)
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
    generator: Optional[ExLlamaV2DynamicGeneratorAsync] = None
    prompt_template: Optional[PromptTemplate] = None
    paged: bool = True

    # Internal config vars
    cache_mode: str = "FP16"
    max_batch_size: int = 20
    generation_config: Optional[GenerationConfig] = None

    # GPU split vars
    gpu_split: Optional[list] = None
    gpu_split_auto: bool = True
    autosplit_reserve: List[float] = [96 * 1024**2]

    # Load state
    model_is_loading: bool = False
    model_loaded: bool = False

    # Load synchronization
    # The lock keeps load tasks sequential
    # The condition notifies any waiting tasks
    load_lock: asyncio.Lock = asyncio.Lock()
    load_condition: asyncio.Condition = asyncio.Condition()

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
        """

        self.quiet = quiet
        self.cache_mode = unwrap(kwargs.get("cache_mode"), "FP16")

        # Turn off GPU split if the user is using 1 GPU
        gpu_count = torch.cuda.device_count()
        gpu_split_auto = unwrap(kwargs.get("gpu_split_auto"), True)
        gpu_device_list = list(range(0, gpu_count))

        if gpu_count > 1 and gpu_split_auto:
            # Auto GPU split parameters
            self.gpu_split_auto = gpu_split_auto

            autosplit_reserve_megabytes = unwrap(kwargs.get("autosplit_reserve"), [96])
            self.autosplit_reserve = [
                int(math.ceil(value * 1024**2)) for value in autosplit_reserve_megabytes
            ]
        elif gpu_count > 1:
            # Manual GPU split
            self.gpu_split = kwargs.get("gpu_split")
            self.gpu_split_auto = False

            gpu_device_list = [
                device_idx
                for device_idx, memory in enumerate(self.gpu_split)
                if memory > 0
            ]
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

        # Enable fasttensors loading if present
        self.config.fasttensors = unwrap(kwargs.get("fasttensors"), False)

        # Disable paged mode if the user's min GPU isn't supported (ampere and up)
        min_compute_capability = min(
            torch.cuda.get_device_capability(device=device_idx)[0]
            for device_idx in gpu_device_list
        )

        # Compute capability < 8 is not supported by FA2
        # AMD is also unsupported until ROCm updates its FA2 fork
        if torch.version.hip or min_compute_capability < 8:
            logger.warning(
                "An unsupported GPU is found in this configuration. "
                "Switching to compatibility mode. \n"
                "This disables parallel batching "
                "and features that rely on it (ex. CFG). \n"
                "To disable compatability mode, all GPUs must be ampere "
                "(30 series) or newer. AMD GPUs are not supported."
            )
            self.config.no_flash_attn = True
            self.paged = False
            self.max_batch_size = 1

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

    async def wait_for_jobs(self, skip_wait: bool = False):
        """Polling mechanism to wait for pending generation jobs."""

        if not self.generator:
            return

        # Immediately abort all jobs if asked
        if skip_wait:
            # Requires a copy to avoid errors during iteration
            jobs_copy = self.generator.jobs.copy()
            for job in jobs_copy.values():
                await job.cancel()

        while self.generator.jobs:
            await asyncio.sleep(0.01)

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

    async def load_gen(self, progress_callback=None, **kwargs):
        """Loads a model and streams progress via a generator."""

        # Indicate that model load has started
        # Do this operation under the load lock's context
        try:
            await self.load_lock.acquire()
            self.model_is_loading = True

            # Wait for existing generation jobs to finish
            await self.wait_for_jobs(kwargs.get("skip_wait"))

            # Streaming gen for model load progress
            model_load_generator = self.load_model_sync(progress_callback)
            async for value in iterate_in_threadpool(model_load_generator):
                yield value

            # Create async generator
            self.generator = ExLlamaV2DynamicGeneratorAsync(
                model=self.model,
                cache=self.cache,
                draft_model=self.draft_model,
                draft_cache=self.draft_cache,
                tokenizer=self.tokenizer,
                max_batch_size=self.max_batch_size,
                paged=self.paged,
            )

            # Clean up any extra vram usage from torch and cuda
            # (Helps reduce VRAM bottlenecking on Windows)
            gc.collect()
            torch.cuda.empty_cache()

            # Cleanup and update model load state
            self.model_loaded = True
            logger.info("Model successfully loaded.")
        finally:
            self.load_lock.release()
            self.model_is_loading = False

            async with self.load_condition:
                self.load_condition.notify_all()

    @torch.inference_mode()
    def load_model_sync(self, progress_callback=None):
        """
        Load model, generator function

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int)

        Runs under a shared inference mode context.
        """

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

        if self.cache_mode == "Q4":
            self.cache = ExLlamaV2Cache_Q4(
                self.model, lazy=self.gpu_split_auto, batch_size=1
            )
        elif self.cache_mode == "FP8":
            self.cache = ExLlamaV2Cache_8bit(
                self.model, lazy=self.gpu_split_auto, batch_size=1
            )
        else:
            self.cache = ExLlamaV2Cache(
                self.model, lazy=self.gpu_split_auto, batch_size=1
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

    def get_loras(self):
        """Convenience function to get all loras."""

        return unwrap(self.generator.generator.current_loras, [])

    async def load_loras(self, lora_directory: pathlib.Path, **kwargs):
        """
        Load loras
        """

        loras = unwrap(kwargs.get("loras"), [])

        try:
            await self.load_lock.acquire()

            # Wait for existing generation jobs to finish
            await self.wait_for_jobs(kwargs.get("skip_wait"))

            loras_to_load: List[ExLlamaV2Lora] = []
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

                logger.info(f"Adding lora: {lora_name} at scaling {lora_scaling}")
                lora_path = lora_directory / lora_name

                loras_to_load.append(
                    ExLlamaV2Lora.from_directory(self.model, lora_path, lora_scaling)
                )
                logger.info(f"Lora successfully added: {lora_name}")
                success.append(lora_name)

            self.generator.generator.set_loras(loras_to_load)
            logger.info("All loras successfully loaded")

            # Return success and failure names
            return {"success": success, "failure": failure}
        finally:
            self.load_lock.release()

            async with self.load_condition:
                self.load_condition.notify_all()

    async def unload(self, loras_only: bool = False, **kwargs):
        """
        Free all VRAM resources used by this model
        """

        try:
            await self.load_lock.acquire()

            # Wait for other jobs to finish
            await self.wait_for_jobs(kwargs.get("skip_wait"))

            if self.generator and self.generator.generator.current_loras:
                for lora in self.generator.generator.current_loras:
                    lora.unload()

                self.generator.generator.set_loras([])

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

                # Cleanup the generator from any pending jobs
                await self.generator.close()
                self.generator = None

                # Set all model state variables to False
                self.model_is_loading = False
                self.model_loaded = False

            gc.collect()
            torch.cuda.empty_cache()

            logger.info("Loras unloaded." if loras_only else "Model unloaded.")
        finally:
            self.load_lock.release()

            async with self.load_condition:
                self.load_condition.notify_all()

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
        top_tokens = [
            self.tokenizer.extended_id_to_piece.get(
                index, self.tokenizer.id_to_piece[index]
            )
            for index in token_ids.flatten().tolist()
        ]

        top_values = torch.log(token_probs).flatten().tolist()

        # Cannot return -inf in JSON
        cleaned_values = [
            -1000 if value == float("-inf") else value for value in top_values
        ]

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
            joined_generation["generated_tokens"] = unwrap(
                generations[-1].get("generated_tokens"), 0
            )

        return joined_generation

    def check_unsupported_settings(self, **kwargs):
        """Check and warn the user if a sampler is unsupported. Meant for dev wheels!"""

        return kwargs

    async def generate_gen(
        self, prompt: str, abort_event: Optional[asyncio.Event] = None, **kwargs
    ):
        """
        Create generator function for prompt completion.

        for kwargs, check common/sampling.py
        """

        # Wait for load lock to be freed before processing
        async with self.load_condition:
            await self.load_condition.wait_for(lambda: not self.load_lock.locked())

        prompts = [prompt]

        token_healing = unwrap(kwargs.get("token_healing"), False)
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
        gen_settings.skew = unwrap(kwargs.get("skew"), 0)

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
            if self.paged:
                gen_settings.cfg_scale = cfg_scale

                # If the negative prompt is empty, use the BOS token
                negative_prompt = unwrap(
                    kwargs.get("negative_prompt"), self.tokenizer.bos_token
                )

                prompts.append(negative_prompt)
            else:
                logger.warning(
                    "CFG is currently disabled because paged mode is disabled. "
                    "Please use an ampere (30 series) or higher GPU for CFG support."
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

        # Add JSON schema filter if it exists
        json_schema = unwrap(kwargs.get("json_schema"))
        if json_schema:
            grammar_handler.add_json_schema_filter(
                json_schema, self.model, self.tokenizer
            )

        # Add regex filter if it exists
        regex_pattern = unwrap(kwargs.get("regex_pattern"))
        if regex_pattern:
            grammar_handler.add_regex_filter(regex_pattern, self.tokenizer)

        # Add EBNF filter if it exists
        grammar_string = unwrap(kwargs.get("grammar_string"))
        if grammar_string:
            grammar_handler.add_ebnf_filter(grammar_string, self.model, self.tokenizer)

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

        # Encode both positive and negative prompts
        input_ids = [
            self.tokenizer.encode(
                prompt, add_bos=add_bos_token, encode_special_tokens=True
            )
            for prompt in prompts
        ]

        # The first index will always be the positive prompt
        context_len = input_ids[0].size(dim=-1)
        if context_len > self.config.max_seq_len:
            logger.warning(
                f"Context length {context_len} is greater than max_seq_len "
                f"{self.config.max_seq_len}. Generation is truncated and "
                "metrics may not be accurate."
            )

        # Automatically set max_tokens to fill up the context
        # This should be an OK default, but may be changed in the future
        max_tokens = unwrap(
            kwargs.get("max_tokens"), self.config.max_seq_len - context_len
        )

        # Set min_tokens to generate while keeping EOS banned
        min_tokens = unwrap(kwargs.get("min_tokens"), 0)

        # This is an inverse of skip_special_tokens
        decode_special_tokens = unwrap(not kwargs.get("skip_special_tokens"), False)

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
            filters=grammar_handler.filters,
        )

        # Log prompt to console
        log_prompt(prompt, negative_prompt)

        # Create and add a new job
        job_id = uuid.uuid4().hex
        job = ExLlamaV2DynamicJobAsync(
            self.generator,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            gen_settings=gen_settings,
            stop_conditions=stop_conditions,
            decode_special_tokens=decode_special_tokens,
            filters=grammar_handler.filters,
            filter_prefer_eos=bool(grammar_handler.filters),
            return_probs=request_logprobs > 0,
            return_top_tokens=request_logprobs,
            return_logits=request_logprobs > 0,
            banned_strings=banned_strings,
            token_healing=token_healing,
            identifier=job_id,
        )

        # Save generated tokens and full response
        # Copy over max seq len incase model is unloaded and stored jobs can complete
        # Full response is required for offset calculation
        max_seq_len = self.config.max_seq_len
        generated_tokens = 0
        full_response = ""

        # Get the generation status once it's ready
        try:
            async for result in job:
                # Abort if the event is set while streaming
                if abort_event and abort_event.is_set():
                    await job.cancel()
                    break

                stage = result.get("stage")
                result_id = result.get("identifier")

                if stage == "streaming" and result_id == job_id:
                    chunk = unwrap(result.get("text"), "")
                    full_response += chunk

                    chunk_tokens = result.get("token_ids")
                    if chunk_tokens is not None:
                        generated_tokens += chunk_tokens.size(dim=0)

                    generation = {
                        "text": chunk,
                        "prompt_tokens": context_len,
                        "generated_tokens": generated_tokens,
                        "offset": len(full_response),
                    }

                    if request_logprobs > 0:
                        # Get top tokens and probs
                        top_tokens = unwrap(
                            result.get("top_k_tokens"),
                            torch.empty((1, 0, 1), dtype=torch.long),
                        )

                        top_probs = unwrap(
                            result.get("top_k_probs"),
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

                    # Second yield if eos is true
                    if result.get("eos"):
                        log_response(full_response)

                        eos_reason = result.get("eos_reason")
                        finish_reason = (
                            "length" if eos_reason == "max_new_tokens" else "stop"
                        )

                        log_metrics(
                            result.get("time_enqueued"),
                            result.get("prompt_tokens"),
                            result.get("time_prefill"),
                            result.get("new_tokens"),
                            result.get("time_generate"),
                            context_len,
                            max_seq_len,
                        )

                        # Remove the token text
                        generation = {
                            "prompt_tokens": generation.get("prompt_tokens"),
                            "generated_tokens": generation.get("generated_tokens"),
                            "finish_reason": finish_reason,
                        }

                        yield generation
                        break
        except asyncio.CancelledError:
            await job.cancel()
