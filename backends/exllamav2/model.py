"""The model container class for ExLlamaV2 models."""

import asyncio
import gc
import math
import pathlib
import traceback
import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2CacheBase,
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Cache_TP,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
    ExLlamaV2VisionTower,
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2DynamicGeneratorAsync,
    ExLlamaV2DynamicJobAsync,
)
from itertools import zip_longest
from loguru import logger
from typing import Dict, List, Optional

from backends.base_model_container import BaseModelContainer
from backends.exllamav2.grammar import (
    ExLlamaV2Grammar,
    clear_grammar_func_cache,
)
from backends.exllamav2.utils import exllama_disabled_flash_attn
from backends.exllamav2.vision import clear_image_embedding_cache
from common.concurrency import iterate_in_threadpool
from common.gen_logging import (
    log_generation_params,
    log_metrics,
    log_prompt,
    log_response,
)
from common.hardware import hardware_supports_flash_attn
from common.health import HealthManager
from common.multimodal import MultimodalEmbeddingWrapper
from common.sampling import BaseSamplerRequest
from common.templating import PromptTemplate, find_prompt_template
from common.transformers_utils import GenerationConfig, TokenizerConfig
from common.utils import calculate_rope_alpha, coalesce, unwrap
from endpoints.core.types.model import ModelCard, ModelCardParameters


class ExllamaV2Container(BaseModelContainer):
    """The model container class for ExLlamaV2 models."""

    # Model directories
    model_dir: pathlib.Path = pathlib.Path("models")
    draft_model_dir: pathlib.Path = pathlib.Path("models")

    # Exl2 vars
    config: Optional[ExLlamaV2Config] = None
    model: Optional[ExLlamaV2] = None
    cache: Optional[ExLlamaV2Cache] = None
    tokenizer: Optional[ExLlamaV2Tokenizer] = None
    generator: Optional[ExLlamaV2DynamicGeneratorAsync] = None
    prompt_template: Optional[PromptTemplate] = None
    paged: bool = True

    # Draft model vars
    use_draft_model: bool = False
    draft_config: Optional[ExLlamaV2Config] = None
    draft_model: Optional[ExLlamaV2] = None
    draft_cache: Optional[ExLlamaV2Cache] = None

    # Internal config vars
    cache_size: int = None
    cache_mode: str = "FP16"
    draft_cache_mode: str = "FP16"
    max_batch_size: Optional[int] = None
    generation_config: Optional[GenerationConfig] = None
    tokenizer_config: Optional[TokenizerConfig] = None

    # GPU split vars
    gpu_split: List[float] = []
    draft_gpu_split: List[float] = []
    gpu_split_auto: bool = True
    autosplit_reserve: List[float] = [96 * 1024**2]
    use_tp: bool = False

    # Vision vars
    use_vision: bool = False
    vision_model: Optional[ExLlamaV2VisionTower] = None

    # Load synchronization
    active_job_ids: Dict[str, Optional[ExLlamaV2DynamicJobAsync]] = {}
    loaded: bool = False
    load_lock: asyncio.Lock = asyncio.Lock()
    load_condition: asyncio.Condition = asyncio.Condition()

    @classmethod
    async def create(cls, model_directory: pathlib.Path, **kwargs):
        """
        Primary asynchronous initializer for model container.

        Kwargs are located in config_sample.yml
        """

        # Create a new instance as a "fake self"
        self = cls()

        # Initialize config
        self.config = ExLlamaV2Config()
        self.model_dir = model_directory
        self.config.model_dir = str(model_directory.resolve())

        # Make the max seq len 4096 before preparing the config
        # This is a better default than 2048
        self.config.max_seq_len = 4096

        self.config.prepare()
        print(self.config.max_seq_len)

        # Check if the model arch is compatible with various exl2 features
        self.config.arch_compat_overrides()

        # Load generation config overrides
        generation_config_path = model_directory / "generation_config.json"
        if generation_config_path.exists():
            try:
                self.generation_config = await GenerationConfig.from_file(
                    model_directory
                )
            except Exception:
                logger.error(traceback.format_exc())
                logger.warning(
                    "Skipping generation config load because of an unexpected error."
                )

        # Load tokenizer config overrides
        tokenizer_config_path = model_directory / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            try:
                self.tokenizer_config = await TokenizerConfig.from_file(model_directory)
            except Exception:
                logger.error(traceback.format_exc())
                logger.warning(
                    "Skipping tokenizer config load because of an unexpected error."
                )

        # Set vision state and error if vision isn't supported on the current model
        self.use_vision = unwrap(kwargs.get("vision"), False)
        if self.use_vision and not self.config.vision_model_type:
            raise ValueError(
                "The provided model does not have vision capabilities that are "
                "supported by ExllamaV2. "
                "Please reload with vision disabled."
            )

        # Prepare the draft model config if necessary
        draft_args = unwrap(kwargs.get("draft_model"), {})
        draft_model_name = draft_args.get("draft_model_name")
        self.use_draft_model = draft_args and draft_model_name

        # Always disable draft if params are incorrectly configured
        if draft_args and draft_model_name is None:
            logger.warning(
                "Draft model is disabled because a model name "
                "wasn't provided. Please check your config.yml!"
            )
            self.use_draft_model = False

        if self.use_draft_model:
            self.draft_config = ExLlamaV2Config()
            draft_model_path = pathlib.Path(
                unwrap(draft_args.get("draft_model_dir"), "models")
            )
            draft_model_path = draft_model_path / draft_model_name

            self.draft_gpu_split = unwrap(draft_args.get("draft_gpu_split"), [])
            self.draft_model_dir = draft_model_path
            self.draft_config.model_dir = str(draft_model_path.resolve())
            self.draft_config.prepare()

        # MARK: User configuration

        # Get cache mode
        self.cache_mode = unwrap(kwargs.get("cache_mode"), "FP16")

        # Turn off GPU split if the user is using 1 GPU
        gpu_count = torch.cuda.device_count()
        gpu_split_auto = unwrap(kwargs.get("gpu_split_auto"), True)
        use_tp = unwrap(kwargs.get("tensor_parallel"), False)
        gpu_split = unwrap(kwargs.get("gpu_split"), [])
        gpu_device_list = list(range(0, gpu_count))

        # Set GPU split options
        if gpu_count == 1:
            self.gpu_split_auto = False
            logger.info("Disabling GPU split because one GPU is in use.")
        else:
            # Set tensor parallel
            if use_tp:
                self.use_tp = True

                # TP has its own autosplit loader
                self.gpu_split_auto = False

            # Enable manual GPU split if provided
            if gpu_split:
                self.gpu_split_auto = False
                self.gpu_split = gpu_split

                gpu_device_list = [
                    device_idx
                    for device_idx, memory in enumerate(self.gpu_split)
                    if memory > 0
                ]
            elif gpu_split_auto and not self.use_tp:
                # Otherwise fallback to autosplit settings
                self.gpu_split_auto = gpu_split_auto

                autosplit_reserve_megabytes = unwrap(
                    kwargs.get("autosplit_reserve"), [96]
                )

                # Reserve VRAM for each GPU
                self.autosplit_reserve = [
                    int(math.ceil(value * 1024**2))
                    for value in autosplit_reserve_megabytes
                ]

            # Change the GPU device list only if gpu_split's list is too small
            # This allows for an uneven list specification
            if self.draft_gpu_split and len(self.draft_gpu_split) > len(self.gpu_split):
                gpu_device_list = [
                    device_idx
                    for device_idx, memory in enumerate(self.draft_gpu_split)
                    if memory > 0
                ]

        # Hardcode max output length to 16
        self.config.max_output_len = 16

        # Grab the base model's sequence length before overrides for
        # rope calculations
        base_seq_len = self.config.max_seq_len

        # Set the target seq len if present
        # Fallback to base_seq_len if not provided
        target_seq_len = unwrap(kwargs.get("max_seq_len"), base_seq_len)

        # Set the rope scale
        self.config.scale_pos_emb = unwrap(
            kwargs.get("rope_scale"), self.config.scale_pos_emb
        )

        # Sets rope alpha value.
        # Automatically calculate if unset or defined as an "auto" literal.
        rope_alpha = unwrap(kwargs.get("rope_alpha"), "auto")
        if rope_alpha == "auto":
            self.config.scale_alpha_value = calculate_rope_alpha(
                base_seq_len, target_seq_len
            )
        else:
            self.config.scale_alpha_value = rope_alpha

        # Set the max seq len if specified
        if target_seq_len:
            self.config.max_seq_len = target_seq_len

        # Set max batch size to the config override
        self.max_batch_size = unwrap(kwargs.get("max_batch_size"))

        # Check whether the user's configuration supports flash/paged attention
        # Also check if exl2 has disabled flash attention
        if exllama_disabled_flash_attn(
            self.config.no_flash_attn
        ) or not hardware_supports_flash_attn(gpu_device_list):
            gpu_unsupported_message = (
                "An unsupported GPU is found in this configuration. "
                "Switching to compatibility mode. \n"
                "This disables parallel batching "
                "and features that rely on it (ex. CFG). \n"
                "To disable compatability mode, all GPUs must be ampere "
                "(30 series) or newer. AMD GPUs are not supported."
            )

            logger.warning(gpu_unsupported_message)

            self.config.no_flash_attn = True
            if self.draft_config:
                self.draft_config.no_flash_attn = True
            self.paged = False
            self.max_batch_size = 1
            torch.backends.cuda.enable_flash_sdp(False)

        # Set k/v cache size
        # cache_size is only relevant when paged mode is enabled
        if self.paged:
            cache_size = unwrap(kwargs.get("cache_size"), self.config.max_seq_len)

            if cache_size < self.config.max_seq_len:
                logger.warning(
                    f"The given cache_size ({cache_size}) is smaller than the "
                    "desired context length.\n"
                    "Overriding cache_size to max_seq_len. "
                )

                cache_size = self.config.max_seq_len

            # Enforce a multiple of 256 for cache size
            # Overestimate to ensure that the cache isn't below max_seq_len
            cache_remainder = cache_size % 256
            if cache_remainder != 0:
                rounded_cache_size = int(
                    256 * ((cache_size - cache_remainder) / 256 + 1)
                )

                logger.warning(
                    f"The given cache size ({cache_size}) is "
                    "not a multiple of 256.\n"
                    "Overriding cache_size with an overestimated value of "
                    f"{rounded_cache_size} tokens."
                )

                cache_size = rounded_cache_size

            # Warn user if cache size may be inadequate for CFG
            if cache_size < 2 * self.config.max_seq_len:
                logger.warning(
                    f"The given cache_size ({cache_size}) is less than 2 * max_seq_len "
                    "and may be too small for requests using CFG. \n"
                    "Ignore this warning if you do not plan on using CFG."
                )

            self.cache_size = cache_size
        else:
            self.cache_size = self.config.max_seq_len

        # Try to set prompt template
        self.prompt_template = await find_prompt_template(
            kwargs.get("prompt_template"), model_directory
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

        # Make sure chunk size is >= 256, keep near or below max seq len
        user_chunk_size = unwrap(kwargs.get("chunk_size"), 2048)
        chunk_size = sorted((256, user_chunk_size, self.config.max_seq_len))[1]
        chunk_remainder = chunk_size % 256
        if chunk_remainder != 0:
            rounded_chunk_size = int(256 * ((chunk_size - chunk_remainder) / 256 + 1))

            logger.warning(
                f"The given chunk size ({chunk_size}) is "
                "not a multiple of 256.\n"
                "Overriding chunk_size with an overestimated value of "
                f"{rounded_chunk_size} tokens."
            )
            chunk_size = rounded_chunk_size
        self.config.max_input_len = chunk_size
        self.config.max_attention_size = chunk_size**2

        # Set user-configured draft model values
        if self.use_draft_model:
            self.draft_config.max_seq_len = self.config.max_seq_len

            self.draft_config.scale_pos_emb = unwrap(
                draft_args.get("draft_rope_scale"), 1.0
            )

            # Set draft rope alpha. Follows same behavior as model rope alpha.
            # Use the base sequence length of the model
            draft_rope_alpha = unwrap(draft_args.get("draft_rope_alpha"), "auto")
            if draft_rope_alpha == "auto":
                self.draft_config.scale_alpha_value = calculate_rope_alpha(
                    base_seq_len, self.draft_config.max_seq_len
                )
            else:
                self.draft_config.scale_alpha_value = draft_rope_alpha

            # Set draft cache mode
            self.draft_cache_mode = unwrap(draft_args.get("draft_cache_mode"), "FP16")

            # Edit the draft config size
            if chunk_size:
                self.draft_config.max_input_len = chunk_size
                self.draft_config.max_attention_size = chunk_size**2

        # Return the created instance
        return self

    def model_info(self):
        draft_model_card: ModelCard = None
        if self.draft_config:
            draft_model_params = ModelCardParameters(
                max_seq_len=self.draft_config.max_seq_len,
                rope_scale=self.draft_config.scale_pos_emb,
                rope_alpha=self.draft_config.scale_alpha_value,
                cache_mode=self.draft_cache_mode,
            )

            draft_model_card = ModelCard(
                id=self.draft_model_dir.name,
                parameters=draft_model_params,
            )

        model_params = ModelCardParameters(
            max_seq_len=self.config.max_seq_len,
            cache_size=self.cache_size,
            rope_scale=self.config.scale_pos_emb,
            rope_alpha=self.config.scale_alpha_value,
            max_batch_size=self.max_batch_size,
            cache_mode=self.cache_mode,
            chunk_size=self.config.max_input_len,
            use_vision=self.use_vision,
            draft=draft_model_card,
        )

        if self.prompt_template:
            model_params.prompt_template = self.prompt_template.name
            model_params.prompt_template_content = self.prompt_template.raw_template

        model_card = ModelCard(
            id=self.model_dir.name,
            parameters=model_params,
        )

        return model_card

    async def wait_for_jobs(self, skip_wait: bool = False):
        """Polling mechanism to wait for pending generation jobs."""

        if not self.generator:
            return

        # Immediately abort all jobs if asked
        if skip_wait:
            logger.warning(
                "Immediately terminating all jobs. "
                "Clients will have their requests cancelled.\n"
            )

            for job in self.active_job_ids.values():
                if job:
                    await job.cancel()

        while len(self.active_job_ids) > 0:
            await asyncio.sleep(0.01)

    async def load(self, progress_callback=None):
        """
        Load model

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded.

                Prototype:
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

            # Wait for existing generation jobs to finish
            await self.wait_for_jobs(kwargs.get("skip_wait"))

            # Streaming gen for model load progress
            model_load_generator = self.load_model_sync(progress_callback)
            async for value in iterate_in_threadpool(model_load_generator):
                yield value

            # Create async generator
            await self.create_generator()

            # Clean up any extra vram usage from torch and cuda
            # (Helps reduce VRAM bottlenecking on Windows)
            gc.collect()
            torch.cuda.empty_cache()

            # Cleanup and update model load state
            self.loaded = True
            logger.info("Model successfully loaded.")
        finally:
            self.load_lock.release()

            async with self.load_condition:
                self.load_condition.notify_all()

    @torch.inference_mode()
    def load_model_sync(self, progress_callback=None):
        """
        Synchronous generator for loading.

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded.

                Prototype:
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
            logger.info("Loading draft model: " + self.draft_config.model_dir)

            # Draft uses the autosplit loader, so create a cache that reflects this
            draft_cache_class = self.get_cache_class(self.draft_cache_mode)

            if self.draft_gpu_split:
                logger.info("Loading with a manual GPU split (or a one GPU setup)")

                for value in self.draft_model.load_gen(
                    self.draft_gpu_split,
                    callback_gen=progress_callback,
                ):
                    if value:
                        yield value

                self.draft_cache = self.create_cache(
                    cache_class=draft_cache_class,
                    autosplit=False,
                    use_tp=False,
                    model=self.draft_model,
                )
            else:
                logger.info("Loading with autosplit")

                self.draft_cache = self.create_cache(
                    cache_class=draft_cache_class,
                    autosplit=True,
                    use_tp=False,
                    model=self.draft_model,
                )

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

        # Load vision tower if it exists
        if self.use_vision:
            self.vision_model = ExLlamaV2VisionTower(self.config)

            for value in self.vision_model.load_gen(callback_gen=progress_callback):
                if value:
                    yield value

        self.model = ExLlamaV2(self.config)
        logger.info("Loading model: " + self.config.model_dir)

        # Get class of the model cache
        cache_class = self.get_cache_class(self.cache_mode)

        # Load model with manual split
        # Entrypoint for single GPU users
        if self.use_tp:
            logger.info("Loading with tensor parallel")

            # GPU split must be None if the array is empty
            # Otherwise the TP loader fails
            for value in self.model.load_tp_gen(
                self.gpu_split or None,
                callback_gen=progress_callback,
                expect_cache_base=cache_class,
                expect_cache_tokens=self.cache_size,
            ):
                if value:
                    yield value
        elif not self.gpu_split_auto:
            logger.info("Loading with a manual GPU split (or a one GPU setup)")

            for value in self.model.load_gen(
                self.gpu_split,
                callback_gen=progress_callback,
            ):
                if value:
                    yield value

        # Create the model cache
        self.cache = self.create_cache(
            cache_class=cache_class,
            autosplit=self.gpu_split_auto,
            use_tp=self.use_tp,
            model=self.model,
        )

        # Load model with autosplit (without TP)
        if self.gpu_split_auto and not self.use_tp:
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

    # TODO: Maybe make a wrapper class with an ID instead of a utility function
    def get_cache_class(self, cache_mode: str):
        """Utility function to get a cache class based on user preference."""

        match cache_mode:
            case "Q4":
                return ExLlamaV2Cache_Q4
            case "Q6":
                return ExLlamaV2Cache_Q6
            case "Q8":
                return ExLlamaV2Cache_Q8
            case _:
                return ExLlamaV2Cache

    def create_cache(
        self,
        cache_class: ExLlamaV2CacheBase,
        autosplit: bool,
        use_tp: bool,
        model: ExLlamaV2,
    ):
        """Utility function to create a model cache."""

        if use_tp:
            return ExLlamaV2Cache_TP(
                model,
                base=cache_class,
                max_seq_len=self.cache_size,
                batch_size=1,
            )
        else:
            return cache_class(
                model,
                max_seq_len=self.cache_size,
                lazy=autosplit,
                batch_size=1,
            )

    async def create_generator(self):
        """Create and save a Exllama generator class."""

        try:
            # Don't acquire locks unless a model is loaded
            if self.loaded:
                await self.load_lock.acquire()

                # Immediately cancel all jobs
                await self.wait_for_jobs(skip_wait=True)

            # Create new generator
            self.generator = ExLlamaV2DynamicGeneratorAsync(
                model=self.model,
                cache=self.cache,
                draft_model=self.draft_model,
                draft_cache=self.draft_cache,
                tokenizer=self.tokenizer,
                max_batch_size=self.max_batch_size,
                paged=self.paged,
            )

            # Update the state of the container var
            if self.max_batch_size is None:
                self.max_batch_size = self.generator.generator.max_batch_size
        finally:
            # This means the generator is being recreated
            # The load lock is already released in the load function
            if self.loaded:
                self.load_lock.release()

                async with self.load_condition:
                    self.load_condition.notify_all()

    def get_loras(self):
        """Convenience function to get all loras."""

        return unwrap(self.generator.generator.current_loras, [])

    async def load_loras(self, lora_directory: pathlib.Path, **kwargs):
        """Load loras."""

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
        """Free all VRAM resources used by the model (and loras)."""

        # Shutdown immediately unloads and bypasses all locks
        do_shutdown = kwargs.get("shutdown")

        try:
            if not do_shutdown:
                await self.load_lock.acquire()

                # Wait for other jobs to finish
                await self.wait_for_jobs(kwargs.get("skip_wait"))

            # Delete references held in the grammar module
            clear_grammar_func_cache()

            # Clear the image embedding cache
            clear_image_embedding_cache()

            # Unload LoRAs
            if self.generator and self.generator.generator.current_loras:
                for lora in self.generator.generator.current_loras:
                    lora.unload()

                self.generator.generator.set_loras([])

            # Unload the entire model if not just unloading loras
            if not loras_only:
                if self.model:
                    self.model.unload()
                self.model = None

                if self.vision_model:
                    self.vision_model.unload()

                self.vision_model = None

                if self.draft_model:
                    self.draft_model.unload()
                self.draft_model = None

                self.config = None
                self.cache = None
                self.tokenizer = None

                # Cleanup the generator from any pending jobs
                if self.generator is not None:
                    await self.generator.close()
                    self.generator = None

                # Set all model state variables to False
                self.loaded = False

            gc.collect()
            torch.cuda.empty_cache()

            logger.info("Loras unloaded." if loras_only else "Model unloaded.")
        finally:
            if not do_shutdown:
                self.load_lock.release()

                async with self.load_condition:
                    self.load_condition.notify_all()

    def encode_tokens(self, text: str, **kwargs):
        """Wrapper to encode tokens from a text string."""

        mm_embeddings: MultimodalEmbeddingWrapper = kwargs.get("embeddings")
        mm_embeddings_content = mm_embeddings.content if mm_embeddings else []

        return (
            self.tokenizer.encode(
                text,
                add_bos=unwrap(
                    kwargs.get("add_bos_token"), self.tokenizer_config.add_bos_token
                ),
                encode_special_tokens=unwrap(kwargs.get("encode_special_tokens"), True),
                embeddings=mm_embeddings_content,
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

    def get_special_tokens(self):
        return {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
        }

    def get_logprobs(self, token_ids: torch.Tensor, token_probs: torch.Tensor):
        top_tokens = [
            self.tokenizer.extended_id_to_piece.get(
                index, self.tokenizer.get_id_to_piece_list(True)[index]
            )
            for index in token_ids.flatten().tolist()
        ]

        top_values = torch.log(token_probs).flatten().tolist()

        # Cannot return -inf in JSON
        cleaned_values = [
            -1000 if value == float("-inf") else value for value in top_values
        ]

        return dict(zip_longest(top_tokens, cleaned_values))

    async def generate(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        abort_event: Optional[asyncio.Event] = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ):
        """Generate a response to a prompt."""
        generations = []
        async for generation in self.stream_generate(
            request_id,
            prompt,
            params,
            abort_event,
            mm_embeddings,
        ):
            generations.append(generation)

        joined_generation = {
            "text": "",
            "prompt_tokens": 0,
            "generation_tokens": 0,
            "tool_calls": None,
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
                joined_generation["stop_str"] = finish_reason_gen.get("stop_str")
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

    async def stream_generate(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        abort_event: Optional[asyncio.Event] = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ):
        try:
            # Wait for load lock to be freed before processing
            # Mainly used for loras and other operations where the class is available
            async with self.load_condition:
                await self.load_condition.wait_for(lambda: not self.load_lock.locked())

            # If the model is being unloaded, don't accept new requests
            if not self.loaded:
                raise RuntimeError(
                    "Model is being unloaded. Cannot process new generation requests."
                )

            # Mark that the job is running
            self.active_job_ids[request_id] = None

            # Yield from the internal generator
            async for generation_chunk in self.generate_gen(
                request_id=request_id,
                prompt=prompt,
                params=params,
                abort_event=abort_event,
                mm_embeddings=mm_embeddings,
            ):
                yield generation_chunk
        finally:
            # Clean up and remove the job from active IDs
            del self.active_job_ids[request_id]

    def check_unsupported_settings(self, params: BaseSamplerRequest):
        """
        Check and warn the user if a sampler is unsupported.

        Meant for dev wheels!
        """

        return params

    def assign_gen_params(
        self,
        params: BaseSamplerRequest,
        gen_settings: ExLlamaV2Sampler.Settings,
        grammar_handler: ExLlamaV2Grammar,
        banned_strings: List[str],
    ):
        # Apply settings
        gen_settings.temperature = params.temperature
        gen_settings.temperature_last = params.temperature_last
        gen_settings.smoothing_factor = params.smoothing_factor
        gen_settings.top_k = params.top_k
        gen_settings.top_p = params.top_p
        gen_settings.top_a = params.top_a
        gen_settings.min_p = params.min_p
        gen_settings.tfs = params.tfs
        gen_settings.typical = params.typical
        gen_settings.mirostat = params.mirostat_mode == 2
        gen_settings.skew = params.skew

        # XTC
        if params.xtc_probability > 0.0:
            gen_settings.xtc_probability = params.xtc_probability

            # 0.1 is the default for this value
            gen_settings.xtc_threshold = params.xtc_threshold

        # DynaTemp settings
        max_temp = params.max_temp
        min_temp = params.min_temp

        if params.max_temp > params.min_temp:
            gen_settings.max_temp = max_temp
            gen_settings.min_temp = min_temp
            gen_settings.temp_exponent = params.temp_exponent
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
        gen_settings.mirostat_tau = params.mirostat_tau
        gen_settings.mirostat_eta = params.mirostat_eta

        # Penalties
        gen_settings.token_repetition_penalty = params.repetition_penalty
        gen_settings.token_frequency_penalty = params.frequency_penalty
        gen_settings.token_presence_penalty = params.presence_penalty

        # Applies for all penalties despite being called token_repetition_range
        gen_settings.token_repetition_range = unwrap(
            params.penalty_range, self.config.max_seq_len
        )

        # Always make sure the fallback is 0 if range < 0
        # It's technically fine to use -1, but this just validates the passed
        # fallback
        # Always default to 0 if something goes wrong
        if gen_settings.token_repetition_range < 0:
            fallback_decay = 0
        else:
            fallback_decay = gen_settings.token_repetition_range
        gen_settings.token_repetition_decay = coalesce(
            params.repetition_decay, fallback_decay, 0
        )

        # DRY options
        dry_multiplier = params.dry_multiplier

        # < 0 = disabled
        if dry_multiplier > 0:
            gen_settings.dry_multiplier = dry_multiplier
            gen_settings.dry_allowed_length = params.dry_allowed_length
            gen_settings.dry_base = params.dry_base

            # Exl2 has dry_range as 0 for unlimited unlike -1 for penalty_range
            # Use max_seq_len as the fallback to stay consistent
            gen_settings.dry_range = unwrap(params.dry_range, self.config.max_seq_len)

            # Tokenize sequence breakers
            if params.dry_sequence_breakers:
                gen_settings.dry_sequence_breakers = {
                    self.encode_tokens(s)[-1] for s in params.dry_sequence_breakers
                }

        # Add JSON schema filter if it exists
        if params.json_schema:
            grammar_handler.add_json_schema_filter(
                params.json_schema, self.model, self.tokenizer
            )

        # Add regex filter if it exists
        if params.regex_pattern:
            grammar_handler.add_regex_filter(
                params.regex_pattern, self.model, self.tokenizer
            )

        # Add EBNF filter if it exists
        if params.grammar_string:
            grammar_handler.add_kbnf_filter(
                params.grammar_string, self.model, self.tokenizer
            )

        # Set banned strings
        banned_strings = params.banned_strings
        if banned_strings and len(grammar_handler.filters) > 0:
            logger.warning(
                "Disabling banned_strings because "
                "they cannot be used with grammar filters."
            )

            banned_strings = []

        # Speculative Ngram
        self.generator.speculative_ngram = params.speculative_ngram

        # Override sampler settings for temp = 0
        if gen_settings.temperature == 0:
            gen_settings.temperature = 1.0
            gen_settings.top_k = 1
            gen_settings.top_p = 0
            gen_settings.typical = 0

            logger.warning(
                "Temperature is set to 0. Overriding temp, "
                "top_k, top_p, and typical to 1.0, 1, 0, and 0."
            )

        # Set banned tokens
        if params.banned_tokens:
            gen_settings.disallow_tokens(self.tokenizer, params.banned_tokens)

        # Set allowed tokens
        if params.allowed_tokens:
            gen_settings.allow_tokens(self.tokenizer, params.allowed_tokens)

        # Set logit bias
        if params.logit_bias:
            # Create a vocab tensor if it doesn't exist for token biasing
            if gen_settings.token_bias is None:
                padding = -self.tokenizer.config.vocab_size % 32
                gen_settings.token_bias = torch.zeros(
                    (self.tokenizer.config.vocab_size + padding,),
                    dtype=torch.float,
                )

            # Map logits to the tensor with their biases
            for token_id, bias in params.logit_bias.items():
                if 0 <= token_id < len(self.tokenizer.get_id_to_piece_list(True)):
                    gen_settings.token_bias[token_id] = bias
                else:
                    logger.warning(
                        f"Logit bias: Token {token_id} not present "
                        "in the model's vocab. Skipping."
                    )

    # Adds logprobs to a generation chunk
    def handle_logprobs(self, result: dict, generation: dict):
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
                token: logprobs[token] for token in list(logprobs.keys())[:1]
            }

    # Creates and returns a finish chunk
    def handle_finish_chunk(self, result: dict, generation: dict):
        eos_reason = result.get("eos_reason")

        stop_str = None
        if eos_reason == "max_new_tokens":
            finish_reason = "length"
        else:
            finish_reason = "stop"
            # Grab stop string if stop was the reason
            if eos_reason == "stop_token":
                stop_str = result.get("eos_triggering_token_str")
            elif eos_reason == "stop_string":
                stop_str = result.get("eos_triggering_string")

        finish_chunk = {
            "prompt_tokens": generation.get("prompt_tokens"),
            "generated_tokens": generation.get("generated_tokens"),
            "finish_reason": finish_reason,
            "stop_str": stop_str,
        }

        return finish_chunk

    async def generate_gen(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        abort_event: Optional[asyncio.Event] = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ):
        """
        Create generator function for prompt completion.

        for kwargs, check common/sampling.py
        """

        prompts = [prompt]
        gen_settings = ExLlamaV2Sampler.Settings()
        grammar_handler = ExLlamaV2Grammar()
        banned_strings = []

        self.assign_gen_params(
            params,
            gen_settings,
            grammar_handler,
            banned_strings,
        )

        # Set CFG scale and negative prompt
        cfg_scale = params.cfg_scale
        negative_prompt = None
        if cfg_scale not in [None, 1.0]:
            if self.paged:
                gen_settings.cfg_scale = cfg_scale

                # If the negative prompt is empty, use the BOS token
                negative_prompt = unwrap(
                    params.negative_prompt, self.tokenizer.bos_token
                )

                prompts.append(negative_prompt)
            else:
                logger.warning(
                    "CFG is currently disabled because paged mode is disabled. "
                    "Please use an ampere (30 series) or higher GPU for CFG support."
                )

        # Dynamically scale penalty range to output tokens
        # Only do this if freq/pres pen is enabled
        # and the repetition range is -1
        auto_scale_penalty_range = (
            gen_settings.token_frequency_penalty != 0
            or gen_settings.token_presence_penalty != 0
        ) and gen_settings.token_repetition_range == -1

        stop_conditions = params.stop
        ban_eos_token = params.ban_eos_token

        # Set add_bos_token for generation
        add_bos_token = unwrap(
            params.add_bos_token, self.tokenizer_config.add_bos_token
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

        # Get multimodal embeddings if present
        mm_embeddings_content = mm_embeddings.content if mm_embeddings else []

        # Encode both positive and negative prompts
        input_ids = [
            self.tokenizer.encode(
                prompt,
                add_bos=add_bos_token,
                encode_special_tokens=True,
                embeddings=mm_embeddings_content,
            )
            for prompt in prompts
        ]

        # The first index will always be the positive prompt
        context_len = input_ids[0].size(dim=-1)

        # The second index will be the negative prompt if CFG is enabled
        negative_context_len = input_ids[1].size(dim=-1) if negative_prompt else 0

        # Automatically set max_tokens to fill up the context
        # This should be an OK default, but may be changed in the future
        max_tokens = unwrap(
            params.max_tokens,
            self.config.max_seq_len - max(context_len, negative_context_len),
        )
        if max_tokens < 1:
            logger.warning("max_tokens must be a positive integer, setting to 1.")
            max_tokens = 1

        # Determine if the negative context or the context length is bigger
        context_to_check = max(negative_context_len, context_len)

        # Check total length of prompt against max context length
        if context_to_check > self.config.max_seq_len:
            preamble = (
                "Negative prompt" if negative_context_len > context_len else "Prompt"
            )

            raise ValueError(
                f"{preamble} length {context_to_check} is greater than "
                f"max_seq_len {self.config.max_seq_len}"
            )

        # Check total required pages for CFG request to avoid overallocation
        if negative_prompt and (
            sum(
                256 * math.ceil((context + max_tokens) / 256)
                for context in (context_len, negative_context_len)
            )
            > self.cache_size
        ):
            raise ValueError(
                f"Total required page size for request "
                f"{context_len} + {negative_context_len} + {max_tokens} * 2 "
                f"is greater than cache_size {self.cache_size}"
            )

        # Log prompt to console. Add the BOS token if specified
        log_prompt(
            f"{self.tokenizer.bos_token if add_bos_token else ''}{prompt}",
            request_id,
            negative_prompt,
        )

        # Create and add a new job
        # Don't use the request ID here as there can be multiple jobs per request
        job = ExLlamaV2DynamicJobAsync(
            self.generator,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            min_new_tokens=params.min_tokens,
            gen_settings=gen_settings,
            stop_conditions=stop_conditions,
            decode_special_tokens=not params.skip_special_tokens,
            filters=grammar_handler.filters,
            filter_prefer_eos=bool(grammar_handler.filters),
            return_probs=params.logprobs > 0,
            return_top_tokens=params.logprobs,
            return_logits=params.logprobs > 0,
            banned_strings=banned_strings,
            token_healing=params.token_healing,
            identifier=request_id,
            embeddings=mm_embeddings_content,
        )

        # Assign the active job to the request ID
        self.active_job_ids[request_id] = job

        # Save generated tokens and full response
        # Copy over max seq len incase model is unloaded and stored jobs can complete
        # Full response is required for offset calculation
        max_seq_len = self.config.max_seq_len
        generated_tokens = 0
        full_response = ""
        metrics_result = {}

        # Get the generation status once it's ready
        try:
            async for result in job:
                # Abort if the event is set while streaming
                if abort_event and abort_event.is_set():
                    await job.cancel()
                    break

                stage = result.get("stage")
                result_id = result.get("identifier")

                if stage == "streaming" and result_id == request_id:
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

                    # Increase penalty range to generated token amount
                    if auto_scale_penalty_range:
                        gen_settings.token_repetition_range = generated_tokens

                    # Handle logprobs
                    if params.logprobs > 0:
                        self.handle_logprobs(result, generation)

                    yield generation

                    # Yield a finish chunk when generation is finished
                    if result.get("eos"):
                        log_response(request_id, full_response)

                        generation = self.handle_finish_chunk(result, generation)

                        # Save the final result for metrics logging
                        metrics_result = result

                        yield generation
                        break
        except asyncio.CancelledError:
            await job.cancel()
        except Exception as ex:
            # Create a new generator since the current state is broken
            # No need to wait for this to finish
            logger.error(
                "FATAL ERROR with generation. "
                "Attempting to recreate the generator. "
                "If this fails, please restart the server.\n"
            )
            asyncio.ensure_future(self.create_generator())

            await HealthManager.add_unhealthy_event(ex)

            raise ex
        finally:
            # Log generation options to console
            # Some options are too large, so log the args instead
            log_generation_params(
                request_id=request_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=eos_tokens,
                prompt=prompt,
                **params.model_dump(exclude={"prompt"}),
                auto_scale_penalty_range=auto_scale_penalty_range,
            )

            # Log the metrics if present
            if metrics_result:
                log_metrics(
                    request_id,
                    metrics_result.get("time_enqueued"),
                    metrics_result.get("prompt_tokens"),
                    metrics_result.get("cached_tokens"),
                    metrics_result.get("time_prefill"),
                    metrics_result.get("new_tokens"),
                    metrics_result.get("time_generate"),
                    context_len,
                    max_seq_len,
                )
