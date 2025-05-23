import asyncio
import gc
import math
import pathlib
import re
import time
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from exllamav3 import (
    AsyncGenerator,
    AsyncJob,
    Cache,
    Config,
    Model,
    Tokenizer,
)
from exllamav3.cache import CacheLayer_quant
from loguru import logger

from backends.base_model_container import BaseModelContainer
from backends.exllamav3.sampler import ExllamaV3SamplerBuilder
from common.concurrency import iterate_in_threadpool
from common.gen_logging import (
    log_generation_params,
    log_metrics,
    log_prompt,
)
from common.hardware import hardware_supports_flash_attn
from common.health import HealthManager
from common.multimodal import MultimodalEmbeddingWrapper
from common.optional_dependencies import check_package_version
from common.sampling import BaseSamplerRequest
from common.templating import PromptTemplate, find_prompt_template
from common.transformers_utils import HFModel
from common.utils import coalesce, unwrap
from endpoints.core.types.model import ModelCard, ModelCardParameters

PAGE_SIZE = 256


class ExllamaV3Container(BaseModelContainer):
    """Abstract base class for model containers."""

    # Exposed model information
    model_dir: pathlib.Path = pathlib.Path("models")
    prompt_template: Optional[PromptTemplate] = None

    # HF Model instance
    hf_model: HFModel

    # Load synchronization
    # The bool is a master switch for accepting requests
    # The lock keeps load tasks sequential
    # The condition notifies any waiting tasks
    active_job_ids: Dict[str, Any] = {}
    loaded: bool = False
    load_lock: asyncio.Lock = asyncio.Lock()
    load_condition: asyncio.Condition = asyncio.Condition()

    # Exl3 vars
    model: Optional[Model] = None
    cache: Optional[Cache] = None
    draft_model: Optional[Model] = None
    draft_cache: Optional[Cache] = None
    tokenizer: Optional[Tokenizer] = None
    config: Optional[Config] = None
    draft_config: Optional[Config] = None
    generator: Optional[AsyncGenerator] = None

    # Class-specific vars
    gpu_split: Optional[List[float]] = None
    gpu_split_auto: bool = True
    autosplit_reserve: Optional[List[float]] = [96 / 1024]
    use_tp: bool = False
    user_max_seq_len: int = 4096  # User-configured max_seq_len
    generator_max_seq_len: int = 4096  # Aligned operational max_seq_len
    cache_size: int = 4096
    cache_mode: str = "FP16"
    draft_cache_mode: str = "FP16"
    chunk_size: int = 2048
    max_batch_size: Optional[int] = None

    # Required methods
    @classmethod
    async def create(cls, model_directory: pathlib.Path, hf_model: HFModel, **kwargs):
        """
        Asynchronously creates and initializes a model container instance.

        Args:
            model_directory: Path to the model files.
            **kwargs: Backend-specific configuration options.

        Returns:
            An instance of the implementing class.
        """

        self = cls()

        # Make sure ExllamaV3 is up to date
        check_package_version("exllamav3", "0.0.2")

        logger.warning(
            "ExllamaV3 is currently in an alpha state. "
            "Please note that all config options may not work."
        )

        self.model_dir = model_directory
        self.hf_model = hf_model
        self.config = Config.from_directory(str(model_directory.resolve()))
        self.model = Model.from_config(self.config)
        self.tokenizer = Tokenizer.from_config(self.config)

        # Fallback to 4096 since exl3 can't fetch from HF's config.json
        # Store user configured max_seq_len
        self.user_max_seq_len = unwrap(kwargs.get("max_seq_len"), 4096)
        # Calculate aligned operational max_seq_len for the generator
        self.generator_max_seq_len = (
            (self.user_max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        ) * PAGE_SIZE
        if self.generator_max_seq_len != self.user_max_seq_len:
            logger.info(
                f"User max_seq_len {self.user_max_seq_len} has been aligned to {self.generator_max_seq_len} for ExllamaV3 operations."
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
            draft_model_path = pathlib.Path(
                unwrap(draft_args.get("draft_model_dir"), "models")
            )
            draft_model_path = draft_model_path / draft_model_name
            self.draft_gpu_split = unwrap(draft_args.get("draft_gpu_split"), [])
            self.draft_model_dir = draft_model_path
            self.draft_config = Config.from_directory(str(draft_model_path.resolve()))
            self.draft_model = Model.from_config(self.draft_config)
            logger.info(f"Using draft model: {str(draft_model_path.resolve())}")
        else:
            self.draft_model = None
            self.draft_cache = None

        # Turn off GPU split if the user is using 1 GPU
        gpu_count = torch.cuda.device_count()
        gpu_split_auto = unwrap(kwargs.get("gpu_split_auto"), True)
        gpu_split = unwrap(kwargs.get("gpu_split"), None)
        gpu_device_list = list(range(0, gpu_count))

        # Set GPU split options
        if gpu_count == 1:
            self.gpu_split_auto = False
            logger.info("Disabling GPU split because one GPU is in use.")
        else:
            # TODO: Set tensor parallel

            # Set GPU split options
            # Enable manual GPU split if provided
            if gpu_split:
                self.gpu_split = gpu_split

                # Causes crash if set with GPU split
                # TODO: Remove when fixed in exllama upstream
                self.autosplit_reserve = None

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
                    value / 1024 for value in autosplit_reserve_megabytes
                ]

        if not hardware_supports_flash_attn(gpu_device_list):
            gpu_unsupported_message = (
                "Unable to run ExllamaV3 because an unsupported GPU is "
                "found in this configuration. \n"
                "All GPUs must be ampere "
                "(30 series) or newer. AMD GPUs are not supported."
            )

            logger.warning(gpu_unsupported_message)

            raise RuntimeError(gpu_unsupported_message)

        # Cache
        user_cache_size = unwrap(
            kwargs.get("cache_size"), self.generator_max_seq_len
        )  # Default to generator_max_seq_len
        self.cache_size = self.adjust_cache_size(user_cache_size)
        self.cache_mode = unwrap(kwargs.get("cache_mode"), "FP16")
        self.cache = self.create_cache(self.cache_mode, self.model)

        # Draft cache
        if self.use_draft_model:
            # Set draft cache mode
            self.draft_cache_mode = unwrap(draft_args.get("draft_cache_mode"), "FP16")
            self.draft_cache = self.create_cache(
                self.draft_cache_mode, self.draft_model
            )

        # Max batch size
        self.max_batch_size = unwrap(kwargs.get("max_batch_size"), 256)

        # Make sure chunk size is >= 256, keep near or below max seq len
        user_chunk_size = unwrap(kwargs.get("chunk_size"), 2048)
        self.chunk_size = self.adjust_chunk_size(user_chunk_size)

        # Template setup
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

        return self

    def adjust_cache_size(self, cache_size) -> int:
        """
        Adjust the cache size to ensure it meets requirements.

        Args:
            cache_size: The requested cache size in tokens.

        Returns:
            An adjusted cache size that's aligned to PAGE_SIZE and meets minimum requirements.
        """
        if cache_size < self.generator_max_seq_len:
            logger.warning(
                f"The given cache_size ({cache_size}) is smaller than the "
                f"generator's operational maximum sequence length ({self.generator_max_seq_len}).\n"
                f"Overriding cache_size to {self.generator_max_seq_len}. "
            )

            cache_size = self.generator_max_seq_len

        # Enforce a multiple of PAGE_SIZE for cache size
        # Overestimate to ensure that the cache isn't below generator_max_seq_len
        cache_remainder = cache_size % PAGE_SIZE
        if cache_remainder != 0:
            rounded_cache_size = int(
                PAGE_SIZE * ((cache_size - cache_remainder) / PAGE_SIZE + 1)
            )

            logger.warning(
                f"The given cache size ({cache_size}) is "
                f"not a multiple of {PAGE_SIZE}.\n"
                "Overriding cache_size with an overestimated value of "
                f"{rounded_cache_size} tokens."
            )

            cache_size = rounded_cache_size

        # Warn user if cache size may be inadequate for CFG
        if cache_size < 2 * self.generator_max_seq_len:
            logger.warning(
                f"The given cache_size ({cache_size}) is less than 2 * generator_max_seq_len "
                f"({2 * self.generator_max_seq_len}) and may be too small for requests using CFG. \n"
                "Ignore this warning if you do not plan on using CFG."
            )

        return cache_size

    def adjust_chunk_size(self, user_chunk_size: int) -> int:
        """
        Adjust the chunk size to ensure it meets requirements.

        Args:
            user_chunk_size: The requested chunk size in tokens.

        Returns:
            An adjusted chunk size that's aligned to PAGE_SIZE and within bounds.
        """
        chunk_size = sorted((PAGE_SIZE, user_chunk_size, self.generator_max_seq_len))[1]
        chunk_remainder = chunk_size % PAGE_SIZE
        if chunk_remainder != 0:
            rounded_chunk_size = int(
                PAGE_SIZE * ((chunk_size - chunk_remainder) / PAGE_SIZE + 1)
            )

            logger.warning(
                f"The given chunk size ({chunk_size}) is "
                f"not a multiple of {PAGE_SIZE}.\n"
                "Overriding chunk_size with an overestimated value of "
                f"{rounded_chunk_size} tokens."
            )

            chunk_size = rounded_chunk_size
        return chunk_size

    def create_cache(self, raw_cache_mode: str, model: Model) -> Cache:
        """
        Create a Cache object with the specified mode.

        Args:
            raw_cache_mode: The cache mode string ('FP16', 'Q4', 'Q6', 'Q8', or bits specification like '4,4').
            model: The model to create the cache for.

        Returns:
            A configured Cache object.
        """
        # Cast exl2 types to exl3
        match raw_cache_mode:
            case "Q4":
                raw_cache_mode = "4,4"
            case "Q6":
                raw_cache_mode = "6,6"
            case "Q8":
                raw_cache_mode = "8,8"

        split_cache_mode = re.search(r"^([2-8])\s*,\s*([2-8])$", raw_cache_mode)

        if split_cache_mode:
            draft_k_bits = int(split_cache_mode.group(1))
            draft_v_bits = int(split_cache_mode.group(2))
            cache = Cache(
                model,
                max_num_tokens=self.cache_size,
                layer_type=CacheLayer_quant,
                k_bits=draft_k_bits,
                v_bits=draft_v_bits,
            )
        else:
            cache = Cache(model, max_num_tokens=self.cache_size)

        return cache

    def supports_logprob_extraction(self) -> bool:
        """Check if this model container supports logprob extraction."""
        return True  # ExllamaV3 supports logprob extraction

    def supports_logit_bias(self) -> bool:  # noqa: D401
        """Return False as logit bias is not implemented."""

        return False

    def model_info(self) -> ModelCard:
        """Return metadata for this model."""
        return ModelCard(
            id=self.hf_model.model_name,
            created=self.hf_model.created_at_timestamp(),
            owned_by=self.hf_model.author(),
            object="model",
            parameters=ModelCardParameters(
                user_max_seq_len=self.user_max_seq_len,
                effective_max_seq_len=self.generator_max_seq_len,
                cache_size=self.cache_size,
                max_batch_size=self.max_batch_size,
                chunk_size=self.chunk_size,
                cache_mode=self.cache_mode,
                gpu_split_auto=self.gpu_split_auto,
                gpu_split=self.gpu_split,
                autosplit_reserve=self.autosplit_reserve,
                prompt_template=self.prompt_template.name
                if self.prompt_template
                else "Unknown",
                backend_name=self.__class__.__name__,
                supports_structured_output=self.supports_structured_output(),
                supports_logprob_extraction=self.supports_logprob_extraction(),
                supports_logit_bias=self.supports_logit_bias(),
                is_vision_model=self.hf_model.is_vision_model(),
                draft_model_name=self.draft_model.config.model_dir.name
                if self.draft_model and self.draft_model.config
                else None,
                draft_cache_mode=self.draft_cache_mode
                if self.use_draft_model
                else None,
            ),
        )

    async def wait_for_jobs(self, skip_wait: bool = False):
        """
        Polling to wait for any active generation jobs to complete.

        Args:
            skip_wait: If True, cancel jobs immediately instead of waiting.
        """

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

    async def load(self, progress_callback=None, **kwargs):
        """
        Loads the model into memory.

        Args:
            progress_callback: Optional callback for progress updates.
            **kwargs: Additional loading options.
        """

        async for _ in self.load_gen(progress_callback):
            pass

    async def load_gen(self, progress_callback=None, **kwargs):
        """
        Loads the model into memory, yielding progress updates.

        Args:
            progress_callback: Optional callback for progress updates.
            **kwargs: Additional loading options.

        Yields:
            Progress updates
        """

        try:
            await self.load_lock.acquire()

            # Wait for existing generation jobs to finish
            await self.wait_for_jobs(kwargs.get("skip_wait"))

            generator = self.load_model_sync(progress_callback)
            async for value in iterate_in_threadpool(generator):
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
        if self.use_draft_model:
            for value in self.draft_model.load_gen(
                reserve_per_device=self.autosplit_reserve,
                callback=progress_callback,
            ):
                if value:
                    yield value

        for value in self.model.load_gen(
            reserve_per_device=self.autosplit_reserve,
            use_per_device=self.gpu_split,
            callback=progress_callback,
        ):
            if value:
                yield value

    async def create_generator(self):
        """Create and save a Exllama generator class."""

        try:
            # Don't acquire locks unless a model is loaded
            if self.loaded:
                await self.load_lock.acquire()

                # Immediately cancel all jobs
                await self.wait_for_jobs(skip_wait=True)

            # Create new generator
            self.generator = AsyncGenerator(
                model=self.model,
                cache=self.cache,
                draft_model=self.draft_model,
                draft_cache=self.draft_cache,
                tokenizer=self.tokenizer,
                max_batch_size=self.max_batch_size,
                max_chunk_size=self.chunk_size,
                max_seq_len_override=self.generator_max_seq_len,
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

    async def unload(self, loras_only: bool = False, **kwargs):
        """
        Unloads the model and associated resources from memory.

        Args:
            loras_only: If True, only unload LoRAs.
            **kwargs: Additional unloading options (e.g., shutdown).
        """

        # Used when shutting down the server
        do_shutdown = kwargs.get("shutdown")

        try:
            if not do_shutdown:
                await self.load_lock.acquire()

                # Wait for other jobs to finish
                await self.wait_for_jobs(kwargs.get("skip_wait"))

            self.model.unload()
            self.model = None
            self.config = None
            self.cache = None
            self.tokenizer = None

            if self.use_draft_model:
                self.draft_model.unload()
                self.draft_model = None
                self.draft_config = None
                self.draft_cache = None

            # Cleanup the generator from any pending jobs
            if self.generator is not None:
                await self.generator.close()
                self.generator = None

            gc.collect()
            torch.cuda.empty_cache()

            logger.info("Model unloaded.")
        finally:
            if not do_shutdown:
                self.load_lock.release()

                async with self.load_condition:
                    self.load_condition.notify_all()

    def encode_tokens(self, text: str, **kwargs) -> List[int]:
        """
        Encodes a string of text into a list of token IDs.

        Args:
            text: The input text string.
            **kwargs: Backend-specific encoding options (e.g., add_bos_token).

        Returns:
            A list of integer token IDs.
        """

        return (
            self.tokenizer.encode(
                text,
                add_bos=unwrap(
                    kwargs.get("add_bos_token"), self.hf_model.add_bos_token()
                ),
                encode_special_tokens=unwrap(kwargs.get("encode_special_tokens"), True),
            )
            .flatten()
            .tolist()
        )

    def decode_tokens(self, ids: List[int], **kwargs) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            ids: A list of integer token IDs.
            **kwargs: Backend-specific decoding options (e.g., decode_special_tokens).

        Returns:
            The decoded text string.
        """
        ids_tensor = torch.as_tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, L)
        return self.tokenizer.decode(
            ids_tensor,
            decode_special_tokens=unwrap(kwargs.get("decode_special_tokens"), True),
        )[0]

    def _normalise_prompt_ids(self, prompt, add_bos: bool, as_tensor: bool = True):
        """
        Accepts str, List[int] or List[List[int]] and returns either:
        - (1, L) LongTensor on CPU if as_tensor=True
        - List[int] if as_tensor=False
        """
        if isinstance(prompt, str):
            ids = self.tokenizer.encode(
                prompt,
                add_bos=add_bos,
                encode_special_tokens=True,
            )
            if not as_tensor:
                if ids.dim() > 1:
                    ids = ids.flatten()
                return ids.tolist()
        elif isinstance(prompt, list):
            if len(prompt) == 1 and isinstance(prompt[0], list):
                prompt = prompt[0]  # unwrap [[...]]
            if not all(isinstance(x, int) for x in prompt):
                raise TypeError("Prompt list must contain ints.")

            ids_list = list(prompt)  # make a copy
            if add_bos and (not ids_list or ids_list[0] != self.tokenizer.bos_token_id):
                ids_list = [self.tokenizer.bos_token_id] + ids_list

            if not as_tensor:
                return ids_list

            ids = torch.tensor(ids_list, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported prompt type {type(prompt)}")

        if as_tensor and ids.dim() == 1:  # ensure (1, L) shape for tensor output
            ids = ids.unsqueeze(0)
        return ids

    def _stringify_prompt(self, prompt, ids):
        """Convert prompt to string for JSON serialization."""
        return (
            prompt
            if isinstance(prompt, str)
            else self.decode_tokens(ids.squeeze(0).tolist())
        )

    def decode_single(self, token_id: int) -> str:
        """
        Decodes a single token ID to a string, optimized for performance.

        Args:
            token_id: Integer token ID to decode.

        Returns:
            The decoded text string for this single token.
        """
        # Initialize token cache with LRU eviction if it doesn't exist
        if not hasattr(self, '_token_cache'):
            from functools import lru_cache
            # Create a larger cache for better performance with multiple choice evals
            self._decode_cache_func = lru_cache(maxsize=50000)(
                lambda tid: self.decode_tokens([tid], decode_special_tokens=True)
            )
            self._token_cache = {}
        
        # Check simple cache first
        if token_id in self._token_cache:
            return self._token_cache[token_id]
        
        # Use LRU cache for decoding
        token_str = self._decode_cache_func(token_id)
        
        # Still maintain a fast access dict for most recent tokens
        # Limit cache size to prevent unbounded memory growth
        if len(self._token_cache) > 10000:
            # Clear the simple cache when it gets too large
            self._token_cache.clear()
        
        self._token_cache[token_id] = token_str
        return token_str

    def get_special_tokens(
        self, add_bos_token: bool = True, ban_eos_token: bool = False
    ):
        """
        Gets special tokens used by the model/tokenizer.

        Args:
            **kwargs: Options like add_bos_token, ban_eos_token.

        Returns:
            A dictionary mapping special token names (e.g., 'bos_token', 'eos_token')
            to their string or ID representation.
        """

        return {
            "bos_token": self.tokenizer.bos_token if add_bos_token else "",
            "eos_token": self.tokenizer.eos_token if not ban_eos_token else "",
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
        }

    @torch.inference_mode()
    def compute_sequence_logprobs(
        self, prompt, params: BaseSamplerRequest, profile: bool = False,
        aggressive_memory_cleanup: bool = True
    ) -> Dict[str, Any]:
        """
        Computes log probabilities for all tokens in a sequence in one operation.

        Args:
            prompt: The input prompt (string, List[int], or List[List[int]]).
            params: Sampling parameters including logprobs (number of top alternatives to return).
            profile: If True, log timing information for performance debugging.
            aggressive_memory_cleanup: If True, aggressively free GPU memory during computation.

        Returns:
            A dictionary containing token log probabilities and other info.
            The 'top_logprobs' field is always present - it's a list of token alternatives
            with their log probabilities, or None for positions where alternatives weren't computed.
        """
        if profile:
            timings = {}
            start_total = time.time()
        
        if not self.loaded:
            raise RuntimeError("Model is not loaded. Cannot compute sequence logprobs.")
        if not self.tokenizer or not self.model:
            raise RuntimeError(
                "Tokenizer or model not initialized. Cannot compute sequence logprobs."
            )

        add_bos = unwrap(params.add_bos_token, self.hf_model.add_bos_token())
        ids = self._normalise_prompt_ids(prompt, add_bos)
        
        if profile:
            timings['tokenization'] = time.time() - start_total

        # ---- pick the model’s actual device ------------------------------------
        device = getattr(self.model, "device", None)
        if device is None:  # some builds don’t set .device
            try:
                device = next(self.model.parameters()).device  # fall back to first weight tensor
            except (StopIteration, AttributeError):
                device = torch.device("cpu")  # CPU model or unusual wrapper

        # ---- move the IDs only if necessary ------------------------------------
        if ids.device != device:
            ids = ids.to(device, non_blocking=True)


        # Ensure prompt isn't too long for the model
        if ids.size(1) > self.generator_max_seq_len:
            raise ValueError(
                f"Prompt length {ids.size(1)} exceeds model's maximum sequence "
                f"length {self.generator_max_seq_len}. Please use a shorter prompt."
            )

        # ExLlamaV3's tokenizer.encode already returns a CPU tensor.
        # If it were a list, we'd do: ids = torch.tensor(ids, dtype=torch.long)

        # Compute log softmax directly from forward pass to avoid storing large logits tensor
        if profile:
            start_forward = time.time()
        
        log_sm = torch.log_softmax(
            self.model.forward(
                ids,
                {
                    "attn_mode": "flash_attn_nc",
                    "position": 0,
                    "last_token_only": False,  # Need full sequence for logprob calculation
                },
            ),
            dim=-1,
            dtype=torch.float32  # Explicit dtype for numerical stability
        )  # Natural log
        
        if profile:
            timings['forward_pass'] = time.time() - start_forward

        # Validate tensor shape before processing
        if log_sm.dim() != 3:
            raise RuntimeError(
                f"Expected (B,T,V) logits but got shape {log_sm.shape}. "
                "Did last_token_only get forced on?"
            )

        seq_len = ids.size(1)
        chosen = ids[0]

        # Vectorised logprob extraction for each token P(t_i | t_<i)
        chosen_next = chosen[1:].to(log_sm.device).unsqueeze(-1)
        gathered = log_sm[0, :-1].gather(1, chosen_next).squeeze(-1)
        # Keep as tensor on GPU for now - defer CPU conversion
        token_logprobs_tensor = gathered
        
        # Calculate ranks for chosen tokens (position in sorted distribution)
        # Rank 1 = highest probability, rank 2 = second highest, etc.
        ranks = (log_sm[0, :-1] >= gathered.unsqueeze(-1)).sum(-1).add_(1)
        # Keep as tensor on GPU for now - defer CPU conversion
        token_ranks_tensor = ranks

        # Top-k alternatives
        k = 1 if params.logprobs is True else int(params.logprobs or 0)
        
        # Pre-allocate top_logprobs list to avoid resizing
        top_logprobs = [None] * seq_len  # Pre-allocate full list

        # Early exit if no top logprobs needed
        if k == 0:
            # List is already pre-allocated with None values
            pass
        else:
            # Process top-k if requested (k > 0)
            if profile:
                start_topk = time.time()
            actual_k = min(k, log_sm.size(-1))
            vals, idxs = torch.topk(log_sm[0, :-1], actual_k)
            
            # Batch decode all unique tokens at once for efficiency
            # First, collect all unique token IDs we'll need to decode
            unique_ids = set()
            # Add all top-k token IDs
            idxs_cpu = idxs.cpu()
            for i in range(seq_len - 1):
                for j in range(actual_k):
                    unique_ids.add(idxs_cpu[i, j].item())
            # Add all chosen token IDs (excluding the first one)
            for i in range(1, seq_len):
                unique_ids.add(chosen[i].item())
            
            # Batch decode all unique tokens
            id_to_str = {}
            for token_id in unique_ids:
                id_to_str[token_id] = self.decode_single(token_id)
            
            # Now build top_logprobs using pre-decoded strings
            vals_cpu = vals.cpu()
            chosen_lp_cpu = gathered.cpu()
            
            for i in range(seq_len - 1):
                current = {}
                for j in range(actual_k):
                    token_id = idxs_cpu[i, j].item()
                    token_str = id_to_str[token_id]
                    current[token_str] = vals_cpu[i, j].item()
                
                # Add chosen token if not in top-k
                chosen_token_id = chosen[i + 1].item()
                chosen_token_str = id_to_str[chosen_token_id]
                if chosen_token_str not in current:
                    current[chosen_token_str] = chosen_lp_cpu[i].item()
                
                top_logprobs[i + 1] = current  # Use direct indexing instead of append
            
            if profile:
                timings['topk_processing'] = time.time() - start_topk
            
            # Clean up top-k tensors
            del vals, idxs, vals_cpu, idxs_cpu
            if aggressive_memory_cleanup and device.type == 'cuda':
                torch.cuda.empty_cache()

        # Clean up log_sm tensor before decoding
        del log_sm, gathered
        if aggressive_memory_cleanup and device.type == 'cuda':
            torch.cuda.empty_cache()

        # Batch decode all prompt tokens efficiently
        if profile:
            start_decode = time.time()
        
        # Decode all prompt tokens using the cached decode_single method
        prompt_token_strings = [self.decode_single(tid) for tid in chosen.tolist()]

        # Calculate character offsets for each token (OpenAI compatibility)
        # Vectorized offset calculation
        token_lengths = [len(s) for s in prompt_token_strings]
        text_offsets = [0]
        for length in token_lengths[:-1]:
            text_offsets.append(text_offsets[-1] + length)
        
        if profile:
            timings['token_decode'] = time.time() - start_decode
            timings['total'] = time.time() - start_total
            logger.info(f"Logprob computation timings (seq_len={seq_len}): {timings}")

        # Convert tensors to lists only at the final return (deferred CPU conversion)
        token_logprobs = [None] + token_logprobs_tensor.cpu().tolist()
        token_ranks = [None] + token_ranks_tensor.cpu().tolist()
        
        return {
            "text": self._stringify_prompt(prompt, ids),
            "prompt_tokens": seq_len,
            "generated_tokens": 0,
            "prompt_token_strings": prompt_token_strings,
            "prompt_token_logprobs": token_logprobs,
            "prompt_token_ranks": token_ranks,
            "top_logprobs": top_logprobs,
            "offset": text_offsets,
            "finish_reason": "stop",
        }
    
    @torch.inference_mode()
    def compute_batch_logprobs(
        self, prompts: List[Union[str, List[int]]], params: BaseSamplerRequest,
        profile: bool = False, aggressive_memory_cleanup: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Computes log probabilities for multiple sequences in a single batch.
        Optimized for multiple choice evaluations.
        
        Args:
            prompts: List of prompts (strings or token lists) to evaluate
            params: Sampling parameters including logprobs settings
            profile: If True, log timing information
            aggressive_memory_cleanup: If True, aggressively free GPU memory
            
        Returns:
            List of dictionaries containing logprobs for each prompt
        """
        if profile:
            timings = {}
            start_total = time.time()
            
        if not self.loaded:
            raise RuntimeError("Model is not loaded. Cannot compute batch logprobs.")
            
        # Get model device
        device = getattr(self.model, "device", None)
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")
                
        add_bos = unwrap(params.add_bos_token, self.hf_model.add_bos_token())
        
        # Tokenize all prompts
        if profile:
            start_tokenize = time.time()
            
        all_ids = []
        max_length = 0
        for prompt in prompts:
            ids = self._normalise_prompt_ids(prompt, add_bos)
            all_ids.append(ids)
            max_length = max(max_length, ids.size(1))
            
        if profile:
            timings['tokenization'] = time.time() - start_tokenize
            
        # Pad sequences to same length for batching
        padded_ids = []
        attention_masks = []
        for ids in all_ids:
            seq_len = ids.size(1)
            if seq_len < max_length:
                # Pad with pad token or 0
                pad_token = getattr(self.tokenizer, 'pad_token_id', 0)
                padding = torch.full((1, max_length - seq_len), pad_token, dtype=ids.dtype, device=ids.device)
                padded = torch.cat([ids, padding], dim=1)
                # Create attention mask (1 for real tokens, 0 for padding)
                mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.bool, device=ids.device),
                    torch.zeros(max_length - seq_len, dtype=torch.bool, device=ids.device)
                ])
            else:
                padded = ids
                mask = torch.ones(max_length, dtype=torch.bool, device=ids.device)
            padded_ids.append(padded)
            attention_masks.append(mask)
            
        # Stack into batch
        batch_ids = torch.cat(padded_ids, dim=0).to(device)
        batch_masks = torch.stack(attention_masks).to(device)
        
        # Single forward pass for all sequences
        if profile:
            start_forward = time.time()
            
        log_sm = torch.log_softmax(
            self.model.forward(
                batch_ids,
                {
                    "attn_mode": "flash_attn_nc",
                    "position": 0,
                    "last_token_only": False,
                },
            ),
            dim=-1,
            dtype=torch.float32
        )
        
        if profile:
            timings['forward_pass'] = time.time() - start_forward
            
        # Process results for each sequence
        results = []
        k = 1 if params.logprobs is True else int(params.logprobs or 0)
        
        if profile:
            start_process = time.time()
            
        for i, (prompt, ids, mask) in enumerate(zip(prompts, all_ids, attention_masks)):
            seq_len = mask.sum().item()
            chosen = batch_ids[i, :seq_len]
            seq_log_sm = log_sm[i, :seq_len]
            
            # Extract logprobs for this sequence
            if seq_len > 1:
                chosen_next = chosen[1:].unsqueeze(-1)
                gathered = seq_log_sm[:-1].gather(1, chosen_next).squeeze(-1)
                token_logprobs_tensor = gathered
                
                # Calculate ranks
                ranks = (seq_log_sm[:-1] >= gathered.unsqueeze(-1)).sum(-1).add_(1)
                token_ranks_tensor = ranks
            else:
                token_logprobs_tensor = torch.tensor([], device=device)
                token_ranks_tensor = torch.tensor([], device=device)
                
            # Top-k processing if needed
            # Pre-allocate top_logprobs list
            top_logprobs = [None] * seq_len
            
            if k > 0 and seq_len > 1:
                actual_k = min(k, seq_log_sm.size(-1))
                vals, idxs = torch.topk(seq_log_sm[:-1], actual_k)
                
                # Batch decode tokens for this sequence
                unique_ids = set()
                for j in range(seq_len - 1):
                    for tok_idx in range(actual_k):
                        unique_ids.add(idxs[j, tok_idx].item())
                for j in range(1, seq_len):
                    unique_ids.add(chosen[j].item())
                    
                id_to_str = {tid: self.decode_single(tid) for tid in unique_ids}
                
                # Build top_logprobs using direct indexing
                for j in range(seq_len - 1):
                    current = {}
                    for tok_idx in range(actual_k):
                        token_id = idxs[j, tok_idx].item()
                        token_str = id_to_str[token_id]
                        current[token_str] = vals[j, tok_idx].item()
                        
                    chosen_token_id = chosen[j + 1].item()
                    chosen_token_str = id_to_str[chosen_token_id]
                    if chosen_token_str not in current:
                        current[chosen_token_str] = gathered[j].item()
                        
                    top_logprobs[j + 1] = current  # Direct indexing
            # else: List is already pre-allocated with None values
                
            # Decode tokens
            prompt_token_strings = [self.decode_single(tid) for tid in chosen.tolist()]
            
            # Calculate offsets
            token_lengths = [len(s) for s in prompt_token_strings]
            text_offsets = [0]
            for length in token_lengths[:-1]:
                text_offsets.append(text_offsets[-1] + length)
                
            # Convert tensors to lists (deferred CPU conversion)
            token_logprobs = [None] + token_logprobs_tensor.cpu().tolist()
            token_ranks = [None] + token_ranks_tensor.cpu().tolist()
            
            results.append({
                "text": self._stringify_prompt(prompt, ids),
                "prompt_tokens": seq_len,
                "generated_tokens": 0,
                "prompt_token_strings": prompt_token_strings,
                "prompt_token_logprobs": token_logprobs,
                "prompt_token_ranks": token_ranks,
                "top_logprobs": top_logprobs,
                "offset": text_offsets,
                "finish_reason": "stop",
            })
            
        # Clean up
        del log_sm, batch_ids, batch_masks
        if aggressive_memory_cleanup and device.type == 'cuda':
            torch.cuda.empty_cache()
            
        if profile:
            timings['result_processing'] = time.time() - start_process
            timings['total'] = time.time() - start_total
            logger.info(f"Batch logprob computation timings (batch_size={len(prompts)}, max_len={max_length}): {timings}")
            
        return results
    
    @torch.inference_mode()
    def compute_perplexity_efficient(
        self, token_ids: torch.Tensor, chunk_size: Optional[int] = None,
        aggressive_memory_cleanup: bool = True
    ) -> Tuple[float, int]:
        """
        Efficiently compute perplexity using cross-entropy loss.
        Processes entire sequences at once instead of token-by-token.
        
        Args:
            token_ids: Tensor of token IDs, shape (seq_len,) or (batch, seq_len)
            chunk_size: Maximum sequence length to process at once (default: model max)
            aggressive_memory_cleanup: If True, aggressively free memory after each chunk
            
        Returns:
            Tuple of (perplexity, number_of_tokens_evaluated)
        """
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")
            
        # Ensure 2D tensor
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            
        batch_size, seq_len = token_ids.shape
        
        # Get model device
        device = getattr(self.model, "device", None)
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")
                
        # Move to device if needed
        if token_ids.device != device:
            token_ids = token_ids.to(device)
            
        # Use model's max length if not specified
        if chunk_size is None:
            chunk_size = self.generator_max_seq_len - 1  # -1 for safety
            
        total_loss = 0.0
        total_tokens = 0
        
        # Process in chunks if sequence is too long
        for start_idx in range(0, seq_len - 1, chunk_size):
            end_idx = min(start_idx + chunk_size + 1, seq_len)  # +1 for targets
            chunk_ids = token_ids[:, start_idx:end_idx]
            
            # Forward pass to get logits
            logits = self.model.forward(
                chunk_ids,
                {
                    "attn_mode": "flash_attn_nc",
                    "position": start_idx,
                    "last_token_only": False,
                }
            )
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_ids[:, 1:].contiguous()
            
            # Use cross-entropy for efficient -log P(correct token)
            import torch.nn.functional as F
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            num_tokens = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += num_tokens
            
            # Clean up tensors from this chunk
            del logits, shift_logits, shift_labels, loss, chunk_ids
            
            # Clear CUDA cache after each chunk to prevent VRAM accumulation
            if aggressive_memory_cleanup and device.type == 'cuda':
                torch.cuda.empty_cache()
                # For very long sequences, also trigger Python garbage collection
                if total_tokens > 10000:  # Every ~10k tokens
                    gc.collect()
            
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity, total_tokens

    async def _safe_logprob_fallback(
        self, prompt: str, full_text: str, params: BaseSamplerRequest
    ) -> Optional[Dict[str, Any]]:
        """
        Safe wrapper for fallback logprob calculation with proper error handling.

        Args:
            prompt: The original prompt.
            full_text: The full text (prompt + generated text).
            params: Sampling parameters.

        Returns:
            Logprob data dictionary or None if calculation failed.
        """
        try:
            return self.compute_sequence_logprobs(full_text, params)
        except Exception as e:
            logger.warning(f"Fallback logprob calculation failed: {e}")
            return None

    async def generate(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        abort_event: Optional[asyncio.Event] = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.

        Args:
            request_id: Unique identifier for the generation request.
            prompt: The input prompt string.
            params: Sampling and generation parameters.
            abort_event: An asyncio Event to signal cancellation.
            mm_embeddings: Optional multimodal embeddings.

        Returns:
            A dictionary containing the generated text and metadata.
        """

        # Special case for logprob-only calculation
        logprob_only = params.extract_prompt_logprobs and params.max_tokens == 0
        if logprob_only:
            return self.compute_sequence_logprobs(prompt, params)

        # For normal generation, collect all chunks
        output_text = ""
        all_results_data = []
        last_chunk = {}
        prompt_tokens_count = 0
        generated_tokens_count = 0
        all_generated_token_ids = []
        all_token_logprobs = []
        all_alternative_logprobs = []

        async for chunk in self.generate_gen(
            request_id,
            prompt,
            params,
            abort_event=abort_event,
            mm_embeddings=mm_embeddings,
        ):
            output_text += chunk.get("text", "")
            all_results_data.append(chunk)
            last_chunk = chunk

            if not prompt_tokens_count:
                prompt_tokens_count = chunk.get("prompt_tokens", 0)

            generated_tokens_count = chunk.get(
                "generated_tokens", generated_tokens_count
            )

            # Collect token data for logprobs
            if chunk.get("token_ids_list"):
                all_generated_token_ids.extend(chunk["token_ids_list"])

            # Collect logprob data if available
            if "token_logprob" in chunk:
                all_token_logprobs.append(chunk["token_logprob"])
            if "alternative_logprobs" in chunk:
                all_alternative_logprobs.append(chunk["alternative_logprobs"])

        # Determine finish_reason
        finish_reason = last_chunk.get("finish_reason", "stop")
        if abort_event and abort_event.is_set():
            finish_reason = "abort"
        elif last_chunk.get("eos"):
            finish_reason = last_chunk.get("eos_reason", "stop")
        elif generated_tokens_count >= params.max_tokens:
            finish_reason = "length"

        final_generation_dict = {
            "text": output_text,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens_count,
            "generated_tokens": generated_tokens_count,
            "stop_str": last_chunk.get("stop_str"),
            "tool_calls": last_chunk.get("tool_calls"),
        }

        # Add logprob data if requested and available
        logprobs_enabled = params.logprobs is not None and (
            (isinstance(params.logprobs, bool) and params.logprobs)
            or (isinstance(params.logprobs, int) and params.logprobs > 0)
        )
        if logprobs_enabled:
            if all_token_logprobs:
                # Use collected logprobs from streaming
                generated_token_strings_list = [
                    self.decode_tokens([token_id])
                    for token_id in all_generated_token_ids
                ]
                final_generation_dict["token_logprobs_list"] = all_token_logprobs
                final_generation_dict["alternative_logprobs_list"] = (
                    all_alternative_logprobs
                )
                final_generation_dict["generated_token_strings_list"] = (
                    generated_token_strings_list
                )
            else:
                # Fallback to computing logprobs if not available from streaming
                # Only compute for generated tokens to avoid O(n²) complexity
                if output_text and generated_tokens_count > 0:
                    fallback_result = await self._safe_logprob_fallback(
                        prompt, f"{prompt}{output_text}", params
                    )
                    if fallback_result:
                        # Extract only the generated portion
                        all_tokens = fallback_result.get("prompt_token_strings", [])
                        all_logprobs = fallback_result.get("prompt_token_logprobs", [])
                        all_top_logprobs = fallback_result.get("top_logprobs", [])

                        # Skip prompt tokens, only include generated tokens
                        prompt_token_count = prompt_tokens_count
                        if len(all_tokens) > prompt_token_count:
                            final_generation_dict["generated_token_strings_list"] = (
                                all_tokens[prompt_token_count:]
                            )
                            final_generation_dict["token_logprobs_list"] = all_logprobs[
                                prompt_token_count:
                            ]
                            final_generation_dict["alternative_logprobs_list"] = (
                                all_top_logprobs[prompt_token_count:]
                            )

        return final_generation_dict

    async def stream_generate(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        abort_event: Optional[asyncio.Event] = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generates a response iteratively (streaming) for a given prompt.

        Args:
            request_id: Unique identifier for the generation request.
            prompt: The input prompt string.
            params: Sampling and generation parameters.
            abort_event: An asyncio Event to signal cancellation.
            mm_embeddings: Optional multimodal embeddings.

        Yields:
            Generation chunks containing text and metadata.
        """

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

    def handle_finish_chunk(self, result: dict, generation: dict) -> dict:
        """
        Handle the final chunk of a generation.

        Args:
            result: The final result from the generator.
            generation: The current generation state.

        Returns:
            A formatted finish chunk with appropriate metadata.
        """
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
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Actual generator for the generation process.

        Args:
            request_id: Unique identifier for the generation request.
            prompt: The input prompt string.
            params: Sampling and generation parameters.
            abort_event: An asyncio Event to signal cancellation.
            mm_embeddings: Optional multimodal embeddings.

        Yields:
            Generation chunks containing text and metadata.
        """

        # Build sampler settings
        sampler_builder = ExllamaV3SamplerBuilder()

        # Check if temperature is 0 (greedy)
        # Temperature can be None, so coalesce to 1.0 for default
        temp = coalesce(params.temperature, 1.0)
        is_greedy = temp == 0

        if not is_greedy:
            sampler_builder.temperature(temp)
            sampler_builder.top_k(params.top_k)
            sampler_builder.top_p(params.top_p)
            sampler_builder.min_p(params.min_p)
            sampler_builder.penalties(
                params.repetition_penalty,
                params.frequency_penalty,
                params.presence_penalty,
                params.penalty_range,
                params.repetition_decay,
            )
        else:
            # Ensure top_k is at least 1 for greedy, otherwise exl3 might error or misbehave.
            # Typically greedy implies top_k = 1.
            sampler_builder.top_k(max(params.top_k, 1))

        sampler_settings = sampler_builder.build(greedy=is_greedy)

        # Create a job
        # Use shared helper to normalize prompt - generate_gen needs List[int]
        actual_prompt_token_ids = self._normalise_prompt_ids(
            prompt,
            add_bos=unwrap(params.add_bos_token, self.hf_model.add_bos_token()),
            as_tensor=False,
        )

        if not actual_prompt_token_ids:
            logger.error(
                f"Empty token sequence for prompt: {prompt}. Cannot start generation."
            )
            raise ValueError("Cannot create AsyncJob with empty token IDs from prompt.")

        # AsyncJob expects input_ids as a CPU tensor, typically (batch_size, sequence_length)
        # For generate_gen, batch_size is implicitly 1.
        final_input_ids_for_job = torch.tensor(
            [actual_prompt_token_ids], dtype=torch.long
        ).to("cpu")
        
        # Calculate context length
        context_len = len(actual_prompt_token_ids)
        
        # Automatically set max_tokens to fill up the context
        # This should be an OK default, but may be changed in the future
        max_tokens = unwrap(
            params.max_tokens,
            self.generator_max_seq_len - context_len,
        )
        if max_tokens < 1:
            logger.warning("max_tokens must be a positive integer, setting to 1.")
            max_tokens = 1

        job = AsyncJob(
            self.generator,
            input_ids=final_input_ids_for_job,  # pass the tensor on CPU
            max_new_tokens=max_tokens,
            min_new_tokens=params.min_tokens,
            sampler=sampler_settings,
            stop_conditions=params.stop,
            token_healing=params.token_healing,
            decode_special_tokens=False,  # We handle this manually for finer control
            # NOTE: Requesting return_probs or return_top_tokens > 0 relies on the
            # ExLlamaV3 library's AsyncJob.receive_logits method correctly implementing
            # the TODO sections for populating these values. As of the last review,
            # these TODOs were not implemented, potentially leading to missing logprob data
            # or errors if the library is not patched/updated.
            return_probs=params.logprobs is not None
            and (
                (isinstance(params.logprobs, bool) and params.logprobs)
                or (isinstance(params.logprobs, int) and params.logprobs > 0)
            ),  # Request probs if logprobs is enabled
            return_top_tokens=getattr(params, "top_logprobs", None)
            or (
                params.logprobs
                if isinstance(params.logprobs, int) and params.logprobs > 0
                else 0
            ),  # Number of top tokens from top_logprobs or fallback to logprobs
            identifier=request_id,
            banned_strings=params.banned_strings,
        )

        # Log generation parameters
        log_generation_params(
            request_id=request_id, **params.model_dump(exclude={"prompt"})
        )
        log_prompt(request_id, prompt)

        stream_start_time = time.time()
        tokens_generated = 0
        output_text = ""
        finish_reason = None
        stop_str = None

        # Initialize event_dict before try block to ensure it exists in finally
        event_dict = {}

        try:
            async for event_dict in job:
                if abort_event and abort_event.is_set():
                    logger.info(f"Generation {request_id} aborted by API.")
                    await job.cancel()
                    finish_reason = "abort"
                    break

                # Common information
                event_type = event_dict.get("event")
                text_chunk = event_dict.get("text", "")
                eos = event_dict.get("eos", False)
                token_ids = event_dict.get(
                    "token_ids"
                )  # List of token IDs for this chunk

                generation_chunk = {
                    "text": text_chunk,
                    "prompt_tokens": context_len,
                    "generated_tokens": event_dict.get("generated_tokens", 0),
                    "offset": len(output_text),
                }
                
                # Update output_text for offset calculation
                output_text += text_chunk

                # Add optional fields for logprobs support
                if token_ids is not None:
                    generation_chunk["token_ids_list"] = token_ids.tolist()

                # Extract logprobs from the ExllamaV3 event if available
                logprobs_enabled = params.logprobs is not None and (
                    (isinstance(params.logprobs, bool) and params.logprobs)
                    or (isinstance(params.logprobs, int) and params.logprobs > 0)
                )
                if logprobs_enabled:
                    # Try to get logprobs from the event data
                    # NOTE: The availability and correctness of the following fields (token_probs,
                    # top_k_tokens, top_k_probs) from event_dict depend on the ExLlamaV3
                    # library's AsyncJob.receive_logits method. If that method has not
                    # been patched to implement its TODOs for logprob calculation,
                    # these fields may be None or missing, even if requested.
                    token_probs_tensor = event_dict.get("token_probs")
                    top_k_tokens = event_dict.get("top_k_tokens")
                    top_k_probs = event_dict.get("top_k_probs")

                    if (
                        token_probs_tensor is not None
                        and token_probs_tensor.numel() > 0
                    ):
                        # Get the probability of the sampled token
                        sampled_token_probability = token_probs_tensor[0].item()
                        # Convert to natural log (ExllamaV3 may return probabilities)
                        if sampled_token_probability > 0:
                            # Convert to natural log
                            generation_chunk["token_logprob"] = math.log(
                                sampled_token_probability
                            )
                        else:
                            # Assign -inf for zero probabilities (robustness)
                            generation_chunk["token_logprob"] = float("-inf")

                    # Get top-k alternatives if available and requested
                    top_logprobs_count = getattr(params, "top_logprobs", None)
                    if top_logprobs_count is None:
                        # Fallback to logprobs value if it's an integer
                        top_logprobs_count = (
                            params.logprobs if isinstance(params.logprobs, int) else 0
                        )

                    if (
                        top_logprobs_count > 0
                        and top_k_tokens is not None
                        and top_k_probs is not None
                    ):
                        alternative_logprobs_dict = {}
                        num_alternatives = min(top_logprobs_count, top_k_tokens.numel())

                        for i in range(num_alternatives):
                            try:
                                token_id = top_k_tokens[i].item()
                                prob_val = top_k_probs[i].item()
                                token_str = self.decode_single(token_id)
                                if prob_val > 0:
                                    # Convert to natural log
                                    alternative_logprobs_dict[token_str] = math.log(
                                        prob_val
                                    )
                                else:
                                    # Assign -inf for zero probabilities (robustness)
                                    alternative_logprobs_dict[token_str] = float("-inf")
                            except (IndexError, ValueError) as e:
                                logger.debug(f"Error processing top-k token {i}: {e}")
                                continue

                        if alternative_logprobs_dict:
                            generation_chunk["alternative_logprobs"] = (
                                alternative_logprobs_dict
                            )

                yield generation_chunk
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
                eos_token_id=self.hf_model.eos_tokens(),
                prompt=prompt,
                **params.model_dump(exclude={"prompt"}),
            )

            # Log the metrics if present
            if event_dict.get("metrics"):
                log_metrics(
                    request_id,
                    event_dict.get("time_enqueued"),
                    event_dict.get("prompt_tokens"),
                    event_dict.get("cached_tokens"),
                    event_dict.get("time_prefill"),
                    event_dict.get("new_tokens"),
                    event_dict.get("time_generate"),
                    event_dict.get("prompt_tokens"),
                    self.generator_max_seq_len,
                )

    @torch.inference_mode()
    async def get_logprobs_for_chosen_tokens(
        self, prompt: str, choice_token_ids: List[int]
    ) -> List[float]:
        """
        Computes the natural log probabilities of a list of specified target tokens
        immediately following a given prompt.
        Used for lm-evaluation-harness multiple-choice.

        Args:
            prompt: The input prompt string.
            choice_token_ids: List of token IDs to calculate probabilities for.

        Returns:
            A list of log probabilities, one for each token in choice_token_ids.
        """
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")
        if not self.tokenizer or not self.model:
            raise RuntimeError(
                "Tokenizer or model not available for logprob calculation."
            )
        if not choice_token_ids:
            logger.warning(
                "get_logprobs_for_chosen_tokens called with empty choice_token_ids."
            )
            return []

        # Tokenize the prompt (ExLlamaV3 tokenizer returns CPU tensor by default)
        # Use model's default BOS behavior, which is typically what lm-eval harness expects for conditioning.
        prompt_ids = self.tokenizer.encode(
            prompt, add_bos=self.hf_model.add_bos_token(), encode_special_tokens=True
        )

        # normalise shape to (1, L) - some tokenizer versions return 1D tensor
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if prompt_ids.numel() == 0:
            logger.error(
                "Prompt tokenized to an empty sequence in get_logprobs_for_chosen_tokens. "
                "This can happen if the prompt is empty and add_bos_token is False. "
                "Cannot condition on an empty sequence for next token probabilities."
            )
            return [float("-inf")] * len(choice_token_ids)

        current_seq_len = prompt_ids.size(1)
        if current_seq_len > self.generator_max_seq_len:
            raise ValueError(
                f"Prompt length {current_seq_len} exceeds model's pre-configured maximum sequence length {self.generator_max_seq_len}."
            )

        # Perform a single forward pass of the prompt to get logits for the *next* token.
        # ExLlamaV3 model handles tensor device placement internally.
        # The `forward` method with last_token_only=True (default) returns logits for the token *after* the input sequence.
        # So, logits will be for the position immediately following the prompt.
        logits = self.model.forward(
            prompt_ids,
            {"attn_mode": "flash_attn_nc", "position": 0, "last_token_only": True},
        )

        # If last_token_only=True, logits shape should be (batch_size, vocab_size)
        # Otherwise, it would be (batch_size, seq_len, vocab_size)
        if logits.dim() == 2 and logits.size(0) == 1:  # (batch, vocab_size)
            next_token_logits = logits[0, :]  # Direct access, no need to get position 0
        elif logits.dim() == 3 and logits.size(1) == 1:  # (batch, 1, vocab_size)
            next_token_logits = logits[0, 0, :]  # Squeeze out the sequence length of 1
        elif (
            logits.dim() == 3 and logits.size(1) == current_seq_len
        ):  # (batch, seq_len, vocab_size)
            # This happens if model.forward doesn't respect last_token_only, or if flash_attn_nc changes behavior.
            # We need the logits from the *last* position of the input sequence to predict the next token.
            next_token_logits = logits[0, -1, :]
        else:
            logger.error(
                f"Unexpected logits shape from model.forward: {logits.shape}. Expected (1, vocab_size) or (1, 1, vocab_size) or (1, {current_seq_len}, vocab_size)."
            )
            return [float("-inf")] * len(choice_token_ids)

        log_softmax_for_next_token = torch.log_softmax(
            next_token_logits, dim=-1
        )  # Natural log

        results = []
        vocab_size = log_softmax_for_next_token.size(-1)
        for token_id in choice_token_ids:
            if 0 <= token_id < vocab_size:
                results.append(log_softmax_for_next_token[token_id].item())
            else:
                logger.warning(
                    f"Choice token ID {token_id} is out of vocab range ({vocab_size}). Assigning -inf logprob."
                )
                results.append(float("-inf"))

        return results
