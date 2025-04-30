import asyncio
import gc
import math
import pathlib
from loguru import logger
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
)

import torch

from backends.base_model_container import BaseModelContainer
from common.concurrency import iterate_in_threadpool
from common.multimodal import MultimodalEmbeddingWrapper
from common.sampling import BaseSamplerRequest
from common.templating import PromptTemplate, find_prompt_template
from common.transformers_utils import GenerationConfig
from common.utils import unwrap
from endpoints.core.types.model import ModelCard

from exllamav3 import AsyncGenerator, AsyncJob, Config, Model, Cache, Tokenizer


class ExllamaV3Container(BaseModelContainer):
    """Abstract base class for model containers."""

    # Exposed model information
    model_dir: pathlib.Path = pathlib.Path("models")
    prompt_template: Optional[PromptTemplate] = None
    generation_config: Optional[GenerationConfig] = None

    # Load synchronization
    # The bool is a master switch for accepting requests
    # The lock keeps load tasks sequential
    # The condition notifies any waiting tasks
    active_job_ids: Dict[str, Any] = {}
    loaded: bool = False
    load_lock: asyncio.Lock = asyncio.Lock()
    load_condition: asyncio.Condition = asyncio.Condition()

    # Exl3 vars
    model: Model
    cache: Cache
    tokenizer: Tokenizer
    config: Config
    gpu_split: List[float] | None = None
    gpu_split_auto: bool = True
    autosplit_reserve: List[float] = [96 / 1024]
    max_seq_len: int
    use_tp: bool = False

    # Required methods
    @classmethod
    async def create(cls, model_directory: pathlib.Path, **kwargs):
        """
        Asynchronously creates and initializes a model container instance.

        Args:
            model_directory: Path to the model files.
            **kwargs: Backend-specific configuration options.

        Returns:
            An instance of the implementing class.
        """

        self = cls()

        logger.warning(
            "ExllamaV3 is currently in an alpha state. "
            "Please note that all config options may not work."
        )

        self.config = Config.from_directory(model_directory.resolve())
        self.model = Model.from_config(self.config)
        self.tokenizer = Tokenizer.from_config(self.config)

        self.max_seq_len = kwargs.get("max_seq_len")
        self.cache = Cache(self.model, max_num_tokens=self.max_seq_len)

        # Try to set prompt template
        self.prompt_template = await find_prompt_template(
            kwargs.get("prompt_template"), model_directory
        )

        # Turn off GPU split if the user is using 1 GPU
        gpu_count = torch.cuda.device_count()
        gpu_split_auto = unwrap(kwargs.get("gpu_split_auto"), True)
        gpu_split = unwrap(kwargs.get("gpu_split"), None)

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
            elif gpu_split_auto and not self.use_tp:
                # Otherwise fallback to autosplit settings
                self.gpu_split_auto = gpu_split_auto

                autosplit_reserve_megabytes = unwrap(
                    kwargs.get("autosplit_reserve"), [96]
                )

                # Reserve VRAM for each GPU
                self.autosplit_reserve = [
                    int(math.ceil(value/1024))
                    for value in autosplit_reserve_megabytes
                ]
        # TODO: speculative decoding

        return self

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
            async for module, modules in iterate_in_threadpool(generator):
                yield module, modules

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

    # TODO: Add draft loading
    @torch.inference_mode()
    def load_model_sync(self, progress_callback=None):
        for value in self.model.load_gen(
            reserve_per_device=self.autosplit_reserve,
            use_per_device=self.gpu_split,
            callback=progress_callback
        ):
            if value:
                yield value

    async def unload(self, loras_only: bool = False, **kwargs):
        """
        Unloads the model and associated resources from memory.

        Args:
            loras_only: If True, only unload LoRAs.
            **kwargs: Additional unloading options (e.g., shutdown).
        """

        try:
            await self.load_lock.acquire()

            # Wait for other jobs to finish
            await self.wait_for_jobs(kwargs.get("skip_wait"))

            self.model.unload()
            self.model = None

            self.config = None
            self.cache = None
            self.tokenizer = None

            gc.collect()
            torch.cuda.empty_cache()

            logger.info("Model unloaded.")
        finally:
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

        return self.tokenizer.encode(
            text,
            add_bos=unwrap(kwargs.get("add_bos_token"), True),
            encode_special_tokens=unwrap(kwargs.get("encode_special_tokens"), True),
        ).flatten().tolist()

    def decode_tokens(self, ids: List[int], **kwargs) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            ids: A list of integer token IDs.
            **kwargs: Backend-specific decoding options (e.g., decode_special_tokens).

        Returns:
            The decoded text string.
        """

        ids = torch.tensor([ids])
        return self.tokenizer.decode(
            ids,
            decode_special_tokens=unwrap(kwargs.get("decode_special_tokens"), True),
        )[0]

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

    def model_info(self) -> ModelCard:
        """
        Returns a dictionary of the current model's configuration parameters.

        Returns:
            Model parameters provided by the backend
        """

        pass

    async def wait_for_jobs(self, skip_wait: bool = False):
        """
        Waits for any active generation jobs to complete.

        Args:
            skip_wait: If True, cancel jobs immediately instead of waiting.
        """

        pass

    async def generate(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        abort_event: Optional[asyncio.Event] = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ) -> Dict[str, Any]:
        """
        Generates a complete response for a given prompt and parameters.

        Args:
            request_id: Unique identifier for the generation request.
            prompt: The input prompt string.
            params: Sampling and generation parameters.
            abort_event: An asyncio Event to signal cancellation.
            mm_embeddings: Optional multimodal embeddings.

        Returns:
            A dictionary containing the generation info
        """

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
            Generation chunks
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
        chunk_tokens: torch.Tensor | tuple[torch.Tensor, torch.Tensor]

        prompts = [prompt]
        stop_conditions = params.stop
        add_bos_token = params.add_bos_token

        # Fetch EOS tokens from generation_config if they exist
        eos_tokens = (
            self.generation_config.eos_tokens()
            if self.generation_config
            else [self.tokenizer.eos_token_id]
        )

        stop_conditions += eos_tokens

        input_ids = [
            self.tokenizer.encode(
                prompt,
                add_bos=add_bos_token,
                encode_special_tokens=True,
            )
            for prompt in prompts
        ]

        # The first index will always be the positive prompt
        context_len = input_ids[0].size(dim=-1)

        # Automatically set max_tokens to fill up the context
        # This should be an OK default, but may be changed in the future
        max_tokens = unwrap(
            params.max_tokens,
            self.max_seq_len - context_len,
        )
        if max_tokens < 1:
            logger.warning("max_tokens must be a positive integer, setting to 1.")
            max_tokens = 1

        # Determine if the negative context or the context length is bigger
        context_to_check = context_len

        # Check total length of prompt against max context length
        if context_to_check > self.max_seq_len:
            preamble = "Prompt"

            raise ValueError(
                f"{preamble} length {context_to_check} is greater than "
                f"max_seq_len {self.max_seq_len}"
            )

        self.generator = AsyncGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )

        generation = {}
        print(max_tokens)
        job = AsyncJob(
            self.generator,
            input_ids=self.tokenizer.encode(prompt, add_bos=False),
            max_new_tokens=max_tokens,
            stop_conditions=stop_conditions,
        )
        generated_tokens = 0
        full_response = ""
        async for result in job:
            chunk = unwrap(result.get("text"), "")
            if chunk:
                chunk_tokens = result.get("token_ids", self.tokenizer.encode(chunk))
                full_response += chunk
                if isinstance(chunk_tokens, torch.Tensor):
                    generated_tokens += chunk_tokens.size(dim=0)
                generation = {
                    "text": chunk,
                    "prompt_tokens": context_len,
                    "generated_tokens": generated_tokens,
                    "offset": len(full_response),
                }
                yield generation

            if result.get("eos"):
                generation = self.handle_finish_chunk(result, generation)
                yield generation
        # Assign the active job to the request ID
        self.active_job_ids[request_id] = job
