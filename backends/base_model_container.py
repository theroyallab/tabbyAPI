import abc
import asyncio
import pathlib
from loguru import logger
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
)

from common.multimodal import MultimodalEmbeddingWrapper
from common.sampling import BaseSamplerRequest
from common.templating import PromptTemplate
from common.transformers_utils import GenerationConfig
from endpoints.core.types.model import ModelCard


class BaseModelContainer(abc.ABC):
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
    load_lock: asyncio.Lock
    load_condition: asyncio.Condition

    # Required methods
    @classmethod
    @abc.abstractmethod
    async def create(cls, model_directory: pathlib.Path, **kwargs):
        """
        Asynchronously creates and initializes a model container instance.

        Args:
            model_directory: Path to the model files.
            **kwargs: Backend-specific configuration options.

        Returns:
            An instance of the implementing class.
        """

        pass

    @abc.abstractmethod
    async def load(self, progress_callback=None, **kwargs):
        """
        Loads the model into memory.

        Args:
            progress_callback: Optional callback for progress updates.
            **kwargs: Additional loading options.
        """

        pass

    # NOTE: Might be an optional method
    @abc.abstractmethod
    async def load_gen(self, progress_callback=None, **kwargs) -> AsyncIterator[Any]:
        """
        Loads the model into memory, yielding progress updates.

        Args:
            progress_callback: Optional callback for progress updates.
            **kwargs: Additional loading options.

        Yields:
            Progress updates
        """

        if False:
            yield

    @abc.abstractmethod
    async def unload(self, loras_only: bool = False, **kwargs):
        """
        Unloads the model and associated resources from memory.

        Args:
            loras_only: If True, only unload LoRAs.
            **kwargs: Additional unloading options (e.g., shutdown).
        """

        pass

    @abc.abstractmethod
    def encode_tokens(self, text: str, **kwargs) -> List[int]:
        """
        Encodes a string of text into a list of token IDs.

        Args:
            text: The input text string.
            **kwargs: Backend-specific encoding options (e.g., add_bos_token).

        Returns:
            A list of integer token IDs.
        """

        pass

    @abc.abstractmethod
    def decode_tokens(self, ids: List[int], **kwargs) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            ids: A list of integer token IDs.
            **kwargs: Backend-specific decoding options (e.g., decode_special_tokens).

        Returns:
            The decoded text string.
        """

        pass

    @abc.abstractmethod
    def get_special_tokens(self, **kwargs) -> Dict[str, Any]:
        """
        Gets special tokens used by the model/tokenizer.

        Args:
            **kwargs: Options like add_bos_token, ban_eos_token.

        Returns:
            A dictionary mapping special token names (e.g., 'bos_token', 'eos_token')
            to their string or ID representation.
        """

        pass

    @abc.abstractmethod
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

        pass

    @abc.abstractmethod
    async def generate_gen(
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

        if False:
            yield

    @abc.abstractmethod
    def model_info(self) -> ModelCard:
        """
        Returns a dictionary of the current model's configuration parameters.

        Returns:
            Model parameters provided by the backend
        """

        pass

    @abc.abstractmethod
    async def wait_for_jobs(self, skip_wait: bool = False):
        """
        Waits for any active generation jobs to complete.

        Args:
            skip_wait: If True, cancel jobs immediately instead of waiting.
        """

        pass

    # Optional methods
    async def load_loras(
        self, lora_directory: pathlib.Path, **kwargs
    ) -> Dict[str, List[str]]:
        """
        Loads LoRA adapters. Base implementation does nothing or raises error.

        Args:
            lora_directory: Path to the directory containing LoRA files.
            **kwargs: LoRA configuration (e.g., list of loras, scaling).

        Returns:
            A dictionary indicating success/failure for each LoRA.
        """

        logger.warning("LoRA loading not implemented for this backend.")  # type: ignore
        return {
            "success": [],
            "failure": [
                lora.get("name", "unknown") for lora in kwargs.get("loras", [])
            ],
        }

    def get_loras(self) -> List[Any]:
        """
        Gets the currently loaded LoRA adapters. Base implementation returns empty list.

        Returns:
            A list representing the loaded LoRAs (backend-specific format).
        """

        return []
