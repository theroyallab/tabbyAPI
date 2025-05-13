import aiofiles
import json
import pathlib
from loguru import logger
from pydantic import BaseModel
from typing import Dict, List, Optional, Set, Union


class GenerationConfig(BaseModel):
    """
    An abridged version of HuggingFace's GenerationConfig.
    Will be expanded as needed.
    """

    eos_token_id: Optional[Union[int, List[int]]] = None

    @classmethod
    async def from_directory(cls, model_directory: pathlib.Path):
        """Create an instance from a generation config file."""

        generation_config_path = model_directory / "generation_config.json"
        async with aiofiles.open(
            generation_config_path, "r", encoding="utf8"
        ) as generation_config_json:
            contents = await generation_config_json.read()
            generation_config_dict = json.loads(contents)
            return cls.model_validate(generation_config_dict)

    def eos_tokens(self):
        """Wrapper method to fetch EOS tokens."""

        if isinstance(self.eos_token_id, list):
            return self.eos_token_id
        elif isinstance(self.eos_token_id, int):
            return [self.eos_token_id]
        else:
            return []


class HuggingFaceConfig(BaseModel):
    """
    DEPRECATED: Currently a stub and doesn't do anything.

    An abridged version of HuggingFace's model config.
    Will be expanded as needed.
    """

    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization_config: Optional[Dict] = None

    @classmethod
    async def from_directory(cls, model_directory: pathlib.Path):
        """Create an instance from a generation config file."""

        hf_config_path = model_directory / "config.json"
        async with aiofiles.open(
            hf_config_path, "r", encoding="utf8"
        ) as hf_config_json:
            contents = await hf_config_json.read()
            hf_config_dict = json.loads(contents)
            return cls.model_validate(hf_config_dict)

    def quant_method(self):
        """Wrapper method to fetch quant type"""

        if isinstance(self.quantization_config, Dict):
            return self.quantization_config.get("quant_method")
        else:
            return None

    def eos_tokens(self):
        """Wrapper method to fetch EOS tokens."""

        if isinstance(self.eos_token_id, list):
            return self.eos_token_id
        elif isinstance(self.eos_token_id, int):
            return [self.eos_token_id]
        else:
            return []


class TokenizerConfig(BaseModel):
    """
    An abridged version of HuggingFace's tokenizer config.
    """

    add_bos_token: Optional[bool] = True

    @classmethod
    async def from_directory(cls, model_directory: pathlib.Path):
        """Create an instance from a tokenizer config file."""

        tokenizer_config_path = model_directory / "tokenizer_config.json"
        async with aiofiles.open(
            tokenizer_config_path, "r", encoding="utf8"
        ) as tokenizer_config_json:
            contents = await tokenizer_config_json.read()
            tokenizer_config_dict = json.loads(contents)
            return cls.model_validate(tokenizer_config_dict)


class HFModel:
    """
    Unified container for HuggingFace model configuration files.
    These are abridged for hyper-specific model parameters not covered
    by most backends.

    Includes:
      - config.json
      - generation_config.json
      - tokenizer_config.json
    """

    hf_config: HuggingFaceConfig
    tokenizer_config: Optional[TokenizerConfig] = None
    generation_config: Optional[GenerationConfig] = None

    @classmethod
    async def from_directory(cls, model_directory: pathlib.Path):
        """Create an instance from a model directory"""

        self = cls()

        # A model must have an HF config
        try:
            self.hf_config = await HuggingFaceConfig.from_directory(model_directory)
        except Exception as exc:
            raise ValueError(
                f"Failed to load config.json from {model_directory}"
            ) from exc

        try:
            self.generation_config = await GenerationConfig.from_directory(
                model_directory
            )
        except Exception:
            logger.warning(
                "Generation config file not found in model directory, skipping."
            )

        try:
            self.tokenizer_config = await TokenizerConfig.from_directory(
                model_directory
            )
        except Exception:
            logger.warning(
                "Tokenizer config file not found in model directory, skipping."
            )

        return self

    def quant_method(self):
        """Wrapper for quantization method"""

        return self.hf_config.quant_method()

    def eos_tokens(self):
        """Combines and returns EOS tokens from various configs"""

        eos_ids: Set[int] = set()

        eos_ids.update(self.hf_config.eos_tokens())

        if self.generation_config:
            eos_ids.update(self.generation_config.eos_tokens())

        # Convert back to a list
        return list(eos_ids)

    def add_bos_token(self):
        """Wrapper for tokenizer config"""

        if self.tokenizer_config:
            return self.tokenizer_config.add_bos_token

        # Expected default
        return True
