import json
import pathlib
from typing import List, Optional, Union
from loguru import logger
from pydantic import BaseModel


class GenerationConfig(BaseModel):
    """
    An abridged version of HuggingFace's GenerationConfig.
    Will be expanded as needed.
    """

    eos_token_id: Optional[Union[int, List[int]]] = None
    bad_words_ids: Optional[List[List[int]]] = None

    @classmethod
    def from_file(self, model_directory: pathlib.Path):
        """Create an instance from a generation config file."""

        generation_config_path = model_directory / "generation_config.json"
        with open(
            generation_config_path, "r", encoding="utf8"
        ) as generation_config_json:
            generation_config_dict = json.load(generation_config_json)
            return self.model_validate(generation_config_dict)

    def eos_tokens(self):
        """Wrapper method to fetch EOS tokens."""

        if isinstance(self.eos_token_id, int):
            return [self.eos_token_id]
        else:
            return self.eos_token_id


class HuggingFaceConfig(BaseModel):
    """
    An abridged version of HuggingFace's model config.
    Will be expanded as needed.
    """

    badwordsids: Optional[str] = None

    @classmethod
    def from_file(self, model_directory: pathlib.Path):
        """Create an instance from a generation config file."""

        hf_config_path = model_directory / "config.json"
        with open(hf_config_path, "r", encoding="utf8") as hf_config_json:
            hf_config_dict = json.load(hf_config_json)
            return self.model_validate(hf_config_dict)

    def get_badwordsids(self):
        """Wrapper method to fetch badwordsids."""

        if self.badwordsids:
            try:
                bad_words_list = json.loads(self.badwordsids)
                return bad_words_list
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping badwordsids from config.json "
                    "since it's not a valid array."
                )

                return []
        else:
            return []
