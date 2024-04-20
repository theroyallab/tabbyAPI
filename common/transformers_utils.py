import json
import pathlib
from typing import List, Optional, Union
from pydantic import BaseModel


class GenerationConfig(BaseModel):
    """
    An abridged version of HuggingFace's GenerationConfig.
    Will be expanded as needed.
    """

    eos_token_id: Optional[Union[int, List[int]]] = None

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
