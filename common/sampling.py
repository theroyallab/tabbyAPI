"""Common functions for sampling parameters"""

import pathlib
from typing import Dict, List, Optional, Union
from pydantic import AliasChoices, BaseModel, Field
import yaml

from common.logger import init_logger
from common.utils import unwrap


logger = init_logger(__name__)


# Common class for sampler params
class SamplerParams(BaseModel):
    """Common class for sampler params that are used in APIs"""

    max_tokens: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("max_tokens", 150)
    )

    stop: Optional[Union[str, List[str]]] = Field(
        default_factory=lambda: get_default_sampler_value("stop", [])
    )

    token_healing: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("token_healing", False)
    )

    temperature: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("temperature", 1.0)
    )

    temperature_last: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("temperature_last", False)
    )

    top_k: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("top_k", 0)
    )

    top_p: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("top_p", 1.0)
    )

    top_a: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("top_a", 0.0)
    )

    min_p: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("min_p", 0.0)
    )

    tfs: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("tfs", 0.0)
    )

    frequency_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("frequency_penalty", 0.0)
    )

    presence_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("presence_penalty", 0.0)
    )

    repetition_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("repetition_penalty", 1.0)
    )

    repetition_decay: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("repetition_decay", 0)
    )

    mirostat_mode: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("mirostat_mode", 0)
    )

    mirostat_tau: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("mirostat_tau", 1.5)
    )

    mirostat_eta: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("mirostat_eta", 0.3)
    )

    add_bos_token: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("add_bos_token", True)
    )

    ban_eos_token: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("ban_eos_token", False)
    )

    logit_bias: Optional[Dict[int, float]] = Field(
        default_factory=lambda: get_default_sampler_value("logit_bias"),
        examples=[[{"1": 10}]],
    )

    negative_prompt: Optional[str] = Field(
        default_factory=lambda: get_default_sampler_value("negative_prompt")
    )

    # Aliased variables
    typical: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("typical", 1.0),
        validation_alias=AliasChoices("typical", "typical_p"),
        description="Aliases: typical_p",
    )

    penalty_range: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("penalty_range", -1),
        validation_alias=AliasChoices(
            "penalty_range",
            "repetition_range",
            "repetition_penalty_range",
        ),
        description="Aliases: repetition_range, repetition_penalty_range",
    )

    cfg_scale: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("cfg_scale", 1.0),
        validation_alias=AliasChoices("cfg_scale", "guidance_scale"),
        description="Aliases: guidance_scale",
    )

    def to_gen_params(self):
        """Converts samplers to internal generation params"""

        # Add forced overrides if present
        apply_forced_sampler_overrides(self)

        # Convert stop to an array of strings
        if isinstance(self.stop, str):
            self.stop = [self.stop]

        return {
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "add_bos_token": self.add_bos_token,
            "ban_eos_token": self.ban_eos_token,
            "token_healing": self.token_healing,
            "logit_bias": self.logit_bias,
            "temperature": self.temperature,
            "temperature_last": self.temperature_last,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "top_a": self.top_a,
            "typical": self.typical,
            "min_p": self.min_p,
            "tfs": self.tfs,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "penalty_range": self.penalty_range,
            "repetition_decay": self.repetition_decay,
            "mirostat": self.mirostat_mode == 2,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "cfg_scale": self.cfg_scale,
            "negative_prompt": self.negative_prompt,
        }


# Global for default overrides
DEFAULT_OVERRIDES = {}


def set_overrides_from_dict(new_overrides: dict):
    """Wrapper function to update sampler overrides"""

    global DEFAULT_OVERRIDES

    if isinstance(new_overrides, dict):
        DEFAULT_OVERRIDES = new_overrides
    else:
        raise TypeError("new sampler overrides must be a dict!")


def get_overrides_from_file(preset_name: str):
    """Fetches an override preset from a file"""

    preset_path = pathlib.Path(f"sampler_overrides/{preset_name}.yml")
    if preset_path.exists():
        with open(preset_path, "r", encoding="utf8") as raw_preset:
            preset = yaml.safe_load(raw_preset)
            set_overrides_from_dict(preset)

            logger.info("Applied sampler overrides from file.")
    else:
        logger.warn(
            f"Sampler override file named \"{preset_name}\" was not found. "
            + "Make sure it's located in the sampler_overrides folder."
        )


# TODO: Maybe move these into the class
# Classmethods aren't recognized in pydantic default_factories
def get_default_sampler_value(key, fallback=None):
    """Gets an overridden default sampler value"""

    return unwrap(DEFAULT_OVERRIDES.get(key, {}).get("override"), fallback)


def apply_forced_sampler_overrides(params: SamplerParams):
    """Forcefully applies overrides if specified by the user"""

    for var, value in DEFAULT_OVERRIDES.items():
        override = value.get("override")
        force = unwrap(value.get("force"), False)
        if force and override:
            setattr(params, var, override)
