"""Common functions for sampling parameters"""

import pathlib
import yaml
from loguru import logger
from pydantic import AliasChoices, BaseModel, Field
from typing import Dict, List, Optional, Union

from common.utils import unwrap, prune_dict


# Common class for sampler params
class BaseSamplerRequest(BaseModel):
    """Common class for sampler params that are used in APIs"""

    max_tokens: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("max_tokens"),
        examples=[150],
    )

    min_tokens: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("min_tokens", 0),
        examples=[0],
    )

    generate_window: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("generate_window"),
        examples=[512],
    )

    stop: Optional[Union[str, List[str]]] = Field(
        default_factory=lambda: get_default_sampler_value("stop", [])
    )

    banned_strings: Optional[Union[str, List[str]]] = Field(
        default_factory=lambda: get_default_sampler_value("banned_strings", [])
    )

    token_healing: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("token_healing", False)
    )

    temperature: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("temperature", 1.0),
        examples=[1.0],
    )

    temperature_last: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("temperature_last", False)
    )

    smoothing_factor: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("smoothing_factor", 0.0),
    )

    top_k: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("top_k", 0),
    )

    top_p: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("top_p", 1.0),
        examples=[1.0],
    )

    top_a: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("top_a", 0.0)
    )

    min_p: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("min_p", 0.0)
    )

    tfs: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("tfs", 1.0),
        examples=[1.0],
    )

    frequency_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("frequency_penalty", 0.0)
    )

    presence_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("presence_penalty", 0.0)
    )

    repetition_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("repetition_penalty", 1.0),
        examples=[1.0],
    )

    repetition_decay: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("repetition_decay", 0)
    )

    mirostat_mode: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("mirostat_mode", 0)
    )

    mirostat_tau: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("mirostat_tau", 1.5),
        examples=[1.5],
    )

    mirostat_eta: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("mirostat_eta", 0.3),
        examples=[0.3],
    )

    add_bos_token: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("add_bos_token", True)
    )

    ban_eos_token: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("ban_eos_token", False),
        examples=[False],
    )

    skip_special_tokens: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("ban_eos_token", True),
        examples=[True],
    )

    logit_bias: Optional[Dict[int, float]] = Field(
        default_factory=lambda: get_default_sampler_value("logit_bias"),
        examples=[{"1": 10, "2": 50}],
    )

    negative_prompt: Optional[str] = Field(
        default_factory=lambda: get_default_sampler_value("negative_prompt")
    )

    json_schema: Optional[object] = Field(
        default_factory=lambda: get_default_sampler_value("json_schema"),
    )

    grammar_string: Optional[str] = Field(
        default_factory=lambda: get_default_sampler_value("grammar_string"),
    )

    speculative_ngram: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("speculative_ngram"),
    )

    # Aliased variables
    typical: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("typical", 1.0),
        validation_alias=AliasChoices("typical", "typical_p"),
        description="Aliases: typical_p",
        examples=[1.0],
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
        examples=[1.0],
    )

    max_temp: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("max_temp", 1.0),
        validation_alias=AliasChoices("max_temp", "dynatemp_high"),
        description="Aliases: dynatemp_high",
        examples=[1.0],
    )

    min_temp: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("min_temp", 1.0),
        validation_alias=AliasChoices("min_temp", "dynatemp_low"),
        description="Aliases: dynatemp_low",
        examples=[1.0],
    )

    temp_exponent: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("temp_exponent", 1.0),
        validation_alias=AliasChoices("temp_exponent", "dynatemp_exponent"),
        examples=[1.0],
    )

    banned_tokens: Optional[Union[List[int], str]] = Field(
        default_factory=lambda: get_default_sampler_value("banned_tokens", []),
        validation_alias=AliasChoices("banned_tokens", "custom_token_bans"),
        description="Aliases: custom_token_bans",
        examples=[[128, 330]],
    )

    # TODO: Return back to adaptable class-based validation But that's just too much
    # abstraction compared to simple if statements at the moment
    def validate_params(self):
        """
        Validates sampler parameters to be within sane ranges.
        """

        # Temperature
        if self.temperature < 0.0:
            raise ValueError(
                "Temperature must be a non-negative value. " f"Got {self.temperature}"
            )

        # Smoothing factor
        if self.smoothing_factor < 0.0:
            raise ValueError(
                "Smoothing factor must be a non-negative value. "
                f"Got {self.smoothing_factor}"
            )

        # Top K
        if self.top_k < 0:
            raise ValueError("Top K must be a non-negative value. " f"Got {self.top_k}")

        # Top P
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("Top P must be in [0, 1]. " f"Got {self.top_p}")

        # Repetition Penalty
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                "Repetition penalty must be a positive value. "
                f"Got {self.repetition_penalty}"
            )

        # Typical
        if self.typical <= 0 and self.typical > 1:
            raise ValueError("Typical must be in (0, 1]. " f"Got {self.typical}")

        # Dynatemp values
        if self.max_temp < 0.0:
            raise ValueError(
                "Max temp must be a non-negative value. ", f"Got {self.max_temp}"
            )

        if self.min_temp < 0.0:
            raise ValueError(
                "Min temp must be a non-negative value. ", f"Got {self.min_temp}"
            )

        if self.temp_exponent < 0.0:
            raise ValueError(
                "Temp exponent must be a non-negative value. ",
                f"Got {self.temp_exponent}",
            )

    def to_gen_params(self, **kwargs):
        """Converts samplers to internal generation params"""

        # Add forced overrides if present
        apply_forced_sampler_overrides(self)

        self.validate_params()

        # Convert stop to an array of strings
        if self.stop and isinstance(self.stop, str):
            self.stop = [self.stop]

        # Convert banned_strings to an array of strings
        if self.banned_strings and isinstance(self.banned_strings, str):
            self.banned_strings = [self.banned_strings]

        # Convert string banned tokens to an integer list
        if self.banned_tokens and isinstance(self.banned_tokens, str):
            self.banned_tokens = [
                int(x) for x in self.banned_tokens.split(",") if x.isdigit()
            ]

        gen_params = {
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "generate_window": self.generate_window,
            "stop": self.stop,
            "banned_strings": self.banned_strings,
            "add_bos_token": self.add_bos_token,
            "ban_eos_token": self.ban_eos_token,
            "skip_special_tokens": self.skip_special_tokens,
            "token_healing": self.token_healing,
            "logit_bias": self.logit_bias,
            "banned_tokens": self.banned_tokens,
            "temperature": self.temperature,
            "temperature_last": self.temperature_last,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "temp_exponent": self.temp_exponent,
            "smoothing_factor": self.smoothing_factor,
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
            "json_schema": self.json_schema,
            "grammar_string": self.grammar_string,
            "speculative_ngram": self.speculative_ngram,
        }

        return {**gen_params, **kwargs}


# Global for default overrides
overrides = {}


def overrides_from_dict(new_overrides: dict):
    """Wrapper function to update sampler overrides"""

    global overrides

    if isinstance(new_overrides, dict):
        overrides = prune_dict(new_overrides)
    else:
        raise TypeError("New sampler overrides must be a dict!")


def overrides_from_file(preset_name: str):
    """Fetches an override preset from a file"""

    preset_path = pathlib.Path(f"sampler_overrides/{preset_name}.yml")
    if preset_path.exists():
        with open(preset_path, "r", encoding="utf8") as raw_preset:
            preset = yaml.safe_load(raw_preset)
            overrides_from_dict(preset)

            logger.info("Applied sampler overrides from file.")
    else:
        error_message = (
            f'Sampler override file named "{preset_name}" was not found. '
            + "Make sure it's located in the sampler_overrides folder."
        )

        raise FileNotFoundError(error_message)


# TODO: Maybe move these into the class
# Classmethods aren't recognized in pydantic default_factories
def get_default_sampler_value(key, fallback=None):
    """Gets an overridden default sampler value"""

    return unwrap(overrides.get(key, {}).get("override"), fallback)


def apply_forced_sampler_overrides(params: BaseSamplerRequest):
    """Forcefully applies overrides if specified by the user"""

    for var, value in overrides.items():
        override = value.get("override")
        original_value = getattr(params, var, None)

        # Force takes precedence over additive
        # Additive only works on lists and doesn't remove duplicates
        if override:
            if unwrap(value.get("force"), False):
                setattr(params, var, override)
            elif (
                unwrap(value.get("additive"), False)
                and isinstance(override, list)
                and isinstance(original_value, list)
            ):
                setattr(params, var, override + original_value)
