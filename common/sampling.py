"""Common functions for sampling parameters"""

import aiofiles
import json
import pathlib
from pydantic_core import ValidationError
from ruamel.yaml import YAML
from copy import deepcopy
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from typing import Dict, List, Optional, Union

from common.utils import filter_none_values, unwrap


# Common class for sampler params
class BaseSamplerRequest(BaseModel):
    """Common class for sampler params that are used in APIs"""

    max_tokens: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("max_tokens"),
        validation_alias=AliasChoices(
            "max_tokens", "max_completion_tokens", "max_length"
        ),
        description="Aliases: max_length",
        examples=[150],
        ge=0,
    )

    min_tokens: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("min_tokens", 0),
        validation_alias=AliasChoices("min_tokens", "min_length"),
        description="Aliases: min_length",
        examples=[0],
        ge=0,
    )

    stop: Optional[Union[str, List[Union[str, int]]]] = Field(
        default_factory=lambda: get_default_sampler_value("stop", []),
        validation_alias=AliasChoices("stop", "stop_sequence"),
        description="Aliases: stop_sequence",
    )

    banned_strings: Optional[Union[str, List[str]]] = Field(
        default_factory=lambda: get_default_sampler_value("banned_strings", [])
    )

    banned_tokens: Optional[Union[List[int], str]] = Field(
        default_factory=lambda: get_default_sampler_value("banned_tokens", []),
        validation_alias=AliasChoices("banned_tokens", "custom_token_bans"),
        description="Aliases: custom_token_bans",
        examples=[[128, 330]],
    )

    allowed_tokens: Optional[Union[List[int], str]] = Field(
        default_factory=lambda: get_default_sampler_value("allowed_tokens", []),
        validation_alias=AliasChoices("allowed_tokens", "allowed_token_ids"),
        description="Aliases: allowed_token_ids",
        examples=[[128, 330]],
    )

    token_healing: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("token_healing", False)
    )

    temperature: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("temperature", 1.0),
        examples=[1.0],
        ge=0,
        le=10,
    )

    temperature_last: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("temperature_last", False),
    )

    smoothing_factor: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("smoothing_factor", 0.0),
        ge=0,
    )

    top_k: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("top_k", 0),
        ge=-1,
    )

    top_p: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("top_p", 1.0),
        ge=0,
        le=1,
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

    typical: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("typical", 1.0),
        validation_alias=AliasChoices("typical", "typical_p"),
        description="Aliases: typical_p",
        examples=[1.0],
        gt=0,
        le=1,
    )

    skew: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("skew", 0.0),
        examples=[0.0],
    )

    xtc_probability: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("xtc_probability", 0.0),
    )

    xtc_threshold: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("xtc_threshold", 0.1)
    )

    frequency_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("frequency_penalty", 0.0),
        ge=0,
    )

    presence_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("presence_penalty", 0.0),
        ge=0,
    )

    repetition_penalty: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("repetition_penalty", 1.0),
        validation_alias=AliasChoices("repetition_penalty", "rep_pen"),
        description="Aliases: rep_pen",
        examples=[1.0],
        gt=0,
    )

    penalty_range: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("penalty_range", -1),
        validation_alias=AliasChoices(
            "penalty_range",
            "repetition_range",
            "repetition_penalty_range",
            "rep_pen_range",
        ),
        description=(
            "Aliases: repetition_range, repetition_penalty_range, rep_pen_range"
        ),
    )

    repetition_decay: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("repetition_decay", 0)
    )

    dry_multiplier: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("dry_multiplier", 0.0)
    )

    dry_base: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("dry_base", 0.0)
    )

    dry_allowed_length: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("dry_allowed_length", 0)
    )

    dry_range: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("dry_range", 0),
        validation_alias=AliasChoices("dry_range", "dry_penalty_last_n"),
        description=("Aliases: dry_penalty_last_n"),
    )

    dry_sequence_breakers: Optional[Union[str, List[str]]] = Field(
        default_factory=lambda: get_default_sampler_value("dry_sequence_breakers", [])
    )

    mirostat_mode: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("mirostat_mode", 0),
        alias=AliasChoices("mirostat_mode", "mirostat"),
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
        default_factory=lambda: get_default_sampler_value("add_bos_token")
    )

    ban_eos_token: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("ban_eos_token", False),
        validation_alias=AliasChoices("ban_eos_token", "ignore_eos"),
        description="Aliases: ignore_eos",
        examples=[False],
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

    regex_pattern: Optional[str] = Field(
        default_factory=lambda: get_default_sampler_value("regex_pattern"),
    )

    grammar_string: Optional[str] = Field(
        default_factory=lambda: get_default_sampler_value("grammar_string"),
    )

    speculative_ngram: Optional[bool] = Field(
        default_factory=lambda: get_default_sampler_value("speculative_ngram"),
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
        ge=0,
    )

    min_temp: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("min_temp", 1.0),
        validation_alias=AliasChoices("min_temp", "dynatemp_low"),
        description="Aliases: dynatemp_low",
        examples=[1.0],
        ge=0,
    )

    temp_exponent: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("temp_exponent", 1.0),
        validation_alias=AliasChoices("temp_exponent", "dynatemp_exponent"),
        examples=[1.0],
        ge=0,
    )

    logprobs: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("logprobs", 0),
        ge=0,
    )

    @field_validator("top_k", mode="before")
    def convert_top_k(cls, v):
        """Fixes instance if Top-K is -1."""

        if v == -1:
            logger.warning("Provided a top-k value of -1. Converting to 0 instead.")
            return 0

        return v

    @field_validator("stop", "banned_strings", mode="before")
    def convert_str_to_list(cls, v):
        """Convert single string to list of strings."""

        if isinstance(v, str):
            return [v]

        return v

    @field_validator("banned_tokens", "allowed_tokens", mode="before")
    def convert_tokens_to_int_list(cls, v):
        """Convert comma-separated string of numbers to a list of integers."""

        if isinstance(v, str):
            return [int(x) for x in v.replace(" ", "").split(",") if x.isdigit()]

        return v

    @field_validator("dry_sequence_breakers", mode="before")
    def parse_json_if_needed(cls, v):
        """Parse dry_sequence_breakers string to JSON array."""

        if isinstance(v, str) and not v.startswith("["):
            v = f"[{v}]"

        try:
            return json.loads(v) if isinstance(v, str) else v
        except Exception:
            logger.warning(
                "Could not parse DRY sequence breakers. Using an empty array."
            )
            return []  # Return empty list if parsing fails

    @model_validator(mode="after")
    def after_validate(self):
        # FIXME: find a better way to register this
        # Maybe make a function to assign values to the
        # model if they do not exist post creation
        apply_forced_sampler_overrides(self)

        if self.min_temp and self.max_temp and self.min_temp > self.max_temp:
            raise ValidationError("min temp cannot be more then max temp")

        if self.min_tokens and self.max_tokens and self.min_tokens > self.max_tokens:
            raise ValidationError("min tokens cannot be more then max tokens")

        return self


class SamplerOverridesContainer(BaseModel):
    selected_preset: Optional[str] = None
    overrides: dict = {}


# Global for default overrides
overrides_container = SamplerOverridesContainer()


def overrides_from_dict(new_overrides: dict):
    """Wrapper function to update sampler overrides"""

    if isinstance(new_overrides, dict):
        overrides_container.overrides = filter_none_values(new_overrides)
    else:
        raise TypeError("New sampler overrides must be a dict!")


async def overrides_from_file(preset_name: str):
    """Fetches an override preset from a file"""

    preset_path = pathlib.Path(f"sampler_overrides/{preset_name}.yml")
    if preset_path.exists():
        overrides_container.selected_preset = preset_path.stem
        async with aiofiles.open(preset_path, "r", encoding="utf8") as raw_preset:
            contents = await raw_preset.read()

            # Create a temporary YAML parser
            yaml = YAML(typ="safe")
            preset = yaml.load(contents)
            overrides_from_dict(preset)

            logger.info("Applied sampler overrides from file.")
    else:
        error_message = (
            f'Sampler override file named "{preset_name}" was not found. '
            + "Make sure it's located in the sampler_overrides folder."
        )

        raise FileNotFoundError(error_message)


def get_all_presets():
    """Fetches all sampler override presets from the overrides directory"""

    override_directory = pathlib.Path("sampler_overrides")
    preset_files = [file.stem for file in override_directory.glob("*.yml")]

    return preset_files


# TODO: Maybe move these into the class
# Classmethods aren't recognized in pydantic default_factories
def get_default_sampler_value(key, fallback=None):
    """Gets an overridden default sampler value"""

    default_value = unwrap(
        deepcopy(overrides_container.overrides.get(key, {}).get("override")),
        fallback,
    )

    return default_value


def apply_forced_sampler_overrides(params: BaseSamplerRequest):
    """Forcefully applies overrides if specified by the user"""

    for var, value in overrides_container.overrides.items():
        override = deepcopy(value.get("override"))
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
