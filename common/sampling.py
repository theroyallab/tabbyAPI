"""Common functions for sampling parameters"""

import aiofiles
import json
import pathlib
from pydantic_core import ValidationError
from ruamel.yaml import YAML
from copy import deepcopy
from common.logger import xlogger
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from typing import Dict, List, Optional, Union

from common.utils import filter_none_values, unwrap


# Params that are accepted for API compatibility but not implemented by the
# exllamav3 backend, mapped to the neutral value that leaves them inactive.
# Requests that activate any of these get a warning and the param is ignored.
UNSUPPORTED_PARAMS = {
    "ban_eos_token": False,
    "allowed_tokens": [],
    "smoothing_factor": 0.0,
    "top_a": 0.0,
    "tfs": 1.0,
    "typical": 1.0,
    "skew": 0.0,
    "mirostat_mode": 0,
    "temp_exponent": 1.0,
}


# Common class for sampler params
class BaseSamplerRequest(BaseModel):
    """Common class for sampler params that are used in APIs"""

    max_tokens: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("max_tokens"),
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens", "max_length"),
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

    top_a: Optional[float] = Field(default_factory=lambda: get_default_sampler_value("top_a", 0.0))

    min_p: Optional[float] = Field(default_factory=lambda: get_default_sampler_value("min_p", 0.0))

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
        ge=0.0,
        le=1.0,
    )

    xtc_threshold: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("xtc_threshold", 0.1),
        ge=0.0,
        le=1.0,
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
        description=("Aliases: repetition_range, repetition_penalty_range, rep_pen_range"),
    )

    repetition_decay: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("repetition_decay", 0)
    )

    dry_multiplier: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("dry_multiplier", 0.0),
        description="DRY repetition penalty scale. 0 disables DRY.",
        examples=[0.8],
        ge=0,
    )

    dry_base: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("dry_base", 1.75),
        description="Base of the exponential DRY penalty growth per repeated token.",
        examples=[1.75],
        ge=0,
    )

    dry_allowed_length: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("dry_allowed_length", 2),
        description="Longest repeated sequence DRY leaves unpenalized.",
        examples=[2],
        ge=0,
    )

    dry_range: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("dry_range", 0),
        description="Number of recent tokens DRY scans. 0 means the whole context.",
        examples=[0],
    )

    dry_penalty_last_n: Optional[int] = Field(
        None,
        description=(
            "llama.cpp-style DRY window: -1 scans the whole context, 0 disables DRY, "
            "a positive value scans that many recent tokens. Takes precedence over dry_range."
        ),
        examples=[-1],
    )

    dry_sequence_breakers: Optional[Union[str, List[str]]] = Field(
        default_factory=lambda: get_default_sampler_value("dry_sequence_breakers", []),
        description=(
            "Strings that repeated sequences cannot span; every token containing one "
            "of them is a breaker. An empty list uses the backend's default set "
            "(punctuation, brackets, quotes, newlines and special tokens)."
        ),
        examples=[["\n", ":", '"', "*"]],
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

    json_schema: Optional[object] = Field(
        default_factory=lambda: get_default_sampler_value("json_schema"),
    )

    regex_pattern: Optional[str] = Field(
        default_factory=lambda: get_default_sampler_value("regex_pattern"),
    )

    grammar_string: Optional[str] = Field(
        default_factory=lambda: get_default_sampler_value("grammar_string"),
        description=(
            "Constrain generation with a context-free grammar in Lark or "
            "llama.cpp GBNF syntax (auto-detected)."
        ),
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

    # Valid for OAI requests
    top_logprobs: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("top_logprobs", 0),
        ge=0,
    )

    adaptive_target: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("adaptive_target", 1.0)
    )

    adaptive_decay: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("adaptive_decay", 0.9)
    )

    loop_detect_window: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("loop_detect_window", 800),
        description=(
            "ExLlamaV3 only. Stops generation when the last N tokens are made "
            "up of a repeating pattern. Set 0 or null to disable."
        ),
        ge=0,
    )

    stall_guard_tokens: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("stall_guard_tokens", 8),
        description=(
            "ExLlamaV3 only. Re-rails grammar-constrained generation "
            "(json_schema, regex_pattern, grammar_string) out of legal "
            "whitespace runs: after N consecutive whitespace-only tokens while "
            "the grammar is still incomplete, the next logit mask excludes "
            "whitespace so sampling lands on a legal content token or EOS. "
            "Set 0 or null to disable."
        ),
        ge=0,
    )

    def param_source(self, name: str) -> str:
        """
        Where the effective value of a sampler param came from: "req" (sent with the
        request), "forced" or "preset" (sampler override preset), or "default".
        """

        override = overrides_container.overrides.get(name)
        if isinstance(override, dict) and override.get("override") and override.get("force"):
            return "forced"

        if name in self.model_fields_set:
            return "req"

        if isinstance(override, dict) and override.get("override") is not None:
            return "preset"

        return "default"

    def get_stop_on_loop(self) -> tuple[int, int] | None:
        """Get ExLlamaV3 loop detection parameters."""

        if self.loop_detect_window and self.loop_detect_window > 1:
            return self.loop_detect_window, 2

        return None

    @field_validator("top_k", mode="before")
    def convert_top_k(cls, v):
        """Fixes instance if Top-K is -1."""

        if v == -1:
            xlogger.warning("Provided a top-k value of -1. Converting to 0 instead.")
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
            xlogger.warning("Could not parse DRY sequence breakers. Using an empty array.")
            return []  # Return empty list if parsing fails

    @model_validator(mode="after")
    def after_validate(self):
        # For OAI requests, logprobs is a boolean and top_logprobs is integer
        # if self.logprobs and self.top_logprobs:
        #     self.logprobs = self.top_logprobs

        # FIXME: find a better way to register this
        # Maybe make a function to assign values to the
        # model if they do not exist post creation
        apply_forced_sampler_overrides(self)

        if self.min_temp and self.max_temp and self.min_temp > self.max_temp:
            raise ValidationError("min temp cannot be more then max temp")

        if self.min_tokens and self.max_tokens and self.min_tokens > self.max_tokens:
            raise ValidationError("min tokens cannot be more then max tokens")

        # llama.cpp's window parameter uses 0 for "off" where dry_range uses 0
        # for "whole context", so it maps onto dry_range explicitly
        if self.dry_penalty_last_n is not None:
            if self.dry_penalty_last_n == 0:
                self.dry_multiplier = 0.0
            else:
                self.dry_range = max(self.dry_penalty_last_n, 0)

        self.warn_unsupported_params()

        return self

    def warn_unsupported_params(self):
        """Warn when the request activates params the backend doesn't implement."""

        active = [
            name
            for name, neutral in UNSUPPORTED_PARAMS.items()
            if getattr(self, name) and getattr(self, name) != neutral
        ]

        # Dynamic temperature is only in effect when the bounds differ
        if (
            self.min_temp is not None
            and self.max_temp is not None
            and self.min_temp != self.max_temp
        ):
            active.append("min_temp/max_temp")

        if active:
            xlogger.warning(
                "Ignoring sampler params not supported by the exllamav3 backend: "
                + ", ".join(active)
            )


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


def resolve_preset_path(preset_name: str) -> Optional[pathlib.Path]:
    """
    Locate a sampler override preset in the sampler_overrides folder.

    The name is accepted with or without its .yml/.yaml extension, so both
    "my_preset" and "my_preset.yml" find the same file.
    """

    override_directory = pathlib.Path("sampler_overrides")
    name = preset_name.strip()

    candidates = [override_directory / name]
    if not name.lower().endswith((".yml", ".yaml")):
        candidates += [override_directory / f"{name}.yml", override_directory / f"{name}.yaml"]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def describe_overrides(overrides: dict) -> str:
    """One-line summary of a set of sampler overrides, e.g. "temperature 0.8, top_k 40"."""

    parts = []
    for key, value in overrides.items():
        if not isinstance(value, dict) or "override" not in value:
            continue

        item = f"{key} {value['override']}"
        if value.get("force"):
            item += " (forced)"
        elif value.get("additive"):
            item += " (additive)"
        parts.append(item)

    return ", ".join(parts) if parts else "no overrides"


async def overrides_from_file(preset_name: str):
    """Fetches an override preset from a file"""

    preset_path = resolve_preset_path(preset_name)
    if preset_path is None:
        raise FileNotFoundError(
            f'Sampler override preset "{preset_name}" was not found in the '
            "sampler_overrides folder."
        )

    overrides_container.selected_preset = preset_path.stem
    async with aiofiles.open(preset_path, "r", encoding="utf8") as raw_preset:
        contents = await raw_preset.read()

        # Create a temporary YAML parser
        yaml = YAML(typ="safe")
        preset = yaml.load(contents)
        overrides_from_dict(preset)

        xlogger.info(
            f'Sampler preset "{preset_path.stem}": '
            + describe_overrides(overrides_container.overrides),
            {"preset": preset},
        )


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

    # Tolerate older OAI standard for logprobs
    if isinstance(params.logprobs, int) and params.logprobs > 1:
        params.top_logprobs = params.logprobs

    for var, value in overrides_container.overrides.items():
        if var not in BaseSamplerRequest.model_fields:
            xlogger.warning(f'Skipping unknown sampler override key "{var}"')
            continue

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
