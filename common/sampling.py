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
    PrivateAttr,
    ValidationInfo,
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

# Settings that can differ between a response's reasoning block and its
# content. Given as a nested set of sampler settings under this key, in a
# request (plain values) or in a sampling config / override preset (override
# entries). Anything not set there is inherited from the regular settings.
REASONING_OVERRIDE_KEY = "reasoning_override"

# The settings the backend can swap mid-generation: the sampler stack and the
# banned strings. Limits, stop conditions and grammars apply to the whole job
REASONING_OVERRIDE_FIELDS = frozenset(
    {
        "banned_strings",
        "banned_tokens",
        "temperature",
        "temperature_last",
        "top_k",
        "top_p",
        "min_p",
        "xtc_probability",
        "xtc_threshold",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "penalty_range",
        "repetition_decay",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_range",
        "dry_penalty_last_n",
        "dry_sequence_breakers",
        "logit_bias",
        "adaptive_target",
        "adaptive_decay",
    }
)


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
        description=(
            "Constrain generation to a JSON schema (also settable via "
            'response_format). Output is single-line JSON with ": " and ", " '
            "separators: optional whitespace is disabled in the grammar so "
            "generation cannot stall on whitespace. Add "
            '{"x-guidance": {"whitespace_flexible": true}} to the schema to allow it.'
        ),
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

    # Private: set by /v1/decisions to read the raw next-token logits at the
    # answer position. Deliberately not a request field: a client-supplied flag
    # would force full-vocab logits materialization on every generated token
    # of public endpoints with no consumer there.
    _return_logits: bool = PrivateAttr(default=False)

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

    reasoning_override: Optional[dict] = Field(
        default=None,
        validation_alias=AliasChoices("reasoning_override", "reasoning_overrides"),
        description=(
            "Sampler settings that apply only while the model is reasoning, e.g. "
            '{"temperature": 1.0, "banned_strings": ["but wait,"]}. Settings left '
            "out are the same as for the rest of the response. Chat completions only."
        ),
        examples=[{"temperature": 1.0}],
    )

    # For a reasoning-phase copy: where each overridden value came from
    _reasoning_sources: dict = PrivateAttr(default_factory=dict)

    def reasoning_params(self) -> Optional["BaseSamplerRequest"]:
        """
        The sampler settings for the reasoning block: this request's settings
        with the reasoning overrides (from the request and the sampling config)
        applied on top. None when nothing is overridden, i.e. reasoning and
        content are sampled alike.
        """

        values, sources = resolve_reasoning_overrides(self)
        if not values:
            return None

        merged = {name: getattr(self, name) for name in REASONING_OVERRIDE_FIELDS}
        # Already folded into dry_range / dry_multiplier for the content phase
        merged["dry_penalty_last_n"] = None
        merged.update(values)

        result = BaseSamplerRequest.model_validate(merged, context={"reasoning_phase": True})
        result._reasoning_sources = sources
        return result

    def reasoning_settings(self) -> list:
        """(name, value, source) for each setting a reasoning-phase copy overrides."""

        return [
            (name, getattr(self, name), source) for name, source in self._reasoning_sources.items()
        ]

    def param_source(self, name: str) -> str:
        """
        Where the effective value of a sampler param came from: "req" (sent with the
        request), "forced" or "preset" (sampler override preset), or "default".
        """

        override = overrides_container.effective().get(name)
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

    @field_validator("reasoning_override", mode="before")
    def check_reasoning_override(cls, v):
        """Keep the settings that can be swapped for the reasoning block."""

        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("reasoning_override must be an object of sampler settings")

        checked = {}
        for name, value in v.items():
            if name not in REASONING_OVERRIDE_FIELDS:
                xlogger.warning(
                    f'Ignoring "{name}" in reasoning_override: '
                    + (
                        "it applies to the whole response and can't differ for reasoning."
                        if name in cls.model_fields
                        else "not a known sampler setting."
                    )
                )
                continue
            checked[name] = value

        return checked

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
    def after_validate(self, info: ValidationInfo):
        # For OAI requests, logprobs is a boolean and top_logprobs is integer
        # if self.logprobs and self.top_logprobs:
        #     self.logprobs = self.top_logprobs

        # A reasoning-phase copy (see reasoning_params) is already resolved:
        # the regular forced overrides must not be laid over it again
        reasoning_copy = bool((info.context or {}).get("reasoning_phase"))

        # FIXME: find a better way to register this
        # Maybe make a function to assign values to the
        # model if they do not exist post creation
        if not reasoning_copy:
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

        if not reasoning_copy:
            self.warn_unsupported_params()

            # Resolve the reasoning overrides once here, so a bad value fails
            # the request up front rather than when generation starts
            self.reasoning_params()

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
    """
    Two layers of sampler overrides. The global layer comes from the startup
    config (a preset and/or inline overrides under `sampling`) or the override
    API; the model layer comes from the loaded model's own `sampling` section
    (same syntax, under `model.sampling`) and is replaced or cleared
    whenever the model changes. Within a layer, inline overrides win over the
    preset; a key in the model layer wins over the same key globally; and a
    request always wins over both unless the override is forced.
    """

    selected_preset: Optional[str] = None
    overrides: dict = {}
    model_preset: Optional[str] = None
    model_overrides: dict = {}

    def effective(self) -> dict:
        merged = {**self.overrides, **self.model_overrides}
        merged.pop(REASONING_OVERRIDE_KEY, None)
        return merged

    def effective_reasoning(self) -> dict:
        """The reasoning overrides of both layers; a model key replaces the global one."""

        merged = {}
        for layer in (self.overrides, self.model_overrides):
            section = layer.get(REASONING_OVERRIDE_KEY)
            if isinstance(section, dict):
                merged.update(section)
        return merged


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

    def describe(section: dict) -> list:
        parts = []
        for key, value in section.items():
            if key == REASONING_OVERRIDE_KEY or not isinstance(value, dict):
                continue
            if "override" not in value:
                continue

            item = f"{key} {value['override']}"
            if value.get("force"):
                item += " (forced)"
            elif value.get("additive"):
                item += " (additive)"
            parts.append(item)
        return parts

    text = ", ".join(describe(overrides)) or "no overrides"

    reasoning = overrides.get(REASONING_OVERRIDE_KEY)
    if isinstance(reasoning, dict) and (reasoning_parts := describe(reasoning)):
        text += "; while reasoning: " + ", ".join(reasoning_parts)

    return text


async def _read_preset(preset_name: str) -> tuple[str, dict]:
    """Read a preset file from the sampler_overrides folder as (name, overrides)."""

    preset_path = resolve_preset_path(preset_name)
    if preset_path is None:
        raise FileNotFoundError(
            f'Sampler override preset "{preset_name}" was not found in the '
            "sampler_overrides folder."
        )

    async with aiofiles.open(preset_path, "r", encoding="utf8") as raw_preset:
        contents = await raw_preset.read()

    preset = YAML(typ="safe").load(contents)
    if preset is None:
        preset = {}
    if not isinstance(preset, dict):
        raise TypeError(f'Sampler override preset "{preset_name}" must be a mapping')

    return preset_path.stem, filter_none_values(preset)


def split_sampling_section(section: Optional[dict], where: str) -> tuple[Optional[str], dict]:
    """
    Split a `sampling` config section into (override_preset, inline overrides).
    The same syntax serves the global section and a model's `model.sampling`.
    """

    if not section:
        return None, {}
    if not isinstance(section, dict):
        raise TypeError(f"The sampling section in {where} must be a mapping")

    section = dict(section)
    preset = section.pop("override_preset", None)
    if preset is not None and not isinstance(preset, str):
        raise TypeError(f"override_preset in {where} must be a preset name")

    return (preset or None), validate_inline_overrides(section, where)


def validate_inline_overrides(inline: Optional[dict], where: str) -> dict:
    """
    Check inline overrides written directly in a config section: a mapping of
    sampler name to {override, force, additive}. Unknown sampler names are
    warned about here rather than on every request.
    """

    if not inline:
        return {}
    if not isinstance(inline, dict):
        raise TypeError(f"Inline sampler overrides in {where} must be a mapping")

    checked = {}
    for name, value in inline.items():
        if name == REASONING_OVERRIDE_KEY:
            reasoning = validate_reasoning_overrides(value, where)
            if reasoning:
                checked[name] = reasoning
            continue
        if not isinstance(value, dict) or "override" not in value:
            raise TypeError(
                f'Inline sampler override "{name}" in {where} must be a mapping with an '
                '"override" key (and optionally "force" / "additive")'
            )
        if name not in BaseSamplerRequest.model_fields:
            xlogger.warning(f'Skipping unknown sampler override key "{name}" in {where}')
            continue
        checked[name] = value

    return checked


def validate_reasoning_overrides(section: Optional[dict], where: str) -> dict:
    """
    Check a `reasoning_override` section: the same override entries as the
    section around it, limited to the settings that can be swapped for the
    reasoning block.
    """

    if not section:
        return {}
    if not isinstance(section, dict):
        raise TypeError(f"{REASONING_OVERRIDE_KEY} in {where} must be a mapping")

    checked = {}
    for name, value in section.items():
        if not isinstance(value, dict) or "override" not in value:
            raise TypeError(
                f'"{name}" under {REASONING_OVERRIDE_KEY} in {where} must be a mapping with '
                'an "override" key (and optionally "force" / "additive")'
            )
        if name not in REASONING_OVERRIDE_FIELDS:
            reason = (
                "it applies to the whole response and can't differ for reasoning"
                if name in BaseSamplerRequest.model_fields
                else "not a known sampler setting"
            )
            xlogger.warning(
                f'Skipping "{name}" under {REASONING_OVERRIDE_KEY} in {where}: {reason}'
            )
            continue
        checked[name] = value

    return checked


def resolve_reasoning_overrides(params: BaseSamplerRequest) -> tuple[dict, dict]:
    """
    The values that differ while reasoning, with where each came from. Per
    setting: a forced config override wins, then the request's
    reasoning_override, then a plain config override. A config override here
    outranks the request's regular (non-reasoning) value for the same setting,
    being the more specific of the two.
    """

    layer = overrides_container.effective_reasoning()
    request = params.reasoning_override or {}
    if not layer and not request:
        return {}, {}

    values, sources = {}, {}
    for name in BaseSamplerRequest.model_fields:
        if name not in REASONING_OVERRIDE_FIELDS:
            continue

        entry = layer.get(name)
        entry = entry if isinstance(entry, dict) else {}
        override = deepcopy(entry.get("override"))
        additive = unwrap(entry.get("additive"), False) and isinstance(override, list)

        if override is not None and unwrap(entry.get("force"), False):
            values[name], sources[name] = override, "forced"
        elif name in request:
            value = request[name]
            if additive and isinstance(value, list):
                value = override + value
            values[name], sources[name] = value, "req"
        elif override is not None:
            inherited = getattr(params, name, None)
            if additive and isinstance(inherited, list):
                override = override + inherited
            values[name], sources[name] = override, "preset"

    return values, sources


async def _build_layer(preset_name: Optional[str], inline: dict) -> tuple[Optional[str], dict]:
    """A layer's (preset name, merged overrides): inline entries win over the preset's."""

    name, overrides = None, {}
    if preset_name:
        name, overrides = await _read_preset(preset_name)

    # The reasoning sections merge per setting rather than one replacing the other
    preset_reasoning = validate_reasoning_overrides(
        overrides.get(REASONING_OVERRIDE_KEY), f'the preset "{name}"'
    )
    reasoning = {**preset_reasoning, **(inline.get(REASONING_OVERRIDE_KEY) or {})}

    merged = {**overrides, **inline}
    merged.pop(REASONING_OVERRIDE_KEY, None)
    if reasoning:
        merged[REASONING_OVERRIDE_KEY] = reasoning
    return name, merged


async def overrides_from_file(preset_name: str):
    """Load the global override preset from a file (the override API's switch)"""

    await set_global_overrides(preset_name, {})


async def set_global_overrides(preset_name: Optional[str], inline: Optional[dict] = None):
    """Set the global layer from a preset and/or inline overrides"""

    inline = validate_inline_overrides(inline, "the sampling config")
    name, overrides = await _build_layer(preset_name, inline)
    overrides_container.selected_preset = name
    overrides_from_dict(overrides)

    xlogger.info(
        _describe_layer("Sampler overrides", name, inline, overrides_container.overrides),
        {"preset": name, "inline": inline},
    )


async def set_model_overrides(preset_name: Optional[str], inline: Optional[dict] = None):
    """Set the model layer from a preset and/or inline overrides, on top of the global one"""

    inline = validate_inline_overrides(inline, "the model's sampling config")
    name, overrides = await _build_layer(preset_name, inline)
    overrides_container.model_preset = name
    overrides_container.model_overrides = filter_none_values(overrides)

    xlogger.info(
        _describe_layer("Sampler overrides for this model", name, inline, overrides),
        {"preset": name, "inline": inline},
    )


async def model_overrides_from_file(preset_name: str):
    """Load a model's own override preset on top of the global one"""

    await set_model_overrides(preset_name, {})


def clear_model_overrides():
    """Drop the model layer, e.g. when the model is unloaded"""

    overrides_container.model_preset = None
    overrides_container.model_overrides = {}


def _describe_layer(label: str, preset: Optional[str], inline: dict, merged: dict) -> str:
    sources = []
    if preset:
        sources.append(f'preset "{preset}"')
    if inline:
        sources.append("inline")
    source_text = " + ".join(sources) if sources else "none"
    return f"{label} ({source_text}): {describe_overrides(merged)}"


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
        deepcopy(overrides_container.effective().get(key, {}).get("override")),
        fallback,
    )

    return default_value


def apply_forced_sampler_overrides(params: BaseSamplerRequest):
    """Forcefully applies overrides if specified by the user"""

    # Tolerate older OAI standard for logprobs
    if isinstance(params.logprobs, int) and params.logprobs > 1:
        params.top_logprobs = params.logprobs

    for var, value in overrides_container.effective().items():
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
