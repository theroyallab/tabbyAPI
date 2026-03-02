"""Helpers for tokenizer compatibility mode detection."""

import json
import pathlib
from typing import Iterable

VLLM_COMPAT_TOKENIZER_MODES = {
    "auto",
    "hf",
    "slow",
    "mistral",
    "deepseek_v32",
}


def normalize_tokenizer_mode(tokenizer_mode: str | None) -> tuple[str, str | None]:
    mode = str(tokenizer_mode or "auto").lower()
    if mode not in VLLM_COMPAT_TOKENIZER_MODES:
        return (
            "auto",
            f"Unknown tokenizer_mode '{mode}' requested. Falling back to 'auto'.",
        )

    if mode == "slow":
        return (
            "hf",
            "tokenizer_mode='slow' requested, but ExLlama backends do not expose "
            "a distinct slow tokenizer path. Using 'hf' compatibility mode.",
        )

    return mode, None


def _read_model_type(model_directory: pathlib.Path) -> str:
    config_path = model_directory / "config.json"
    if not config_path.exists():
        return ""

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return str(json.load(config_file).get("model_type", "")).lower()
    except Exception:
        return ""


def has_mistral_tokenizer_assets(model_directory: pathlib.Path) -> bool:
    return (
        (model_directory / "tekken.json").exists()
        or (model_directory / "tokenizer.model").exists()
        or any(model_directory.glob("tokenizer.model.v*"))
    )


def supports_mistral_tokenizer_mode(model_directory: pathlib.Path) -> bool:
    """
    Return True when mistral tokenizer mode is safe to enable for this model.

    vLLM uses mistral-common only for Mistral-family models in auto mode.
    Match that intent by requiring both:
    1. A mistral-family model type.
    2. Mistral tokenizer assets.
    """

    model_type = _read_model_type(model_directory)
    is_mistral_family = model_type.startswith("mistral") or model_type.startswith(
        "mixtral"
    )

    return is_mistral_family and has_mistral_tokenizer_assets(model_directory)


def _matches_allowlist(
    model_directory: pathlib.Path, mistral_tokenizer_models: Iterable[str]
) -> bool:
    model_name = model_directory.name.lower()
    model_path = model_directory.as_posix().lower().rstrip("/")

    for entry in mistral_tokenizer_models:
        normalized = str(entry).strip().lower().strip("/")
        if not normalized:
            continue

        if model_name == normalized:
            return True

        if model_path.endswith(normalized):
            return True

    return False


def should_enable_mistral_tokenizer_mode(
    model_directory: pathlib.Path,
    mistral_tokenizer_models: Iterable[str] | None = None,
) -> bool:
    """
    Decide whether mistral tokenizer mode should be enabled.

    If an explicit allowlist is configured, only listed Mistral-family models
    can use mistral mode. If no allowlist is provided, fallback to auto
    detection (mistral-family model + tokenizer assets).
    """

    allowlist = list(mistral_tokenizer_models or [])
    if allowlist:
        return _matches_allowlist(model_directory, allowlist) and (
            supports_mistral_tokenizer_mode(model_directory)
        )

    return supports_mistral_tokenizer_mode(model_directory)
