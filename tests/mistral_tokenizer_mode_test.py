"""Tests for mistral tokenizer mode auto-detection."""

import json

from common.tokenizer_modes import (
    normalize_tokenizer_mode,
    should_enable_mistral_tokenizer_mode,
    supports_mistral_tokenizer_mode,
)


def _write_config(directory, model_type: str) -> None:
    with open(directory / "config.json", "w", encoding="utf-8") as config_file:
        json.dump({"model_type": model_type}, config_file)


def test_supports_mistral_with_tekken(tmp_path):
    _write_config(tmp_path, "mistral3")
    (tmp_path / "tekken.json").write_text("{}", encoding="utf-8")

    assert supports_mistral_tokenizer_mode(tmp_path) is True


def test_supports_mistral_with_sentencepiece_variant(tmp_path):
    _write_config(tmp_path, "mistral")
    (tmp_path / "tokenizer.model.v3").write_text("dummy", encoding="utf-8")

    assert supports_mistral_tokenizer_mode(tmp_path) is True


def test_rejects_non_mistral_with_sentencepiece_tokenizer(tmp_path):
    _write_config(tmp_path, "gemma2")
    (tmp_path / "tokenizer.model").write_text("dummy", encoding="utf-8")

    assert supports_mistral_tokenizer_mode(tmp_path) is False


def test_allowlist_enables_listed_mistral_model(tmp_path):
    _write_config(tmp_path, "mistral3")
    (tmp_path / "tekken.json").write_text("{}", encoding="utf-8")

    assert should_enable_mistral_tokenizer_mode(
        tmp_path, [tmp_path.name, "other-model"]
    )


def test_allowlist_disables_non_mistral_even_if_listed(tmp_path):
    _write_config(tmp_path, "gemma2")
    (tmp_path / "tokenizer.model").write_text("dummy", encoding="utf-8")

    assert not should_enable_mistral_tokenizer_mode(tmp_path, [tmp_path.name])


def test_allowlist_disables_unlisted_mistral_model(tmp_path):
    _write_config(tmp_path, "mistral3")
    (tmp_path / "tekken.json").write_text("{}", encoding="utf-8")

    assert not should_enable_mistral_tokenizer_mode(tmp_path, ["another-model"])


def test_normalize_tokenizer_mode_accepts_deepseek_v32():
    normalized, message = normalize_tokenizer_mode("deepseek_v32")
    assert normalized == "deepseek_v32"
    assert message is None


def test_normalize_tokenizer_mode_maps_slow_to_hf():
    normalized, message = normalize_tokenizer_mode("slow")
    assert normalized == "hf"
    assert message is not None


def test_normalize_tokenizer_mode_unknown_falls_back_to_auto():
    normalized, message = normalize_tokenizer_mode("unknown_mode")
    assert normalized == "auto"
    assert message is not None
