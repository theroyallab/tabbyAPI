"""Tests for reasoning parser registry parity with vLLM."""

from endpoints.OAI.reasoning import ReasoningParserManager


VLLM_CANONICAL_REASONING_PARSERS = {
    "deepseek_r1",
    "deepseek_v3",
    "ernie45",
    "glm45",
    "openai_gptoss",
    "granite",
    "holo2",
    "hunyuan_a13b",
    "kimi_k2",
    "minimax_m2",
    "minimax_m2_append_think",
    "mistral",
    "olmo3",
    "qwen3",
    "seed_oss",
    "step3",
    "step3p5",
}


def test_reasoning_registry_contains_all_vllm_canonical_parsers():
    registered = set(ReasoningParserManager.list_registered())
    missing = sorted(VLLM_CANONICAL_REASONING_PARSERS - registered)
    assert missing == []


def test_reasoning_registry_allows_local_extensions():
    registered = set(ReasoningParserManager.list_registered())
    # Local compatibility/default parsers that may not exist in vLLM.
    assert "identity" in registered
    assert "basic" in registered
    assert "exaone4" in registered
