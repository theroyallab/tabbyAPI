"""JSON schema constraints get single-line llguidance options by default."""

import pytest

pytest.importorskip("exllamav3")

from backends.exllamav3.grammar import (  # noqa: E402
    JSON_SCHEMA_GUIDANCE_DEFAULTS,
    json_schema_lark_with_lead,
    prepare_json_schema,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


def test_default_disables_optional_whitespace():
    out = prepare_json_schema(SCHEMA)
    assert out["x-guidance"] == JSON_SCHEMA_GUIDANCE_DEFAULTS
    assert out["x-guidance"]["whitespace_flexible"] is False
    # The separators models write by habit; bare ones corrupt string values
    assert out["x-guidance"]["key_separator"] == ": "
    assert out["x-guidance"]["item_separator"] == ", "
    assert "x-guidance" not in SCHEMA  # input not mutated
    assert out["properties"] is SCHEMA["properties"]


def test_client_x_guidance_wins():
    schema = {**SCHEMA, "x-guidance": {"whitespace_flexible": True}}
    assert prepare_json_schema(schema)["x-guidance"] == {"whitespace_flexible": True}


def test_named_schema_is_unwrapped_then_defaulted():
    out = prepare_json_schema({"name": "thing", "schema": SCHEMA})
    assert out["type"] == "object"
    assert out["x-guidance"] == JSON_SCHEMA_GUIDANCE_DEFAULTS


def test_non_dict_passes_through():
    assert prepare_json_schema('{"type": "object"}') == '{"type": "object"}'


def accepts(grammar, tokenizer, raw, text):
    from llguidance import LLMatcher

    matcher = LLMatcher(tokenizer, grammar)
    for token in raw.encode(text, add_special_tokens=False).ids:
        if not matcher.consume_token(token):
            return False
    return matcher.is_accepting()


def test_grammars_accept_single_line_json_only():
    """Compiled against a small real vocabulary, if one can be built."""

    llguidance = pytest.importorskip("llguidance")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers

    # Byte-level BPE with no merges: every byte is its own token
    vocab = {token: i for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    raw = tokenizers.Tokenizer(models.BPE(vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    try:
        tokenizer = llguidance.LLTokenizer(raw.to_str())
    except Exception as exc:  # pragma: no cover - depends on the llguidance build
        pytest.skip(f"llguidance can't wrap the test tokenizer: {exc}")

    schema = prepare_json_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        }
    )
    plain = llguidance.LLMatcher.grammar_from_json_schema(schema)
    lead = llguidance.LLMatcher.grammar_from_lark(json_schema_lark_with_lead(schema))

    good = '{"a": "x", "b": 1}'
    assert accepts(plain, tokenizer, raw, good)
    assert not accepts(plain, tokenizer, raw, '{"a":"x","b":1}')
    assert not accepts(plain, tokenizer, raw, '{\n  "a": "x", "b": 1}')
    assert not accepts(plain, tokenizer, raw, "\n\n" + good)

    # After a reasoning block: a blank line at most, then the same JSON
    for prefix in ("", " ", "\n", "\n\n"):
        assert accepts(lead, tokenizer, raw, prefix + good), repr(prefix)
    assert not accepts(lead, tokenizer, raw, "\n\n\n" + good)
    assert not accepts(lead, tokenizer, raw, "\n\nx" + good)
    assert not accepts(lead, tokenizer, raw, good + "\n")
