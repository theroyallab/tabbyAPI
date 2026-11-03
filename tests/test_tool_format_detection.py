"""Tests for tool call format auto-detection and the Hermes, Kimi and OLMo 3 parsers."""

import json

import pytest

from endpoints.OAI.utils.tools import (
    ALL_TOOLCALL_FORMATS,
    canonical_format_name,
    detect_reasoning_tags,
    detect_tool_format,
    get_toolcall_tags,
    parse_toolcalls,
)

# Minimal template fragments carrying each format's markers, the way real
# templates render assistant tool-call history
TEMPLATES = {
    "qwen3_coder": "{{ '<tool_call>\\n<function=' ~ name ~ '>\\n<parameter=' }}",
    "glm4_5": "{{ '<tool_call>' ~ name }}{{ '<arg_key>' ~ k ~ '</arg_key><arg_value>' }}",
    "hermes": 'return a json object within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{"name": <function-name>, "arguments": <args-json-object>}\\n</tool_call>',
    "lfm2": "{{ '<|tool_call_start|>' ~ calls ~ '<|tool_call_end|>' }}",
    "harmony": "<|start|>assistant<|channel|>commentary to=functions.x<|message|>{}<|call|>",
    "muse_glimmer": "<|start|>assistant to=x<|message|>...<|eom|>",
    "mistral": "[TOOL_CALLS]{{ name }}[ARGS]{{ args | tojson }}",
    "mistral_old": "[TOOL_CALLS][{{ calls | tojson }}]",
    "minimax_m2": '<minimax:tool_call>\\n<invoke name=\\"{{ name }}\\">',
    "gemma4": "<|tool_call>call:{{ name }}{...}<tool_call|>",
    "hy3": "<tool_calls:opensource><tool_call:opensource>{{ name }}",
    "deepseek_v4": '<｜DSML｜tool_calls><｜DSML｜invoke name=\\"{{ name }}\\">',
    "deepseek_v41": '<｜DSML｜ calls><｜DSML｜ invoke name=\\"{{ name }}\\">',
    "kimi": "<|tool_calls_section_begin|><|tool_call_begin|>{{ id }}<|tool_call_argument_begin|>{{ args }}<|tool_call_end|>",
    "olmo3": "<function_calls>{{ name }}({{ arguments }})</function_calls>",
}


def no_tokens(_token):
    return False


@pytest.mark.parametrize("expected, template", list(TEMPLATES.items()))
def test_each_format_is_detected_from_its_template_alone(expected, template):
    detection = detect_tool_format(template, no_tokens, None)
    assert detection.tool_format == expected, detection
    assert detection.evidence == ["template"]


def test_all_registered_formats_carry_a_signature():
    missing = [
        name for name, module in ALL_TOOLCALL_FORMATS.items() if not hasattr(module, "DETECT")
    ]
    assert missing == []


def test_tokenizer_alone_selects_special_token_formats():
    harmony = {"<|channel|>", "<|message|>", "<|call|>", "<|return|>"}
    detection = detect_tool_format(None, lambda t: t in harmony, "GptOssForCausalLM")
    assert detection.tool_format == "harmony"
    assert detection.evidence == ["tokenizer", "architecture"]


def test_architecture_alone_is_not_enough():
    detection = detect_tool_format("no tools here", no_tokens, "Qwen3_5ForConditionalGeneration")
    assert detection.tool_format is None
    assert not detection.ambiguous


def test_nothing_detected_for_unknown_template():
    detection = detect_tool_format("{{ tools | tojson }} <custom_call>", no_tokens, "Foo")
    assert detection.tool_format is None


def test_canonical_names():
    assert canonical_format_name("qwen3_5") == "qwen3_coder"
    assert canonical_format_name("lfm") == "lfm2"
    assert canonical_format_name("spark2_5") == "glm4_5"
    assert canonical_format_name("ling3") == "glm4_5"
    assert canonical_format_name("kimi_k2") == "kimi"
    assert canonical_format_name("dsv41") == "deepseek_v41"
    assert canonical_format_name("nope") == "nope"


def test_reasoning_tags_from_format_then_template_then_tokenizer():
    assert detect_reasoning_tags("<think>x</think>", no_tokens, "qwen3_coder") == (
        ("<think>", "</think>"),
        "template",
    )
    # The format's tags are only used when something confirms them
    assert detect_reasoning_tags("no tags", no_tokens, "qwen3_coder") == (None, None)
    assert detect_reasoning_tags("no tags", lambda t: t in {"<think>", "</think>"}, None) == (
        ("<think>", "</think>"),
        "tokenizer",
    )
    assert detect_reasoning_tags("[THINK]...[/THINK]", no_tokens, "mistral") == (
        ("[THINK]", "[/THINK]"),
        "template",
    )


# Parsers


def test_hermes_parser():
    text = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}\n</tool_call>'
        '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>'
    )
    calls = parse_toolcalls(text, "hermes")
    assert [(c.function.name, json.loads(c.function.arguments)) for c in calls] == [
        ("get_weather", {"city": "Paris", "days": 3}),
        ("add", {"a": 2, "b": 3}),
    ]
    assert get_toolcall_tags("smollm3") == ("<tool_call>", "</tool_call>")


def test_hermes_parser_tolerates_fences_and_string_arguments():
    text = '<tool_call>\n```json\n{"name": "f", "arguments": "{\\"x\\": 1}"}\n```\n</tool_call>'
    calls = parse_toolcalls(text, "hermes")
    assert json.loads(calls[0].function.arguments) == {"x": 1}
    assert parse_toolcalls("<tool_call>not json</tool_call>", "hermes") == []


def test_kimi_parser():
    text = (
        "<|tool_calls_section_begin|>"
        '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>'
        '<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{"city": "Tokyo"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )
    calls = parse_toolcalls(text, "kimi")
    assert [(c.id, c.function.name, c.function.arguments) for c in calls] == [
        ("functions.get_weather:0", "get_weather", '{"city": "Paris"}'),
        ("functions.get_weather:1", "get_weather", '{"city": "Tokyo"}'),
    ]
    assert get_toolcall_tags("kimi_linear") == (
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
    )


def test_olmo3_parser():
    text = (
        "<function_calls>\n"
        'get_weather(city="Paris", unit="c")\n'
        'write_file(path="a.txt", content="x, y (z)", tags=["a", "b"], meta={"k": null}, n=3, ok=true)\n'
        "list_files()\n"
        "</function_calls>"
    )
    calls = parse_toolcalls(text, "olmo3")
    assert [(c.function.name, json.loads(c.function.arguments)) for c in calls] == [
        ("get_weather", {"city": "Paris", "unit": "c"}),
        (
            "write_file",
            {
                "path": "a.txt",
                "content": "x, y (z)",
                "tags": ["a", "b"],
                "meta": {"k": None},
                "n": 3,
                "ok": True,
            },
        ),
        ("list_files", {}),
    ]
    assert get_toolcall_tags("olmo") == ("<function_calls>", "</function_calls>")


def test_olmo3_parser_unterminated_block_and_python_literals():
    calls = parse_toolcalls("<function_calls>\nf(a=True, b='x')", "olmo3")
    assert json.loads(calls[0].function.arguments) == {"a": True, "b": "x"}


def test_ling3_uses_the_glm4_5_format():
    """inclusionAI's Ling 3.x (BailingMoeV3) renders tool calls exactly like
    GLM-4.5, down to the arg_key/arg_value tokens; its template puts no
    newline between the function name and the first <arg_key> in history, and
    indents it, while the model itself is instructed to write one."""

    template = (
        "{{- '<tool_call>' + tc.name }}\n    {%- for k, v in _args.items() %}"
        "{{- '<arg_key>' + k + '</arg_key>' }}{{- '\\n<arg_value>' }}{{- v }}{{- '</arg_value>' }}"
        "{%- endfor %}{{- '\\n</tool_call>' }} tools"
    )
    tokens = {
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "<arg_value>",
        "<think>",
        "</think>",
        "<role>",
    }
    detection = detect_tool_format(template, lambda t: t in tokens, "BailingMoeV3ForCausalLM")
    assert detection.tool_format == "glm4_5"
    assert detection.evidence == ["template", "tokenizer", "architecture"]

    rendered = (
        "<tool_call>get_weather\n                <arg_key>city</arg_key>\n<arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>"
    )
    generated = (
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
        "<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>"
    )
    for text in (rendered, generated):
        (call,) = parse_toolcalls(text, "ling3")
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == {"city": "Paris", "days": 3}
