"""Parameter values typed from the request's tool schema, for parsing and for delta streaming."""

import asyncio
import json
import unittest

from endpoints.OAI.types.tools import ToolSpec
from endpoints.OAI.utils import tools as tools_module
from endpoints.OAI.utils.toolcall_formats import glm4_5, hy3, minimax_m2, qwen3_coder
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value, param_types_from_tools
from endpoints.OAI.utils.toolcall_stream import QwenToolCallDeltaStreamer, make_delta_streamer

LT, GT = chr(60), chr(62)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "integer"},
                    "flags": {"anyOf": [{"type": "string"}, {"type": "array"}]},
                    "meta": {"type": "object"},
                },
            },
        },
    },
    {"type": "function", "function": {"name": "no_params"}},
]


def qwen_call(name, params, wrapped=True):
    body = "".join(f"{LT}parameter={k}{GT}\n{v}\n{LT}/parameter{GT}\n" for k, v in params)
    block = f"{LT}function={name}{GT}\n{body}{LT}/function{GT}"
    return f"{LT}tool_call{GT}\n{block}\n{LT}/tool_call{GT}" if wrapped else block


class ParamTypesTests(unittest.TestCase):
    def test_from_dicts_and_models(self):
        expected = {
            "write_file": {
                "path": "string",
                "content": "string",
                "mode": "integer",
                "flags": None,
                "meta": "object",
            },
            "no_params": {},
        }
        self.assertEqual(param_types_from_tools(TOOLS), expected)
        self.assertEqual(param_types_from_tools([ToolSpec(**t) for t in TOOLS]), expected)
        self.assertEqual(param_types_from_tools(None), {})

    def test_coercion_follows_the_declared_type(self):
        self.assertEqual(coerce_param_value("\n123\n", "string"), "123")
        self.assertEqual(coerce_param_value('{"a": 1}', "string"), '{"a": 1}')
        self.assertEqual(coerce_param_value("true", "string"), "true")
        self.assertEqual(coerce_param_value("123", "integer"), 123)
        self.assertEqual(coerce_param_value("not a number", "integer"), "not a number")
        # Untyped: the old guess
        self.assertEqual(coerce_param_value("123"), 123)
        self.assertEqual(coerce_param_value("123 apples"), "123 apples")


class TypedParsingTests(unittest.TestCase):
    def test_qwen_string_parameter_stays_a_string(self):
        text = qwen_call("write_file", [("path", "123"), ("mode", "644"), ("flags", "[1]")])
        types = param_types_from_tools(TOOLS)
        (call,) = qwen3_coder.parse_toolcalls(text, param_types=types)
        self.assertEqual(
            json.loads(call.function.arguments), {"path": "123", "mode": 644, "flags": [1]}
        )
        # Without the schema the path is guessed as a number, as before
        (call,) = qwen3_coder.parse_toolcalls(text)
        self.assertEqual(json.loads(call.function.arguments)["path"], 123)

    def test_dispatcher_passes_types_only_where_accepted(self):
        types = param_types_from_tools(TOOLS)
        text = qwen_call("write_file", [("path", "123")])
        (call,) = tools_module.parse_toolcalls(text, "qwen3_coder", types)
        self.assertEqual(json.loads(call.function.arguments), {"path": "123"})
        # A JSON format ignores the types and still parses
        hermes = (
            LT + "tool_call" + GT + '{"name": "f", "arguments": {"a": 1}}' + LT + "/tool_call" + GT
        )
        (call,) = tools_module.parse_toolcalls(hermes, "hermes", types)
        self.assertEqual(call.function.name, "f")

    def test_other_untyped_text_formats_accept_the_schema(self):
        types = {"f": {"a": "string"}}
        mm = (
            f'{LT}minimax:tool_call{GT}{LT}invoke name="f"{GT}'
            f'{LT}parameter name="a"{GT}12{LT}/parameter{GT}{LT}/invoke{GT}{LT}/minimax:tool_call{GT}'
        )
        glm = f"{LT}tool_call{GT}f{LT}arg_key{GT}a{LT}/arg_key{GT}{LT}arg_value{GT}12{LT}/arg_value{GT}{LT}/tool_call{GT}"
        for module, text in ((minimax_m2, mm), (glm4_5, glm), (hy3, None)):
            if text is None:
                continue
            (call,) = module.parse_toolcalls(text, param_types=types)
            self.assertEqual(json.loads(call.function.arguments), {"a": "12"}, module.__name__)
            (call,) = module.parse_toolcalls(text)
            self.assertEqual(json.loads(call.function.arguments), {"a": 12}, module.__name__)


def stream(text, param_types, n=3):
    streamer = QwenToolCallDeltaStreamer(param_types)
    frames = [streamer.feed(text[i : i + n]) for i in range(0, len(text), n)]
    frames = [f for f in frames if f]
    assembled = {}
    for frame in frames:
        for delta in frame:
            entry = assembled.setdefault(delta["index"], {"name": None, "args": ""})
            if "name" in delta.get("function", {}):
                entry["name"] = delta["function"]["name"]
            entry["args"] += delta["function"].get("arguments", "")
    return streamer, frames, assembled


class TypedStreamingTests(unittest.TestCase):
    types = param_types_from_tools(TOOLS)

    def argument_frames(self, frames, index=0):
        return [
            d["function"]["arguments"]
            for f in frames
            for d in f
            if d["index"] == index and "arguments" in d.get("function", {})
        ]

    def test_declared_string_streams_from_the_first_character(self):
        content = "2024 report\n- first\n- second\n" + "x" * 200
        text = qwen_call("write_file", [("path", "123"), ("content", content)])
        streamer, frames, assembled = stream(text, self.types)

        expected = qwen3_coder.parse_toolcalls(text, param_types=self.types)[0].function.arguments
        self.assertEqual(assembled[0]["args"], expected)
        self.assertEqual(json.loads(expected)["path"], "123")
        # The content arrived in many fragments, not one piece at the close tag
        self.assertGreater(len(self.argument_frames(frames)), 20)

    def test_untyped_value_is_held_until_it_cannot_be_json(self):
        content = "2024 report " + "y" * 200
        text = qwen_call("write_file", [("content", content)])
        streamer, frames, assembled = stream(text, None)
        expected = qwen3_coder.parse_toolcalls(text)[0].function.arguments
        self.assertEqual(assembled[0]["args"], expected)
        # Starts with a digit: without a schema the whole value waits for the close tag
        self.assertLessEqual(len(self.argument_frames(frames)), 4)

    def test_typed_non_string_is_emitted_whole(self):
        text = qwen_call("write_file", [("mode", "644"), ("meta", '{"a": [1, 2]}')])
        streamer, frames, assembled = stream(text, self.types)
        expected = qwen3_coder.parse_toolcalls(text, param_types=self.types)[0].function.arguments
        self.assertEqual(assembled[0]["args"], expected)
        self.assertEqual(json.loads(expected), {"mode": 644, "meta": {"a": [1, 2]}})

    def test_verify_agrees_with_the_typed_parse(self):
        text = qwen_call("write_file", [("path", "123"), ("mode", "7")])
        streamer, frames, assembled = stream(text, self.types)
        errors = []
        import endpoints.OAI.utils.toolcall_stream as tcs

        original = tcs.xlogger.error
        tcs.xlogger.error = lambda *a, **k: errors.append(a)
        try:
            streamer.verify(text, "test")
        finally:
            tcs.xlogger.error = original
        self.assertEqual(errors, [])

    def test_factory(self):
        self.assertIsInstance(make_delta_streamer("qwen3_5", self.types), QwenToolCallDeltaStreamer)
        self.assertIsNone(make_delta_streamer("hermes", self.types))
        self.assertIsNone(make_delta_streamer(None, self.types))


class WiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_schema_reaches_the_streamer(self):
        import test_tool_delta_streaming as wiring

        content = "2024 report " + "z" * 120
        text = (
            wiring.TC_S
            + f"{LT}function=get_weather{GT}"
            + f"{LT}parameter=city{GT}\n{content}{LT}/parameter{GT}"
            + f"{LT}/function{GT}"
            + wiring.TC_E
        )
        chunks = wiring.pieces(wiring.RS + "thinking" + wiring.RE) + wiring.pieces(text)
        frames, _ = await wiring.run_collector(wiring.make_mc(chunks), wiring.make_request())
        tool_frames = [f for f in frames if isinstance(f, dict) and f.get("delta_tool_calls")]
        # The city parameter is declared string in the request, so the digit-led
        # value streams in fragments instead of arriving whole at the close tag
        self.assertGreater(len(tool_frames), 10)
        args = "".join(
            d["function"].get("arguments", "") for f in tool_frames for d in f["delta_tool_calls"]
        )
        self.assertEqual(json.loads(args), {"city": content})


if __name__ == "__main__":
    unittest.main()
