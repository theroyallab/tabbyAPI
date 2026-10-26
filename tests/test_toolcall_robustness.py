import json
import unittest

from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.toolcall_formats.common import (
    unwrap_tool_args,
    parse_json_tool_call,
    strip_code_fences,
)
from endpoints.OAI.utils.toolcall_formats import qwen3_coder, hermes
from endpoints.OAI.utils.tools import parse_toolcalls, is_supported_format


class ToolCallRobustnessTests(unittest.TestCase):
    def test_strip_code_fences(self):
        text = "```json\n{\"foo\": 123}\n```"
        self.assertEqual(strip_code_fences(text), '{"foo": 123}')

    def test_unwrap_tool_args_nested_arguments(self):
        raw = {"arguments": {"url": "https://example.com", "depth": 2}}
        unwrapped = unwrap_tool_args(raw, "web__read_page")
        self.assertEqual(unwrapped, {"url": "https://example.com", "depth": 2})

    def test_unwrap_tool_args_nested_parameters_string(self):
        raw = {"parameters": json.dumps({"query": "python", "limit": 10})}
        unwrapped = unwrap_tool_args(raw, "search")
        self.assertEqual(unwrapped, {"query": "python", "limit": 10})

    def test_unwrap_tool_args_redundant_name(self):
        raw = {
            "name": "web__read_page",
            "arguments": {"url": "https://example.com"},
        }
        unwrapped = unwrap_tool_args(raw, "web__read_page")
        self.assertEqual(unwrapped, {"url": "https://example.com"})

    def test_unwrap_tool_args_flat_remains_intact(self):
        raw = {"url": "https://example.com", "options": {"timeout": 30}}
        unwrapped = unwrap_tool_args(raw, "fetch")
        self.assertEqual(unwrapped, {"url": "https://example.com", "options": {"timeout": 30}})

    def test_parse_json_tool_call(self):
        block = '```json\n{"name": "calculator", "arguments": {"expression": "2 + 2"}}\n```'
        result = parse_json_tool_call(block)
        self.assertIsNotNone(result)
        name, args = result
        self.assertEqual(name, "calculator")
        self.assertEqual(args, {"expression": "2 + 2"})

    def test_qwen3_coder_standard(self):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>London</parameter>\n"
            "<parameter=unit>celsius</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = qwen3_coder.parse_toolcalls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(json.loads(calls[0].function.arguments), {"location": "London", "unit": "celsius"})

    def test_qwen3_coder_nested_arguments_parameter(self):
        text = (
            "<tool_call>\n"
            "<function=web__read_page>\n"
            "<parameter=arguments>\n"
            '{"url": "https://example.com"}\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = qwen3_coder.parse_toolcalls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "web__read_page")
        self.assertEqual(json.loads(calls[0].function.arguments), {"url": "https://example.com"})

    def test_qwen3_coder_redundant_name_and_arguments(self):
        text = (
            "<tool_call>\n"
            "<function=web__read_page>\n"
            "<parameter=name>web__read_page</parameter>\n"
            "<parameter=arguments>\n"
            '{"url": "https://example.com"}\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = qwen3_coder.parse_toolcalls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "web__read_page")
        self.assertEqual(json.loads(calls[0].function.arguments), {"url": "https://example.com"})

    def test_qwen3_coder_raw_json_body(self):
        text = (
            "<tool_call>\n"
            "<function=web__read_page>\n"
            '{"url": "https://example.com"}\n'
            "</function>\n"
            "</tool_call>"
        )
        calls = qwen3_coder.parse_toolcalls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "web__read_page")
        self.assertEqual(json.loads(calls[0].function.arguments), {"url": "https://example.com"})

    def test_qwen3_coder_hermes_json_fallback(self):
        text = (
            '<tool_call>\n'
            '{"name": "fetch", "arguments": {"url": "https://example.com"}}\n'
            '</tool_call>'
        )
        calls = qwen3_coder.parse_toolcalls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "fetch")
        self.assertEqual(json.loads(calls[0].function.arguments), {"url": "https://example.com"})

    def test_hermes_nested_arguments(self):
        text = (
            '<tool_call>\n'
            '{"name": "fetch", "arguments": {"arguments": {"url": "https://example.com"}}}\n'
            '</tool_call>'
        )
        calls = hermes.parse_toolcalls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "fetch")
        self.assertEqual(json.loads(calls[0].function.arguments), {"url": "https://example.com"})

    def test_hermes_qwen_xml_fallback(self):
        text = (
            "<tool_call>\n"
            "<function=fetch>\n"
            "<parameter=url>https://example.com</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = hermes.parse_toolcalls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "fetch")
        self.assertEqual(json.loads(calls[0].function.arguments), {"url": "https://example.com"})

    def test_hermes_bare_json_fallback(self):
        text = '{"name": "fetch", "arguments": {"url": "https://example.com"}}'
        calls = hermes.parse_toolcalls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "fetch")
        self.assertEqual(json.loads(calls[0].function.arguments), {"url": "https://example.com"})

    def test_nanbeige_alias_registered(self):
        self.assertTrue(is_supported_format("nanbeige"))
        self.assertTrue(is_supported_format("nanbeige4"))

        text = (
            "<tool_call>\n"
            "<function=calc>\n"
            "<parameter=x>10</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = parse_toolcalls(text, "nanbeige")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "calc")
        self.assertEqual(json.loads(calls[0].function.arguments), {"x": 10})

    def test_max_tool_calls_capping(self):
        text = "".join(
            f"<tool_call><function=f{i}><parameter=v>{i}</parameter></function></tool_call>"
            for i in range(10)
        )
        # Without limit:
        all_calls = parse_toolcalls(text, "qwen3_coder")
        self.assertEqual(len(all_calls), 10)

        # With limit 3:
        capped_calls = parse_toolcalls(text, "qwen3_coder", max_tool_calls=3)
        self.assertEqual(len(capped_calls), 3)
        self.assertEqual([c.function.name for c in capped_calls], ["f0", "f1", "f2"])

    def test_chat_completion_request_model(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tool_calls=5,
        )
        self.assertEqual(req.max_tool_calls, 5)


if __name__ == "__main__":
    unittest.main()
