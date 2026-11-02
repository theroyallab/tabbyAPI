"""Tests for string-typed parameters in the qwen3_coder tool call format.

A parameter the request declares as ``"type": "string"`` must reach the tool
verbatim, even when its text happens to be valid JSON: writing a package.json
through a `write` tool must not turn its content into an object. Values whose
schema is not a string, and requests that pass no tools at all, keep the
previous coercion behaviour.
"""

import json
import unittest

from endpoints.OAI.types.tools import Function, ToolSpec
from endpoints.OAI.utils.tools import parse_toolcalls

PACKAGE_JSON = '{"name": "demo", "private": true}'


def write_tool(content_type="string"):
    """A `write` tool whose `content` parameter is declared as `content_type`."""
    return ToolSpec(
        type="function",
        function=Function(
            name="write",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": content_type},
                    "limit": {"type": "number"},
                },
            },
        ),
    )


def write_call(content=PACKAGE_JSON):
    """The official template's shape, newlines around the values included."""
    return (
        "<tool_call>\n<function=write>\n"
        "<parameter=path>\npackage.json\n</parameter>\n"
        f"<parameter=content>\n{content}\n</parameter>\n"
        "<parameter=limit>\n100\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def parse_args(text, tools=None):
    parsed = parse_toolcalls(text, "qwen3_coder", tools)
    assert len(parsed) == 1, parsed
    return json.loads(parsed[0].function.arguments)


class Qwen3CoderStringParamTests(unittest.TestCase):
    def test_string_param_holding_json_is_kept_verbatim(self):
        args = parse_args(write_call(), [write_tool()])
        self.assertEqual(args["content"], PACKAGE_JSON)
        self.assertIsInstance(args["content"], str)

    def test_other_params_are_still_coerced(self):
        # The fix must not disable coercion for values the schema types otherwise
        args = parse_args(write_call(), [write_tool()])
        self.assertEqual(args["limit"], 100)
        self.assertEqual(args["path"], "package.json")

    def test_non_string_schema_still_coerces(self):
        args = parse_args(write_call(), [write_tool(content_type="object")])
        self.assertEqual(args["content"], {"name": "demo", "private": True})

    def test_without_tools_behaviour_is_unchanged(self):
        # Back-compat: callers that pass no schema get the old coercion
        args = parse_args(write_call())
        self.assertEqual(args["content"], {"name": "demo", "private": True})

    def test_param_absent_from_the_schema_is_coerced(self):
        text = (
            "<tool_call>\n<function=write>\n"
            "<parameter=undeclared>\n42\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        self.assertEqual(parse_args(text, [write_tool()])["undeclared"], 42)

    def test_a_string_param_that_is_not_json_is_untouched(self):
        args = parse_args(write_call("hello world"), [write_tool()])
        self.assertEqual(args["content"], "hello world")


if __name__ == "__main__":
    unittest.main()
