"""Tests for sorting tool-result messages into tool call order."""

import unittest

from endpoints.OAI.utils.chat_completion import _sort_tool_messages


def assistant(*ids):
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": i, "type": "function", "function": {"name": "f", "arguments": {}}} for i in ids
        ],
    }


def tool(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": call_id}


class SortToolMessagesTests(unittest.TestCase):
    def test_out_of_order_results_sorted(self):
        messages = [
            {"role": "user", "content": "hi"},
            assistant("a", "b", "c"),
            tool("c"),
            tool("a"),
            tool("b"),
        ]
        _sort_tool_messages(messages)
        self.assertEqual([m["tool_call_id"] for m in messages[2:]], ["a", "b", "c"])

    def test_in_order_results_untouched(self):
        messages = [assistant("a", "b"), tool("a"), tool("b")]
        _sort_tool_messages(messages)
        self.assertEqual([m["tool_call_id"] for m in messages[1:]], ["a", "b"])

    def test_unknown_ids_keep_relative_order_at_end(self):
        messages = [assistant("a", "b"), tool("x"), tool("b"), tool("y"), tool("a")]
        _sort_tool_messages(messages)
        self.assertEqual([m["tool_call_id"] for m in messages[1:]], ["a", "b", "x", "y"])

    def test_runs_sorted_independently(self):
        messages = [
            assistant("a", "b"),
            tool("b"),
            tool("a"),
            {"role": "assistant", "content": "done"},
            assistant("c", "d"),
            tool("d"),
            tool("c"),
        ]
        _sort_tool_messages(messages)
        self.assertEqual([m["tool_call_id"] for m in messages[1:3]], ["a", "b"])
        self.assertEqual([m["tool_call_id"] for m in messages[5:]], ["c", "d"])

    def test_calls_without_ids_left_alone(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "f"}}],
            },
            tool("b"),
            tool("a"),
        ]
        _sort_tool_messages(messages)
        self.assertEqual([m["tool_call_id"] for m in messages[1:]], ["b", "a"])

    def test_single_result_and_no_tools_noop(self):
        messages = [
            {"role": "user", "content": "hi"},
            assistant("a"),
            tool("a"),
            {"role": "assistant", "content": "done"},
        ]
        before = [dict(m) for m in messages]
        _sort_tool_messages(messages)
        self.assertEqual(messages, before)


if __name__ == "__main__":
    unittest.main()
