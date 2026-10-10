import unittest

from endpoints.OAI.utils.chat_completion import _compose_serialize_stream_chunk

TOOL_CALL = [
    {
        "id": "call_0",
        "type": "function",
        "index": 0,
        "function": {"name": "web_search", "arguments": '{"query": "x"}'},
    }
]


class StreamRoleTests(unittest.TestCase):
    def test_first_delta_carries_assistant_role(self):
        _, data, _, is_empty = _compose_serialize_stream_chunk(
            "request-id",
            {"index": 0, "delta_tool_calls": TOOL_CALL, "finish_reason": "tool_calls"},
            include_role=True,
        )

        self.assertFalse(is_empty)
        delta = data["choices"][0]["delta"]
        self.assertEqual(delta["role"], "assistant")
        self.assertEqual(delta["tool_calls"], TOOL_CALL)

    def test_later_deltas_carry_no_role(self):
        _, data, _, _ = _compose_serialize_stream_chunk(
            "request-id", {"index": 0, "delta_content": "Hi"}
        )

        self.assertEqual(data["choices"][0]["delta"], {"content": "Hi"})

    def test_empty_delta_stays_empty(self):
        _, data, _, is_empty = _compose_serialize_stream_chunk(
            "request-id", {"index": 0}, include_role=True
        )

        self.assertTrue(is_empty)
        self.assertEqual(data["choices"][0]["delta"], {})


if __name__ == "__main__":
    unittest.main()
