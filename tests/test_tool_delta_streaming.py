"""
Wiring tests for incremental tool_calls delta streaming in
_chat_stream_collector: fragment frames are emitted while generating, the
authoritative end-of-stream parse is skipped when fragments were streamed
(and kept as fallback when they were not), non-streaming and non-qwen
backends are unaffected.

The backend container is mocked, following the mock style of the other
unit tests in this directory.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest
from endpoints.OAI.utils import chat_completion as cc
from endpoints.OAI.utils.toolcall_formats import qwen3_coder
from endpoints.OAI.utils.tools import get_toolcall_tags

LT, GT = chr(60), chr(62)
# Wrapper tags as the server actually routes on (qwen3_coder's own tags), not
# invented ones: fixtures must match the format under test.
TC_S, TC_E = get_toolcall_tags("qwen3_coder")
RS, RE = LT + "|think|" + GT, LT + "|/think|" + GT

CITY = "Amsterdam and, well, Paris"


def tool_text():
    return (
        TC_S
        + f"{LT}function=get_weather{GT}"
        + f"{LT}parameter=city{GT}\n{CITY}{LT}/parameter{GT}"
        + f"{LT}/function{GT}"
        + TC_E
    )


def expected_args():
    return qwen3_coder.parse_toolcalls(tool_text())[0].function.arguments


def pieces(text, n=5):
    return [text[i : i + n] for i in range(0, len(text), n)]


def make_mc(chunks, **overrides):
    mc = SimpleNamespace(
        harmony=False,
        muse_glimmer=False,
        tool_format="qwen3_coder",
        reasoning=True,
        reasoning_start_token=RS,
        reasoning_end_token=RE,
        reasoning_budget_tokens=None,
        reasoning_budget_message=None,
        tool_calls_in_reasoning=True,
    )

    async def stream_generate(*args, **kwargs):
        for chunk in chunks:
            yield {"text": chunk, "token_ids": [1] * max(1, len(chunk) // 4)}
        yield {"text": "", "finish_reason": "stop", "eos_reason": "eot"}

    mc.stream_generate = stream_generate
    mc.constrain_generation_output = lambda *a, **k: False
    for key, value in overrides.items():
        setattr(mc, key, value)
    return mc


def make_request(**kwargs):
    kwargs.setdefault("messages", [ChatCompletionMessage(role="user", content="weather?")])
    kwargs.setdefault(
        "tools",
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )
    kwargs.setdefault("tool_choice", "auto")
    return ChatCompletionRequest(**kwargs)


async def run_collector(mc, params, streaming=True):
    queue = asyncio.Queue() if streaming else None
    with patch.object(cc, "model") as mocked:
        mocked.container = mc
        result = await cc._chat_stream_collector(
            0, queue, "req-1", "PROMPT", params, False, None, streaming, None
        )
    frames = []
    if queue is not None:
        while not queue.empty():
            frames.append(queue.get_nowait())
    return frames, result


class StreamingDeltaWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_streams_incremental_frames(self):
        chunks = pieces(RS + "thinking" + RE) + pieces(tool_text())
        mc = make_mc(chunks)

        frames, _ = await run_collector(mc, make_request())

        tool_frames = [f for f in frames if f.get("delta_tool_calls")]
        self.assertGreaterEqual(len(tool_frames), 2, "expected multiple delta frames")

        # reasoning arrives before any tool fragment
        first_reasoning = next(i for i, f in enumerate(frames) if f.get("delta_reasoning_content"))
        first_tool = next(i for i, f in enumerate(frames) if f.get("delta_tool_calls"))
        self.assertLess(first_reasoning, first_tool)

        # fragments assemble to the authoritative arguments
        assembled = ""
        ids = []
        for frame in tool_frames:
            for delta in frame["delta_tool_calls"]:
                if delta.get("id"):
                    ids.append(delta["id"])
                assembled += delta["function"].get("arguments", "")
        self.assertEqual(assembled, expected_args())
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids[0].startswith("call_"))

        # final frame: finish reason set, authoritative parse skipped
        final = frames[-1]
        self.assertEqual(final["finish_reason"], "tool_calls")
        self.assertEqual(final["delta_tool_calls"], "")

    async def test_fallback_when_streamer_emits_nothing(self):
        # tool-channel text without any function block: full_tool is
        # non-empty but the streamer produces no fragments, so the
        # authoritative end-of-stream parse still runs
        chunks = pieces(RS + RE) + pieces(TC_S + "just noise" + TC_E)
        mc = make_mc(chunks)

        frames, _ = await run_collector(mc, make_request())

        self.assertFalse([f for f in frames if f.get("delta_tool_calls")])
        final = frames[-1]
        self.assertEqual(final["finish_reason"], "tool_calls")
        self.assertEqual(final["delta_tool_calls"], [])  # authoritative parse: none

    async def test_tool_choice_none_keeps_old_path(self):
        chunks = pieces(RS + "thinking" + RE) + pieces(tool_text())
        mc = make_mc(chunks)

        frames, _ = await run_collector(mc, make_request(tool_choice="none"))

        self.assertFalse([f for f in frames if f.get("delta_tool_calls")])
        self.assertEqual(frames[-1]["finish_reason"], "stop")

    async def test_non_streaming_path_unchanged(self):
        chunks = pieces(RS + "thinking" + RE) + pieces(tool_text())
        mc = make_mc(chunks)

        _, result = await run_collector(mc, make_request(), streaming=False)

        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(result["reasoning_content"], "thinking")
        self.assertEqual(result["tool_calls"][0]["function"]["arguments"], expected_args())
        self.assertNotIn("delta_tool_calls", result)

    async def test_plain_content_no_tools(self):
        chunks = pieces(RS + "thinking" + RE + "plain answer")
        mc = make_mc(chunks)

        frames, _ = await run_collector(mc, make_request())

        content = "".join(f.get("delta_content", "") for f in frames)
        self.assertEqual(content, "plain answer")
        self.assertFalse([f for f in frames if f.get("delta_tool_calls")])
        self.assertEqual(frames[-1]["finish_reason"], "stop")

    async def test_glimmer_backend_unaffected(self):
        # regression: the muse_glimmer branch must define use_tool as well
        chunks = ["hello there"]
        mc = make_mc(chunks, muse_glimmer=True)

        frames, _ = await run_collector(mc, make_request())

        self.assertEqual(frames[-1]["finish_reason"], "stop")
        self.assertFalse([f for f in frames if f.get("delta_tool_calls")])

    async def test_harmony_backend_unaffected(self):
        # regression: harmony/glimmer branches must define use_tool
        chunks = ["hello there"]
        mc = make_mc(chunks, harmony=True)

        frames, _ = await run_collector(mc, make_request())

        self.assertEqual(frames[-1]["finish_reason"], "stop")
        self.assertFalse([f for f in frames if f.get("delta_tool_calls")])


if __name__ == "__main__":
    unittest.main()
