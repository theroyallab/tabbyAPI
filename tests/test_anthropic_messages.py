import asyncio
import json
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from sse_starlette import EventSourceResponse

import common.model  # noqa: F401 - resolve import cycle ordering
from common.tabby_config import config
from endpoints.Anthropic.errors import AnthropicHTTPException, exception_to_response
from endpoints.Anthropic.router import count_tokens_request, messages_request as messages_endpoint
from endpoints.Anthropic.types.messages import (
    CountTokensRequest,
    MessagesRequest,
    ResponseTextBlock,
    ResponseThinkingBlock,
)
from endpoints.Anthropic.utils.convert import (
    _sampler_params,
    convert_count_tokens_request,
    convert_messages_request,
)
from endpoints.Anthropic.utils.messages import convert_response, stop_reason
from endpoints.Anthropic.utils.stream import ContentBlockTracker, stream_generate_message
from endpoints.OAI.types.chat_completion import (
    ChatCompletionMessage,
    ChatCompletionRespChoice,
    ChatCompletionResponse,
)
from endpoints.OAI.types.common import UsageStats

MODEL_DIR = pathlib.Path("/models/test-model")


def messages_request(**kwargs):
    kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
    kwargs.setdefault("max_tokens", 64)
    return MessagesRequest(**kwargs)


def choice(
    content="Hello",
    reasoning_content=None,
    finish_reason="stop",
    eos_reason="stop_token",
    stop_str=None,
):
    return ChatCompletionRespChoice(
        finish_reason=finish_reason,
        eos_reason=eos_reason,
        stop_str=stop_str,
        message=ChatCompletionMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
        ),
    )


def completion(usage=None, **kwargs):
    return ChatCompletionResponse(
        choices=[choice(**kwargs)],
        model="test-model",
        usage=usage,
    )


class SystemPromptTests(unittest.TestCase):
    def test_string_system(self):
        converted = convert_messages_request(messages_request(system="You are helpful."))
        self.assertEqual(converted.messages[0].role, "system")
        self.assertEqual(converted.messages[0].content, "You are helpful.")

    def test_block_list_system_joined_on_blank_line(self):
        converted = convert_messages_request(
            messages_request(
                system=[
                    {"type": "text", "text": "You are helpful."},
                    {"type": "text", "text": "<env>cwd=/tmp</env>"},
                ]
            )
        )
        self.assertEqual(converted.messages[0].content, "You are helpful.\n\n<env>cwd=/tmp</env>")

    def test_no_system_message_added_when_absent(self):
        converted = convert_messages_request(messages_request())
        self.assertEqual([m.role for m in converted.messages], ["user"])

    def test_empty_system_string_adds_no_message(self):
        converted = convert_messages_request(messages_request(system=""))
        self.assertEqual([m.role for m in converted.messages], ["user"])

    def test_cache_control_is_ignored(self):
        converted = convert_messages_request(
            messages_request(
                system=[
                    {
                        "type": "text",
                        "text": "cached",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            )
        )
        self.assertEqual(converted.messages[0].content, "cached")


class MessageContentTests(unittest.TestCase):
    def test_string_content(self):
        converted = convert_messages_request(
            messages_request(messages=[{"role": "user", "content": "hi"}])
        )
        self.assertEqual(converted.messages[0].content, "hi")

    def test_multiple_text_blocks_joined(self):
        converted = convert_messages_request(
            messages_request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "part one"},
                            {"type": "text", "text": "part two"},
                        ],
                    }
                ]
            )
        )
        self.assertEqual(converted.messages[0].content, "part one\n\npart two")

    def test_thinking_block_becomes_reasoning_content(self):
        converted = convert_messages_request(
            messages_request(
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                            {"type": "text", "text": "Hello!"},
                        ],
                    }
                ]
            )
        )
        message = converted.messages[0]
        self.assertEqual(message.reasoning_content, "hmm")
        self.assertEqual(message.content, "Hello!")

    def test_redacted_thinking_dropped(self):
        converted = convert_messages_request(
            messages_request(
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "redacted_thinking", "data": "encrypted"},
                            {"type": "text", "text": "Hello!"},
                        ],
                    }
                ]
            )
        )
        message = converted.messages[0]
        self.assertIsNone(message.reasoning_content)
        self.assertEqual(message.content, "Hello!")

    def test_unsupported_block_raises_invalid_request(self):
        with self.assertRaises(AnthropicHTTPException) as ctx:
            convert_messages_request(
                messages_request(
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "image", "source": {"type": "base64"}}],
                        }
                    ]
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.error_type, "invalid_request_error")
        self.assertIn("image", ctx.exception.detail)

    def test_tool_blocks_are_reported_as_unsupported(self):
        # Until tool support lands, a tool_result must fail loudly rather than
        # drop the tool output out of the conversation
        with self.assertRaises(AnthropicHTTPException) as ctx:
            convert_messages_request(
                messages_request(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "x", "content": "42"}
                            ],
                        }
                    ]
                )
            )
        self.assertIn("tool_result", ctx.exception.detail)


class SamplerMappingTests(unittest.TestCase):
    def test_unset_samplers_are_omitted(self):
        params = _sampler_params(messages_request())
        self.assertEqual(set(params), {"max_tokens"})

    def test_set_samplers_are_mapped(self):
        params = _sampler_params(
            messages_request(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                stop_sequences=["STOP"],
                metadata={"user_id": "u1"},
            )
        )
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_p"], 0.9)
        self.assertEqual(params["top_k"], 40)
        self.assertEqual(params["stop"], ["STOP"])
        self.assertEqual(params["user"], "u1")

    def test_request_shape(self):
        converted = convert_messages_request(messages_request(max_tokens=64))
        self.assertEqual(converted.max_tokens, 64)
        self.assertEqual(converted.n, 1)
        self.assertTrue(converted.stream_options.include_usage)

    def test_thinking_enabled(self):
        converted = convert_messages_request(messages_request(thinking={"type": "enabled"}))
        self.assertIs(converted.template_vars["enable_thinking"], True)

    def test_thinking_disabled(self):
        converted = convert_messages_request(messages_request(thinking={"type": "disabled"}))
        self.assertIs(converted.template_vars["enable_thinking"], False)

    def test_thinking_absent_sets_no_template_var(self):
        converted = convert_messages_request(messages_request())
        self.assertNotIn("enable_thinking", converted.template_vars)

    def test_count_tokens_request_conversion(self):
        converted = convert_count_tokens_request(
            CountTokensRequest(system="sys", messages=[{"role": "user", "content": "hi"}])
        )
        self.assertEqual([m.role for m in converted.messages], ["system", "user"])


class StopReasonTests(unittest.TestCase):
    def test_length_maps_to_max_tokens(self):
        reason, sequence = stop_reason("length", "max_new_tokens", None, None)
        self.assertEqual(reason, "max_tokens")
        self.assertIsNone(sequence)

    def test_tool_calls_maps_to_tool_use(self):
        reason, _ = stop_reason("tool_calls", None, None, None)
        self.assertEqual(reason, "tool_use")

    def test_eos_token_maps_to_end_turn(self):
        reason, sequence = stop_reason("stop", "stop_token", "<|im_end|>", ["STOP"])
        self.assertEqual(reason, "end_turn")
        self.assertIsNone(sequence)

    def test_client_stop_sequence_is_reported(self):
        reason, sequence = stop_reason("stop", "stop_string", "STOP", ["STOP"])
        self.assertEqual(reason, "stop_sequence")
        self.assertEqual(sequence, "STOP")

    def test_template_stop_string_is_not_reported_as_stop_sequence(self):
        # The prompt template contributes stop strings the client never sent;
        # naming one in stop_sequence would be a lie
        reason, sequence = stop_reason("stop", "stop_string", "<|end|>", ["STOP"])
        self.assertEqual(reason, "end_turn")
        self.assertIsNone(sequence)

    def test_streaming_and_non_streaming_agree(self):
        # Both paths must derive the stop reason from the same inputs
        c = choice(finish_reason="length")
        response = convert_response(
            ChatCompletionResponse(choices=[c], model="test-model"),
            messages_request(),
            "test-model",
        )
        streamed, _ = stop_reason(c.finish_reason, c.eos_reason, c.stop_str, None)
        self.assertEqual(response.stop_reason, streamed)


class ConvertResponseTests(unittest.TestCase):
    def test_thinking_precedes_text(self):
        response = convert_response(
            completion(content="Hello", reasoning_content="hmm"),
            messages_request(),
            "test-model",
        )
        self.assertIsInstance(response.content[0], ResponseThinkingBlock)
        self.assertEqual(response.content[0].thinking, "hmm")
        self.assertEqual(response.content[0].signature, "")
        self.assertIsInstance(response.content[1], ResponseTextBlock)
        self.assertEqual(response.content[1].text, "Hello")

    def test_text_only(self):
        response = convert_response(completion(), messages_request(), "test-model")
        self.assertEqual(len(response.content), 1)
        self.assertEqual(response.content[0].type, "text")

    def test_empty_content(self):
        response = convert_response(completion(content=None), messages_request(), "test-model")
        self.assertEqual(response.content, [])

    def test_envelope_fields(self):
        response = convert_response(completion(), messages_request(), "test-model")
        self.assertTrue(response.id.startswith("msg_"))
        self.assertEqual(response.type, "message")
        self.assertEqual(response.role, "assistant")
        self.assertEqual(response.model, "test-model")

    def test_usage_mapping(self):
        response = convert_response(
            completion(usage=UsageStats(prompt_tokens=12, completion_tokens=5, total_tokens=17)),
            messages_request(),
            "test-model",
        )
        self.assertEqual(response.usage.input_tokens, 12)
        self.assertEqual(response.usage.output_tokens, 5)
        self.assertEqual(response.usage.cache_read_input_tokens, 0)

    def test_missing_usage_defaults_to_zero(self):
        response = convert_response(completion(usage=None), messages_request(), "test-model")
        self.assertEqual(response.usage.input_tokens, 0)


class FakeRequest:
    """Minimal stand-in for the parts of Request the endpoints touch."""

    def __init__(self, body=None):
        self.body = body or {}
        self.state = SimpleNamespace(id="test-request")

    async def json(self):
        return self.body

    async def is_disconnected(self):
        return False


class EndpointTests(unittest.TestCase):
    """Checks over the endpoint functions with inference stubbed out."""

    def setUp(self):
        container = SimpleNamespace(
            prompt_template=SimpleNamespace(name="test"),
            model_dir=MODEL_DIR,
            encode_tokens=lambda text, **kwargs: list(range(7)),
            validate_context_length=lambda *args, **kwargs: 12,
        )

        async def fake_apply_chat_template(data):
            return "PROMPT", None

        async def fake_generate(*args, **kwargs):
            return completion(
                usage=UsageStats(prompt_tokens=12, completion_tokens=5, total_tokens=17)
            )

        async def fake_check_model_container():
            return None

        async def fake_load_inline_model(model_name, request):
            return None

        patches = [
            patch.object(common.model, "container", container),
            patch("endpoints.Anthropic.router.check_model_container", fake_check_model_container),
            patch("endpoints.Anthropic.router.load_inline_model", fake_load_inline_model),
            patch("endpoints.Anthropic.router.apply_chat_template", fake_apply_chat_template),
            patch("endpoints.Anthropic.router.generate_chat_completion", fake_generate),
            patch(
                "endpoints.Anthropic.utils.messages.apply_chat_template",
                fake_apply_chat_template,
            ),
        ]
        for entry in patches:
            entry.start()
            self.addCleanup(entry.stop)

    def test_messages_success(self):
        data = messages_request(model="test-model")
        response = asyncio.run(messages_endpoint(FakeRequest(), data))

        body = response.model_dump()
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertEqual(body["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(body["stop_reason"], "end_turn")
        self.assertEqual(body["model"], MODEL_DIR.name)
        self.assertEqual(body["usage"]["input_tokens"], 12)
        self.assertEqual(body["usage"]["output_tokens"], 5)

    def test_count_tokens(self):
        data = CountTokensRequest(messages=[{"role": "user", "content": "hi"}])
        response = asyncio.run(count_tokens_request(FakeRequest(), data))

        self.assertEqual(response.input_tokens, 7)

    def test_streaming_returns_an_event_stream(self):
        response = asyncio.run(messages_endpoint(FakeRequest(), messages_request(stream=True)))

        self.assertIsInstance(response, EventSourceResponse)

    def test_streaming_rejected_when_disabled_in_config(self):
        # Returning a non-streaming body to a client expecting SSE would fail
        # in the client's parser rather than say what went wrong
        with patch.object(config.developer, "disable_request_streaming", True):
            with self.assertRaises(AnthropicHTTPException) as ctx:
                asyncio.run(messages_endpoint(FakeRequest(), messages_request(stream=True)))

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.error_type, "invalid_request_error")

    def test_missing_prompt_template_rejected(self):
        with patch.object(
            common.model,
            "container",
            SimpleNamespace(prompt_template=None, model_dir=MODEL_DIR),
        ):
            with self.assertRaises(AnthropicHTTPException) as ctx:
                asyncio.run(messages_endpoint(FakeRequest(), messages_request()))

        self.assertEqual(ctx.exception.status_code, 422)

    def test_load_lock_released_on_error(self):
        # A failure past the lock must not wedge every later request
        from endpoints.Anthropic.router import load_lock

        with patch.object(
            common.model,
            "container",
            SimpleNamespace(prompt_template=None, model_dir=MODEL_DIR),
        ):
            with self.assertRaises(AnthropicHTTPException):
                asyncio.run(messages_endpoint(FakeRequest(), messages_request()))

        self.assertFalse(load_lock.locked())


class ContentBlockTrackerTests(unittest.TestCase):
    def names(self, events):
        return [event.event for event in events]

    def test_blocks_open_lazily(self):
        # A response without reasoning must put its text at index 0
        tracker = ContentBlockTracker()
        events = tracker.write("text", "Hello")

        self.assertEqual(self.names(events), ["content_block_start", "content_block_delta"])
        self.assertEqual(json.loads(events[0].data)["index"], 0)
        self.assertEqual(json.loads(events[0].data)["content_block"]["type"], "text")

    def test_empty_text_emits_nothing(self):
        tracker = ContentBlockTracker()
        self.assertEqual(tracker.write("text", ""), [])
        self.assertEqual(tracker.write("thinking", ""), [])

    def test_switching_kind_closes_previous_block(self):
        tracker = ContentBlockTracker()
        events = tracker.write("thinking", "hm") + tracker.write("text", "Hi")

        self.assertEqual(
            self.names(events),
            [
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
            ],
        )
        self.assertEqual(json.loads(events[2].data)["index"], 0)
        self.assertEqual(json.loads(events[3].data)["index"], 1)

    def test_same_kind_reuses_open_block(self):
        tracker = ContentBlockTracker()
        events = tracker.write("text", "a") + tracker.write("text", "b")

        self.assertEqual(
            self.names(events),
            ["content_block_start", "content_block_delta", "content_block_delta"],
        )
        self.assertTrue(all(json.loads(e.data)["index"] == 0 for e in events))

    def test_close_without_open_block_emits_nothing(self):
        self.assertEqual(ContentBlockTracker().close(), [])

    def test_close_is_not_repeated(self):
        tracker = ContentBlockTracker()
        tracker.write("text", "a")
        self.assertEqual(len(tracker.close()), 1)
        self.assertEqual(tracker.close(), [])

    def test_thinking_block_carries_empty_signature(self):
        events = ContentBlockTracker().write("thinking", "hm")
        self.assertEqual(json.loads(events[0].data)["content_block"]["signature"], "")


class FakeDisconnectHandler:
    def __init__(self):
        self.cleaned = False

    async def cleanup(self):
        self.cleaned = True


def run_stream(data, chunks, input_tokens=12):
    """Drive the stream generator with a stubbed collector."""

    async def fake_collector(task_idx, gen_queue, *args, **kwargs):
        for chunk in chunks:
            await gen_queue.put(chunk)

    handler = FakeDisconnectHandler()

    async def drive():
        events = []
        with (
            patch("endpoints.Anthropic.utils.stream._chat_stream_collector", fake_collector),
            patch(
                "endpoints.Anthropic.utils.stream._resolve_start_in_reasoning",
                lambda prompt, params: False,
            ),
        ):
            generator = stream_generate_message(
                "PROMPT",
                None,
                data,
                convert_messages_request(data),
                FakeRequest(),
                MODEL_DIR,
                handler,
                input_tokens,
            )
            async for event in generator:
                events.append((event.event, json.loads(event.data)))

        return events

    return asyncio.run(drive()), handler


def finish_chunk(**kwargs):
    chunk = {
        "index": 0,
        "finish_reason": "stop",
        "eos_reason": "stop_token",
        "stop_str": None,
        "prompt_tokens": 12,
        "gen_tokens": 5,
        "delta_content": "",
        "delta_reasoning_content": "",
    }
    chunk.update(kwargs)
    return chunk


class StreamEventTests(unittest.TestCase):
    def test_full_event_sequence(self):
        events, handler = run_stream(
            messages_request(),
            [
                {"index": 0, "delta_content": "Hel", "delta_reasoning_content": ""},
                {"index": 0, "delta_content": "lo", "delta_reasoning_content": ""},
                finish_chunk(),
            ],
        )

        self.assertEqual(
            [name for name, _ in events],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        self.assertTrue(handler.cleaned)

    def test_message_start_carries_input_tokens(self):
        events, _ = run_stream(messages_request(), [finish_chunk()], input_tokens=99)

        name, payload = events[0]
        self.assertEqual(name, "message_start")
        self.assertEqual(payload["message"]["usage"]["input_tokens"], 99)
        self.assertEqual(payload["message"]["usage"]["output_tokens"], 0)
        self.assertEqual(payload["message"]["role"], "assistant")
        self.assertEqual(payload["message"]["content"], [])
        self.assertTrue(payload["message"]["id"].startswith("msg_"))
        self.assertEqual(payload["message"]["model"], MODEL_DIR.name)

    def test_no_done_sentinel(self):
        # Anthropic terminates on message_stop; a [DONE] would be an OAI-ism
        events, _ = run_stream(messages_request(), [finish_chunk()])
        self.assertEqual(events[-1][0], "message_stop")

    def test_reasoning_then_content_uses_two_blocks(self):
        events, _ = run_stream(
            messages_request(),
            [
                {"index": 0, "delta_reasoning_content": "hm", "delta_content": ""},
                {"index": 0, "delta_reasoning_content": "", "delta_content": "Hi"},
                finish_chunk(),
            ],
        )

        starts = [p for n, p in events if n == "content_block_start"]
        self.assertEqual([s["content_block"]["type"] for s in starts], ["thinking", "text"])
        self.assertEqual([s["index"] for s in starts], [0, 1])

        deltas = [p["delta"] for n, p in events if n == "content_block_delta"]
        self.assertEqual(deltas[0], {"type": "thinking_delta", "thinking": "hm"})
        self.assertEqual(deltas[1], {"type": "text_delta", "text": "Hi"})

    def test_message_delta_reports_stop_reason_and_usage(self):
        events, _ = run_stream(
            messages_request(),
            [{"index": 0, "delta_content": "Hi", "delta_reasoning_content": ""}, finish_chunk()],
        )

        payload = dict(events)["message_delta"]
        self.assertEqual(payload["delta"]["stop_reason"], "end_turn")
        self.assertIsNone(payload["delta"]["stop_sequence"])
        self.assertEqual(payload["usage"]["output_tokens"], 5)
        self.assertEqual(payload["usage"]["input_tokens"], 12)

    def test_max_tokens_stop_reason(self):
        events, _ = run_stream(
            messages_request(),
            [finish_chunk(finish_reason="length", eos_reason="max_new_tokens")],
        )
        self.assertEqual(dict(events)["message_delta"]["delta"]["stop_reason"], "max_tokens")

    def test_client_stop_sequence_reported_in_message_delta(self):
        events, _ = run_stream(
            messages_request(stop_sequences=["END"]),
            [finish_chunk(eos_reason="stop_string", stop_str="END")],
        )

        delta = dict(events)["message_delta"]["delta"]
        self.assertEqual(delta["stop_reason"], "stop_sequence")
        self.assertEqual(delta["stop_sequence"], "END")

    def test_collector_exception_becomes_error_event(self):
        events, handler = run_stream(messages_request(), [RuntimeError("backend exploded")])

        names = [name for name, _ in events]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "error")
        self.assertEqual(dict(events)["error"]["type"], "error")
        self.assertEqual(dict(events)["error"]["error"]["type"], "api_error")
        self.assertTrue(handler.cleaned)

    def test_collector_finishing_without_finish_chunk_closes_stream(self):
        # Must not wait forever on a chunk that will never arrive
        events, _ = run_stream(
            messages_request(),
            [{"index": 0, "delta_content": "Hi", "delta_reasoning_content": ""}],
        )

        names = [name for name, _ in events]
        self.assertEqual(names[-1], "message_stop")
        self.assertIn("content_block_stop", names)

    def test_open_block_is_closed_before_message_delta(self):
        events, _ = run_stream(
            messages_request(),
            [{"index": 0, "delta_content": "Hi", "delta_reasoning_content": ""}, finish_chunk()],
        )

        names = [name for name, _ in events]
        self.assertLess(names.index("content_block_stop"), names.index("message_delta"))


class ErrorEnvelopeTests(unittest.TestCase):
    def test_anthropic_exception_keeps_its_type(self):
        status, content = exception_to_response(
            AnthropicHTTPException(429, "slow down", "rate_limit_error")
        )
        self.assertEqual(status, 429)
        self.assertEqual(content["type"], "error")
        self.assertEqual(content["error"]["type"], "rate_limit_error")
        self.assertEqual(content["error"]["message"], "slow down")

    def test_shared_http_exception_is_reshaped(self):
        # Raised by check_api_key, which knows nothing about this API
        status, content = exception_to_response(HTTPException(401, "Invalid API key"))
        self.assertEqual(status, 401)
        self.assertEqual(content["error"]["type"], "authentication_error")
        self.assertEqual(content["error"]["message"], "Invalid API key")

    def test_template_failure_maps_to_invalid_request(self):
        status, content = exception_to_response(HTTPException(422, "TemplateError: boom"))
        self.assertEqual(status, 422)
        self.assertEqual(content["error"]["type"], "invalid_request_error")

    def test_unloaded_model_maps_to_api_error(self):
        _, content = exception_to_response(HTTPException(503, "no model"))
        self.assertEqual(content["error"]["type"], "api_error")

    def test_validation_error_is_reshaped(self):
        status, content = exception_to_response(RequestValidationError([]))
        self.assertEqual(status, 422)
        self.assertEqual(content["error"]["type"], "invalid_request_error")

    def test_unknown_exception_is_reraised(self):
        with self.assertRaises(ValueError):
            exception_to_response(ValueError("not an HTTP error"))


if __name__ == "__main__":
    unittest.main()
