import asyncio
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

import common.model  # noqa: F401 - resolve import cycle ordering
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
from endpoints.Anthropic.utils.messages import _stop_reason, convert_response
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
        reason, sequence = _stop_reason(choice(finish_reason="length"), None)
        self.assertEqual(reason, "max_tokens")
        self.assertIsNone(sequence)

    def test_tool_calls_maps_to_tool_use(self):
        reason, _ = _stop_reason(choice(finish_reason="tool_calls"), None)
        self.assertEqual(reason, "tool_use")

    def test_eos_token_maps_to_end_turn(self):
        reason, sequence = _stop_reason(
            choice(eos_reason="stop_token", stop_str="<|im_end|>"), ["STOP"]
        )
        self.assertEqual(reason, "end_turn")
        self.assertIsNone(sequence)

    def test_client_stop_sequence_is_reported(self):
        reason, sequence = _stop_reason(choice(eos_reason="stop_string", stop_str="STOP"), ["STOP"])
        self.assertEqual(reason, "stop_sequence")
        self.assertEqual(sequence, "STOP")

    def test_template_stop_string_is_not_reported_as_stop_sequence(self):
        # The prompt template contributes stop strings the client never sent;
        # naming one in stop_sequence would be a lie
        reason, sequence = _stop_reason(
            choice(eos_reason="stop_string", stop_str="<|end|>"), ["STOP"]
        )
        self.assertEqual(reason, "end_turn")
        self.assertIsNone(sequence)


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

    def test_streaming_rejected(self):
        data = messages_request(stream=True)

        with self.assertRaises(AnthropicHTTPException) as ctx:
            asyncio.run(messages_endpoint(FakeRequest(), data))

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
