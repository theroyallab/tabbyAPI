import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException

from common import model
from common.errors import (
    ContextLengthExceededError,
    ContextLengthHTTPException,
    context_length_exception_handler,
    validate_context_requirements,
)
from common.networking import get_context_length_generator_error
from endpoints.Kobold.utils import generation as kobold_generation
from endpoints.OAI.utils import chat_completion, completion


class DummyDisconnectHandler:
    async def cleanup(self):
        pass


class DummyRequestData:
    n = 1
    stream_options = None

    def model_copy(self, deep=False):
        return self

    def model_dump(self, mode=None):
        return {}


def request_with_id(request_id="request-id"):
    return SimpleNamespace(state=SimpleNamespace(id=request_id))


class ContextLengthErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_length_http_error_uses_openai_error_shape(self):
        app = FastAPI()
        app.add_exception_handler(ContextLengthHTTPException, context_length_exception_handler)

        @app.get("/")
        async def raise_context_error():
            raise ContextLengthHTTPException("Prompt exceeds the available context size")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "message": "Prompt exceeds the available context size",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "context_length_exceeded",
                }
            },
        )

    def test_context_length_stream_error_uses_openai_error_shape(self):
        message = "Prompt exceeds the available context size"
        self.assertEqual(
            httpx.Response(200, text=get_context_length_generator_error(message)).json(),
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "context_length_exceeded",
                }
            },
        )

    def test_prompt_limit_message_is_litellm_compatible(self):
        with self.assertRaises(ContextLengthExceededError) as raised:
            validate_context_requirements(4097, 4096, 4096)

        self.assertIn("exceeds the available context size", str(raised.exception))
        self.assertIn("maximum context length", str(raised.exception))

    def test_preflight_only_requires_room_to_start(self):
        """The engine caps the response at the room left in the cache, so a request
        asking for more than fits is served up to the room, not rejected; only a
        prompt that leaves no room at all is."""

        validate_context_requirements(3500, 8192, 4096)
        # The engine fills the cache to one token short of the end (its budget is
        # max_new_tokens + 1), so a 4,094-token prompt can still produce one token
        validate_context_requirements(4094, 8192, 4096)
        with self.assertRaises(ContextLengthExceededError) as raised:
            validate_context_requirements(4095, 8192, 4096)
        self.assertIn("leaves no room to generate", str(raised.exception))
        self.assertIn("needs 4097 cache tokens", str(raised.exception))
        self.assertIn("exceeds the available context size of 4096 tokens", str(raised.exception))

    def test_streaming_preflight_returns_400_for_context_length_error(self):
        error = ContextLengthExceededError("Prompt length 9 is greater than max_seq_len 8")
        container = SimpleNamespace(
            validate_context_length=lambda *args: (_ for _ in ()).throw(error)
        )

        original_container = model.container
        model.container = container
        try:
            with self.assertRaises(HTTPException) as raised:
                model.check_context_length("prompt", DummyRequestData())
        finally:
            model.container = original_container

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, str(error))

    def test_streaming_preflight_checks_each_batched_prompt(self):
        checked_prompts = []
        container = SimpleNamespace(
            validate_context_length=lambda prompt, *args: checked_prompts.append(prompt)
        )

        original_container = model.container
        model.container = container
        try:
            model.check_context_length(["first", "second"], DummyRequestData())
        finally:
            model.container = original_container

        self.assertEqual(checked_prompts, ["first", "second"])

    async def test_completion_returns_400_for_context_length_error(self):
        error = ContextLengthExceededError("Prompt length 9 is greater than max_seq_len 8")

        async def collector(*args, **kwargs):
            return error

        with patch.object(completion, "_stream_collector", collector):
            with self.assertRaises(HTTPException) as raised:
                await completion.generate_completion(
                    "prompt",
                    DummyRequestData(),
                    request_with_id(),
                    Path("model"),
                    DummyDisconnectHandler(),
                )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, str(error))

    async def test_chat_completion_returns_400_for_context_length_error(self):
        error = ContextLengthExceededError("Prompt length 9 is greater than max_seq_len 8")

        async def collector(*args, **kwargs):
            return error

        original_container = model.container
        model.container = SimpleNamespace(reasoning=False, harmony=False, muse_glimmer=False)
        try:
            with patch.object(chat_completion, "_chat_stream_collector", collector):
                with self.assertRaises(HTTPException) as raised:
                    await chat_completion.generate_chat_completion(
                        "prompt",
                        None,
                        DummyRequestData(),
                        request_with_id(),
                        Path("model"),
                        DummyDisconnectHandler(),
                    )
        finally:
            model.container = original_container

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, str(error))

    async def test_kobold_generation_returns_400_for_context_length_error(self):
        error = ContextLengthExceededError("Prompt length 9 is greater than max_seq_len 8")
        data = SimpleNamespace(genkey=None)

        async def collector(*args, **kwargs):
            raise error
            yield

        with patch.object(kobold_generation, "_stream_collector", collector):
            with self.assertRaises(HTTPException) as raised:
                await kobold_generation.get_generation(data, request_with_id())

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, str(error))

    async def test_other_completion_errors_remain_503(self):
        async def collector(*args, **kwargs):
            return ValueError("backend failure")

        with patch.object(completion, "_stream_collector", collector):
            with self.assertRaises(HTTPException) as raised:
                await completion.generate_completion(
                    "prompt",
                    DummyRequestData(),
                    request_with_id(),
                    Path("model"),
                    DummyDisconnectHandler(),
                )

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()


class DraftHeadroomTests(unittest.TestCase):
    """A drafter's speculative window is written past the budget, so it has to fit
    in the cache beyond the prompt as well (rq_headroom = max_draft_tokens)."""

    def test_the_opencode_case_now_starts(self):
        # 61,446 prompt tokens in a 65,536 cache with a 31-token hybrid draft window: the
        # aligned chunk budget reached the end of the cache and the engine refused the job
        # outright; the engine now cuts the final segment to fit, so the prompt is accepted
        validate_context_requirements(61446, 65536, 65536, draft_headroom=31)

    def test_window_counts_against_the_room(self):
        validate_context_requirements(65503, 65536, 65536, draft_headroom=31)
        with self.assertRaises(ContextLengthExceededError) as raised:
            validate_context_requirements(65504, 65536, 65536, draft_headroom=31)
        self.assertIn("31-token speculative window", str(raised.exception))
        self.assertIn("needs 65537 cache tokens", str(raised.exception))
        # Without a drafter the same prompt fits
        validate_context_requirements(65504, 65536, 65536)
