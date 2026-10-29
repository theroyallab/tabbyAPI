import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from common import model
from common.errors import (
    LoopDetectedError,
    LoopDetectedHTTPException,
    loop_detected_exception_handler,
)
from common.networking import get_loop_detected_generator_error
from common.sampling import BaseSamplerRequest
from endpoints.Kobold.utils import generation as kobold_generation
from endpoints.OAI.utils import chat_completion, completion
from endpoints.OAI.utils.chat_completion import _compose_serialize_stream_chunk
from endpoints.OAI.utils.completion import _compose_response

LOOP_MESSAGE = "Generation failed: the model started repeating itself"
LOOP_ERROR = {
    "error": {
        "message": LOOP_MESSAGE,
        "type": "server_error",
        "param": None,
        "code": "loop_detected",
    }
}


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


class LoopDetectWindowTests(unittest.TestCase):
    def test_default_loop_detect_window_maps_to_exllamav3_stop_on_loop(self):
        request = BaseSamplerRequest()

        self.assertEqual(request.loop_detect_window, 800)
        self.assertEqual(request.get_stop_on_loop(), (800, 2))
        self.assertFalse(request.loop_detect_error)

    def test_loop_detect_window_can_be_disabled(self):
        request = BaseSamplerRequest(loop_detect_window=0)

        self.assertIsNone(request.get_stop_on_loop())

    def test_completion_response_exposes_loop_detected_eos_reason(self):
        response = _compose_response(
            "request-id",
            [
                {
                    "index": 0,
                    "finish_reason": "repetition",
                    "eos_reason": "loop_detected",
                    "content": "",
                }
            ],
            "model",
            False,
        )

        self.assertEqual(response.choices[0].finish_reason, "repetition")
        self.assertEqual(response.choices[0].eos_reason, "loop_detected")

    def test_stream_chunk_exposes_loop_detected_eos_reason(self):
        chunk, data, finish_reason, is_empty = _compose_serialize_stream_chunk(
            "request-id",
            {
                "index": 0,
                "finish_reason": "repetition",
                "eos_reason": "loop_detected",
            },
        )

        self.assertEqual(finish_reason, "repetition")
        self.assertFalse(is_empty)
        self.assertIn('"eos_reason": "loop_detected"', chunk)
        self.assertEqual(data["choices"][0]["eos_reason"], "loop_detected")


@unittest.skipUnless(importlib.util.find_spec("exllamav3"), "needs exllamav3")
class LoopFinishReasonTests(unittest.TestCase):
    def test_loop_finishes_with_repetition(self):
        from backends.exllamav3.model import ExllamaV3Container

        container = ExllamaV3Container.__new__(ExllamaV3Container)
        result = {
            "eos_reason": "loop_detected",
            "prompt_tokens": 10,
            "cached_tokens": 0,
            "time_prefill": 0.1,
            "new_tokens": 800,
            "time_generate": 1.0,
            "time_enqueued": 0.0,
        }

        chunk = container.handle_finish_chunk(result, "request-id", "text")
        self.assertEqual(chunk["finish_reason"], "repetition")
        self.assertEqual(chunk["eos_reason"], "loop_detected")

        result["eos_reason"] = "max_new_tokens"
        self.assertEqual(
            container.handle_finish_chunk(result, "request-id", "text")["finish_reason"],
            "length",
        )


class LoopDetectErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_error_is_a_500_in_openai_shape(self):
        app = FastAPI()
        app.add_exception_handler(LoopDetectedHTTPException, loop_detected_exception_handler)

        @app.get("/")
        async def raise_loop_error():
            raise LoopDetectedHTTPException(LOOP_MESSAGE)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), LOOP_ERROR)

    def test_stream_error_uses_the_same_shape(self):
        self.assertEqual(
            httpx.Response(200, text=get_loop_detected_generator_error(LOOP_MESSAGE)).json(),
            LOOP_ERROR,
        )

    async def test_completion_fails_with_loop_error(self):
        async def collector(*args, **kwargs):
            return LoopDetectedError(LOOP_MESSAGE)

        with patch.object(completion, "_stream_collector", collector):
            with self.assertRaises(LoopDetectedHTTPException) as raised:
                await completion.generate_completion(
                    "prompt",
                    DummyRequestData(),
                    request_with_id(),
                    Path("model"),
                    DummyDisconnectHandler(),
                )

        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail, LOOP_MESSAGE)

    async def test_chat_completion_fails_with_loop_error(self):
        async def collector(*args, **kwargs):
            return LoopDetectedError(LOOP_MESSAGE)

        original_container = model.container
        model.container = SimpleNamespace(reasoning=False, harmony=False, muse_glimmer=False)
        try:
            with patch.object(chat_completion, "_chat_stream_collector", collector):
                with self.assertRaises(LoopDetectedHTTPException) as raised:
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

        self.assertEqual(raised.exception.status_code, 500)

    async def test_chat_stream_ends_with_loop_error_after_the_output(self):
        async def collector(index, gen_queue, *args, **kwargs):
            await gen_queue.put({"index": 0, "delta_content": "again and again"})
            await gen_queue.put(LoopDetectedError(LOOP_MESSAGE))

        original_container = model.container
        model.container = SimpleNamespace(reasoning=False, harmony=False, muse_glimmer=False)
        try:
            with patch.object(chat_completion, "_chat_stream_collector", collector):
                chunks = [
                    chunk
                    async for chunk in chat_completion.stream_generate_chat_completion(
                        "prompt",
                        None,
                        DummyRequestData(),
                        request_with_id(),
                        Path("model"),
                        DummyDisconnectHandler(),
                    )
                ]
        finally:
            model.container = original_container

        self.assertIn("again and again", chunks[0])
        self.assertEqual(httpx.Response(200, text=chunks[-1]).json(), LOOP_ERROR)

    async def test_kobold_generation_fails_with_loop_error(self):
        data = SimpleNamespace(genkey=None)

        async def collector(*args, **kwargs):
            raise LoopDetectedError(LOOP_MESSAGE)
            yield

        with patch.object(kobold_generation, "_stream_collector", collector):
            with self.assertRaises(LoopDetectedHTTPException) as raised:
                await kobold_generation.get_generation(data, request_with_id())

        self.assertEqual(raised.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
