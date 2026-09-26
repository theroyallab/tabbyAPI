import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from backends.exllamav3.grammar import ExLlamaV3Grammar
from common import model
from common.errors import GrammarParseError
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


def _raising_filter(message="Invalid grammar: boom"):
    def factory(*args, **kwargs):
        raise ValueError(message)

    return factory


class GrammarParseErrorUnitTests(unittest.TestCase):
    """A rejected schema/regex/grammar must fail the request, not silently
    degrade to unconstrained generation."""

    def setUp(self):
        self.handler = ExLlamaV3Grammar()
        self.tokenizer = SimpleNamespace(
            actual_vocab_size=4,
            extended_id_to_piece={0: "a", 1: "b", 2: " ", 3: "\n"},
        )

    def test_bad_json_schema_raises(self):
        with patch("backends.exllamav3.grammar.LLGuidanceFilter", _raising_filter()):
            with self.assertRaises(GrammarParseError):
                self.handler.add_json_schema_filter({"type": "object"}, self.tokenizer)
        self.assertEqual(self.handler.filters, [])

    def test_bad_regex_raises(self):
        with patch("backends.exllamav3.grammar.LLGuidanceFilter", _raising_filter()):
            with self.assertRaises(GrammarParseError):
                self.handler.add_regex_filter("[unclosed", self.tokenizer)
        self.assertEqual(self.handler.filters, [])

    def test_bad_grammar_raises(self):
        with patch("backends.exllamav3.grammar.LLGuidanceFilter", _raising_filter()):
            with self.assertRaises(GrammarParseError):
                self.handler.add_grammar_filter("rule ::= ", self.tokenizer)
        self.assertEqual(self.handler.filters, [])

    def test_valid_schema_appends_filter(self):
        sentinel = SimpleNamespace(
            name="filter",
            tokenizer=None,
            trigger_token=None,
            prefix_str=None,
            eos_after_completed=True,
        )
        with patch(
            "backends.exllamav3.grammar.LLGuidanceFilter",
            lambda *args, **kwargs: sentinel,
        ):
            self.handler.add_json_schema_filter({"type": "object"}, self.tokenizer)
        self.assertEqual(self.handler.filters, [sentinel])

    def test_missing_llguidance_dependency_is_not_masked(self):
        # A server-side environment problem must keep its original identity
        # (it maps to a server error), not a client-facing parse error.
        with patch("backends.exllamav3.grammar.LLGuidanceFilter", _raising_filter()):
            with patch(
                "backends.exllamav3.grammar._llguidance_ready", return_value=False
            ):
                with self.assertRaises(ValueError) as raised:
                    self.handler.add_json_schema_filter({"type": "object"}, self.tokenizer)
        self.assertNotIsInstance(raised.exception, GrammarParseError)


class GrammarParseErrorEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Grammar parse errors are client errors (400), unlike generation faults."""

    async def test_chat_completion_returns_400_for_grammar_parse_error(self):
        async def collector(*args, **kwargs):
            raise GrammarParseError(
                "The JSON schema could not be compiled: Invalid grammar: boom"
            )

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
        self.assertIn("could not be compiled", raised.exception.detail)

    async def test_completion_returns_400_for_grammar_parse_error(self):
        async def collector(*args, **kwargs):
            raise GrammarParseError(
                "The JSON schema could not be compiled: Invalid grammar: boom"
            )

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
        self.assertIn("could not be compiled", raised.exception.detail)

    async def test_kobold_generation_returns_400_for_grammar_parse_error(self):
        async def collector(*args, **kwargs):
            raise GrammarParseError("The JSON schema could not be compiled: Invalid grammar: boom")
            yield

        data = SimpleNamespace(genkey=None)
        with patch.object(kobold_generation, "_stream_collector", collector):
            with self.assertRaises(HTTPException) as raised:
                await kobold_generation.get_generation(data, request_with_id())

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("could not be compiled", raised.exception.detail)


class GrammarParseErrorStreamTests(unittest.IsolatedAsyncioTestCase):
    """Streaming requests surface the grammar rejection as an error event."""

    async def test_chat_stream_yields_error_event_for_grammar_parse_error(self):
        error = GrammarParseError(
            "The JSON schema could not be compiled: Invalid grammar: boom"
        )

        async def collector(task_idx, gen_queue=None, *args, **kwargs):
            # Mirror the real collector's contract: failures are pushed to the
            # queue as exception objects and re-raised by the consumer loop.
            await gen_queue.put(error)

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

        joined = "".join(chunks)
        self.assertIn("could not be compiled", joined)
        self.assertNotIn('"content"', joined)


if __name__ == "__main__":
    unittest.main()