import json
import unittest

from endpoints.OAI.types.common import Timings
from endpoints.OAI.utils.chat_completion import (
    _compose_response as compose_chat_response,
)
from endpoints.OAI.utils.chat_completion import (
    _compose_serialize_stream_chunk as compose_chat_chunk,
)
from endpoints.OAI.utils.chat_completion import (
    _compose_serialize_stream_usage_chunk as compose_chat_usage_chunk,
)
from endpoints.OAI.utils.common_ import get_timings, get_usage_stats
from endpoints.OAI.utils.completion import (
    _compose_response as compose_text_response,
)
from endpoints.OAI.utils.completion import (
    _compose_serialize_stream_chunk as compose_text_chunk,
)
from endpoints.OAI.utils.completion import (
    _compose_serialize_stream_usage_chunk as compose_text_usage_chunk,
)


def generation(**overrides):
    """A finish chunk carrying the fields exllamav3 reports for a completed generation."""
    base = {
        "finish_reason": "stop",
        "index": 0,
        "content": "Hello!",
        "prompt_tokens": 1000,
        "cached_tokens": 900,
        "prompt_time": 0.06,
        "prompt_tokens_per_sec": 1666.67,
        "gen_tokens": 50,
        "gen_time": 1.2,
        "gen_tokens_per_sec": 41.7,
        "total_time": 1.26,
        "draft_accept": 40,
        "draft_reject": 8,
    }
    base.update(overrides)
    return base


class GetTimingsTests(unittest.TestCase):
    def test_finish_chunk_maps_to_llama_cpp_keys(self):
        timings = get_timings(generation())

        self.assertEqual(timings.cache_n, 900)
        self.assertEqual(timings.prompt_n, 100)
        self.assertEqual(timings.prompt_ms, 60.0)
        self.assertEqual(timings.prompt_per_token_ms, 0.6)
        self.assertEqual(timings.prompt_per_second, 1666.6666666666667)
        self.assertEqual(timings.predicted_n, 50)
        self.assertEqual(timings.predicted_ms, 1200.0)
        # gen_time covers every generated token, so per-token rates divide by gen_tokens
        self.assertEqual(timings.predicted_per_token_ms, 24.0)
        self.assertEqual(timings.predicted_per_second, 41.66666666666667)
        self.assertEqual(timings.draft_n, 48)
        self.assertEqual(timings.draft_n_accepted, 40)

    def test_draft_keys_are_absent_when_no_draft_tokens_were_produced(self):
        # llama.cpp's guard is n_draft_tokens > 0, not "a draft model is loaded"
        timings = get_timings(generation(draft_accept=0, draft_reject=0))

        payload = timings.model_dump(mode="json")
        self.assertNotIn("draft_n", payload)
        self.assertNotIn("draft_n_accepted", payload)

    def test_non_finish_chunk_reports_nothing(self):
        self.assertIsNone(get_timings({"prompt_tokens": 10, "gen_tokens": 5}))

    def test_draft_keys_are_absent_when_no_draft_ran(self):
        chunk = generation()
        del chunk["draft_accept"]
        del chunk["draft_reject"]

        timings = get_timings(chunk)

        self.assertIsNone(timings.draft_n)
        self.assertIsNone(timings.draft_n_accepted)
        payload = timings.model_dump(mode="json")
        self.assertNotIn("draft_n", payload)
        self.assertNotIn("draft_n_accepted", payload)
        # The unconditional keys stay present
        for key in (
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
        ):
            self.assertIn(key, payload)

    def test_zero_times_report_zero_rates_without_indeterminate(self):
        # The backend reports the per-sec fields as the string "Indeterminate"
        # when a time is zero; timings is computed from the times instead
        chunk = generation(
            prompt_time=0,
            prompt_tokens_per_sec="Indeterminate",
            gen_tokens=1,
            gen_time=0,
            gen_tokens_per_sec="Indeterminate",
        )

        timings = get_timings(chunk)

        self.assertEqual(timings.prompt_ms, 0.0)
        self.assertEqual(timings.prompt_per_token_ms, 0.0)
        self.assertEqual(timings.prompt_per_second, 0.0)
        self.assertEqual(timings.predicted_ms, 0.0)
        self.assertEqual(timings.predicted_per_token_ms, 0.0)
        self.assertEqual(timings.predicted_per_second, 0.0)
        self.assertNotIn("Indeterminate", json.dumps(timings.model_dump(mode="json")))

    def test_generation_rate_counts_every_generated_token(self):
        # exllamav3's gen_time starts before the pass that produces the first
        # token, so one token over 0.5 s is 2 tokens/s, not llama.cpp's n - 1 = 0
        timings = get_timings(generation(gen_tokens=1, gen_time=0.5))

        self.assertEqual(timings.predicted_per_token_ms, 500.0)
        self.assertEqual(timings.predicted_per_second, 2.0)

    def test_zero_predicted_n_reports_zero_rates(self):
        timings = get_timings(generation(gen_tokens=0, gen_time=0.5))

        self.assertEqual(timings.predicted_n, 0)
        self.assertEqual(timings.predicted_per_token_ms, 0.0)
        self.assertEqual(timings.predicted_per_second, 0.0)

    def test_fractional_cached_tokens_round_like_usage_does(self):
        chunk = generation(prompt_tokens=2000, cached_tokens=1792.33)

        timings = get_timings(chunk)
        usage = get_usage_stats(chunk)

        self.assertEqual(timings.cache_n, 1792)
        self.assertEqual(timings.prompt_n, 208)
        self.assertEqual(timings.cache_n, usage.prompt_tokens_details.cached_tokens)

    def test_cache_larger_than_prompt_floors_prompt_n_at_zero(self):
        timings = get_timings(generation(prompt_tokens=850))

        self.assertEqual(timings.cache_n, 900)
        self.assertEqual(timings.prompt_n, 0)
        self.assertEqual(timings.prompt_per_token_ms, 0.0)
        self.assertEqual(timings.prompt_per_second, 0.0)

    def test_missing_numeric_inputs_report_zeros(self):
        chunk = generation()
        for key in ("cached_tokens", "prompt_time", "gen_time"):
            del chunk[key]

        timings = get_timings(chunk)

        self.assertEqual(timings.cache_n, 0)
        self.assertEqual(timings.prompt_n, 1000)
        self.assertEqual(timings.prompt_ms, 0.0)
        self.assertEqual(timings.prompt_per_token_ms, 0.0)
        self.assertEqual(timings.prompt_per_second, 0.0)
        self.assertEqual(timings.predicted_ms, 0.0)
        self.assertEqual(timings.predicted_per_token_ms, 0.0)
        self.assertEqual(timings.predicted_per_second, 0.0)


class ResponseAttachTests(unittest.TestCase):
    def test_chat_response_carries_timings_for_one_generation(self):
        response = compose_chat_response("id", [generation()], "model", True)

        self.assertIsInstance(response.timings, Timings)
        self.assertEqual(response.timings.prompt_n, 100)
        self.assertIsNotNone(response.usage)

    def test_chat_response_carries_timings_without_usage(self):
        # llama.cpp emits timings unconditionally; usage stays behind
        # stream_options.include_usage as before
        response = compose_chat_response("id", [generation()], "model", False)

        self.assertIsInstance(response.timings, Timings)
        self.assertIsNone(response.usage)

    def test_chat_response_omits_timings_for_two_generations(self):
        generations = [generation(), generation(index=1, gen_tokens=30)]

        response = compose_chat_response("id", generations, "model", True)

        self.assertIsNone(response.timings)
        self.assertIsNotNone(response.usage)

    def test_text_response_carries_timings_for_one_generation(self):
        response = compose_text_response("id", [generation()], "model", True)

        self.assertIsInstance(response.timings, Timings)
        self.assertEqual(response.timings.predicted_n, 50)

    def test_text_response_omits_timings_for_two_generations(self):
        generations = [generation(), generation(index=1, gen_tokens=30)]

        response = compose_text_response("id", generations, "model", True)

        self.assertIsNone(response.timings)
        self.assertIsNotNone(response.usage)

    def test_response_payload_drops_draft_keys_when_no_draft_ran(self):
        chunk = generation()
        del chunk["draft_accept"]
        del chunk["draft_reject"]

        response = compose_chat_response("id", [chunk], "model", True)

        payload = response.model_dump(mode="json")
        self.assertNotIn("draft_n", payload["timings"])
        self.assertNotIn("draft_n_accepted", payload["timings"])


class StreamAttachTests(unittest.TestCase):
    def test_chat_finish_chunk_carries_timings_when_no_usage_chunk_follows(self):
        # Without include_usage the finish_reason chunk is the stream's last
        serialized, data, _, _ = compose_chat_chunk(
            "id", generation(), "model", False, get_timings(generation())
        )

        self.assertIn("timings", data)
        self.assertEqual(data["timings"]["cache_n"], 900)
        self.assertIn("timings", serialized)

    def test_chat_usage_chunk_carries_timings(self):
        # With include_usage the usage chunk is the stream's last
        usage = get_usage_stats(generation())

        serialized, data = compose_chat_usage_chunk(
            "id", usage, 0, "stop", "model", get_timings(generation())
        )

        self.assertIn("timings", data)
        self.assertEqual(data["timings"]["predicted_n"], 50)
        self.assertIn("timings", serialized)

    def test_chat_usage_chunk_without_timings_leaves_the_key_out(self):
        usage = get_usage_stats(generation())

        serialized, data = compose_chat_usage_chunk("id", usage, 0, "stop", "model")

        self.assertNotIn("timings", data)
        self.assertNotIn("timings", serialized)

    def test_chat_delta_chunks_do_not_carry_timings(self):
        chunk = {"index": 0, "delta_content": "Hel"}
        serialized, data, _, _ = compose_chat_chunk("id", chunk, "model", False)

        self.assertNotIn("timings", data)
        self.assertNotIn("timings", serialized)

    def test_text_finish_chunk_carries_timings_when_no_usage_chunk_follows(self):
        serialized, data, _, _ = compose_text_chunk(
            "id", generation(), "model", False, get_timings(generation())
        )

        self.assertIn("timings", data)
        self.assertEqual(data["timings"]["prompt_per_second"], 1666.6666666666667)
        self.assertIn("timings", serialized)

    def test_text_usage_chunk_carries_timings(self):
        usage = get_usage_stats(generation())

        serialized, data = compose_text_usage_chunk(
            "id", usage, 0, "stop", "model", get_timings(generation())
        )

        self.assertIn("timings", data)
        self.assertIn("timings", serialized)

    def test_text_delta_chunks_do_not_carry_timings(self):
        chunk = {"index": 0, "delta_content": "Hel"}
        serialized, data, _, _ = compose_text_chunk("id", chunk, "model", False)

        self.assertNotIn("timings", data)
        self.assertNotIn("timings", serialized)

    def test_stream_chunk_draft_keys_follow_the_timings_object(self):
        chunk = generation()
        del chunk["draft_accept"]
        del chunk["draft_reject"]

        _, data, _, _ = compose_chat_chunk("id", chunk, "model", False, get_timings(chunk))

        self.assertNotIn("draft_n", data["timings"])
        self.assertNotIn("draft_n_accepted", data["timings"])


if __name__ == "__main__":
    unittest.main()
