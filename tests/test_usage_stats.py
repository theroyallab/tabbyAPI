import unittest

from endpoints.OAI.types.common import (
    CompletionTokensDetails,
    PromptTokensDetails,
    UsageStats,
)
from endpoints.OAI.utils.common_ import aggregate_usage_stats, get_usage_stats


def generation(**overrides):
    """A finish chunk carrying the fields exllamav3 reports for a completed generation."""
    base = {
        "finish_reason": "stop",
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


class GetUsageStatsTests(unittest.TestCase):
    def test_non_finish_chunk_reports_nothing(self):
        self.assertIsNone(get_usage_stats({"prompt_tokens": 10, "gen_tokens": 5}))

    def test_finish_chunk_reports_cache_and_draft_counters(self):
        stats = get_usage_stats(generation())

        self.assertEqual(stats.prompt_tokens, 1000)
        self.assertEqual(stats.prompt_tokens_details.cached_tokens, 900)
        self.assertEqual(stats.completion_tokens, 50)
        self.assertEqual(stats.total_tokens, 1050)
        self.assertEqual(stats.prompt_time, 0.06)
        self.assertEqual(stats.completion_time, 1.2)
        self.assertEqual(stats.completion_tokens_details.accepted_prediction_tokens, 40)
        self.assertEqual(stats.completion_tokens_details.rejected_prediction_tokens, 8)

    def test_absent_cache_and_draft_fields_stay_none(self):
        chunk = generation()
        for key in ("cached_tokens", "draft_accept", "draft_reject"):
            del chunk[key]

        stats = get_usage_stats(chunk)

        self.assertIsNone(stats.prompt_tokens_details)
        self.assertIsNone(stats.completion_tokens_details)

    def test_fractional_cached_tokens_are_rounded_to_an_int(self):
        stats = get_usage_stats(generation(cached_tokens=899.6))

        self.assertEqual(stats.prompt_tokens_details.cached_tokens, 900)

    def test_zero_draft_counters_are_reported_not_dropped(self):
        stats = get_usage_stats(generation(draft_accept=0, draft_reject=0))

        self.assertEqual(stats.completion_tokens_details.accepted_prediction_tokens, 0)
        self.assertEqual(stats.completion_tokens_details.rejected_prediction_tokens, 0)


class AggregateUsageStatsTests(unittest.TestCase):
    def test_single_entry_is_returned_unchanged(self):
        only = get_usage_stats(generation())

        self.assertIs(aggregate_usage_stats([only]), only)

    def test_draft_counters_sum_while_prompt_stats_come_from_the_shared_prompt(self):
        # n>1 generations share one prompt, so prompt-side stats are taken from the
        # first entry while generation-side stats accumulate.
        first = get_usage_stats(generation())
        second = get_usage_stats(
            generation(
                cached_tokens=0, gen_tokens=30, gen_time=0.8, draft_accept=25, draft_reject=5
            )
        )

        aggregated = aggregate_usage_stats([first, second])

        self.assertEqual(aggregated.prompt_tokens, 1000)
        self.assertEqual(aggregated.prompt_tokens_details.cached_tokens, 900)
        self.assertEqual(aggregated.completion_tokens, 80)
        self.assertEqual(aggregated.total_tokens, 1080)
        self.assertEqual(aggregated.completion_tokens_details.accepted_prediction_tokens, 65)
        self.assertEqual(aggregated.completion_tokens_details.rejected_prediction_tokens, 13)

    def test_draft_counters_stay_none_when_no_entry_reports_them(self):
        chunk = generation()
        del chunk["draft_accept"]
        del chunk["draft_reject"]
        stats = get_usage_stats(chunk)

        aggregated = aggregate_usage_stats([stats, get_usage_stats(chunk)])

        self.assertIsNone(aggregated.completion_tokens_details)

    def test_partially_reported_draft_counters_sum_the_present_entries(self):
        with_draft = get_usage_stats(generation())
        without = generation()
        del without["draft_accept"]
        del without["draft_reject"]

        aggregated = aggregate_usage_stats([with_draft, get_usage_stats(without)])

        self.assertEqual(aggregated.completion_tokens_details.accepted_prediction_tokens, 40)
        self.assertEqual(aggregated.completion_tokens_details.rejected_prediction_tokens, 8)

    def test_absent_prompt_token_details_survive_aggregation(self):
        chunk = generation()
        del chunk["cached_tokens"]
        stats = get_usage_stats(chunk)

        aggregated = aggregate_usage_stats([stats, get_usage_stats(chunk)])

        self.assertIsNone(aggregated.prompt_tokens_details)


class UsageStatsSerializationTests(unittest.TestCase):
    def test_draft_and_cache_fields_serialize_under_their_openai_style_names(self):
        stats = UsageStats(
            prompt_tokens=10,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
            completion_tokens=5,
            completion_tokens_details=CompletionTokensDetails(
                accepted_prediction_tokens=3,
                rejected_prediction_tokens=1,
            ),
            total_tokens=15,
        )

        payload = stats.model_dump()

        self.assertEqual(payload["prompt_tokens_details"]["cached_tokens"], 4)
        self.assertEqual(payload["completion_tokens_details"]["accepted_prediction_tokens"], 3)
        self.assertEqual(payload["completion_tokens_details"]["rejected_prediction_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
