import unittest

from common.metrics import MetricsManagerClass


class NoPrefillThroughputGaugeTests(unittest.TestCase):
    """/metrics must not publish a prefill throughput gauge.

    Prefill is timed as one span per request and that span cannot be split into
    time spent on cached versus newly processed tokens, so any lifetime average
    of tokens over prefill seconds decays as prefix cache reuse accumulates.
    Restricting the sample to requests that computed enough tokens to swamp the
    fixed overhead fixes the bias but leaves too few samples to be worth
    publishing on a workload with a stable system prefix. The counters are
    exposed instead, so a windowed rate can be taken at query time.
    """

    def setUp(self):
        self.metrics = MetricsManagerClass()

    def test_no_prefill_throughput_series_is_published(self):
        self.metrics.record_generation(
            prompt_tokens=4000,
            cached_tokens=0,
            gen_tokens=10,
            prompt_time=2.0,
            gen_time=1.0,
        )

        rendered = self.metrics.render_prometheus()

        for name in (
            "tabbyapi:prompt_tokens_seconds",
            "tabbyapi:prompt_compute_tokens_seconds",
            "tabbyapi:prompt_cold_tokens_total",
            "tabbyapi:prompt_cold_seconds_total",
        ):
            self.assertNotIn(name, rendered)

    def test_the_counters_a_windowed_rate_needs_are_published(self):
        self.metrics.record_generation(
            prompt_tokens=10000,
            cached_tokens=9900,
            gen_tokens=10,
            prompt_time=0.5,
            gen_time=1.0,
        )

        rendered = self.metrics.render_prometheus()

        # Tokens actually processed, the time prefill took, and the cache hits
        # that explain the difference.
        self.assertIn("tabbyapi:prompt_tokens_total 100", rendered)
        self.assertIn("tabbyapi:prompt_seconds_total 0.5", rendered)
        self.assertIn("tabbyapi:cached_tokens_total 9900", rendered)

    def test_decode_throughput_gauge_is_kept(self):
        # Decode time has no cached-token analogue, so this one measures what
        # it claims to and stays.
        self.metrics.record_generation(
            prompt_tokens=4000,
            cached_tokens=0,
            gen_tokens=100,
            prompt_time=2.0,
            gen_time=4.0,
        )

        self.assertIn("tabbyapi:predicted_tokens_seconds 25.0", self.metrics.render_prometheus())


if __name__ == "__main__":
    unittest.main()
