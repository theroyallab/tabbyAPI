import unittest

from common.metrics import MetricsManagerClass


class PeakSizeCounterTests(unittest.TestCase):
    def setUp(self):
        self.metrics = MetricsManagerClass()

    def record(self, prompt_tokens: int, gen_tokens: int):
        self.metrics.record_generation(
            prompt_tokens=prompt_tokens,
            cached_tokens=0,
            gen_tokens=gen_tokens,
            prompt_time=1.0,
            gen_time=1.0,
        )

    def test_peaks_track_the_largest_request(self):
        self.record(1000, 20)
        self.record(500, 8000)
        self.record(9000, 15)

        self.assertEqual(self.metrics.n_tokens_max, 9000)
        self.assertEqual(self.metrics.tokens_predicted_max, 8000)

        rendered = self.metrics.render_prometheus()
        self.assertIn("tabbyapi:n_tokens_max 9000", rendered)
        self.assertIn("tabbyapi:tokens_predicted_max 8000", rendered)

    def test_peaks_never_fall(self):
        self.record(9000, 8000)
        for _ in range(50):
            self.record(20, 15)

        self.assertEqual(self.metrics.n_tokens_max, 9000)
        self.assertEqual(self.metrics.tokens_predicted_max, 8000)

    def test_peak_survives_where_percentiles_cannot(self):
        # The reason this counter exists: one large completion among many small
        # ones sits above every percentile the histogram can report, so the
        # distribution alone cannot show it.
        self.record(100, 8000)
        for _ in range(344):
            self.record(100, 15)

        gen_hist = self.metrics.hist_gen_tokens
        self.assertEqual(gen_hist.count, 345)
        # The outlier is one sample in 345, i.e. above the 99th percentile.
        self.assertLess(gen_hist.count * 0.99, 344)
        self.assertEqual(self.metrics.tokens_predicted_max, 8000)

    def test_missing_counts_do_not_break_the_peaks(self):
        self.metrics.record_generation(
            prompt_tokens=None,
            cached_tokens=None,
            gen_tokens=None,
            prompt_time=None,
            gen_time=None,
        )

        self.assertEqual(self.metrics.n_tokens_max, 0)
        self.assertEqual(self.metrics.tokens_predicted_max, 0)


if __name__ == "__main__":
    unittest.main()
