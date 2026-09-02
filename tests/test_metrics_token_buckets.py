import unittest

from common.metrics import MetricsManagerClass, build_1_2_5_buckets


class TokenBucketTests(unittest.TestCase):
    """Token histogram boundaries, built from the model's context length.

    A fixed ladder is not merely imprecise, it produces percentiles above any
    request the server can accept: histogram_quantile interpolates across the
    bucket a sample landed in, so a 300k-wide bucket holding prompts clustered
    at its floor reports a p99 near its ceiling.
    """

    def test_ladder_is_a_1_2_5_progression(self):
        self.assertEqual(build_1_2_5_buckets(100), [1, 2, 5, 10, 20, 50, 100])
        self.assertEqual(build_1_2_5_buckets(1), [1])
        # 9 is off the ladder, so it becomes the final boundary itself
        self.assertEqual(build_1_2_5_buckets(9), [1, 2, 5, 9])

    def test_limit_is_always_a_boundary(self):
        # Stopping at the last mantissa below the limit strands everything
        # between there and the real limit in +Inf. At 262144 that is the top
        # 24% of the usable range.
        buckets = build_1_2_5_buckets(262144)
        self.assertEqual(buckets[-1], 262144)
        self.assertEqual(buckets[-2], 200000)

    def test_no_bucket_exceeds_the_limit(self):
        for limit in (4096, 32768, 131072, 262144, 344064, 1_000_000):
            self.assertLessEqual(max(build_1_2_5_buckets(limit)), limit)
            self.assertEqual(sorted(set(build_1_2_5_buckets(limit))), build_1_2_5_buckets(limit))

    def test_configure_resizes_the_histograms(self):
        metrics = MetricsManagerClass()
        metrics.configure_token_buckets(262144)

        self.assertEqual(metrics.token_buckets[-1], 262144)
        self.assertEqual(metrics.hist_prompt_tokens.buckets, metrics.token_buckets)
        self.assertEqual(metrics.hist_gen_tokens.buckets, metrics.token_buckets)

    def test_reconfiguring_to_the_same_length_keeps_the_samples(self):
        # Rebuilding drops accumulated counts, so an unchanged ladder must not.
        metrics = MetricsManagerClass()
        metrics.configure_token_buckets(262144)
        metrics.record_generation(
            prompt_tokens=1000, cached_tokens=0, gen_tokens=10, prompt_time=1.0, gen_time=1.0
        )
        metrics.configure_token_buckets(262144)

        self.assertEqual(metrics.hist_prompt_tokens.count, 1)

    def test_a_different_length_resets_the_histograms(self):
        # Counts against a different ladder cannot be carried over.
        metrics = MetricsManagerClass()
        metrics.record_generation(
            prompt_tokens=1000, cached_tokens=0, gen_tokens=10, prompt_time=1.0, gen_time=1.0
        )
        metrics.configure_token_buckets(4096)

        self.assertEqual(metrics.hist_prompt_tokens.count, 0)

    def test_nonsense_lengths_are_ignored(self):
        metrics = MetricsManagerClass()
        before = list(metrics.token_buckets)
        for bad in (0, -1, None):
            metrics.configure_token_buckets(bad)
        self.assertEqual(metrics.token_buckets, before)

    def test_percentiles_stay_inside_the_context_limit(self):
        # The regression this exists for: every reported percentile must be a
        # length the server could actually have been asked for.
        metrics = MetricsManagerClass()
        metrics.configure_token_buckets(262144)
        for _ in range(31):
            metrics.record_generation(
                prompt_tokens=203000, cached_tokens=0, gen_tokens=1, prompt_time=1.0, gen_time=1.0
            )

        rendered = metrics.render_prometheus()
        boundaries = [
            float(line.split('le="')[1].split('"')[0])
            for line in rendered.splitlines()
            if line.startswith("tabbyapi:request_prompt_tokens_bucket") and "+Inf" not in line
        ]
        self.assertLessEqual(max(boundaries), 262144)


if __name__ == "__main__":
    unittest.main()
