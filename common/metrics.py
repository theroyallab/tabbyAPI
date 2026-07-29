"""Global inference metrics for the Prometheus-compatible /metrics endpoint.

Modeled after llama.cpp's `/metrics` exporter: process-lifetime counters are
accumulated as generations finish, while gauges (throughput, in-flight and
queued requests, KV-cache usage) are computed live at scrape time. Per-request
latency and size distributions are recorded as histograms, following vLLM's
metric set. All access happens on the single asyncio event loop, so plain
attributes are safe without locking.
"""

import time


# Bucket boundaries borrowed from vLLM's exporter so its dashboards work after a
# prefix swap. Seconds-valued latency histograms share one coarse set; the
# time-to-first-token histogram gets a finer sub-second set since prefill is
# often fast. Per-request token counts use a 1-2-5 progression.
LATENCY_BUCKETS = [
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0,
    60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
]
TTFT_BUCKETS = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5,
    5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
]
TOKEN_BUCKETS = [
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
    100000, 200000, 500000, 1000000,
]
# Per-request draft acceptance is a ratio in [0, 1], so it gets its own evenly
# spaced buckets rather than the token or latency sets.
ACCEPTANCE_BUCKETS = [
    0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0,
]


class _Histogram:
    """A minimal cumulative-bucket histogram for the Prometheus text format."""

    def __init__(self, buckets: list[float]):
        # Upper bounds in ascending order. observe() tallies each value into the
        # first bucket it fits; render sums them cumulatively, as Prometheus
        # histogram buckets are "less than or equal" and cumulative.
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float):
        self.sum += value
        self.count += 1
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[i] += 1
                return
        # A value above the last bucket is reflected only in +Inf / _count.


class MetricsManagerClass:
    """Tracks process-lifetime inference stats for the /metrics endpoint."""

    def __init__(self):
        self.process_start_time = time.time()

        # Counters (monotonic over the process lifetime)
        # prompt_tokens_total counts only tokens actually processed, matching
        # llama.cpp's counter of the same name. Tokens served from the prefix
        # cache are tracked separately, so the full prompt length over the
        # process lifetime is prompt_tokens_total + cached_tokens_total.
        self.prompt_tokens_total = 0
        self.cached_tokens_total = 0
        self.gen_tokens_total = 0
        self.prompt_seconds_total = 0.0
        self.gen_seconds_total = 0.0
        self.requests_total = 0
        self.n_tokens_max = 0

        # No prefill throughput gauge is derived from the two counters above,
        # deliberately. The backend times prefill as one span per request, and
        # that span cannot be split into time spent on cached versus newly
        # processed tokens: a prefix cache hit takes tokens out of
        # prompt_tokens_total while the lookup, page allocation and per-chunk
        # overhead it still paid for stay in prompt_seconds_total. Their ratio
        # therefore reads low and keeps sinking as reuse accumulates. Measured
        # on a 27B model, a warm request contributing 12 tokens in 37ms drags a
        # lifetime average towards ~320 T/s against a real rate of ~1646 T/s.
        #
        # Restricting the sample to requests that computed enough tokens for the
        # ~30ms of fixed per-request overhead to vanish does fix the bias, but a
        # server fronting a harness with a stable system prefix may never see
        # more than one such request, and the first one carries the autotuning
        # pass (measured 1.6% slow), so the estimator is pinned to its single
        # worst sample. Prefill speed is a benchmark quantity; measure it with
        # exllamav3's eval/perf.py or from the per-request log line.
        #
        # What is well defined here is the rate of work over wall clock, which
        # rate(prompt_tokens_total[5m]) gives without any of this reasoning.

        # Speculative decoding counters, following vLLM's spec-decode metric
        # names. A "draft token" is one the drafter proposed; it is accepted
        # when the target model samples the same token, otherwise it and every
        # draft position after it are rejected, so accepted + rejected is the
        # number of tokens drafted. Only requests served with drafting enabled
        # contribute, tracked by draft_requests_total so the acceptance rate is
        # not diluted by non-drafted requests.
        self.draft_tokens_accepted_total = 0
        self.draft_tokens_rejected_total = 0
        self.draft_requests_total = 0
        # Decode steps over drafted requests only. exllamav3 does not count
        # draft rounds, but every decode step emits exactly one token from the
        # target model and the accepted drafts ride along on top of it, so
        # (gen_tokens - accepted) recovers the step count exactly.
        self.draft_decode_steps_total = 0

        # Per-request distributions (vLLM-style histograms). Latency is split
        # into the queue / prefill / decode phases the backend already times,
        # plus derived time-to-first-token (queue + prefill) and end-to-end
        # totals; token counts cover the full prompt and the generation.
        self.hist_queue_time = _Histogram(LATENCY_BUCKETS)
        self.hist_prefill_time = _Histogram(LATENCY_BUCKETS)
        self.hist_decode_time = _Histogram(LATENCY_BUCKETS)
        self.hist_ttft = _Histogram(TTFT_BUCKETS)
        self.hist_e2e = _Histogram(LATENCY_BUCKETS)
        self.hist_prompt_tokens = _Histogram(TOKEN_BUCKETS)
        self.hist_gen_tokens = _Histogram(TOKEN_BUCKETS)
        self.hist_draft_acceptance = _Histogram(ACCEPTANCE_BUCKETS)

    def record_generation(
        self,
        prompt_tokens: int,
        cached_tokens: float,
        gen_tokens: int,
        prompt_time: float,
        gen_time: float,
        queue_time: float = 0.0,
        accepted_draft_tokens: int = None,
        rejected_draft_tokens: int = None,
    ):
        """Accumulate stats from a single finished generation.

        `prompt_tokens` is the full prompt length and `cached_tokens` the part
        of it served from the prefix cache; only the difference was processed.
        `queue_time`, `prompt_time` and `gen_time` are the queue, prefill and
        decode phase durations in seconds.

        `accepted_draft_tokens` / `rejected_draft_tokens` are the speculative
        decoding tallies, and are None when the request ran without a drafter.
        """

        prompt_tokens = prompt_tokens or 0
        cached_tokens = cached_tokens or 0
        gen_tokens = gen_tokens or 0
        prompt_time = prompt_time or 0.0
        gen_time = gen_time or 0.0
        queue_time = queue_time or 0.0

        self.prompt_tokens_total += prompt_tokens - cached_tokens
        self.cached_tokens_total += cached_tokens
        self.gen_tokens_total += gen_tokens
        self.prompt_seconds_total += prompt_time
        self.gen_seconds_total += gen_time
        self.requests_total += 1
        self.n_tokens_max = max(self.n_tokens_max, prompt_tokens)

        # Time to first token is the wait in queue plus prefill; end-to-end adds
        # the decode phase on top.
        self.hist_queue_time.observe(queue_time)
        self.hist_prefill_time.observe(prompt_time)
        self.hist_decode_time.observe(gen_time)
        self.hist_ttft.observe(queue_time + prompt_time)
        self.hist_e2e.observe(queue_time + prompt_time + gen_time)
        self.hist_prompt_tokens.observe(prompt_tokens)
        self.hist_gen_tokens.observe(gen_tokens)

        # Drafting stats are absent when no drafter is configured; a request
        # that ran with one but happened to draft nothing still counts, so the
        # None check has to stay distinct from a zero tally.
        if accepted_draft_tokens is not None:
            accepted = accepted_draft_tokens or 0
            rejected = rejected_draft_tokens or 0
            drafted = accepted + rejected

            self.draft_tokens_accepted_total += accepted
            self.draft_tokens_rejected_total += rejected
            self.draft_requests_total += 1
            self.draft_decode_steps_total += max(gen_tokens - accepted, 0)

            if drafted > 0:
                self.hist_draft_acceptance.observe(accepted / drafted)

    def _live_request_counts(self) -> tuple[int, int]:
        """Read (processing, deferred) request counts from the generator.

        Returns zeros if no model is loaded or the backend does not expose
        job counts.
        """

        # Imported lazily to avoid a circular import (common.model pulls in the
        # backends, which import this module).
        from common import model

        container = model.container
        generator = getattr(container, "generator", None) if container else None
        sync_generator = getattr(generator, "generator", None) if generator else None

        if sync_generator is None:
            return 0, 0

        try:
            return sync_generator.num_active_jobs(), sync_generator.num_pending_jobs()
        except Exception:
            return 0, 0

    def _live_kv_cache(self) -> tuple[int, int]:
        """Read (used_tokens, max_tokens) of the paged KV cache from the generator.

        Usage is measured over pages currently referenced by in-flight jobs, the
        instantaneous KV load. Unreferenced pages may still hold reusable prompt
        prefixes but are free to be evicted, so they count as headroom rather than
        usage (matching llama.cpp's kv_cache_usage_ratio). Returns zeros if no
        model is loaded or the backend does not expose a page table.
        """

        # Imported lazily to avoid a circular import (common.model pulls in the
        # backends, which import this module).
        from common import model

        container = model.container
        generator = getattr(container, "generator", None) if container else None
        sync_generator = getattr(generator, "generator", None) if generator else None
        pagetable = getattr(sync_generator, "pagetable", None) if sync_generator else None

        if pagetable is None:
            return 0, 0

        try:
            max_pages = pagetable.max_pages
            max_tokens = sync_generator.max_total_tokens
            page_size = max_tokens // max_pages if max_pages else 0
            used_tokens = len(pagetable.referenced_pages) * page_size
            return used_tokens, max_tokens
        except Exception:
            return 0, 0

    def render_prometheus(self) -> str:
        """Render all metrics in the Prometheus text exposition format."""

        requests_processing, requests_deferred = self._live_request_counts()
        kv_cache_tokens, kv_cache_max_tokens = self._live_kv_cache()
        kv_cache_usage_ratio = (
            kv_cache_tokens / kv_cache_max_tokens if kv_cache_max_tokens > 0 else 0.0
        )

        # Prefix-cache effectiveness is exposed as the raw queries/hits token
        # counters (vLLM-style), leaving the hit ratio to be computed at query
        # time with rate() so it reflects recent behavior rather than a
        # lifetime average.
        prefix_cache_queries = self.prompt_tokens_total + self.cached_tokens_total
        prefix_cache_hits = self.cached_tokens_total

        # There is no prefill counterpart to this gauge on purpose; see the
        # counter definitions. Decode time has no cached-token analogue to skew
        # it, so generation tokens over decode seconds is what it claims to be.
        predicted_tokens_seconds = (
            self.gen_tokens_total / self.gen_seconds_total if self.gen_seconds_total > 0 else 0.0
        )

        # Speculative decoding effectiveness. The two raw counters are the
        # vLLM-style primitives to rate() over; these gauges are the lifetime
        # summary, cheap to read without a query language. Acceptance rate is
        # per drafted token, mean accepted length is per decode step (how many
        # drafts a step gets for free), and tokens per step adds the target
        # model's own token, so it is the decode speedup factor over no drafter.
        draft_tokens_total = self.draft_tokens_accepted_total + self.draft_tokens_rejected_total
        draft_acceptance_rate = (
            self.draft_tokens_accepted_total / draft_tokens_total
            if draft_tokens_total > 0
            else 0.0
        )
        draft_mean_accepted_len = (
            self.draft_tokens_accepted_total / self.draft_decode_steps_total
            if self.draft_decode_steps_total > 0
            else 0.0
        )
        draft_tokens_per_step = (
            1.0 + draft_mean_accepted_len if self.draft_decode_steps_total else 0.0
        )

        # (type, name, help, value)
        # Names and help text of the shared metrics are kept verbatim from
        # llama.cpp's exporter so its dashboards work after a prefix swap.
        # tabbyAPI-only metrics follow the ones they relate to. The one
        # deliberate omission from that set is prompt_tokens_seconds; see the
        # counter definitions for why it is not a quantity worth publishing.
        metrics = [
            (
                "counter",
                "prompt_tokens_total",
                "Number of prompt tokens processed.",
                self.prompt_tokens_total,
            ),
            (
                "counter",
                "cached_tokens_total",
                "Number of prompt tokens skipped via the prefix cache.",
                self.cached_tokens_total,
            ),
            (
                "counter",
                "prefix_cache_queries",
                "Prefix cache queries, in terms of number of queried tokens.",
                prefix_cache_queries,
            ),
            (
                "counter",
                "prefix_cache_hits",
                "Prefix cache hits, in terms of number of cached tokens.",
                prefix_cache_hits,
            ),
            (
                "counter",
                "prompt_seconds_total",
                "Prompt process time",
                self.prompt_seconds_total,
            ),
            (
                "counter",
                "tokens_predicted_total",
                "Number of generation tokens processed.",
                self.gen_tokens_total,
            ),
            (
                "counter",
                "tokens_predicted_seconds_total",
                "Predict process time",
                self.gen_seconds_total,
            ),
            (
                "counter",
                "n_tokens_max",
                "Largest observed n_tokens.",
                self.n_tokens_max,
            ),
            (
                "counter",
                "requests_total",
                "Number of finished generation requests.",
                self.requests_total,
            ),
            (
                "gauge",
                "predicted_tokens_seconds",
                "Average generation throughput in tokens/s.",
                predicted_tokens_seconds,
            ),
            (
                "counter",
                "spec_decode_num_draft_tokens_total",
                "Number of tokens proposed by the drafter.",
                draft_tokens_total,
            ),
            (
                "counter",
                "spec_decode_num_accepted_tokens_total",
                "Number of drafted tokens accepted by the target model.",
                self.draft_tokens_accepted_total,
            ),
            (
                "counter",
                "spec_decode_num_decode_steps_total",
                "Number of decode steps over requests served with a drafter.",
                self.draft_decode_steps_total,
            ),
            (
                "counter",
                "spec_decode_requests_total",
                "Number of finished requests served with a drafter.",
                self.draft_requests_total,
            ),
            (
                "gauge",
                "spec_decode_draft_acceptance_rate",
                "Fraction of drafted tokens accepted. 1 means every draft was accepted.",
                draft_acceptance_rate,
            ),
            (
                "gauge",
                "spec_decode_mean_accepted_length",
                "Average drafted tokens accepted per decode step.",
                draft_mean_accepted_len,
            ),
            (
                "gauge",
                "spec_decode_tokens_per_step",
                "Average tokens emitted per decode step, including the target model's own.",
                draft_tokens_per_step,
            ),
            (
                "gauge",
                "requests_processing",
                "Number of requests processing.",
                requests_processing,
            ),
            (
                "gauge",
                "requests_deferred",
                "Number of requests deferred.",
                requests_deferred,
            ),
            (
                "gauge",
                "kv_cache_usage_ratio",
                "KV-cache usage. 1 means 100 percent usage.",
                kv_cache_usage_ratio,
            ),
            (
                "gauge",
                "kv_cache_tokens",
                "KV-cache tokens.",
                kv_cache_tokens,
            ),
            (
                "gauge",
                "kv_cache_max_tokens",
                "Total KV-cache token capacity.",
                kv_cache_max_tokens,
            ),
        ]

        # (name, help, histogram)
        histograms = [
            (
                "request_queue_time_seconds",
                "Histogram of time spent in the queue before prefill, in seconds.",
                self.hist_queue_time,
            ),
            (
                "request_prefill_time_seconds",
                "Histogram of prefill (prompt processing) time in seconds.",
                self.hist_prefill_time,
            ),
            (
                "request_decode_time_seconds",
                "Histogram of decode (generation) time in seconds.",
                self.hist_decode_time,
            ),
            (
                "time_to_first_token_seconds",
                "Histogram of time to first token in seconds.",
                self.hist_ttft,
            ),
            (
                "e2e_request_latency_seconds",
                "Histogram of end to end request latency in seconds.",
                self.hist_e2e,
            ),
            (
                "request_prompt_tokens",
                "Histogram of number of prompt tokens per request.",
                self.hist_prompt_tokens,
            ),
            (
                "request_generation_tokens",
                "Histogram of number of generation tokens per request.",
                self.hist_gen_tokens,
            ),
            (
                "spec_decode_acceptance_rate",
                "Histogram of per-request draft acceptance rate.",
                self.hist_draft_acceptance,
            ),
        ]

        lines = []
        for metric_type, name, help_text, value in metrics:
            full_name = f"tabbyapi:{name}"
            lines.append(f"# HELP {full_name} {help_text}")
            lines.append(f"# TYPE {full_name} {metric_type}")
            lines.append(f"{full_name} {value}")

        for name, help_text, hist in histograms:
            full_name = f"tabbyapi:{name}"
            lines.append(f"# HELP {full_name} {help_text}")
            lines.append(f"# TYPE {full_name} histogram")
            cumulative = 0
            for upper, count in zip(hist.buckets, hist.counts, strict=True):
                cumulative += count
                lines.append(f'{full_name}_bucket{{le="{upper}"}} {cumulative}')
            lines.append(f'{full_name}_bucket{{le="+Inf"}} {hist.count}')
            lines.append(f"{full_name}_sum {hist.sum}")
            lines.append(f"{full_name}_count {hist.count}")

        return "\n".join(lines) + "\n"


# Create an instance of the global metrics manager
MetricsManager = MetricsManagerClass()
