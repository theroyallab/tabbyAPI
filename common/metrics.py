"""Inference metrics for the optional Prometheus-compatible /metrics endpoint.

Names, help text and scrape behaviour follow llama.cpp's exporter under a
tabbyapi: prefix, so its dashboards work after a prefix swap. What llama.cpp
doesn't report uses vLLM's names (the latency histograms) or its own older ones
(KV cache usage).

Nothing here runs per token. Token counts, job states and cache statistics are
read from the status display's snapshot at scrape time, which also keeps the
generation counters moving during a request, and the rest is recorded once when
a request finishes.

Everything here runs on the single asyncio event loop, so the plain attributes
need no locking.
"""

import time
from typing import Optional

# The boundaries vLLM's dashboards expect, except that the backend's phase times
# are rounded to 10 ms, so nothing finer is worth a bucket
LATENCY_BUCKETS = (
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1920.0,
    7680.0,
)
FIRST_TOKEN_BUCKETS = (
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    640.0,
    2560.0,
)


class _Histogram:
    """A cumulative-bucket histogram for the Prometheus text format."""

    def __init__(self, buckets: tuple):
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

    def render(self, full_name: str) -> list[str]:
        lines = []
        cumulative = 0
        for upper, count in zip(self.buckets, self.counts, strict=True):
            cumulative += count
            lines.append(f'{full_name}_bucket{{le="{upper}"}} {cumulative}')
        lines.append(f'{full_name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{full_name}_sum {self.sum}")
        lines.append(f"{full_name}_count {self.count}")
        return lines


class MetricsManagerClass:
    """Process-lifetime inference counters, rendered as Prometheus text."""

    def __init__(self):
        self.process_start_time = time.time()

        self.prompt_tokens_cached_total = 0
        self.prompt_seconds_total = 0.0
        self.n_tokens_max = 0

        self.draft_tokens_total = 0
        self.draft_tokens_accepted_total = 0
        self.drafts_total = 0

        self.queue_time = _Histogram(LATENCY_BUCKETS)
        self.first_token_time = _Histogram(FIRST_TOKEN_BUCKETS)
        self.e2e_latency = _Histogram(LATENCY_BUCKETS)

        # The prompt tokens prompt_seconds_total belongs to. The live counter
        # also holds the prefill still in progress, whose time isn't known yet
        self._finished_prompt_tokens = 0

        # (tokens, seconds) at the last scrape, by gauge
        self._last_scrape = {}

    def record_finish(self, finish: dict):
        """Record a finished request from the backend's finish chunk."""

        prompt_tokens = finish.get("prompt_tokens") or 0
        cached_tokens = int(finish.get("cached_tokens") or 0)
        gen_tokens = finish.get("gen_tokens") or 0
        queue_time = finish.get("queue_time") or 0.0
        prompt_time = finish.get("prompt_time") or 0.0
        gen_time = finish.get("gen_time") or 0.0

        self.prompt_tokens_cached_total += cached_tokens
        self.prompt_seconds_total += prompt_time
        self._finished_prompt_tokens += prompt_tokens - cached_tokens
        self.n_tokens_max = max(self.n_tokens_max, prompt_tokens + gen_tokens)

        self.queue_time.observe(queue_time)
        self.first_token_time.observe(queue_time + prompt_time)
        self.e2e_latency.observe(queue_time + prompt_time + gen_time)

        # Absent when the request ran without a drafter
        accepted = finish.get("draft_accept")
        if accepted is not None:
            self.draft_tokens_total += accepted + (finish.get("draft_reject") or 0)
            self.draft_tokens_accepted_total += accepted
            # Every decode step emits one token of the target model's own, with
            # the accepted drafts riding along on top of it
            self.drafts_total += max(gen_tokens - accepted, 0)

    def _since_last_scrape(self, gauge: str, tokens: float, seconds: float) -> float:
        """Tokens per second since the last scrape, the window llama.cpp's gauges cover."""

        last_tokens, last_seconds = self._last_scrape.get(gauge, (0, 0.0))
        self._last_scrape[gauge] = (tokens, seconds)
        elapsed = seconds - last_seconds
        return (tokens - last_tokens) / elapsed if elapsed > 0 else 0.0

    def render_prometheus(self, status: dict, model_name: Optional[str] = None) -> str:
        """
        Render all metrics in the Prometheus text exposition format.

        Args:
            status: The status display's snapshot.
            model_name: The loaded model, if any.
        """

        cache = status["cache"] or {}
        max_tokens = cache.get("max_tokens", 0)
        used_tokens = cache.get("used_tokens", 0)
        deferred = sum(job["stage"] in ("queued", "waiting") for job in status["jobs"])

        # (type, name, help, value)
        metrics = [
            (
                "counter",
                "prompt_tokens_total",
                "Number of prompt tokens processed, excluding cached tokens",
                status["prefilled_tokens"],
            ),
            (
                "counter",
                "prompt_tokens_cached_total",
                "Number of prompt tokens reused from the cache",
                self.prompt_tokens_cached_total,
            ),
            (
                "counter",
                "prompt_seconds_total",
                "Total time spent processing prompts",
                self.prompt_seconds_total,
            ),
            (
                "counter",
                "tokens_predicted_total",
                "Number of generation tokens processed",
                status["generated_tokens"],
            ),
            (
                "counter",
                "tokens_predicted_seconds_total",
                "Total time spent generating tokens",
                status["generated_seconds"],
            ),
            (
                "counter",
                "n_tokens_max",
                "Largest observed sequence length (prompt + generation)",
                self.n_tokens_max,
            ),
            (
                "counter",
                "requests_total",
                "Number of finished generation requests, cancelled ones included",
                status["completed"],
            ),
            (
                "counter",
                "spec_decode_num_draft_tokens_total",
                "Speculative: Total draft tokens generated",
                self.draft_tokens_total,
            ),
            (
                "counter",
                "spec_decode_num_accepted_tokens_total",
                "Speculative: Total draft tokens accepted by the target model",
                self.draft_tokens_accepted_total,
            ),
            (
                "counter",
                "spec_decode_num_drafts_total",
                "Speculative: Total speculative decoding verification steps",
                self.drafts_total,
            ),
            (
                "gauge",
                "prompt_tokens_seconds",
                "Average prompt throughput in tokens/s",
                self._since_last_scrape(
                    "prompt", self._finished_prompt_tokens, self.prompt_seconds_total
                ),
            ),
            (
                "gauge",
                "predicted_tokens_seconds",
                "Average generation throughput in tokens/s",
                self._since_last_scrape(
                    "predicted", status["generated_tokens"], status["generated_seconds"]
                ),
            ),
            (
                "gauge",
                "requests_processing",
                "Number of requests processing",
                len(status["jobs"]) - deferred,
            ),
            ("gauge", "requests_deferred", "Number of requests deferred", deferred),
            (
                "gauge",
                "process_start_time_seconds",
                "Start time of the process since the Unix epoch, in seconds",
                self.process_start_time,
            ),
            (
                "gauge",
                "kv_cache_usage_ratio",
                "KV cache usage by running requests. 1 means 100 percent usage",
                used_tokens / max_tokens if max_tokens else 0.0,
            ),
            ("gauge", "kv_cache_tokens", "KV cache tokens in use", used_tokens),
            ("gauge", "kv_cache_max_tokens", "Total KV cache capacity in tokens", max_tokens),
            (
                "gauge",
                "kv_cache_reusable_tokens",
                "Tokens a later request could reuse from the KV cache",
                cache.get("cached_tokens", 0),
            ),
            (
                "gauge",
                "kv_cache_sysmem_tokens",
                "Reusable tokens held only in the system memory tier",
                cache.get("tier_cached_tokens", 0),
            ),
            (
                "gauge",
                "kv_cache_disk_tokens",
                "Reusable tokens held only in the disk tier",
                cache.get("disk_cached_tokens", 0),
            ),
            (
                "gauge",
                "kv_cache_hit_rate",
                "Share of prompt pages served from the cache since the model was loaded",
                cache.get("hit_rate") or 0.0,
            ),
            (
                "gauge",
                "kv_cache_sysmem_hit_rate",
                "Share of prompt pages served from the system memory tier",
                cache.get("tier_hit_rate") or 0.0,
            ),
            (
                "gauge",
                "kv_cache_disk_hit_rate",
                "Share of prompt pages served from the disk tier",
                cache.get("disk_hit_rate") or 0.0,
            ),
        ]

        # (name, help, histogram)
        histograms = [
            (
                "request_queue_time_seconds",
                "Histogram of time spent in the queue before prefill, in seconds",
                self.queue_time,
            ),
            (
                "time_to_first_token_seconds",
                "Histogram of time to first token, in seconds",
                self.first_token_time,
            ),
            (
                "e2e_request_latency_seconds",
                "Histogram of end to end request latency, in seconds",
                self.e2e_latency,
            ),
        ]

        lines = []
        for metric_type, name, help_text, value in metrics:
            full_name = f"tabbyapi:{name}"
            lines.append(f"# HELP {full_name} {help_text}")
            lines.append(f"# TYPE {full_name} {metric_type}")
            lines.append(f"{full_name} {value}")

        for name, help_text, histogram in histograms:
            full_name = f"tabbyapi:{name}"
            lines.append(f"# HELP {full_name} {help_text}")
            lines.append(f"# TYPE {full_name} histogram")
            lines += histogram.render(full_name)

        lines.append("# HELP tabbyapi:model_info The loaded model, in its model_name label")
        lines.append("# TYPE tabbyapi:model_info gauge")
        if model_name:
            label = model_name.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            lines.append(f'tabbyapi:model_info{{model_name="{label}"}} 1')

        return "\n".join(lines) + "\n"


# Create an instance of the global metrics manager
MetricsManager = MetricsManagerClass()
