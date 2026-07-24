"""Global inference metrics for the Prometheus-compatible /metrics endpoint.

Modeled after llama.cpp's `/metrics` exporter: process-lifetime counters are
accumulated as generations finish, while gauges (throughput, in-flight and
queued requests) are computed live at scrape time. All access happens on the
single asyncio event loop, so plain attributes are safe without locking.
"""

import time


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

    def record_generation(
        self,
        prompt_tokens: int,
        cached_tokens: float,
        gen_tokens: int,
        prompt_time: float,
        gen_time: float,
    ):
        """Accumulate stats from a single finished generation.

        `prompt_tokens` is the full prompt length and `cached_tokens` the part
        of it served from the prefix cache; only the difference was processed.
        """

        self.prompt_tokens_total += (prompt_tokens or 0) - (cached_tokens or 0)
        self.cached_tokens_total += cached_tokens or 0
        self.gen_tokens_total += gen_tokens or 0
        self.prompt_seconds_total += prompt_time or 0.0
        self.gen_seconds_total += gen_time or 0.0
        self.requests_total += 1
        self.n_tokens_max = max(self.n_tokens_max, prompt_tokens or 0)

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

        # Throughput is measured over processed (non-cached) prompt tokens, to
        # match how the backend reports per-request prompt speed.
        prompt_tokens_seconds = (
            self.prompt_tokens_total / self.prompt_seconds_total
            if self.prompt_seconds_total > 0
            else 0.0
        )
        predicted_tokens_seconds = (
            self.gen_tokens_total / self.gen_seconds_total if self.gen_seconds_total > 0 else 0.0
        )

        # (type, name, help, value)
        # Names and help text of the shared metrics are kept verbatim from
        # llama.cpp's exporter so its dashboards work after a prefix swap.
        # tabbyAPI-only metrics follow the ones they relate to.
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
                "prompt_tokens_seconds",
                "Average prompt throughput in tokens/s.",
                prompt_tokens_seconds,
            ),
            (
                "gauge",
                "predicted_tokens_seconds",
                "Average generation throughput in tokens/s.",
                predicted_tokens_seconds,
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

        lines = []
        for metric_type, name, help_text, value in metrics:
            full_name = f"tabbyapi:{name}"
            lines.append(f"# HELP {full_name} {help_text}")
            lines.append(f"# TYPE {full_name} {metric_type}")
            lines.append(f"{full_name} {value}")

        return "\n".join(lines) + "\n"


# Create an instance of the global metrics manager
MetricsManager = MetricsManagerClass()
