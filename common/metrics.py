"""Global inference metrics for the optional Prometheus-compatible /metrics endpoint.

The generation counters are written as each piece is delivered to the client
rather than when the request finishes, so a dashboard watching rate() sees
throughput while a long generation runs instead of one step at its end.

Only tokens actually sent are counted: the engine's own tally on the finish
chunk (`new_tokens`) can differ by a token from what the client received, and a
dashboard is after what left the server. Metric names and help text follow
llama.cpp's exporter, so its dashboards work unchanged.

Everything here runs on the single asyncio event loop, so the plain attributes
need no locking.
"""

import time


class MetricsManagerClass:
    """Process-lifetime inference counters, rendered as Prometheus text."""

    def __init__(self):
        self.process_start_time = time.time()

        # prompt_tokens_total counts only tokens actually processed, matching
        # llama.cpp; the part served from the prefix cache is counted apart, so
        # the full prompt length is the sum of the two.
        self.prompt_tokens_total = 0
        self.cached_tokens_total = 0
        self.prompt_seconds_total = 0.0
        self.requests_total = 0

        # Both are advanced per delivered piece, on the same clock, so their
        # ratio stays the rate the tokens were actually sent at. The clock is
        # per request: with continuous batching, several jobs deliver
        # interleaved pieces and a shared clock would count the gaps between
        # them as decode time.
        self.tokens_predicted_total = 0
        self.tokens_predicted_seconds_total = 0.0
        self._last_delivery = {}

        # Drafted tokens are accepted when the target model samples the same
        # token, and (gen_tokens - accepted) recovers the decode step count:
        # every step emits exactly one token from the target model.
        self.draft_tokens_accepted_total = 0
        self.draft_tokens_rejected_total = 0
        self.draft_requests_total = 0
        self.draft_decode_steps_total = 0

    @staticmethod
    def enabled() -> bool:
        """Whether the endpoint is on, read once per request."""

        from common.tabby_config import config

        return bool(config.network.enable_metrics)

    def tokens_delivered(self, request_id: str, count: int):
        """Count a generated piece, and the time since the request's last one."""

        if count <= 0:
            return

        now = time.monotonic()
        last = self._last_delivery.get(request_id, now)
        self._last_delivery[request_id] = now
        self.tokens_predicted_seconds_total += now - last
        self.tokens_predicted_total += count

    def end_request(self, request_id: str, finish: dict):
        """Close a request's decode clock and record its prompt side.

        `finish` is the backend's finish chunk, empty for a request that was
        cancelled or failed: its delivered tokens stay counted, but it is not
        one of the finished requests, and neither its prefill nor its drafting
        tallies are known.
        """

        self._last_delivery.pop(request_id, None)
        if not finish:
            return

        cached_tokens = int(finish.get("cached_tokens") or 0)
        self.prompt_tokens_total += (finish.get("prompt_tokens") or 0) - cached_tokens
        self.cached_tokens_total += cached_tokens
        self.prompt_seconds_total += finish.get("prompt_time") or 0.0
        self.requests_total += 1

        # Absent when the request ran without a drafter
        accepted = finish.get("draft_accept")
        if accepted is not None:
            self.draft_tokens_accepted_total += accepted
            self.draft_tokens_rejected_total += finish.get("draft_reject") or 0
            self.draft_requests_total += 1
            self.draft_decode_steps_total += max((finish.get("gen_tokens") or 0) - accepted, 0)

    @staticmethod
    def _live_job_counts() -> tuple[int, int]:
        """(processing, deferred) requests, as the status line tracks them."""

        from common.status_display import status_display

        # A job that has not left the engine's queue yet is still at "queued"
        queued = sum(1 for job in status_display.jobs.values() if job.stage == "queued")
        return len(status_display.jobs) - queued, queued

    def render_prometheus(self) -> str:
        """Render all metrics in the Prometheus text exposition format."""

        processing, deferred = self._live_job_counts()
        predicted_tokens_seconds = (
            self.tokens_predicted_total / self.tokens_predicted_seconds_total
            if self.tokens_predicted_seconds_total > 0
            else 0.0
        )
        draft_tokens_total = self.draft_tokens_accepted_total + self.draft_tokens_rejected_total

        # (type, name, help, value)
        metrics = [
            (
                "gauge",
                "process_start_time_seconds",
                "Start time of the process since the Unix epoch, in seconds.",
                self.process_start_time,
            ),
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
            ("counter", "prompt_seconds_total", "Prompt process time", self.prompt_seconds_total),
            (
                "counter",
                "tokens_predicted_total",
                "Number of generation tokens processed.",
                self.tokens_predicted_total,
            ),
            (
                "counter",
                "tokens_predicted_seconds_total",
                "Predict process time",
                self.tokens_predicted_seconds_total,
            ),
            (
                "counter",
                "requests_total",
                "Number of finished generation requests.",
                self.requests_total,
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
                "predicted_tokens_seconds",
                "Average generation throughput in tokens/s.",
                predicted_tokens_seconds,
            ),
            (
                "gauge",
                "requests_processing",
                "Number of requests processing.",
                processing,
            ),
            (
                "gauge",
                "requests_deferred",
                "Number of requests deferred.",
                deferred,
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
