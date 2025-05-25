"""Prometheus metrics and OpenTelemetry tracing helpers."""

from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except Exception:  # Module not found or other import error
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = None
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"

    def generate_latest() -> bytes:  # type: ignore
        return b""


try:
    from opentelemetry import trace
    from opentelemetry.trace import Tracer
except Exception:  # Module not found or import error
    trace = None
    Tracer = None

# Metrics definitions
if PROMETHEUS_AVAILABLE:
    LOGPROB_REQUESTS = Counter(
        "tabby_logprob_requests_total",
        "Total number of logprob requests",
    )
    LOGPROB_TOKENS = Counter(
        "tabby_logprob_tokens_total",
        "Total number of tokens processed in logprob requests",
    )
    LOGPROB_LATENCY = Histogram(
        "tabby_logprob_latency_seconds",
        "Latency of logprob requests in seconds",
    )
else:
    LOGPROB_REQUESTS = LOGPROB_TOKENS = LOGPROB_LATENCY = None

if trace is not None:
    TRACER = trace.get_tracer(__name__)
else:
    TRACER = None


def record_logprob_metrics(tokens: int, latency_seconds: float) -> None:
    """Update Prometheus metrics for a logprob request."""
    if LOGPROB_REQUESTS is not None:
        LOGPROB_REQUESTS.inc()
        LOGPROB_TOKENS.inc(tokens)
        LOGPROB_LATENCY.observe(latency_seconds)


def get_tracer() -> Optional[Tracer]:
    """Return the configured OpenTelemetry tracer if available."""
    return TRACER


def metrics_endpoint() -> tuple[bytes, str]:
    """Return a tuple of (body, content_type) for the /metrics endpoint."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(), CONTENT_TYPE_LATEST
    return b"", CONTENT_TYPE_LATEST
