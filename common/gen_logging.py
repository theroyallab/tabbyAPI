"""
Functions for logging generation events.
"""

from typing import Optional

from common.logger import xlogger
from common.sampling import BaseSamplerRequest
from common.tabby_config import config

# Below this many newly processed prompt tokens a prefill rate says more about
# fixed per-request latency than about ingestion speed, so it isn't reported
PREFILL_RATE_MIN_TOKENS = 256


def broadcast_status():
    """Broadcasts the current logging status"""
    enabled = []
    if config.logging.log_prompt:
        enabled.append("prompts")

    if config.logging.log_generation_params:
        enabled.append("generation params")

    if config.logging.log_generation_progress_interval:
        interval = config.logging.log_generation_progress_interval
        enabled.append(f"generation progress every {interval:g} seconds")

    if enabled:
        xlogger.info("Generation logging is enabled for: " + ", ".join(enabled))


def format_settings(settings: list, params: BaseSamplerRequest) -> str:
    """
    Render effective generation settings as "name: value (source)" items, where the
    source says whether the value came from the request, a sampler override preset
    (or a forced override), or a built-in default.
    """

    parts = []
    for name, value in settings:
        if isinstance(value, tuple):
            # (display value, explicit source)
            value, source = value
        else:
            source = params.param_source(name)

        if value is True:
            parts.append(f"{name} ({source})")
        else:
            parts.append(f"{name}: {value} ({source})")

    return ", ".join(parts)


def log_request_start(label: str, context_len: int, settings_text: str, extra: dict):
    """One-line summary of a generation request as the backend starts it."""

    xlogger.info(f"{label}: {context_len:,} prompt tokens · {settings_text}", extra)


def log_generation_params(label: str, **kwargs):
    """Logs the full generation parameter dump to console (opt-in)."""
    if config.logging.log_generation_params:
        xlogger.info(f"{label} generation options:", kwargs, details=f"{kwargs}\n")


def log_prompt(prompt: str, label: str, negative_prompt: Optional[str] = None):
    """Logs the prompt to console."""
    if config.logging.log_prompt:
        xlogger.info(
            f"{label} prompt:",
            {"prompt": prompt},
            details=f"\n{prompt if prompt else 'Empty'}\n",
        )

        if negative_prompt:
            xlogger.info(
                f"{label} negative prompt:",
                {"negative_prompt": negative_prompt},
                details=f"\n{negative_prompt}\n",
            )


def log_response(label: str, response: str):
    """Logs the response to console."""
    if config.logging.log_prompt:
        xlogger.info(
            f"{label} response:",
            {"response": response},
            details=f"\n{response if response else 'Empty'}\n",
        )


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.1f} s"


def _describe_finish(metrics: dict) -> Optional[str]:
    """
    Why generation ended, or None for the unremarkable cases. Ending on a stop
    token or a completed grammar is what normally happens and isn't worth a
    mention; hitting max_tokens, a client stop string or a loop is.
    """

    eos_reason = metrics.get("eos_reason")
    stop_str = metrics.get("stop_str")

    match eos_reason:
        case "stop_token" | "end_filter" | None:
            return None
        case "stop_string":
            return f"stop string {stop_str!r}" if stop_str else "stop string"
        case "max_new_tokens":
            return "max_tokens reached"
        case "loop_detected":
            return "loop detected"
        case _:
            return str(eos_reason)


def log_generation_progress(
    request_id: str,
    stage: str,
    generated_tokens: int,
    elapsed: float,
    generation_elapsed: float,
    idle: float,
):
    """Log a periodic snapshot for an active generation request."""
    elapsed = round(elapsed, 2)
    generation_elapsed = round(generation_elapsed, 2)
    idle = round(idle, 2)
    tokens_per_second = (
        round(generated_tokens / generation_elapsed, 2) if generation_elapsed > 0 else 0.0
    )
    xlogger.info(
        f"Generation progress (ID: {request_id}): {generated_tokens} tokens in {elapsed} seconds",
        {
            "stage": stage,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "generation_elapsed_seconds": generation_elapsed,
            "idle_seconds": idle,
            "tokens_per_second": tokens_per_second,
        },
        details=(f"(Stage: {stage}, Generate: {tokens_per_second} T/s, No activity: {idle} s)"),
    )


def log_metrics(
    label: str,
    metrics: dict,
    context_len: Optional[int],
    max_seq_len: int,
):
    """
    Log the outcome of a generation. Prompt caching is reported as the share of the
    prompt that was already in the cache; the prefill rate covers only the tokens
    that actually had to be processed.
    """

    gen_tokens = metrics.get("gen_tokens") or 0
    gen_time = metrics.get("gen_time") or 0.0
    gen_ts = metrics.get("gen_tokens_per_sec")

    prompt_tokens = metrics.get("prompt_tokens") or 0
    cached_tokens = int(metrics.get("cached_tokens") or 0)
    new_tokens = max(prompt_tokens - cached_tokens, 0)
    prompt_time = metrics.get("prompt_time") or 0.0
    queue_time = metrics.get("queue_time") or 0.0
    total_time = metrics.get("total_time") or (queue_time + prompt_time + gen_time)

    # Generation
    generated = f"{gen_tokens:,} tokens generated"
    if isinstance(gen_ts, (int, float)):
        generated += f" at {gen_ts:,.1f} T/s"
    sections = [generated]

    # Prompt and cache reuse
    if prompt_tokens:
        if cached_tokens:
            cache_part = f"{cached_tokens / prompt_tokens * 100:.0f}% cached"
        else:
            cache_part = "none cached"

        processed = f"{new_tokens:,} new in {_format_seconds(prompt_time)}"
        if new_tokens >= PREFILL_RATE_MIN_TOKENS and prompt_time > 0:
            processed += f" ({new_tokens / prompt_time:,.0f} T/s)"

        sections.append(f"prompt {prompt_tokens:,} tokens, {cache_part}, {processed}")

    # Timing as the client experiences it
    timing = []
    if queue_time >= 0.1:
        timing.append(f"queued {_format_seconds(queue_time)}")
    timing.append(f"first token {_format_seconds(queue_time + prompt_time)}")
    timing.append(f"total {_format_seconds(total_time)}")
    sections.append(", ".join(timing))

    # Speculative decoding
    if "draft_accept" in metrics:
        accept = metrics.get("draft_accept", 0)
        reject = metrics.get("draft_reject", 0)
        total_draft = accept + reject
        accept_rate = accept / total_draft * 100 if total_draft > 0 else 0.0
        sections.append(f"draft {accept}/{total_draft} accepted ({accept_rate:.0f}%)")

    finish = _describe_finish(metrics)
    if finish:
        sections.append(finish)

    message = f"{label}: " + " · ".join(sections)
    if context_len and context_len > max_seq_len:
        message += " (context truncated)"

    xlogger.info(
        message,
        {
            "new_tokens": new_tokens,
            "cached_tokens": cached_tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_time": prompt_time,
            "prompt_tokens_per_second": metrics.get("prompt_tokens_per_sec"),
            "gen_tokens": gen_tokens,
            "gen_time": gen_time,
            "gen_tokens_per_second": gen_ts,
            "queue_time": queue_time,
            "total_time": total_time,
            "eos_reason": metrics.get("eos_reason"),
            "context_len": context_len,
            "max_seq_len": max_seq_len,
        },
    )
