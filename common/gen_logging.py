"""
Functions for logging generation events.
"""

import asyncio
from asyncio import CancelledError
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import torch

from common.logger import xlogger
from common.sampling import BaseSamplerRequest
from common.tabby_config import config

# Below this many newly processed prompt tokens a prefill rate says more about
# fixed per-request latency than about ingestion speed, so it isn't reported
PREFILL_RATE_MIN_TOKENS = 256


_PROGRESS_LOG_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tabby-progress-log")


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
    prompt_tokens: Optional[int] = None,
    prompt_total: Optional[int] = None,
):
    """Log a periodic snapshot for an active generation request."""
    elapsed = round(elapsed, 2)
    generation_elapsed = round(generation_elapsed, 2)
    idle = round(idle, 2)
    tokens_per_second = (
        round(generated_tokens / generation_elapsed, 2) if generation_elapsed > 0 else 0.0
    )
    # The reporter tracks prompt ingestion during the prefill/queue phases and
    # generated tokens once streaming begins; render whichever is active.
    in_prefill = stage in ("queued", "started", "prefill")
    if in_prefill:
        prompt_tokens = prompt_tokens or 0
        prompt_total = max(prompt_total or prompt_tokens, prompt_tokens)
        headline = f"prefill {prompt_tokens:,}/{prompt_total:,} prompt tokens"
        detail = (f"(Stage: {stage}, Prefill: {prompt_tokens:,}/{prompt_total:,} tokens, "
                  f"No activity: {idle} s)")
    else:
        headline = f"{generated_tokens} tokens"
        detail = f"(Stage: {stage}, Generate: {tokens_per_second} T/s, No activity: {idle} s)"

    fields = {
        "stage": stage,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "generation_elapsed_seconds": generation_elapsed,
        "idle_seconds": idle,
        "tokens_per_second": tokens_per_second,
    }
    if in_prefill:
        fields.update({"prompt_tokens": prompt_tokens, "prompt_total": prompt_total})

    xlogger.info(
        f"Generation progress (ID: {request_id}): {headline} in {elapsed} seconds",
        fields,
        details=detail,
    )


def _result_token_count(result: dict) -> int:
    token_ids = result.get("token_ids")
    if isinstance(token_ids, torch.Tensor):
        return token_ids.numel()
    if isinstance(token_ids, tuple) and token_ids:
        token_ids = token_ids[0]
        return token_ids.numel() if isinstance(token_ids, torch.Tensor) else len(token_ids)
    return len(token_ids) if token_ids is not None else 0


class GenerationProgressReporter:
    """Emit bounded progress snapshots without blocking request processing."""

    def __init__(self, request_id, interval, logger=log_generation_progress):
        self.request_id = request_id
        self.interval = interval or 0
        self.logger = logger
        self.started = None
        self.last_activity = None
        self.generation_started = None
        self.generated_tokens = 0
        self.prompt_tokens = 0
        self.prompt_total = 0
        self.stage = "queued"
        self.task = None

    def start(self):
        if self.interval <= 0:
            return
        loop = asyncio.get_running_loop()
        self.started = self.last_activity = loop.time()
        self.task = asyncio.create_task(self._run())

    def observe(self, result):
        if self.task is None or not isinstance(result, dict):
            return
        now = asyncio.get_running_loop().time()
        self.last_activity = now
        self.stage = result.get("stage", self.stage)
        curr_progress = result.get("curr_progress")
        if curr_progress is not None:
            self.prompt_tokens = max(self.prompt_tokens, curr_progress)
            self.prompt_total = max(self.prompt_total, result.get("max_progress", 0))
        self.generated_tokens += _result_token_count(result)
        if self.generated_tokens and self.generation_started is None:
            self.generation_started = now

    def attach(self, result_sink):
        put_result = result_sink.put_result

        def put_result_with_progress(result):
            try:
                self.observe(result)
            except Exception:
                pass
            put_result(result)

        result_sink.put_result = put_result_with_progress

    async def _run(self):
        try:
            while True:
                await asyncio.sleep(self.interval)
                await self._report()
        except CancelledError:
            raise
        except Exception:
            return

    async def _report(self):
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self.generated_tokens and self.generation_started is None:
            self.generation_started = now
        await loop.run_in_executor(
            _PROGRESS_LOG_EXECUTOR,
            partial(
                self.logger,
                request_id=self.request_id,
                stage=self.stage,
                generated_tokens=self.generated_tokens,
                elapsed=now - self.started,
                generation_elapsed=(
                    now - self.generation_started if self.generation_started is not None else 0
                ),
                idle=now - self.last_activity,
                prompt_tokens=self.prompt_tokens,
                prompt_total=self.prompt_total,
            ),
        )

    async def stop(self):
        if self.task is None:
            return
        task = self.task
        self.task = None
        task.cancel()
        try:
            await task
        except CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except Exception:
            pass


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
