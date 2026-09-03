import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError

from common.config_models import LoggingConfig
from common.gen_logging import GenerationProgressReporter
from common import gen_logging


def test_generation_progress_interval_is_disabled_by_default_and_nonnegative():
    assert LoggingConfig().log_generation_progress_interval == 0
    assert LoggingConfig(log_generation_progress_interval=15).log_generation_progress_interval == 15

    with pytest.raises(ValidationError):
        LoggingConfig(log_generation_progress_interval=-1)


def test_broadcast_status_includes_generation_progress(monkeypatch):
    logger = Mock()
    config = SimpleNamespace(
        logging=SimpleNamespace(
            log_prompt=False,
            log_generation_params=False,
            log_generation_progress_interval=15,
        )
    )
    monkeypatch.setattr(gen_logging, "xlogger", logger)
    monkeypatch.setattr(gen_logging, "config", config)

    gen_logging.broadcast_status()

    assert logger.info.call_args.args[0] == (
        "Generation logging is enabled for: generation progress every 15 seconds"
    )


@pytest.mark.asyncio
async def test_progress_reporter_tracks_pending_results_and_stops_cleanly():
    logs = []
    pending = [{"stage": "streaming", "token_ids": torch.tensor([[1, 2, 3]])}]
    reporter = GenerationProgressReporter(
        "request-1", 0.01, lambda: pending, logger=lambda **fields: logs.append(fields)
    )

    reporter.start()
    await asyncio.sleep(0.025)
    reporter.observe(pending.pop())
    reporter.observe({"stage": "streaming", "token_ids": torch.tensor([[4, 5]])})
    await asyncio.sleep(0.015)
    await reporter.stop()
    logged_count = len(logs)
    await asyncio.sleep(0.015)

    assert logged_count >= 3
    assert len(logs) == logged_count
    assert logs[0]["request_id"] == "request-1"
    assert logs[0]["stage"] == "streaming"
    assert logs[0]["generated_tokens"] == 3
    assert logs[-1]["generated_tokens"] == 5
    assert logs[-1]["idle"] >= 0


@pytest.mark.asyncio
async def test_progress_reporter_disabled_interval_never_starts_or_logs():
    logger = Mock()
    reporter = GenerationProgressReporter("request-1", 0, lambda: (), logger=logger)

    reporter.start()
    reporter.observe({"stage": "streaming", "token_ids": torch.tensor([[1]])})
    await asyncio.sleep(0)
    await reporter.stop()

    assert reporter.task is None
    logger.assert_not_called()


@pytest.mark.asyncio
async def test_progress_reporter_isolates_logger_failures_and_concurrent_requests():
    logs = []

    def logger(**fields):
        logs.append(fields)
        if fields["request_id"] == "broken":
            raise RuntimeError("logging failed")

    first = GenerationProgressReporter("healthy", 0.01, lambda: (), logger=logger)
    second = GenerationProgressReporter("broken", 0.01, lambda: (), logger=logger)
    first.start()
    second.start()
    first.observe({"stage": "prefill"})
    second.observe({"stage": "streaming", "token_ids": torch.tensor([[1]])})

    await asyncio.sleep(0.025)
    await first.stop()
    await second.stop()

    assert any(item["request_id"] == "healthy" and item["stage"] == "prefill" for item in logs)
    assert any(item["request_id"] == "broken" and item["generated_tokens"] == 1 for item in logs)


def test_log_generation_progress_reports_stage_rate_and_idle_time(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(gen_logging, "xlogger", logger)

    gen_logging.log_generation_progress(
        request_id="request-1",
        stage="streaming",
        generated_tokens=120,
        elapsed=42.0,
        generation_elapsed=12.0,
        idle=3.5,
    )

    message, fields = logger.info.call_args.args[:2]
    details = logger.info.call_args.kwargs["details"]
    assert message == "Generation progress (ID: request-1): 120 tokens in 42.0 seconds"
    assert fields == {
        "stage": "streaming",
        "generated_tokens": 120,
        "elapsed_seconds": 42.0,
        "generation_elapsed_seconds": 12.0,
        "idle_seconds": 3.5,
        "tokens_per_second": 10.0,
    }
    assert details == "(Stage: streaming, Generate: 10.0 T/s, No activity: 3.5 s)"
