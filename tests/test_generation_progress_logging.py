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
async def test_progress_reporter_tracks_produced_results_and_stops_cleanly():
    logs = []
    reporter = GenerationProgressReporter(
        "request-1", 60, logger=lambda **fields: logs.append(fields)
    )

    reporter.start()
    reporter.observe({"stage": "streaming", "token_ids": torch.tensor([[1, 2, 3]])})
    reporter.last_activity -= 1
    await reporter._report()
    reporter.observe({"stage": "streaming", "token_ids": torch.tensor([[4, 5]])})
    await reporter._report()
    await reporter.stop()

    assert len(logs) == 2
    assert logs[0]["request_id"] == "request-1"
    assert logs[0]["stage"] == "streaming"
    assert logs[0]["generated_tokens"] == 3
    assert logs[0]["idle"] >= 1
    assert logs[1]["generated_tokens"] == 5
    assert logs[1]["idle"] < logs[0]["idle"]


@pytest.mark.asyncio
async def test_progress_reporter_logs_prefill_tokens_instead_of_zero():
    logs = []
    reporter = GenerationProgressReporter(
        "request-1", 60, logger=lambda **fields: logs.append(fields)
    )

    reporter.start()
    reporter.observe({"stage": "prefill", "curr_progress": 500, "max_progress": 1000})
    reporter.observe({"stage": "prefill", "curr_progress": 800, "max_progress": 1000})
    await reporter._report()
    await reporter.stop()

    assert len(logs) == 1
    assert logs[0]["stage"] == "prefill"
    assert logs[0]["generated_tokens"] == 0
    assert logs[0]["prompt_tokens"] == 800
    assert logs[0]["prompt_total"] == 1000


@pytest.mark.asyncio
async def test_progress_reporter_attach_tracks_production_without_blocking_delivery():
    logs = []
    put_result = Mock()
    sink = SimpleNamespace(put_result=put_result)
    reporter = GenerationProgressReporter(
        "request-1", 60, logger=lambda **fields: logs.append(fields)
    )
    reporter.start()
    reporter.attach(sink)

    result = {"stage": "streaming", "token_ids": torch.tensor([[1, 2]])}
    sink.put_result(result)
    error = RuntimeError("generation failed")
    sink.put_result(error)
    await reporter._report()
    await reporter.stop()

    assert put_result.call_args_list[0].args == (result,)
    assert put_result.call_args_list[1].args == (error,)
    assert logs[0]["generated_tokens"] == 2

    reporter.observe = Mock(side_effect=RuntimeError("telemetry failed"))
    sink.put_result(result)
    assert put_result.call_args_list[-1].args == (result,)


@pytest.mark.asyncio
async def test_progress_reporter_disabled_interval_never_starts_or_logs():
    logger = Mock()
    reporter = GenerationProgressReporter("request-1", 0, logger=logger)

    reporter.start()
    reporter.observe({"stage": "streaming", "token_ids": torch.tensor([[1]])})
    await asyncio.sleep(0)
    await reporter.stop()

    assert reporter.task is None
    logger.assert_not_called()


@pytest.mark.asyncio
async def test_progress_reporter_keeps_concurrent_requests_isolated():
    logs = []
    logger = lambda **fields: logs.append(fields)
    first = GenerationProgressReporter("first", 60, logger=logger)
    second = GenerationProgressReporter("second", 60, logger=logger)
    first.start()
    second.start()
    first.observe({"stage": "prefill"})
    second.observe({"stage": "streaming", "token_ids": torch.tensor([[1]])})

    await asyncio.gather(first._report(), second._report())
    await first.stop()
    await second.stop()

    assert any(item["request_id"] == "first" and item["stage"] == "prefill" for item in logs)
    assert any(item["request_id"] == "second" and item["generated_tokens"] == 1 for item in logs)


@pytest.mark.asyncio
async def test_progress_reporter_run_isolates_report_failures():
    reporter = GenerationProgressReporter("request-1", 0, logger=Mock())

    async def fail_report():
        raise RuntimeError("logging failed")

    reporter._report = fail_report
    await reporter._run()


@pytest.mark.asyncio
async def test_progress_reporter_stop_preserves_caller_cancellation():
    async def cancel_while_stopping():
        reporter = GenerationProgressReporter("request-1", 1, logger=Mock())
        reporter.task = asyncio.create_task(asyncio.sleep(60))
        asyncio.current_task().cancel()
        await reporter.stop()

    stop_task = asyncio.create_task(cancel_while_stopping())
    with pytest.raises(asyncio.CancelledError):
        await stop_task


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
