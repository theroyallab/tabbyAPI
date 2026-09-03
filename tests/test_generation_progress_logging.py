from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from common.config_models import LoggingConfig
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
