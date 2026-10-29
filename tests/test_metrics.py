"""The live /metrics counters: written per delivered piece, not per finished request."""

import pytest

import common.metrics as metrics_module
from common.metrics import MetricsManagerClass


class FakeClock:
    """Stand-in for the module's `time`: monotonic seconds the test advances itself."""

    def __init__(self):
        self.monotonic_now = 0.0

    def time(self):
        return 1_700_000_000.0

    def monotonic(self):
        return self.monotonic_now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(metrics_module, "time", fake)
    # Reading the live job counts pulls in the backend; the counters don't need it
    monkeypatch.setattr(MetricsManagerClass, "_live_job_counts", staticmethod(lambda: (1, 2)))
    return fake


def finish_chunk(**overrides):
    chunk = {
        "prompt_tokens": 1000,
        "cached_tokens": 900,
        "prompt_time": 0.5,
        "gen_tokens": 10,
        "draft_accept": 4,
        "draft_reject": 1,
    }
    chunk.update(overrides)
    return chunk


def test_counters_move_with_each_delivered_piece(clock):
    metrics = MetricsManagerClass()

    metrics.tokens_delivered("a", 3)
    assert metrics.tokens_predicted_total == 3
    # The first piece only starts the request's clock
    assert metrics.tokens_predicted_seconds_total == 0.0

    clock.monotonic_now = 0.25
    metrics.tokens_delivered("a", 4)

    assert metrics.tokens_predicted_total == 7
    assert metrics.tokens_predicted_seconds_total == pytest.approx(0.25)

    # An empty delivery counts nothing and does not touch the clock
    metrics.tokens_delivered("a", 0)
    assert metrics.tokens_predicted_total == 7
    assert metrics.tokens_predicted_seconds_total == pytest.approx(0.25)


def test_decode_time_is_tracked_per_request(clock):
    metrics = MetricsManagerClass()

    metrics.tokens_delivered("a", 1)
    clock.monotonic_now = 1.0
    metrics.tokens_delivered("b", 1)
    clock.monotonic_now = 3.0
    metrics.tokens_delivered("a", 1)

    # 3s for a, 0s for b's first piece: a shared clock would have counted 3s for both
    assert metrics.tokens_predicted_seconds_total == pytest.approx(3.0)


def test_end_request_records_the_prompt_side(clock):
    metrics = MetricsManagerClass()

    metrics.tokens_delivered("a", 10)
    metrics.end_request("a", finish_chunk())

    # Only the tokens actually processed, like llama.cpp's counter of that name
    assert metrics.prompt_tokens_total == 100
    assert metrics.cached_tokens_total == 900
    assert metrics.prompt_seconds_total == pytest.approx(0.5)
    assert metrics.requests_total == 1
    assert metrics.draft_tokens_accepted_total == 4
    assert metrics.draft_tokens_rejected_total == 1
    assert metrics.draft_requests_total == 1
    assert metrics.draft_decode_steps_total == 6
    assert "a" not in metrics._last_delivery


def test_request_without_a_drafter_has_no_draft_tallies(clock):
    metrics = MetricsManagerClass()

    metrics.end_request("a", finish_chunk(draft_accept=None, draft_reject=None))

    assert metrics.requests_total == 1
    assert metrics.draft_requests_total == 0
    assert metrics.draft_decode_steps_total == 0


def test_cancelled_request_keeps_its_tokens_but_is_not_a_finished_request(clock):
    metrics = MetricsManagerClass()

    metrics.tokens_delivered("a", 5)
    metrics.end_request("a", {})

    assert metrics.tokens_predicted_total == 5
    assert metrics.requests_total == 0
    assert metrics.prompt_tokens_total == 0
    assert "a" not in metrics._last_delivery


def test_render_exposes_the_names_a_dashboard_reads(clock):
    metrics = MetricsManagerClass()

    metrics.tokens_delivered("a", 10)
    clock.monotonic_now = 0.5
    metrics.tokens_delivered("a", 10)
    rendered = metrics.render_prometheus()

    for name in (
        "tokens_predicted_total",
        "tokens_predicted_seconds_total",
        "prompt_tokens_total",
        "cached_tokens_total",
        "requests_total",
        "requests_processing",
        "requests_deferred",
        "predicted_tokens_seconds",
        "spec_decode_num_draft_tokens_total",
    ):
        assert f"# TYPE tabbyapi:{name}" in rendered, name

    assert "tabbyapi:tokens_predicted_total 20" in rendered
    assert "tabbyapi:requests_processing 1" in rendered
    assert "tabbyapi:requests_deferred 2" in rendered
    # 20 tokens over 0.5s of decode time
    assert "tabbyapi:predicted_tokens_seconds 40.0" in rendered
