"""The /metrics text: read from the status snapshot, recorded once per finished request."""

import pytest

from common.metrics import MetricsManagerClass
from common.status_display import StatusDisplay


def snapshot(**overrides):
    status = {
        "jobs": [],
        "completed": 0,
        "generated_tokens": 0,
        "generated_seconds": 0.0,
        "prefilled_tokens": 0,
        "cache": None,
    }
    status.update(overrides)
    return status


def finish_chunk(**overrides):
    chunk = {
        "prompt_tokens": 1000,
        "cached_tokens": 900,
        "queue_time": 0.25,
        "prompt_time": 0.5,
        "gen_tokens": 10,
        "gen_time": 1.0,
        "draft_accept": 4,
        "draft_reject": 1,
    }
    chunk.update(overrides)
    return chunk


def values(rendered: str) -> dict:
    lines = (line for line in rendered.splitlines() if not line.startswith("#"))
    return {name: float(value) for name, value in (line.rsplit(" ", 1) for line in lines)}


def test_exposes_the_names_of_llama_cpps_exporter():
    rendered = MetricsManagerClass().render_prometheus(snapshot())

    for metric_type, name in (
        ("counter", "prompt_tokens_total"),
        ("counter", "prompt_tokens_cached_total"),
        ("counter", "prompt_seconds_total"),
        ("counter", "tokens_predicted_total"),
        ("counter", "tokens_predicted_seconds_total"),
        ("counter", "n_tokens_max"),
        ("counter", "spec_decode_num_draft_tokens_total"),
        ("counter", "spec_decode_num_accepted_tokens_total"),
        ("counter", "spec_decode_num_drafts_total"),
        ("gauge", "prompt_tokens_seconds"),
        ("gauge", "predicted_tokens_seconds"),
        ("gauge", "requests_processing"),
        ("gauge", "requests_deferred"),
    ):
        assert f"# TYPE tabbyapi:{name} {metric_type}\n" in rendered, name


def test_token_counters_come_from_the_status_snapshot():
    status = snapshot(generated_tokens=20, generated_seconds=0.5, prefilled_tokens=7, completed=3)
    metrics = values(MetricsManagerClass().render_prometheus(status))

    assert metrics["tabbyapi:tokens_predicted_total"] == 20
    assert metrics["tabbyapi:tokens_predicted_seconds_total"] == 0.5
    assert metrics["tabbyapi:prompt_tokens_total"] == 7
    assert metrics["tabbyapi:requests_total"] == 3


def test_jobs_back_in_the_engines_queue_are_deferred():
    jobs = [{"stage": stage} for stage in ("queued", "waiting", "prefill", "generating")]
    metrics = values(MetricsManagerClass().render_prometheus(snapshot(jobs=jobs)))

    assert metrics["tabbyapi:requests_processing"] == 2
    assert metrics["tabbyapi:requests_deferred"] == 2


def test_finish_records_the_prompt_side_and_drafting():
    manager = MetricsManagerClass()
    manager.record_finish(finish_chunk())
    manager.record_finish(finish_chunk(prompt_tokens=200, gen_tokens=5000, cached_tokens=0))

    metrics = values(manager.render_prometheus(snapshot()))

    assert metrics["tabbyapi:prompt_tokens_cached_total"] == 900
    assert metrics["tabbyapi:prompt_seconds_total"] == 1.0
    assert metrics["tabbyapi:n_tokens_max"] == 5200
    assert metrics["tabbyapi:spec_decode_num_draft_tokens_total"] == 10
    assert metrics["tabbyapi:spec_decode_num_accepted_tokens_total"] == 8
    # One step per token of the target model's own
    assert metrics["tabbyapi:spec_decode_num_drafts_total"] == 6 + 4996


def test_request_without_a_drafter_leaves_the_draft_counters_alone():
    manager = MetricsManagerClass()
    manager.record_finish(finish_chunk(draft_accept=None, draft_reject=None))

    metrics = values(manager.render_prometheus(snapshot()))

    assert metrics["tabbyapi:spec_decode_num_draft_tokens_total"] == 0
    assert metrics["tabbyapi:spec_decode_num_drafts_total"] == 0


def test_throughput_gauges_cover_the_window_since_the_last_scrape():
    manager = MetricsManagerClass()

    status = snapshot(generated_tokens=100, generated_seconds=2.0)
    assert values(manager.render_prometheus(status))["tabbyapi:predicted_tokens_seconds"] == 50

    status = snapshot(generated_tokens=400, generated_seconds=3.0)
    assert values(manager.render_prometheus(status))["tabbyapi:predicted_tokens_seconds"] == 300

    # Nothing happened since, which is no throughput rather than the old figure
    assert values(manager.render_prometheus(status))["tabbyapi:predicted_tokens_seconds"] == 0


def test_prompt_gauge_ignores_a_prefill_still_in_progress():
    manager = MetricsManagerClass()
    manager.record_finish(finish_chunk())

    # 100 processed tokens in 0.5 s; the other 5000 have no time to go with yet
    metrics = values(manager.render_prometheus(snapshot(prefilled_tokens=5100)))

    assert metrics["tabbyapi:prompt_tokens_total"] == 5100
    assert metrics["tabbyapi:prompt_tokens_seconds"] == 200


def test_cache_figures_come_from_the_engines_statistics():
    cache = {
        "max_tokens": 1000,
        "used_tokens": 250,
        "cached_tokens": 600,
        "tier_cached_tokens": 40,
        "disk_cached_tokens": 30,
        "hit_rate": 0.75,
        "tier_hit_rate": 0.5,
        "disk_hit_rate": 0.25,
    }
    metrics = values(MetricsManagerClass().render_prometheus(snapshot(cache=cache)))

    assert metrics["tabbyapi:kv_cache_usage_ratio"] == 0.25
    assert metrics["tabbyapi:kv_cache_tokens"] == 250
    assert metrics["tabbyapi:kv_cache_max_tokens"] == 1000
    assert metrics["tabbyapi:kv_cache_reusable_tokens"] == 600
    assert metrics["tabbyapi:kv_cache_sysmem_tokens"] == 40
    assert metrics["tabbyapi:kv_cache_disk_tokens"] == 30
    assert metrics["tabbyapi:kv_cache_hit_rate"] == 0.75
    assert metrics["tabbyapi:kv_cache_sysmem_hit_rate"] == 0.5
    assert metrics["tabbyapi:kv_cache_disk_hit_rate"] == 0.25


def test_every_series_exists_without_a_model_or_a_job():
    rendered = MetricsManagerClass().render_prometheus(snapshot())

    assert "tabbyapi:kv_cache_max_tokens 0" in rendered
    assert "tabbyapi:kv_cache_hit_rate 0.0" in rendered
    assert 'model_name="' not in rendered


def test_histograms_are_cumulative():
    manager = MetricsManagerClass()
    manager.record_finish(finish_chunk(queue_time=0.0, prompt_time=0.1, gen_time=0.1))
    manager.record_finish(finish_chunk(queue_time=0.25, prompt_time=0.25, gen_time=10000.0))

    metrics = values(manager.render_prometheus(snapshot()))

    # 0.1 and 0.5 to the first token
    assert metrics['tabbyapi:time_to_first_token_seconds_bucket{le="0.1"}'] == 1
    assert metrics['tabbyapi:time_to_first_token_seconds_bucket{le="0.5"}'] == 2
    assert metrics['tabbyapi:time_to_first_token_seconds_bucket{le="+Inf"}'] == 2
    assert metrics["tabbyapi:time_to_first_token_seconds_count"] == 2
    assert metrics["tabbyapi:time_to_first_token_seconds_sum"] == pytest.approx(0.6)

    # Past the last boundary, so only in +Inf
    assert metrics['tabbyapi:e2e_request_latency_seconds_bucket{le="7680.0"}'] == 1
    assert metrics['tabbyapi:e2e_request_latency_seconds_bucket{le="+Inf"}'] == 2


def test_model_name_is_escaped_for_its_label():
    rendered = MetricsManagerClass().render_prometheus(snapshot(), 'odd"name\\')

    assert 'tabbyapi:model_info{model_name="odd\\"name\\\\"} 1' in rendered


def test_generation_time_stays_counted_when_a_job_ends(monkeypatch):
    # A counter that dropped at the end of a request would read as a reset
    clock = [100.0]
    monkeypatch.setattr("common.status_display.time.monotonic", lambda: clock[0])

    display = StatusDisplay()
    job = display.add_job("a", "#1", 10)
    assert display.snapshot()["generated_seconds"] == 0

    job.generated(1)
    clock[0] = 102.0
    job.generated(2)
    assert display.snapshot()["generated_seconds"] == 2

    display.remove_job("a")
    assert display.snapshot()["generated_seconds"] == 2
