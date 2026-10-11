"""The persistent disk cache tier (memory.disk_cache_*) is resolved into
generator arguments once per load, gated on the engine having the feature, and
shown in the status line when the engine reports it."""

from types import SimpleNamespace

import pytest

from backends.exllamav3 import model as exl3_model
from common.config_models import MemoryConfig


def memory(**overrides):
    return MemoryConfig(**overrides)


def test_disabled_by_default():
    assert exl3_model.disk_cache_kwargs(memory()) == {}


def test_resolves_engine_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr(exl3_model, "engine_supports_disk_cache", lambda: True)
    cfg = memory(
        sysmem_kv_cache=1024,
        disk_cache_dir=str(tmp_path),
        disk_cache_size=2048,
        disk_cache_flush_on_idle=False,
        disk_cache_checkpoint_interval=0,
    )
    assert exl3_model.disk_cache_kwargs(cfg) == {
        "disk_cache_dir": str(tmp_path),
        "disk_cache_size": 2048 * 1024**2,
        "disk_cache_flush_on_idle": False,
        "disk_cache_checkpoint_interval": 0,
    }


def test_expands_user_home(monkeypatch):
    monkeypatch.setattr(exl3_model, "engine_supports_disk_cache", lambda: True)
    monkeypatch.setenv("HOME", "/home/someone")
    cfg = memory(sysmem_kv_cache=1024, disk_cache_dir="~/kvcache")
    assert exl3_model.disk_cache_kwargs(cfg)["disk_cache_dir"] == "/home/someone/kvcache"


def test_needs_sysmem_tier(monkeypatch, caplog):
    monkeypatch.setattr(exl3_model, "engine_supports_disk_cache", lambda: True)
    warnings = []
    monkeypatch.setattr(exl3_model.xlogger, "warning", lambda msg, *a, **k: warnings.append(msg))
    cfg = memory(sysmem_kv_cache=0, disk_cache_dir="/tmp/kvcache")
    assert exl3_model.disk_cache_kwargs(cfg) == {}
    assert len(warnings) == 1 and "sysmem_kv_cache" in warnings[0]


def test_needs_engine_support(monkeypatch):
    monkeypatch.setattr(exl3_model, "engine_supports_disk_cache", lambda: False)
    warnings = []
    monkeypatch.setattr(exl3_model.xlogger, "warning", lambda msg, *a, **k: warnings.append(msg))
    cfg = memory(sysmem_kv_cache=1024, disk_cache_dir="/tmp/kvcache")
    assert exl3_model.disk_cache_kwargs(cfg) == {}
    assert len(warnings) == 1 and "1.6.2" in warnings[0]


def test_engine_check_is_by_attribute(monkeypatch):
    class Old:
        pass

    class New:
        def flush_cache_to_disk(self):
            pass

    monkeypatch.setattr(exl3_model, "Generator", Old)
    assert exl3_model.engine_supports_disk_cache() is False
    monkeypatch.setattr(exl3_model, "Generator", New)
    assert exl3_model.engine_supports_disk_cache() is True


def test_size_must_be_positive():
    with pytest.raises(ValueError):
        memory(disk_cache_size=0)


def summary_with(stats):
    from common import model
    from common.status_display import StatusDisplay

    display = StatusDisplay()
    display._cache_stats = lambda: stats
    model_container = SimpleNamespace(loaded=True)
    saved = model.container
    model.container = model_container
    try:
        return display._summary_line().plain
    finally:
        model.container = saved


def test_summary_without_disk_tier():
    stats = {
        "max_tokens": 1000,
        "used_tokens": 100,
        "cached_tokens": 50,
        "tier_max_tokens": 2000,
        "tier_cached_tokens": 300,
        "hit_rate": 0.5,
        "tier_hit_rate": 0.25,
        "pending_jobs": 0,
    }
    text = summary_with(stats)
    assert "50 reusable, +300 more in sysmem ·" in text
    assert "hit rate 50% incl. 25% from sysmem ·" in text
    assert "disk" not in text


def test_summary_with_disk_tier():
    stats = {
        "max_tokens": 1000,
        "used_tokens": 100,
        "cached_tokens": 50,
        "tier_max_tokens": 2000,
        "tier_cached_tokens": 300,
        "disk_cached_tokens": 4096,
        "disk_used_bytes": int(1.5 * 1024**3),
        "disk_max_bytes": 50 * 1024**3,
        "hit_rate": 0.5,
        "tier_hit_rate": 0.25,
        "disk_hit_rate": 0.1,
        "pending_jobs": 0,
    }
    text = summary_with(stats)
    assert "50 reusable, +300 more in sysmem, +4,096 more on disk · disk 1.5/50 GB" in text
    # The engine's sysmem rate includes the pages it promoted from disk
    assert "hit rate 50% incl. 15% from sysmem, 10% from disk" in text


def test_summary_restored_from_disk():
    """After a restore the pages are in every tier, so nothing is reusable
    *beyond* VRAM, and every hit came from disk."""

    stats = {
        "max_tokens": 256000,
        "used_tokens": 0,
        "cached_tokens": 11520,
        "tier_max_tokens": 4096,
        "tier_cached_tokens": 0,
        "disk_cached_tokens": 0,
        "disk_used_bytes": int(1.0 * 1024**3),
        "disk_max_bytes": 98 * 1024**3,
        "hit_rate": 0.73,
        "tier_hit_rate": 0.73,
        "disk_hit_rate": 0.73,
        "pending_jobs": 0,
    }
    text = summary_with(stats)
    assert "11,520 reusable, +0 more in sysmem, +0 more on disk · disk 1.0/98 GB" in text
    assert "hit rate 73% incl. 0% from sysmem, 73% from disk" in text
