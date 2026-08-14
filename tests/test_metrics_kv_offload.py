import types
import unittest

from common import model as model_module
from common.metrics import MetricsManagerClass


PAGE_SIZE = 256
SLOT_BYTES = 16 * 1024 * 1024


class FakeCPUPageCache:
    """Stand-in for exllamav3's CPUPageCache, with the surface the reader uses."""

    def __init__(self, pages=100, max_slots=400, **overrides):
        self.pages = pages
        self.max_slots = max_slots
        self.slot_size = SLOT_BYTES
        self.metrics = {
            "pushes": 500,
            "dedup_hits": 40,
            "restores": 390,
            "evictions": 100,
            "cold_allocs": 3,
        }
        self.metrics.update(overrides)

    def __len__(self):
        return self.pages


def install_container(cpu_cache, max_pages=1000, max_total_tokens=1000 * PAGE_SIZE):
    """Point common.model.container at a stand-in generator exposing a CPU page cache."""

    pagetable = types.SimpleNamespace(max_pages=max_pages, referenced_pages={})
    sync_generator = types.SimpleNamespace(
        pagetable=pagetable,
        max_total_tokens=max_total_tokens,
        cpu_page_cache=cpu_cache,
    )
    model_module.container = types.SimpleNamespace(
        generator=types.SimpleNamespace(generator=sync_generator)
    )


class KVOffloadMetricsTests(unittest.TestCase):
    """The /metrics view of the CPU page cache.

    The counters behind these series live in the generator's page cache rather
    than in the metrics manager, so the reader has to tolerate a generator that
    is missing, has offloading disabled, or reports something unexpected,
    without dropping the series or raising through a scrape.
    """

    def setUp(self):
        self.metrics = MetricsManagerClass()
        self.original_container = model_module.container

    def tearDown(self):
        model_module.container = self.original_container

    def test_page_counts_are_reported_as_tokens(self):
        install_container(FakeCPUPageCache())

        live = self.metrics._live_kv_offload()

        self.assertEqual(live["tokens"], 100 * PAGE_SIZE)
        self.assertEqual(live["max_tokens"], 400 * PAGE_SIZE)
        self.assertAlmostEqual(live["usage_ratio"], 0.25)

    def test_restored_tokens_measure_prefill_avoided(self):
        # Every restore reads back one whole page, which is one page of prompt
        # that did not have to be prefilled again.
        install_container(FakeCPUPageCache())

        live = self.metrics._live_kv_offload()

        self.assertEqual(live["restored_tokens"], 390 * PAGE_SIZE)

    def test_byte_counters_are_derived_from_whole_slot_transfers(self):
        # A transfer in either direction always moves exactly one slot, so the
        # byte totals follow from the transfer counts rather than being an
        # estimate.
        install_container(FakeCPUPageCache())

        live = self.metrics._live_kv_offload()

        self.assertEqual(live["bytes"], 100 * SLOT_BYTES)
        self.assertEqual(live["max_bytes"], 400 * SLOT_BYTES)
        self.assertEqual(live["bytes_read"], 390 * SLOT_BYTES)
        self.assertEqual(live["bytes_written"], 500 * SLOT_BYTES)

    def test_synchronous_pinning_is_visible(self):
        # A store that outruns the background pinning thread pins on the
        # generator's own thread at roughly 2.5 GB/s. That has to be observable.
        install_container(FakeCPUPageCache(cold_allocs=17))

        self.assertEqual(self.metrics._live_kv_offload()["cold_allocs"], 17)

    def test_series_are_published_when_offloading_is_disabled(self):
        # Series that appear and disappear across scrapes are awkward to alert
        # on, so a disabled cache reports zeros rather than nothing.
        install_container(None)

        rendered = self.metrics.render_prometheus()

        for name in (
            "tabbyapi:kv_offload_usage_ratio",
            "tabbyapi:kv_offload_tokens",
            "tabbyapi:kv_offload_max_tokens",
            "tabbyapi:kv_offload_bytes",
            "tabbyapi:kv_offload_max_bytes",
            "tabbyapi:kv_offload_restored_tokens_total",
            "tabbyapi:kv_offload_stores_total",
            "tabbyapi:kv_offload_deduped_stores_total",
            "tabbyapi:kv_offload_evictions_total",
            "tabbyapi:kv_offload_cold_allocs_total",
            "tabbyapi:kv_offload_read_bytes_total",
            "tabbyapi:kv_offload_written_bytes_total",
        ):
            self.assertIn(f"{name} 0", rendered)

    def test_no_model_loaded_does_not_raise(self):
        model_module.container = None

        self.assertEqual(self.metrics._live_kv_offload()["tokens"], 0)

    def test_a_broken_cpu_cache_does_not_break_the_scrape(self):
        install_container(types.SimpleNamespace(metrics={}))

        self.assertEqual(self.metrics._live_kv_offload()["tokens"], 0)
        self.assertIn("tabbyapi:kv_offload_tokens 0", self.metrics.render_prometheus())

    def test_no_throughput_gauge_is_published(self):
        # A lifetime average of transfer rate divides by wall clock that
        # includes every interval with no transfers at all. The bytes counters
        # are exposed so a windowed rate can be taken at query time instead.
        install_container(FakeCPUPageCache())

        rendered = self.metrics.render_prometheus()

        for name in (
            "tabbyapi:kv_offload_bytes_seconds",
            "tabbyapi:kv_offload_read_bytes_seconds",
            "tabbyapi:kv_offload_written_bytes_seconds",
            "tabbyapi:kv_offload_hit_ratio",
        ):
            self.assertNotIn(name, rendered)

    def test_rendered_values_match_the_live_read(self):
        install_container(FakeCPUPageCache())

        rendered = self.metrics.render_prometheus()

        self.assertIn(f"tabbyapi:kv_offload_tokens {100 * PAGE_SIZE}", rendered)
        self.assertIn("tabbyapi:kv_offload_usage_ratio 0.25", rendered)
        self.assertIn(f"tabbyapi:kv_offload_restored_tokens_total {390 * PAGE_SIZE}", rendered)
        self.assertIn("tabbyapi:kv_offload_evictions_total 100", rendered)


if __name__ == "__main__":
    unittest.main()
