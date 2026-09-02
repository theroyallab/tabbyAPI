import types
import unittest

from common import model as model_module
from common.metrics import MetricsManagerClass


PAGE_SIZE = 256
CHECKPOINT_BYTES = 155 * 1024**2


class FakeRecurrentCache:
    """Stand-in for exllamav3's RecurrentCache, with the surface the reader uses."""

    def __init__(self, checkpoints=60, max_size=16 * 1024**3, **overrides):
        self.checkpoints = checkpoints
        self.current_size = checkpoints * CHECKPOINT_BYTES
        self.max_size = max_size
        self.metrics = {
            "stash_evictions": 40,
            "stash_evictions_stranded": 25,
            "stash_evictions_live_kv": 12,
            "stash_pruned": 8,
        }
        self.metrics.update(overrides)

    def __len__(self):
        return self.checkpoints


def install_hybrid(recurrent_cache=None, capped_pages=0, stranded_by_kv=0):
    pagetable = types.SimpleNamespace(
        max_pages=1000,
        referenced_pages={},
        metrics={
            "alloc_kv_only_pages": capped_pages,
            "stashes_stranded": stranded_by_kv,
        },
    )
    sync_generator = types.SimpleNamespace(
        pagetable=pagetable,
        max_total_tokens=1000 * PAGE_SIZE,
        cpu_page_cache=None,
        recurrent_cache=recurrent_cache,
    )
    model_module.container = types.SimpleNamespace(
        generator=types.SimpleNamespace(generator=sync_generator)
    )


class RecurrentCacheMetricsTests(unittest.TestCase):
    """The recurrent checkpoint cache, and its coupling to the KV cache.

    On a hybrid model, prompt reuse is capped at the longest prefix that has
    both valid K/V pages and a matching recurrent checkpoint. An undersized
    recurrent cache therefore silently defeats the KV cache, and with offloading
    enabled it wastes PCIe bandwidth doing so.
    """

    def setUp(self):
        self.metrics = MetricsManagerClass()
        self.original_container = model_module.container

    def tearDown(self):
        model_module.container = self.original_container

    def test_checkpoint_accounting(self):
        install_hybrid(FakeRecurrentCache())

        live = self.metrics._live_recurrent()

        self.assertEqual(live["checkpoints"], 60)
        self.assertEqual(live["checkpoint_bytes"], CHECKPOINT_BYTES)
        self.assertEqual(live["max_bytes"], 16 * 1024**3)
        self.assertAlmostEqual(live["usage_ratio"], (60 * CHECKPOINT_BYTES) / (16 * 1024**3))

    def test_an_empty_cache_reports_no_checkpoint_size(self):
        # checkpoint_bytes is a mean, so it has no value before the first store.
        install_hybrid(FakeRecurrentCache(checkpoints=0))

        live = self.metrics._live_recurrent()

        self.assertEqual(live["checkpoints"], 0)
        self.assertEqual(live["checkpoint_bytes"], 0)
        self.assertEqual(live["usage_ratio"], 0.0)

    def test_capped_pages_are_reported_as_tokens(self):
        # The waste metric: valid KV that was re-prefilled for lack of recurrent state.
        install_hybrid(FakeRecurrentCache(), capped_pages=120)

        self.assertEqual(self.metrics._live_recurrent()["capped_tokens"], 120 * PAGE_SIZE)

    def test_eviction_breakdown_separates_harmless_from_costly(self):
        # A stranded checkpoint could never have been resumed, so dropping it is
        # free. One dropped while its anchor page was still cached is the drop
        # that turns into capped tokens later.
        install_hybrid(FakeRecurrentCache(), stranded_by_kv=17)

        live = self.metrics._live_recurrent()

        self.assertEqual(live["evictions"], 40)
        self.assertEqual(live["stranded_evictions"], 25)
        self.assertEqual(live["live_kv_evictions"], 12)
        self.assertEqual(live["pruned"], 8)
        # The mirror case, counted by the page table: KV eviction stranding a
        # checkpoint rather than the other way round.
        self.assertEqual(live["stranded_by_kv"], 17)

    def test_capping_is_reported_without_a_recurrent_cache(self):
        # The page table counts the cap even if the cache object is unavailable;
        # losing the waste figure would hide the very failure it exists to show.
        install_hybrid(None, capped_pages=50)

        live = self.metrics._live_recurrent()
        self.assertEqual(live["capped_tokens"], 50 * PAGE_SIZE)
        self.assertEqual(live["checkpoints"], 0)

    def test_non_hybrid_model_reports_zeros(self):
        install_hybrid(None)

        live = self.metrics._live_recurrent()
        self.assertEqual(live["checkpoints"], 0)
        self.assertEqual(live["capped_tokens"], 0)

    def test_a_broken_cache_does_not_take_the_cap_with_it(self):
        # The cap is counted by the page table, not the cache, and it is the
        # figure these series exist for. A cache in an unexpected state must not
        # cost it.
        install_hybrid(None, capped_pages=50)
        model_module.container.generator.generator.recurrent_cache = types.SimpleNamespace(
            metrics={}
        )

        live = self.metrics._live_recurrent()

        self.assertEqual(live["capped_tokens"], 50 * PAGE_SIZE)
        self.assertEqual(live["checkpoints"], 0)

    def test_series_are_published_and_survive_a_broken_cache(self):
        install_hybrid(None)
        model_module.container.generator.generator.recurrent_cache = types.SimpleNamespace(
            metrics={}
        )

        rendered = self.metrics.render_prometheus()
        for name in (
            "tabbyapi:recurrent_cache_usage_ratio",
            "tabbyapi:recurrent_checkpoints",
            "tabbyapi:recurrent_cache_bytes",
            "tabbyapi:recurrent_cache_max_bytes",
            "tabbyapi:recurrent_checkpoint_bytes",
            "tabbyapi:recurrent_cache_evictions_total",
            "tabbyapi:recurrent_cache_stranded_evictions_total",
            "tabbyapi:recurrent_cache_live_kv_evictions_total",
            "tabbyapi:recurrent_cache_pruned_total",
            "tabbyapi:recurrent_stranded_by_kv_total",
            "tabbyapi:recurrent_capped_tokens_total",
        ):
            self.assertIn(f"{name} 0", rendered)

    def test_rendered_values_match_the_live_read(self):
        install_hybrid(FakeRecurrentCache(), capped_pages=120, stranded_by_kv=17)

        rendered = self.metrics.render_prometheus()

        self.assertIn("tabbyapi:recurrent_checkpoints 60", rendered)
        self.assertIn(f"tabbyapi:recurrent_capped_tokens_total {120 * PAGE_SIZE}", rendered)
        self.assertIn("tabbyapi:recurrent_cache_evictions_total 40", rendered)
        self.assertIn("tabbyapi:recurrent_cache_live_kv_evictions_total 12", rendered)
        self.assertIn("tabbyapi:recurrent_stranded_by_kv_total 17", rendered)


if __name__ == "__main__":
    unittest.main()
