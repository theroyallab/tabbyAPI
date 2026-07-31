import importlib.util
import pathlib
import time
import types
import unittest

from common import model as model_module
from common.metrics import MetricsManagerClass


def load_inspector():
    """The tool is a script rather than a package module, so it is loaded by path."""

    path = pathlib.Path(__file__).parent.parent / "tools" / "inspect_metrics.py"
    spec = importlib.util.spec_from_file_location("inspect_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspector = load_inspector()


def rendered(prefill: float, decode: float, requests: int = 20, uptime: float = 3600.0) -> str:
    metrics = MetricsManagerClass()
    metrics.process_start_time = time.time() - uptime

    pagetable = types.SimpleNamespace(
        max_pages=1000,
        referenced_pages={},
        metrics={"alloc_kv_only_pages": 0, "stashes_stranded": 0},
    )
    model_module.container = types.SimpleNamespace(
        generator=types.SimpleNamespace(
            generator=types.SimpleNamespace(
                pagetable=pagetable,
                max_total_tokens=256000,
                cpu_page_cache=None,
                recurrent_cache=None,
                num_active_jobs=lambda: 0,
                num_pending_jobs=lambda: 0,
            )
        )
    )
    for _ in range(requests):
        metrics.record_generation(20000, 2000, 300, prefill, decode, 0.05)

    scalars, hists = inspector.parse(metrics.render_prometheus())
    return inspector.render(
        scalars, hists, "http://test/metrics", inspector.C(False), show_hists=False, width=200
    )


def block(text: str) -> str:
    """Just the time breakdown, so an assertion cannot match a number from elsewhere."""

    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "Where the time goes" in line)
    end = next(
        (i for i in range(start + 1, len(lines)) if not lines[i].startswith("  ")), len(lines)
    )
    return "\n".join(lines[start:end])


class TimeBreakdownTests(unittest.TestCase):
    """The phase split, which is what decides whether to buy prefill or decode speed.

    A parallelism layout trades one against the other, so the number that
    matters is not how fast either phase is but how much of the server's time
    goes into each.
    """

    def setUp(self):
        self.original_container = model_module.container

    def tearDown(self):
        model_module.container = self.original_container

    def test_the_exchange_rate_is_the_ratio_of_the_shares(self):
        # One percent off the dominant phase pays for (p/d) percent onto the
        # other, which is the whole decision in one number and does not depend
        # on any assumed speedup factor. 93.0 / 7.0 = 13.3.
        text = block(rendered(prefill=8.0, decode=0.6))

        self.assertIn("93.0%", text)
        self.assertIn("1% off prefill pays for 13.3% onto decode", text)

    def test_the_rate_names_whichever_phase_dominates(self):
        text = block(rendered(prefill=0.4, decode=6.0))

        self.assertIn("1% off decode pays for", text)
        self.assertIn("onto prefill", text)

    def test_an_even_split_trades_one_for_one(self):
        text = block(rendered(prefill=3.0, decode=3.0))

        self.assertIn("1% off prefill pays for 1.0% onto decode", text)

    def test_break_even_is_exact_rather_than_marginal(self):
        # The rate above is a linear reading and overstates a large move, so the
        # break-even for a concrete one is computed exactly: after 2x prefill on
        # a 50/50 split, decode may fall to 0.5 / (1 - 0.25) = 66.7% of its speed.
        text = block(rendered(prefill=3.0, decode=3.0))

        self.assertIn("at 2x prefill", text)
        self.assertIn("break-even at 66.7% of decode speed", text)

    def test_a_phase_that_costs_nothing_is_named_as_such(self):
        text = block(rendered(prefill=3.0, decode=0.0))

        self.assertIn("prefill is all of it", text)

    def test_shares_sum_to_the_whole(self):
        text = block(rendered(prefill=3.0, decode=1.0))

        self.assertIn("75.0%", text)
        self.assertIn("25.0%", text)

    def test_utilization_is_against_wall_clock(self):
        # 20 requests of 4s each in an hour of uptime
        text = block(rendered(prefill=3.0, decode=1.0, requests=20, uptime=3600.0))

        self.assertIn("2.2% utilization", text)

    def test_concurrency_is_named_rather_than_reported_as_over_100_percent(self):
        # Summed request time can exceed the interval it ran in. That is
        # concurrency, not a bug, but a utilization over 100% reads as one.
        text = block(rendered(prefill=8.0, decode=8.0, requests=100, uptime=60.0))

        self.assertIn("overlapped", text)
        self.assertNotIn("utilization", text.replace("100.0% utilization", ""))

    def test_the_block_is_absent_before_any_request(self):
        # Nothing to divide, and a 0/0 split would be an invented number
        text = rendered(prefill=0.0, decode=0.0, requests=0)

        self.assertNotIn("Where the time goes", text)

    def test_queue_time_is_kept_out_of_the_split(self):
        # A request waiting is the engine working on another one, so counting
        # queue time as a phase would double-count it.
        text = block(rendered(prefill=3.0, decode=1.0))

        self.assertIn("not engine time", text)


if __name__ == "__main__":
    unittest.main()
