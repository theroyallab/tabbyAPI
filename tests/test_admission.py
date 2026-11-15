"""Tests for the exllamav3 lane-admission gate.

Stdlib unittest only — runs offline:
    python3 -m unittest tests.test_admission
"""

import asyncio
import logging
import unittest

from backends.exllamav3 import admission


# Reference config mirroring the plan's SPEC §2 example, with an extra
# catch-all so `default` and first-declared fallback can be told apart.
SPEC_CONFIG = {
    "classes": {
        "long": {"prompt_tokens_gte": 32768, "max_new_tokens_gte": None, "max_concurrent": 1},
        "mid": {"prompt_tokens_gte": 4096, "max_new_tokens_gte": 1024, "max_concurrent": 1},
        "short": {"max_concurrent": 2},
    },
    "default": "short",
    "hold_timeout": 1800,
}


class DisabledGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = admission

    def test_from_config_none_is_disabled(self):
        gate = self.mod.LaneGate.from_config(None, max_batch_size=None)
        self.assertFalse(gate.enabled)

    def test_default_logger_is_named_admission(self):
        # Ruling 2: logger=None means logging.getLogger("admission").
        gate = self.mod.LaneGate.from_config(None, max_batch_size=None)
        self.assertEqual(gate._logger.name, "admission")
        gate2 = self.mod.LaneGate.from_config(SPEC_CONFIG, max_batch_size=2)
        self.assertEqual(gate2._logger.name, "admission")

    def test_disabled_abort_all_held_waiting_are_noops(self):
        gate = self.mod.LaneGate.from_config({}, max_batch_size=1)
        gate.abort_all()
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})

    def test_from_config_empty_is_disabled(self):
        gate = self.mod.LaneGate.from_config({}, max_batch_size=None)
        self.assertFalse(gate.enabled)

    async def test_disabled_acquire_never_awaits_and_keeps_no_state(self):
        gate = self.mod.LaneGate.from_config(None, max_batch_size=1)

        # Swap the module's asyncio namespace for one that raises on ANY
        # attribute access: if the disabled `async with` completes, the gate
        # provably never touched the event loop (binding: "no await" when
        # unconfigured — the zero-behavior-change / mergeability property).
        class _ForbiddenAsyncio:
            def __getattr__(self, name):
                raise AssertionError(f"disabled gate must not use asyncio.{name}")

        real_asyncio = self.mod.asyncio
        self.mod.asyncio = _ForbiddenAsyncio()
        try:
            async with gate.acquire("long", "req-1"):
                pass
            cls = gate.classify(100_000, None)
        finally:
            self.mod.asyncio = real_asyncio
        self.assertIsInstance(cls, str)
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})
        # abort_all on a disabled/idle gate must be a harmless no-op.
        gate.abort_all()


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.mod = admission

    def test_enabled_for_real_config(self):
        gate = self.mod.LaneGate.from_config(SPEC_CONFIG, max_batch_size=2)
        self.assertTrue(gate.enabled)

    def test_boundary_inclusive(self):
        gate = self.mod.LaneGate.from_config(SPEC_CONFIG, max_batch_size=2)
        self.assertEqual(gate.classify(32768, None), "long")
        self.assertEqual(gate.classify(32767, None), "short")  # via default
        self.assertEqual(gate.classify(100_000, None), "long")

    def test_declaration_order_preserved_first_match_wins(self):
        # Alphabetical sorting would put "zzz" first and swallow everything;
        # insertion order must win instead.
        cfg = {
            "classes": {
                "first": {"prompt_tokens_gte": 32768},
                "zzz": {"max_concurrent": 1},
            },
        }
        gate = self.mod.LaneGate.from_config(cfg, max_batch_size=2)
        self.assertEqual(gate.classify(40000, None), "first")

    def test_default_fallback(self):
        gate = self.mod.LaneGate.from_config(SPEC_CONFIG, max_batch_size=2)
        self.assertEqual(gate.classify(10, None), "short")

    def test_default_absent_falls_back_to_first_declared_class(self):
        cfg = {
            "classes": {
                "long": {"prompt_tokens_gte": 32768},
                "short": {"max_concurrent": 2},
            },
            # deliberately no "default" key
        }
        gate = self.mod.LaneGate.from_config(cfg, max_batch_size=2)
        self.assertEqual(gate.classify(10, None), "long")

    def test_both_predicates_are_anded(self):
        gate = self.mod.LaneGate.from_config(SPEC_CONFIG, max_batch_size=2)
        # mid needs BOTH prompt>=4096 AND max_new>=1024
        self.assertEqual(gate.classify(4096, 1024), "mid")
        self.assertEqual(gate.classify(4096, 1023), "short")  # tokens too small
        self.assertEqual(gate.classify(4095, 1024), "short")  # prompt too small

    def test_null_predicate_is_absent_predicate(self):
        # long declares max_new_tokens_gte: null -> that predicate is absent,
        # so a tiny max_new_tokens must not stop a 40k prompt from classing long.
        gate = self.mod.LaneGate.from_config(SPEC_CONFIG, max_batch_size=2)
        self.assertEqual(gate.classify(40000, 1), "long")

    def test_none_max_new_tokens_never_matches_class_declaring_it(self):
        cfg = {
            "classes": {
                "capped": {"max_new_tokens_gte": 256, "max_concurrent": 1},
                "other": {"max_concurrent": 1},
            },
            "default": "other",
        }
        gate = self.mod.LaneGate.from_config(cfg, max_batch_size=2)
        self.assertEqual(gate.classify(10, None), "other")
        self.assertEqual(gate.classify(10, 255), "other")
        self.assertEqual(gate.classify(10, 256), "capped")

    def test_predicateless_class_matches_nothing(self):
        # "bare" declares no predicates: it must NOT act as a catch-all, so
        # a small request goes to `default` (mid), not to "bare".
        cfg = {
            "classes": {
                "long": {"prompt_tokens_gte": 32768, "max_concurrent": 1},
                "bare": {"max_concurrent": 2},
                "mid": {"prompt_tokens_gte": 64, "max_concurrent": 1},
            },
            "default": "mid",
        }
        gate = self.mod.LaneGate.from_config(cfg, max_batch_size=2)
        self.assertEqual(gate.classify(100, None), "mid")  # matched mid, not bare
        self.assertEqual(gate.classify(10, None), "mid")  # via default, not bare
        self.assertEqual(gate.classify(40000, None), "long")

    def test_max_concurrent_defaults_to_one(self):
        cfg = {"classes": {"solo": {"max_concurrent": 1}, "lone": {}}}
        gate = self.mod.LaneGate.from_config(cfg, max_batch_size=2)
        # "lone" omitted max_concurrent -> 1; observable via semaphores in
        # GateTests; here just ensure classify never raises on it.
        self.assertEqual(gate.classify(10, None), "solo")  # default = first class


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def text(self):
        return "\n".join(r.getMessage() for r in self.records)


def make_gate(mod, cfg, logger, max_batch_size=None):
    return mod.LaneGate.from_config(cfg, max_batch_size=max_batch_size, logger=logger)


def holding(gate, cls, rid, held_evt, release_evt):
    """Task body: acquire, signal `held_evt`, block until `release_evt`."""

    async def body():
        async with gate.acquire(cls, rid):
            held_evt.set()
            await release_evt.wait()

    return asyncio.get_running_loop().create_task(body())


async def _until(predicate, poll=0.005):
    """Poll `predicate` until true; wrap in wait_for so a regression FAILS."""
    while not predicate():
        await asyncio.sleep(poll)


class _GrantBoomLogger:
    """Injected logger stand-in that raises on the grant line of chosen requests.

    The gate never configures logging: the caller injects its own logger
    (Task 3 injects tabbyAPI's), so the grant line is the first place a
    foreign logger can blow up — a custom logger object, a formatter error,
    an ``extra`` key collision. ``boom_ids`` names which requests fail, so a
    test can hold a lane normally and fail only the grant under examination.
    The surface mirrors exactly what this module calls (``info`` /
    ``warning`` / ``debug``); nothing of the module under test is patched or
    mocked.
    """

    BOOM_ON = "admission: granted"

    def __init__(self, boom_ids=("*",)):
        self.boom_ids = set(boom_ids)
        self.messages = []

    @staticmethod
    def _text(msg, args):
        # Same lazy-format contract logging.Logger.info() has — the module
        # under test always passes %-style args, never a pre-formatted string.
        return msg % args if args else str(msg)

    def info(self, msg, *args):
        text = self._text(msg, args)
        self.messages.append(text)
        if not text.startswith(self.BOOM_ON):
            return
        if "*" in self.boom_ids or any(f"request_id={rid} class=" in text for rid in self.boom_ids):
            raise RuntimeError(f"injected logger failed on the grant line: {text}")

    def warning(self, msg, *args):
        self.messages.append(self._text(msg, args))

    def debug(self, msg, *args):
        self.messages.append(self._text(msg, args))


async def _acquire_once(gate, cls, rid):
    """One full hold that exits immediately — propagates any __aenter__ error."""
    async with gate.acquire(cls, rid):
        pass


class GateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = admission
        self.log = _Capture()
        self.logger = logging.getLogger("test.admission.gate")
        self.logger.handlers = [self.log]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    async def test_same_class_semaphore_of_one_blocks_second(self):
        cfg = {"classes": {"long": {"max_concurrent": 1}}, "default": "long"}
        gate = make_gate(self.mod, cfg, self.logger)
        h1, r1, h2, r2 = (asyncio.Event() for _ in range(4))
        t1 = holding(gate, "long", "req-1", h1, r1)
        await h1.wait()
        self.assertEqual(gate.held(), {"long": 1})
        t2 = holding(gate, "long", "req-2", h2, r2)
        await asyncio.sleep(0.01)
        self.assertFalse(h2.is_set(), "second long request must be held back")
        self.assertEqual(gate.waiting(), {"long": 1})
        self.assertEqual(gate.held(), {"long": 1})
        r1.set()
        await t1
        await asyncio.wait_for(h2.wait(), timeout=1)
        self.assertEqual(gate.held(), {"long": 1})  # slot moved, not duplicated
        r2.set()
        await t2
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})

    async def test_short_class_never_blocked_by_long(self):
        cfg = {
            "classes": {
                "long": {"prompt_tokens_gte": 32768, "max_concurrent": 1},
                "short": {"max_concurrent": 2},
            },
            "default": "short",
        }
        gate = make_gate(self.mod, cfg, self.logger, max_batch_size=2)
        h1, r1, h2, r2 = (asyncio.Event() for _ in range(4))
        t1 = holding(gate, "long", "req-1", h1, r1)
        await h1.wait()
        # The 2k query classes short and must be admitted immediately while
        # the 100k agentic session holds the long lane.
        self.assertEqual(gate.classify(2000, None), "short")
        t2 = holding(gate, "short", "req-2", h2, r2)
        await asyncio.wait_for(h2.wait(), timeout=1)
        self.assertEqual(gate.held(), {"long": 1, "short": 1})
        r1.set()
        r2.set()
        await asyncio.gather(t1, t2)
        self.assertEqual(gate.held(), {})

    async def test_two_shorts_admit_third_blocks(self):
        cfg = {
            "classes": {
                "long": {"prompt_tokens_gte": 32768, "max_concurrent": 1},
                "short": {"max_concurrent": 2},
            },
            "default": "short",
        }
        gate = make_gate(self.mod, cfg, self.logger, max_batch_size=3)
        evts = [asyncio.Event() for _ in range(6)]
        h1, r1, h2, r2, h3, r3 = evts
        t1 = holding(gate, "short", "a", h1, r1)
        await h1.wait()
        t2 = holding(gate, "short", "b", h2, r2)
        await h2.wait()
        self.assertEqual(gate.held(), {"short": 2})
        t3 = holding(gate, "short", "c", h3, r3)
        await asyncio.sleep(0.01)
        self.assertFalse(h3.is_set())
        self.assertEqual(gate.waiting(), {"short": 1})
        r1.set()
        await t1
        await asyncio.wait_for(h3.wait(), timeout=1)
        r2.set()
        r3.set()
        await asyncio.gather(t2, t3)
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})

    async def test_fifo_order_within_class(self):
        cfg = {"classes": {"solo": {"max_concurrent": 1}}, "default": "solo"}
        gate = make_gate(self.mod, cfg, self.logger)
        order = []
        release = asyncio.Event()
        first_done = asyncio.Event()

        async def holder(rid):
            async with gate.acquire("solo", rid):
                order.append(rid)
                if rid == "first":
                    first_done.set()
                await release.wait()

        t0 = asyncio.get_running_loop().create_task(holder("first"))
        await first_done.wait()
        tails = [asyncio.get_running_loop().create_task(holder(f"w{i}")) for i in range(3)]
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 3})
        release.set()
        await asyncio.gather(t0, *tails)
        self.assertEqual(order, ["first", "w0", "w1", "w2"])

    async def test_grant_and_hold_log_lines(self):
        cfg = {"classes": {"solo": {"max_concurrent": 1}}, "default": "solo"}
        gate = make_gate(self.mod, cfg, self.logger)
        async with gate.acquire("solo", "req-42", prompt_tokens=1234):
            pass
        text = self.log.text()
        self.assertIn("req-42", text)
        self.assertIn("solo", text)
        self.assertIn("1234", text)
        self.assertIn("waited", text)


class HoldTimeoutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = admission
        self.log = _Capture()
        self.logger = logging.getLogger("test.admission.timeout")
        self.logger.handlers = [self.log]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    async def test_timeout_proceeds_with_warning_never_raises(self):
        cfg = {
            "classes": {"solo": {"max_concurrent": 1}},
            "default": "solo",
            "hold_timeout": 0.05,
        }
        gate = make_gate(self.mod, cfg, self.logger)
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        h2, r2 = asyncio.Event(), asyncio.Event()
        t2 = holding(gate, "solo", "waiter", h2, r2)
        # Fail-open: the waiter proceeds on its own after the timeout.
        await asyncio.wait_for(h2.wait(), timeout=2)
        text = self.log.text()
        self.assertIn("waiter", text)
        self.assertIn("fail-open", text)
        self.assertEqual(gate.held(), {"solo": 2})  # holder + fail-open waiter
        r1.set()
        r2.set()
        await asyncio.gather(t1, t2)
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})

    async def test_timeout_expiry_does_not_consume_slot_for_exits(self):
        # After a fail-open expiry, releasing the real holder must let a
        # NEW request in normally (no stranded permit).
        cfg = {
            "classes": {"solo": {"max_concurrent": 1}},
            "default": "solo",
            "hold_timeout": 0.05,
        }
        gate = make_gate(self.mod, cfg, self.logger)
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        h2, r2 = asyncio.Event(), asyncio.Event()
        t2 = holding(gate, "solo", "expired", h2, r2)
        # Bounded: a fail-open regression must FAIL this test, not hang CI.
        await asyncio.wait_for(h2.wait(), timeout=2)
        r1.set()
        await t1
        h3, r3 = asyncio.Event(), asyncio.Event()
        t3 = holding(gate, "solo", "fresh", h3, r3)
        await asyncio.wait_for(h3.wait(), timeout=2)  # admitted by permit, not timeout
        r2.set()
        r3.set()
        await asyncio.gather(t2, t3)
        self.assertEqual(gate.waiting(), {})

    async def test_zero_hold_timeout_waits_forever(self):
        cfg = {
            "classes": {"solo": {"max_concurrent": 1}},
            "default": "solo",
            "hold_timeout": 0,
        }
        gate = make_gate(self.mod, cfg, self.logger)
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        h2, r2 = asyncio.Event(), asyncio.Event()
        t2 = holding(gate, "solo", "patient", h2, r2)
        await asyncio.sleep(0.1)
        self.assertFalse(h2.is_set(), "hold_timeout 0 must not fail open")
        fail_opens = [
            r
            for r in self.log.records
            if "fail-open" in r.getMessage() and "patient" in r.getMessage()
        ]
        self.assertEqual(fail_opens, [], "hold_timeout 0 must arm no fail-open timer")
        self.assertEqual(gate.waiting(), {"solo": 1})
        r1.set()
        await asyncio.wait_for(h2.wait(), timeout=2)  # eventually granted
        r2.set()
        await asyncio.gather(t1, t2)
        self.assertEqual(gate.held(), {})

    async def test_default_hold_timeout_is_1800(self):
        gate = make_gate(self.mod, {"classes": {"s": {}}}, self.logger)
        self.assertEqual(gate._hold_timeout, 1800.0)
        gate2 = make_gate(self.mod, {"classes": {"s": {}}, "hold_timeout": None}, self.logger)
        self.assertEqual(gate2._hold_timeout, 1800.0)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = admission
        self.log = _Capture()
        self.logger = logging.getLogger("test.admission.cancel")
        self.logger.handlers = [self.log]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def _gate(self, cfg=None):
        cfg = cfg or {"classes": {"solo": {"max_concurrent": 1}}, "default": "solo"}
        return make_gate(self.mod, cfg, self.logger)

    async def test_cancel_while_held_frees_slot(self):
        gate = self._gate()
        body_entered = asyncio.Event()

        async def leaky():
            async with gate.acquire("solo", "victim"):
                body_entered.set()
                await asyncio.sleep(10)  # cancelled here, mid-hold

        t = asyncio.get_running_loop().create_task(leaky())
        await body_entered.wait()
        self.assertEqual(gate.held(), {"solo": 1})
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        self.assertEqual(gate.held(), {}, "cancellation must release the slot")
        self.assertEqual(gate.waiting(), {})
        # The slot is really free: a new request is admitted without waiting.
        h, r = asyncio.Event(), asyncio.Event()
        t2 = holding(gate, "solo", "next", h, r)
        await asyncio.wait_for(h.wait(), timeout=1)
        r.set()
        await t2

    async def test_cancel_while_body_raises_frees_slot(self):
        gate = self._gate()

        async def raiser():
            async with gate.acquire("solo", "boom"):
                raise RuntimeError("handler blew up")

        with self.assertRaises(RuntimeError):
            await raiser()
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})

    async def test_cancel_while_waiting_leaves_no_trace_and_no_leak(self):
        gate = self._gate()
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "blocker", h1, r1)
        await h1.wait()

        async def queued():
            async with gate.acquire("solo", "queued"):
                pass  # never reached

        t2 = asyncio.get_running_loop().create_task(queued())
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 1})
        t2.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t2
        self.assertEqual(gate.waiting(), {}, "cancelled waiter must leave waiting() at 0")
        self.assertEqual(gate.held(), {"solo": 1})  # only the blocker remains
        # No slot was consumed by the cancelled waiter: when the blocker
        # exits, the next arrival is admitted immediately.
        r1.set()
        await t1
        self.assertEqual(gate.held(), {})
        h3, r3 = asyncio.Event(), asyncio.Event()
        t3 = holding(gate, "solo", "after", h3, r3)
        await asyncio.wait_for(h3.wait(), timeout=1)
        r3.set()
        await t3
        self.assertEqual(gate.held(), {})

    async def test_cancel_waiting_with_two_waiters_preserves_queue(self):
        gate = self._gate()
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "blocker", h1, r1)
        await h1.wait()
        order = []
        done = asyncio.Event()

        async def queued(rid):
            async with gate.acquire("solo", rid):
                order.append(rid)
                done.set()
                await asyncio.sleep(10)

        tm = asyncio.get_running_loop().create_task(queued("middle"))
        tl = asyncio.get_running_loop().create_task(queued("last"))
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 2})
        tm.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await tm
        self.assertEqual(gate.waiting(), {"solo": 1})
        r1.set()
        await asyncio.wait_for(done.wait(), timeout=1)
        self.assertEqual(order, ["last"], "queue must survive a middle cancellation")
        tl.cancel()
        await asyncio.gather(tl, return_exceptions=True)
        await t1
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})

    async def test_cancel_after_grant_hands_permit_to_queue(self):
        # The grant LANDED (fut resolved True) but the waiter was cancelled in
        # the one-tick window before it could resume. That permit is in transit
        # and must go back *through the FIFO*: releasing it straight into the
        # Semaphore strands the next waiter, because only _pass_permit()/
        # abort_all() drain the queue — the lane would wedge with a full
        # semaphore and a queued request nobody ever wakes.
        gate = self._gate(
            {
                "classes": {"solo": {"max_concurrent": 1}},
                "default": "solo",
                "hold_timeout": 0,  # wait forever: the wedge must be permanent
            }
        )
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        woke = set()

        async def queued(rid):
            async with gate.acquire("solo", rid):
                woke.add(rid)
                await asyncio.sleep(10)

        t2 = asyncio.get_running_loop().create_task(queued("granted"))
        t3 = asyncio.get_running_loop().create_task(queued("next"))
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 2})
        fut_granted, fut_next = list(gate._queues["solo"])  # FIFO order
        r1.set()  # holder exits -> permit handed to t2
        await asyncio.sleep(0)  # t1's step ran; the grant is in flight to t2
        self.assertTrue(
            fut_granted.done() and fut_granted.result() is True,
            "the grant must have landed on t2's future",
        )
        self.assertFalse(t2.done(), "grant must still be in transit to t2")
        self.assertEqual(len(gate._queues["solo"]), 1, "t2 dequeued, t3 still queued")
        self.assertFalse(fut_next.done(), "t3 must still be pending at cancel time")
        t2.cancel()  # cancel inside the in-transit window
        with self.assertRaises(asyncio.CancelledError):
            await t2
        self.assertNotIn("granted", woke, "cancelled waiter must not reach its body")
        # The permit must arrive at t3 via the queue, not idle in the semaphore.
        await asyncio.wait_for(_until(lambda: "next" in woke), timeout=1)
        self.assertEqual(gate.held(), {"solo": 1})
        self.assertEqual(gate.waiting(), {})
        self.assertLessEqual(gate._sems["solo"]._value, 1, "semaphore must never exceed its limit")
        self.assertEqual(gate._sems["solo"]._value, 0, "t3 consumed the permit")
        t3.cancel()
        await asyncio.gather(t1, t3, return_exceptions=True)
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)

    async def test_cancel_after_abort_all_wake_releases_no_permit(self):
        # abort_all() wakes a waiter with result False: admitted, but owning NO
        # permit. Cancelling that waiter in the one-tick window before it
        # resumes must therefore release nothing — else the lane silently stops
        # limiting (a permit that was never owned enters the Semaphore).
        gate = self._gate()
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        woke2 = asyncio.Event()

        async def queued():
            async with gate.acquire("solo", "aborted"):
                woke2.set()
                await asyncio.sleep(10)

        t2 = asyncio.get_running_loop().create_task(queued())
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 1})
        gate.abort_all()  # fail-open wake (result False), owns no permit
        t2.cancel()  # ... before the wake can be delivered
        with self.assertRaises(asyncio.CancelledError):
            await t2
        self.assertFalse(woke2.is_set())
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate._sems["solo"]._value, 0, "permit-free wake must release nothing")
        # The lane still limits: a third request must NOT be admitted while the
        # original holder is still inside its body.
        h3, r3 = asyncio.Event(), asyncio.Event()
        t3 = holding(gate, "solo", "third", h3, r3)
        await asyncio.sleep(0.05)
        self.assertFalse(h3.is_set(), "over-release admitted a second holder")
        self.assertEqual(gate.held(), {"solo": 1})
        r1.set()
        await t1
        await asyncio.wait_for(h3.wait(), timeout=1)  # admitted normally now
        self.assertEqual(gate.held(), {"solo": 1})
        r3.set()
        await t3
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)


class AbortEventTests(unittest.IsolatedAsyncioTestCase):
    """SPEC §3/§5 — ``abort_event`` is the queued-path twin of poll()."""

    def setUp(self):
        self.mod = admission
        self.log = _Capture()
        self.logger = logging.getLogger("test.admission.abort_event")
        self.logger.handlers = [self.log]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def _gate(self, cfg=None):
        cfg = cfg or {"classes": {"solo": {"max_concurrent": 1}}, "default": "solo"}
        return make_gate(self.mod, cfg, self.logger)

    async def test_abort_set_on_entry_raises_without_acquiring(self):
        gate = self._gate()
        ev = asyncio.Event()
        ev.set()  # client already gone when the request reached the gate
        with self.assertRaises(asyncio.CancelledError):
            async with gate.acquire("solo", "gone", abort_event=ev):
                self.fail("body must not run for a vanished client")
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate._sems["solo"]._value, 1, "no permit may be taken")

    async def test_abort_while_queued_abandons_hold_without_permit(self):
        gate = self._gate()
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        ev = asyncio.Event()
        entered = asyncio.Event()

        async def queued():
            async with gate.acquire("solo", "queued", abort_event=ev):
                entered.set()
                await asyncio.sleep(10)

        t2 = asyncio.get_running_loop().create_task(queued())
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 1})
        ev.set()  # client disconnects while queued
        with self.assertRaises(asyncio.CancelledError):
            await t2
        self.assertFalse(entered.is_set(), "abandoned hold must not reach its body")
        self.assertEqual(gate.waiting(), {}, "aborted waiter must leave waiting() at 0")
        self.assertEqual(gate.held(), {"solo": 1})  # only the real holder
        self.assertEqual(gate._sems["solo"]._value, 0, "abandoning must take/release no permit")
        # The lane still limits: a new request queues behind the holder.
        h3, r3 = asyncio.Event(), asyncio.Event()
        t3 = holding(gate, "solo", "third", h3, r3)
        await asyncio.sleep(0.05)
        self.assertFalse(h3.is_set())
        self.assertEqual(gate.waiting(), {"solo": 1})
        r1.set()
        await t1
        await asyncio.wait_for(h3.wait(), timeout=1)  # admitted on the holder's exit
        r3.set()
        await t3
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)

    async def test_abort_after_grant_before_body_returns_permit_to_queue(self):
        # Grant and disconnect land in the same tick: the abort wins, and the
        # in-transit permit must go back through the FIFO — the in-transit bug
        # class of the cancel-after-grant case, reached through abort_event.
        gate = self._gate(
            {
                "classes": {"solo": {"max_concurrent": 1}},
                "default": "solo",
                "hold_timeout": 0,
            }
        )
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        ev = asyncio.Event()
        woke = set()

        async def queued(rid, abort=None):
            async with gate.acquire("solo", rid, abort_event=abort):
                woke.add(rid)
                await asyncio.sleep(10)

        t2 = asyncio.get_running_loop().create_task(queued("granted", ev))
        t3 = asyncio.get_running_loop().create_task(queued("next"))
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 2})
        r1.set()  # holder exits -> permit handed to t2
        ev.set()  # client vanishes before t2 resumes
        with self.assertRaises(asyncio.CancelledError):
            await t2
        self.assertNotIn("granted", woke)
        await asyncio.wait_for(_until(lambda: "next" in woke), timeout=1)
        self.assertEqual(gate.held(), {"solo": 1})
        self.assertEqual(gate.waiting(), {})
        self.assertLessEqual(gate._sems["solo"]._value, 1)
        self.assertEqual(gate._sems["solo"]._value, 0, "t3 consumed the permit")
        t3.cancel()
        await asyncio.gather(t1, t3, return_exceptions=True)
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)

    async def test_none_abort_event_and_positional_call_keep_today_behaviour(self):
        gate = self._gate()
        async with gate.acquire("solo", "plain"):  # SPEC §3 required call shape
            self.assertEqual(gate.held(), {"solo": 1})
        async with gate.acquire("solo", "none", prompt_tokens=7, abort_event=None):
            self.assertEqual(gate.held(), {"solo": 1})
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)

    async def test_task_cancel_while_racing_abort_event_uses_the_same_giveup(self):
        # Fix 1 + Fix 2 together: the task is *cancelled* (not aborted) while it
        # sits in the grant/abort race with the event attached and an in-transit
        # grant. The give-up path must be identical — permit back through the
        # FIFO, and no leaked disconnect-waiter task.
        gate = self._gate(
            {
                "classes": {"solo": {"max_concurrent": 1}},
                "default": "solo",
                "hold_timeout": 0,
            }
        )
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        woke = set()
        ev = asyncio.Event()  # never set: the client is cancelled, not aborted

        async def queued(rid, abort=None):
            async with gate.acquire("solo", rid, abort_event=abort):
                woke.add(rid)
                await asyncio.sleep(10)

        t2 = asyncio.get_running_loop().create_task(queued("granted", ev))
        t3 = asyncio.get_running_loop().create_task(queued("next"))
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 2})
        r1.set()
        await asyncio.sleep(0)
        self.assertFalse(t2.done(), "grant must still be in transit to t2")
        t2.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t2
        await asyncio.wait_for(_until(lambda: "next" in woke), timeout=1)
        self.assertNotIn("granted", woke)
        self.assertEqual(gate.held(), {"solo": 1})
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate._sems["solo"]._value, 0)
        self.assertFalse(
            getattr(ev, "_waiters", []),
            "cancelled holder must leave no waiter registered on the caller's event",
        )
        t3.cancel()
        await asyncio.gather(t1, t3, return_exceptions=True)
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)

    async def test_abort_event_still_fails_open_on_hold_timeout(self):
        # SPEC §4: an attached disconnect signal must not turn the fail-open
        # timer into a rejection path — expiry still proceeds, permit-free.
        gate = self._gate(
            {
                "classes": {"solo": {"max_concurrent": 1}},
                "default": "solo",
                "hold_timeout": 0.05,
            }
        )
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        ev = asyncio.Event()  # never set
        entered, leave = asyncio.Event(), asyncio.Event()

        async def queued():
            async with gate.acquire("solo", "patient", abort_event=ev):
                entered.set()
                await leave.wait()

        t2 = asyncio.get_running_loop().create_task(queued())
        await asyncio.wait_for(entered.wait(), timeout=2)
        text = self.log.text()
        self.assertIn("patient", text)
        self.assertIn("fail-open", text)
        self.assertEqual(gate.held(), {"solo": 2})  # holder + permit-free waiter
        r1.set()
        await t1
        self.assertEqual(gate._sems["solo"]._value, 1, "fail-open exit releases nothing")
        leave.set()
        await t2
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)

    async def test_disabled_gate_ignores_abort_event_entirely(self):
        # SPEC §8: unconfigured means provably no behaviour change — the
        # no-op CM neither inspects nor awaits the event (a vanished client is
        # still caught by DisconnectHandler.poll() mid-generation).
        gate = self.mod.LaneGate.from_config(None, max_batch_size=None, logger=self.logger)
        ev = asyncio.Event()
        ev.set()
        async with gate.acquire("solo", "req", abort_event=ev):
            pass
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})


class AbortAllTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = admission
        self.log = _Capture()
        self.logger = logging.getLogger("test.admission.abort")
        self.logger.handlers = [self.log]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def _gate(self, n=1, timeout=30):
        cfg = {
            "classes": {
                "long": {"prompt_tokens_gte": 32768, "max_concurrent": n},
                "short": {"max_concurrent": 2},
            },
            "default": "short",
            "hold_timeout": timeout,
        }
        return make_gate(self.mod, cfg, self.logger, max_batch_size=2)

    async def test_abort_all_noop_when_idle(self):
        gate = self._gate()
        gate.abort_all()  # must not raise
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})

    async def test_abort_all_releases_waiters_in_fifo_order(self):
        gate = self._gate()
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "long", "blocker", h1, r1)
        await h1.wait()
        woke = []
        release = asyncio.Event()

        async def waiter(rid):
            async with gate.acquire("long", rid):
                woke.append(rid)
                await release.wait()

        waiters = []
        for i in range(3):
            waiters.append(asyncio.get_running_loop().create_task(waiter(f"w{i}")))
            await asyncio.sleep(0.005)  # establish strict FIFO queue order
        self.assertEqual(gate.waiting(), {"long": 3})
        gate.abort_all()
        # Observe the wake directly (the waiters then park on `release`, so
        # awaiting their tasks here would deadlock the test itself).
        for i in range(3):
            for _ in range(200):
                if len(woke) > i:
                    break
                await asyncio.sleep(0.005)
            self.assertGreater(len(woke), i, f"waiter w{i} must be released")
        self.assertEqual(woke, ["w0", "w1", "w2"], "FIFO release order")
        self.assertEqual(gate.waiting(), {})
        release.set()
        r1.set()
        await asyncio.gather(t1, *waiters)
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate.held(), {})
        # Every wake rode a real or handed-down permit — never an over-release:
        # the semaphore must end exactly at its limit.
        self.assertEqual(gate._sems["long"]._value, 1)

    async def test_abort_all_only_affects_waiters_not_holders(self):
        gate = self._gate()
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "long", "blocker", h1, r1)
        await h1.wait()
        gate.abort_all()  # nothing waiting -> harmless
        self.assertEqual(gate.held(), {"long": 1})
        self.assertFalse(t1.done())
        r1.set()
        await t1

    async def test_abort_all_does_not_exceed_granted_permits(self):
        # Two long slots, two holders, one waiter: abort wakes exactly the
        # waiter; the semaphore count never exceeds its limit at any point.
        gate = self._gate(n=2)
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "long", "h1", h1, r1)
        await h1.wait()
        h2, r2 = asyncio.Event(), asyncio.Event()
        t2 = holding(gate, "long", "h2", h2, r2)
        await h2.wait()
        self.assertEqual(gate.waiting(), {})  # fast path, no queue
        release = asyncio.Event()

        async def waiter(rid):
            async with gate.acquire("long", rid):
                await release.wait()

        w = asyncio.get_running_loop().create_task(waiter("w"))
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"long": 1})
        self.assertLessEqual(gate._sems["long"]._value, 2)
        release.set()  # the waiter parks on this once woken
        gate.abort_all()
        await asyncio.wait_for(w, timeout=1)  # woken through the release, not rejected
        self.assertEqual(gate._sems["long"]._value, 0)  # never over-released
        r1.set()
        r2.set()
        await asyncio.gather(t1, t2)
        self.assertEqual(gate._sems["long"]._value, 2)  # back to limit exactly


class FromConfigValidationTests(unittest.TestCase):
    """Config-load-time errors: ValueError naming the offending key."""

    def setUp(self):
        self.mod = admission

    def _expect(self, raw, key_fragment):
        with self.assertRaises(ValueError) as cm:
            self.mod.LaneGate.from_config(raw, max_batch_size=None)
        msg = str(cm.exception)
        self.assertIn(key_fragment, msg, f"message must name {key_fragment!r}: {msg}")

    def test_non_mapping_raw(self):
        self._expect(["not", "a", "mapping"], "admission")
        self._expect(42, "admission")

    def test_non_mapping_classes(self):
        self._expect({"classes": ["long", "short"]}, "classes")

    def test_empty_classes(self):
        self._expect({"classes": {}}, "classes")
        self._expect({"classes": None}, "classes")

    def test_negative_max_concurrent(self):
        self._expect(
            {"classes": {"long": {"max_concurrent": -1}}},
            "classes.long.max_concurrent",
        )

    def test_zero_max_concurrent(self):
        self._expect(
            {"classes": {"long": {"max_concurrent": 0}}},
            "classes.long.max_concurrent",
        )

    def test_non_int_max_concurrent(self):
        self._expect(
            {"classes": {"long": {"max_concurrent": "two"}}},
            "classes.long.max_concurrent",
        )
        self._expect(
            {"classes": {"long": {"max_concurrent": 1.5}}},
            "classes.long.max_concurrent",
        )

    def test_unknown_default(self):
        self._expect(
            {"classes": {"long": {}}, "default": "nope"},
            "default",
        )

    def test_negative_hold_timeout(self):
        self._expect({"classes": {"long": {}}, "hold_timeout": -5}, "hold_timeout")

    def test_non_mapping_class_body(self):
        self._expect({"classes": {"long": 7}}, "classes.long")

    def test_negative_predicate(self):
        self._expect(
            {"classes": {"long": {"prompt_tokens_gte": -1}}},
            "classes.long.prompt_tokens_gte",
        )

    def test_unknown_key_names_the_key(self):
        self._expect({"classes": {"long": {}}, "typo_key": 1}, "typo_key")
        self._expect({"classes": {"long": {"hold_timout": 3}}}, "hold_timout")


class RequestTimeResilienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = admission
        self.log = _Capture()
        self.logger = logging.getLogger("test.admission.resilient")
        self.logger.handlers = [self.log]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def _gate(self):
        cfg = {
            "classes": {
                "long": {"prompt_tokens_gte": 32768, "max_concurrent": 1},
                "short": {"max_concurrent": 2},
            },
            "default": "short",
        }
        return make_gate(self.mod, cfg, self.logger, max_batch_size=2)

    async def test_negative_prompt_tokens_classes_default_no_raise(self):
        gate = self._gate()
        self.assertEqual(gate.classify(-5, None), "short")
        self.assertEqual(gate.classify(0, -7), "short")

    async def test_garbage_request_values_never_raise(self):
        gate = self._gate()
        self.assertEqual(gate.classify("lots", None), "short")
        self.assertEqual(gate.classify(None, "many"), "short")
        self.assertEqual(gate.classify(float("nan"), None), "short")  # NaN < x is False...
        # ...but a NaN must never match the long lane:
        self.assertNotEqual(gate.classify(float("nan"), None), "long")

    async def test_acquire_undeclared_class_falls_to_default_no_raise(self):
        gate = self._gate()
        async with gate.acquire("nonexistent", "req-x"):
            self.assertEqual(gate.held(), {"short": 1})
        self.assertEqual(gate.held(), {})
        self.assertIn("nonexistent", self.log.text())

    async def test_hold_and_release_balances_under_mixed_exits(self):
        gate = self._gate()
        for i in range(6):
            async with gate.acquire("short", f"r{i}", prompt_tokens=10 * i):
                pass
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate._sems["short"]._value, 2)


class EntryFailureTests(unittest.IsolatedAsyncioTestCase):
    """``__aenter__`` must be *failure-safe*: everything it already owns is
    handed back when it raises, because Python never calls ``__aexit__`` for a
    failed ``async with`` entry. Without that, a permit spent by a grant is
    stranded for the life of the process — on a ``max_concurrent: 1`` lane, one
    occurrence disables the lane forever.

    Trigger: the injected logger raising on the grant line (see
    ``_GrantBoomLogger``). Entry is driven both on the pre-grant fast path and
    through the FIFO, because the two paths reach the grant line from
    different places (outside the try / inside it).
    """

    def setUp(self):
        self.mod = admission

    @staticmethod
    def _cfg(cls="solo", n=1):
        return {"classes": {cls: {"max_concurrent": n}}, "default": cls}

    async def test_grant_log_failure_on_fast_path_returns_permit(self):
        logger = _GrantBoomLogger({"req-1"})
        gate = make_gate(self.mod, self._cfg(), logger)
        with self.assertRaises(RuntimeError):
            async with gate.acquire("solo", "req-1"):
                self.fail("body must not run when the grant line raises")
        self.assertEqual(
            gate._sems["solo"]._value,
            1,
            "the spent permit must be returned when __aenter__ raises",
        )
        self.assertEqual(gate.held(), {}, "failed entry must not stay counted held")
        self.assertEqual(gate.waiting(), {})
        # The lane is still whole: the next request is admitted without waiting.
        h2, r2 = asyncio.Event(), asyncio.Event()
        t2 = holding(gate, "solo", "req-2", h2, r2)
        await asyncio.wait_for(h2.wait(), timeout=1)
        self.assertEqual(gate.held(), {"solo": 1})
        r2.set()
        await t2
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)

    async def test_grant_log_failure_on_queued_path_keeps_lane_alive(self):
        logger = _GrantBoomLogger({"queued"})
        gate = make_gate(self.mod, self._cfg(), logger)
        h1, r1 = asyncio.Event(), asyncio.Event()
        t1 = holding(gate, "solo", "holder", h1, r1)
        await h1.wait()
        # t2 must genuinely wait, then be granted by t1's release.
        t2 = asyncio.get_running_loop().create_task(_acquire_once(gate, "solo", "queued"))
        await asyncio.sleep(0.01)
        self.assertEqual(gate.waiting(), {"solo": 1})
        r1.set()  # the release grants to t2 ...
        with self.assertRaises(RuntimeError):
            await t2  # ... and the grant line raises inside its entry
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate.held(), {}, "failed entry must not stay counted held")
        self.assertLessEqual(gate._sems["solo"]._value, 1, "semaphore must never exceed its limit")
        # A later request is admitted normally AND the lane still limits to 1:
        # an over-release would silently admit a second holder.
        h3, r3 = asyncio.Event(), asyncio.Event()
        t3 = holding(gate, "solo", "third", h3, r3)
        await asyncio.wait_for(h3.wait(), timeout=1)
        self.assertEqual(gate.held(), {"solo": 1})
        h4, r4 = asyncio.Event(), asyncio.Event()
        t4 = holding(gate, "solo", "fourth", h4, r4)
        await asyncio.sleep(0.05)
        self.assertFalse(h4.is_set(), "over-release would admit a second holder")
        self.assertEqual(gate._sems["solo"]._value, 0)
        r3.set()
        await asyncio.wait_for(h4.wait(), timeout=1)
        r4.set()
        await asyncio.gather(t1, t3, t4)
        self.assertEqual(gate.held(), {})
        self.assertEqual(gate.waiting(), {})
        self.assertEqual(gate._sems["solo"]._value, 1)

    async def test_entry_failure_then_aexit_releases_nothing_more(self):
        logger = _GrantBoomLogger({"req-1"})
        gate = make_gate(self.mod, self._cfg(), logger)
        holder = gate.acquire("solo", "req-1")
        with self.assertRaises(RuntimeError):
            await holder.__aenter__()
        self.assertEqual(gate._sems["solo"]._value, 1, "permit already returned")
        self.assertFalse(holder.permit, "ownership flag must be cleared")
        self.assertFalse(holder.admitted)
        # `async with` never calls __aexit__ after a failed entry, but a caller
        # that does (or a second cleanup pass) must not push a permit back in.
        await holder.__aexit__(None, None, None)
        holder._release_owned()
        self.assertEqual(gate._sems["solo"]._value, 1, "must stay at the class limit exactly")
        self.assertEqual(gate.held(), {})


class OversubscriptionTests(unittest.TestCase):
    def setUp(self):
        self.mod = admission
        self.log = _Capture()
        self.logger = logging.getLogger("test.admission.oversub")
        self.logger.handlers = [self.log]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def test_oversubscription_warns_once_and_keeps_working(self):
        cfg = {
            "classes": {
                "long": {"max_concurrent": 2},
                "short": {"max_concurrent": 2},
            },
            "default": "short",
        }
        gate = make_gate(self.mod, cfg, self.logger, max_batch_size=1)
        self.assertTrue(gate.enabled)
        warns = [r for r in self.log.records if r.levelno >= logging.WARNING]
        self.assertEqual(len(warns), 1, "exactly one advisory line")
        self.assertIn("max_batch_size", warns[0].getMessage())
        # Advisory, never fatal: classification still works.
        self.assertEqual(gate.classify(10, None), "short")

    def test_no_warning_when_within_batch_size(self):
        cfg = {
            "classes": {
                "long": {"max_concurrent": 1},
                "short": {"max_concurrent": 1},
            },
            "default": "short",
        }
        gate = make_gate(self.mod, cfg, self.logger, max_batch_size=2)
        warns = [r for r in self.log.records if r.levelno >= logging.WARNING]
        self.assertEqual(warns, [])

    def test_no_warning_without_max_batch_size(self):
        cfg = {"classes": {"long": {"max_concurrent": 4}}}
        gate = make_gate(self.mod, cfg, self.logger, max_batch_size=None)
        warns = [r for r in self.log.records if r.levelno >= logging.WARNING]
        self.assertEqual(warns, [])


class _XLoggerStrict:
    """Mimics tabbyAPI's ``common.logger.XLogger`` — the logger the model.py
    patch injects into the gate in production.

    Unlike stdlib ``logging.Logger``, its ``info``/``warning`` take NO
    printf-style ``*args``: the second positional is a dict. Every gate log
    call must therefore be ONE formatted string. Production incident
    2026-11-14: the 6-positional ``info`` call at admission.py:527 raised
    ``TypeError: XLogger.info() takes from 2 to 4 positional arguments but 6
    were given`` and 500'd every chat request. The offline suite's stdlib
    logger accepted the same call, which is why it missed."""

    def __init__(self):
        self.messages = []

    def info(self, log_message, log_extra=None, details=None):
        self.messages.append(("info", log_message))

    def warning(self, log_message, log_extra=None, details=None):
        self.messages.append(("warning", log_message))


class XLoggerSignatureTests(unittest.IsolatedAsyncioTestCase):
    """Every log-producing path of the gate, driven through a logger with
    tabbyAPI's real signature. A TypeError here is a 500 on every request."""

    @classmethod
    def setUpClass(cls):
        cls.mod = admission

    def _gate(self, cfg, xlog, max_batch_size=2):
        return self.mod.LaneGate.from_config(cfg, max_batch_size=max_batch_size, logger=xlog)

    CFG = {
        "classes": {
            "long": {"max_concurrent": 1, "prompt_tokens_gte": 10000},
            "short": {"max_concurrent": 1},
        },
        "default": "short",
    }

    async def test_hold_and_grant_lines_survive(self):
        xlog = _XLoggerStrict()
        gate = self._gate(self.CFG, xlog)
        async with gate.acquire("short", "r1"):
            pass
        levels = [l for l, _ in xlog.messages]
        self.assertIn("info", levels, "hold and/or grant line must log")
        text = " ".join(m for _, m in xlog.messages)
        self.assertIn("granted", text)

    async def test_unknown_class_warning_survives(self):
        xlog = _XLoggerStrict()
        gate = self._gate(self.CFG, xlog)
        async with gate.acquire("bogus", "r2"):
            pass
        self.assertTrue(
            any(l == "warning" for l, _ in xlog.messages), "unknown class must warn without raising"
        )

    async def test_nonfinite_classification_warning_survives(self):
        xlog = _XLoggerStrict()
        gate = self._gate(self.CFG, xlog)
        cls = gate.classify(float("nan"), None)
        self.assertEqual(cls, "short", "NaN must fall to the default class")
        self.assertTrue(any(l == "warning" for l, _ in xlog.messages))

    async def test_timeout_fail_open_warning_survives(self):
        xlog = _XLoggerStrict()
        cfg = dict(self.CFG, hold_timeout=0.05)
        cfg = {**self.CFG, "hold_timeout": 0.05}
        gate = self._gate(cfg, xlog)
        release = asyncio.Event()

        async def holder():
            async with gate.acquire("short", "h1"):
                await release.wait()

        t = asyncio.get_running_loop().create_task(holder())
        for _ in range(100):
            if gate.held():
                break
            await asyncio.sleep(0.005)
        try:
            async with gate.acquire("short", "r3"):
                pass
        finally:
            release.set()
            await t
        self.assertTrue(
            any("fail-open" in m for _, m in xlog.messages),
            "the hold_timeout warning must log without raising",
        )

    def test_advisory_warning_survives(self):
        xlog = _XLoggerStrict()
        cfg = {
            "classes": {"long": {"max_concurrent": 2}, "short": {"max_concurrent": 1}},
            "default": "short",
        }
        self._gate(cfg, xlog, max_batch_size=2)
        self.assertTrue(any(l == "warning" for l, _ in xlog.messages))


if __name__ == "__main__":
    unittest.main()
