"""Lane admission control for the tabbyAPI exllamav3 backend.

LANE-ADMISSION-v1

A pure-stdlib request-class gate that sits *above* the exllamav3 enqueue
point in ``backends/exllamav3/model.py``. exllamav3 has no per-job priority
and its requeue path re-grants a long job the slot it just freed, so class
separation ("short queries jump the line, long agentic sessions never share
the slot with each other") can only be expressed before the job is built.
The gate holds requests instead of rejecting them: a held request always
eventually proceeds (``hold_timeout`` expiry is fail-open).

No tabbyAPI or third-party imports: the module must import cleanly with a
bare ``python3`` so it is unit-testable on a workstation with no engine
deps. ``logger`` is injected by the caller (tabbyAPI passes its own) so this
module never configures logging itself.

Config block (all keys optional; omitting it entirely, or ``{}``, disables
the feature — the gate is then provably a no-op):

    admission:
      classes:
        long:
          prompt_tokens_gte: 32768     # optional predicate
          max_new_tokens_gte: null     # optional predicate
          max_concurrent: 1
        short:
          max_concurrent: 2
      default: short                   # class when no rule matches
      hold_timeout: 1800               # seconds; 0 = wait forever
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque

DEFAULT_HOLD_TIMEOUT = 1800.0

# Recognised per-class predicates. A key present with value ``None`` is an
# *absent* predicate (the YAML example spells it out to document intent);
# a class with no present predicates matches nothing — use ``default`` for
# the catch-all.
_PREDICATES = ("prompt_tokens_gte", "max_new_tokens_gte")
_CLASS_KEYS = frozenset(_PREDICATES) | {"max_concurrent"}
_CONFIG_KEYS = frozenset({"classes", "default", "hold_timeout"})


def _is_mapping(obj):
    return isinstance(obj, dict)


def _require_int(value, key):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"admission: {key} must be an integer, got {value!r}")
    return value


def _require_nonneg_int(value, key):
    _require_int(value, key)
    if value < 0:
        raise ValueError(f"admission: {key} must be >= 0, got {value!r}")
    return value


class _ClassRule:
    __slots__ = ("name", "prompt_tokens_gte", "max_new_tokens_gte", "max_concurrent")

    def __init__(self, name, body):
        if not _is_mapping(body):
            raise ValueError(f"admission: classes.{name} must be a mapping, got {body!r}")
        for key in body:
            if key not in _CLASS_KEYS:
                raise ValueError(f"admission: classes.{name} has unknown key {key!r}")
        self.name = name
        self.max_concurrent = body.get("max_concurrent", 1)
        if isinstance(self.max_concurrent, bool) or not isinstance(self.max_concurrent, int):
            raise ValueError(
                f"admission: classes.{name}.max_concurrent must be an integer, "
                f"got {self.max_concurrent!r}"
            )
        if self.max_concurrent < 1:
            raise ValueError(
                f"admission: classes.{name}.max_concurrent must be >= 1, "
                f"got {self.max_concurrent!r}"
            )
        self.prompt_tokens_gte = body.get("prompt_tokens_gte")
        self.max_new_tokens_gte = body.get("max_new_tokens_gte")
        if self.prompt_tokens_gte is not None:
            _require_nonneg_int(self.prompt_tokens_gte, f"classes.{name}.prompt_tokens_gte")
        if self.max_new_tokens_gte is not None:
            _require_nonneg_int(self.max_new_tokens_gte, f"classes.{name}.max_new_tokens_gte")

    def matches(self, prompt_tokens, max_new_tokens):
        """True iff every *present* predicate matches. Anded. Inclusive bounds.

        A class with no present predicates *matches nothing* (ruling 5) — it
        is a capacity reservation, not a catch-all; use ``default`` for that.
        ``max_new_tokens`` of ``None`` (client sent no limit) never matches a
        class that declares ``max_new_tokens_gte``.
        """
        matched_any = False
        if self.prompt_tokens_gte is not None:
            if prompt_tokens < self.prompt_tokens_gte:
                return False
            matched_any = True
        if self.max_new_tokens_gte is not None:
            if max_new_tokens is None or max_new_tokens < self.max_new_tokens_gte:
                return False
            matched_any = True
        return matched_any


class _NoopCM:
    """Async context manager that never awaits and never touches state."""

    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc_info):
        return False


_NOOP_CM = _NoopCM()


class _Holder:
    """One in-flight ``async with gate.acquire(...)`` — the slot handle.

    Lifecycle: queued in the class FIFO as a future; granted by a releasing
    holder or :meth:`LaneGate.abort_all`; releases (or hands on) its permit
    on *every* exit path — normal, exception, cancellation, timeout — and on a
    failed entry too (:meth:`__aenter__` raising, where Python never calls
    :meth:`__aexit__`). Both routes go through :meth:`_release_owned`, which is
    idempotent.
    """

    __slots__ = (
        "gate",
        "cls",
        "request_id",
        "prompt_tokens",
        "max_new_tokens",
        "fut",
        "counted",
        "permit",
        "admitted",
        "t_enqueued",
        "t_granted",
        "abort_event",
    )
    # ``permit`` means this holder OWNS a semaphore permit and must hand it
    # on / return it at exit. Fail-open wakes (hold_timeout expiry, abort_all)
    # set admitted WITHOUT a permit — they proceed but never release one, so
    # the semaphore count can never exceed its limit.

    def __init__(self, gate, cls, request_id, prompt_tokens, max_new_tokens, abort_event=None):
        self.gate = gate
        self.cls = cls
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.fut = asyncio.Future()
        self.counted = False  # counted in gate._waiting
        self.permit = False  # consumed a semaphore permit (holds a slot)
        self.admitted = False  # __aexit__ must free the slot
        self.t_enqueued = 0.0
        self.t_granted = 0.0
        self.abort_event = abort_event

    async def __aenter__(self):
        try:
            return await self._enter()
        except BaseException:
            # ``async with`` does NOT call __aexit__ when __aenter__ raises, so
            # whatever this holder already owns — a permit spent by a grant,
            # the ``_held`` counter bumped by _note_grant() — would be stranded
            # for the life of the process. One occurrence permanently shrinks
            # the lane, and on a ``max_concurrent: 1`` lane it disables it.
            # Hand back exactly what is owned, then let the original exception
            # propagate untouched (never swallowed).
            self._release_owned()
            raise

    async def _enter(self):
        gate = self.gate
        if not gate._enabled:
            return self
        abort_event = self.abort_event
        if abort_event is not None and abort_event.is_set():
            # The client is already gone. Raise the same error the endpoint
            # would see from DisconnectHandler.poll() mid-generation (SPEC §5)
            # — before queueing, so this request never consumes a slot.
            raise asyncio.CancelledError()
        queue = gate._queues[self.cls]
        # Pre-grant fast path: an idle permit is taken directly; the FIFO is
        # only formed once the lane is saturated (FIFO fairness applies among
        # actual waiters).
        if not queue and not gate._sems[self.cls].locked():
            await gate._sems[self.cls].acquire()
            self.permit = True
            self._note_grant(0.0)
            return self
        queue.append(self.fut)
        self.counted = True
        gate._waiting[self.cls] = gate._waiting.get(self.cls, 0) + 1
        self.t_enqueued = gate._clock()
        abort_fut = None
        try:
            timeout = gate._hold_timeout
            if abort_event is None:
                if timeout and timeout > 0:
                    try:
                        # NOTE: asyncio.wait_for cancels self.fut on expiry, so
                        # a grant racing the timer shows up as a *cancelled*
                        # future here; the permit is reconciled below, never
                        # stranded.
                        await asyncio.wait_for(self.fut, timeout)
                    except asyncio.TimeoutError:
                        if self.fut.done() and not self.fut.cancelled():
                            # A wake landed before the timer's cancel of the
                            # future. result True  -> a real grant (permit was
                            # consumed by the releaser on our behalf);
                            # result False -> an abort_all fail-open wake.
                            self.permit = bool(self.fut.result())
                            self._note_grant(gate._clock() - self.t_enqueued)
                            return self
                        # No permit exists for us (a grant could not have landed:
                        # wait_for cancelled the future first). Give up the queue
                        # position and proceed fail-open with a warning — admitted
                        # but permit-free, so our exit releases nothing.
                        if self.fut in queue:
                            queue.remove(self.fut)
                        self._warn_fail_open(timeout)
                        self._note_grant(timeout)
                        return self
                else:
                    # hold_timeout == 0: wait forever, no fail-open timer.
                    await self.fut
            else:
                # SPEC §3/§5 — race the FIFO wake and the fail-open timer
                # against the disconnect signal, so a request that vanishes
                # while *queued* reaches the endpoint through the same path as
                # one that vanishes mid-generation.
                abort_fut = asyncio.ensure_future(abort_event.wait())
                done, _ = await asyncio.wait(
                    (self.fut, abort_fut),
                    timeout=(timeout if timeout and timeout > 0 else None),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_event.is_set():
                    # Abandon the hold, permit-free. Raising here (rather than
                    # returning) makes the handler below do the giving-up: a
                    # grant that landed in the same tick re-enters the FIFO
                    # instead of being spent on a request with no client.
                    raise asyncio.CancelledError()
                if self.fut in done:
                    self.permit = bool(self.fut.result())
                    self._note_grant(gate._clock() - self.t_enqueued)
                    return self
                # Timer expired with neither a grant nor an abort: the same
                # fail-open exit as the branch above (SPEC §4 — hold, never
                # reject; a vanished client is the only early way out).
                if self.fut in queue:
                    queue.remove(self.fut)
                self._warn_fail_open(timeout)
                self._note_grant(timeout)
                return self
            # Woken via the FIFO. result True  -> a real grant: we consumed
            # the releaser's permit. result False -> abort_all fail-open wake:
            # admitted, but we own no permit.
            self.permit = bool(self.fut.result())
            self._note_grant(gate._clock() - self.t_enqueued)
            return self
        except BaseException:
            # Cancellation (or any other exception) while waiting: give up the
            # queue position and any permit that nonetheless landed, so a
            # cancelled waiter neither consumes nor leaks a slot. Consult the
            # wake *flag*: a False wake (abort_all / fail-open) owns no permit,
            # so releasing one here would over-release and silently un-limit
            # the lane; a True wake is an in-transit permit and must re-enter
            # the FIFO, because a bare Semaphore.release() never wakes a
            # queued waiter (only _pass_permit/abort_all drain the queue).
            if self.fut.done() and not self.fut.cancelled() and self.fut.result():
                gate._pass_permit(self.cls)
                # That in-transit permit is back in the lane now; forget the
                # ownership flag so the entry-failure cleanup in __aenter__
                # cannot hand the same permit back a second time.
                self.permit = False
            if self.fut in queue:
                queue.remove(self.fut)
            if self.counted:
                self.counted = False
                gate._dec_waiting(self.cls)
            raise
        finally:
            if abort_fut is not None:
                # The disconnect waiter is ours; never leave it pending on the
                # caller's event after this holder has settled.
                abort_fut.cancel()
            if self.counted:
                self.counted = False
                gate._dec_waiting(self.cls)

    def _warn_fail_open(self, timeout):
        """One advisory line per fail-open grant (SPEC §4: never reject)."""
        # One formatted string: the injected logger is tabbyAPI's XLogger,
        # whose info/warning take (msg, log_extra: dict | None, details), not
        # printf *args — extra positional values TypeError (verified against
        # common.logger.XLogger on the pinned image).
        self.gate._logger.warning(
            f"admission: request_id={self.request_id} class={self.cls} "
            f"waited >{timeout:.1f}s for a {self.cls} slot "
            f"(prompt_tokens={self.prompt_tokens} "
            f"max_new_tokens={self.max_new_tokens}); "
            f"proceeding fail-open, never rejected",
        )

    def _note_grant(self, waited):
        self.t_granted = self.gate._clock()
        self.gate._held[self.cls] = self.gate._held.get(self.cls, 0) + 1
        self.admitted = True
        self.gate._logger.info(
            f"admission: granted request_id={self.request_id} "
            f"class={self.cls} prompt_tokens={self.prompt_tokens} "
            f"waited={waited:.3f}s (held={self.gate._held[self.cls]})",
        )

    def _release_owned(self):
        """Return everything this holder currently owns. Idempotent.

        The single exit path for *both* :meth:`__aexit__` and the failed-entry
        path in :meth:`__aenter__`: each owned thing is handed back once and
        its flag cleared, so calling this again — or calling it after the
        normal exit — releases nothing further. A permit-free fail-open wake
        owns no permit, so it returns only its ``_held`` count.
        """
        gate = self.gate
        if not gate._enabled:
            return
        if self.admitted:
            self.admitted = False
            gate._dec_held(self.cls)
        if self.permit:
            # Hand the slot straight to the next live waiter (FIFO), else
            # return the permit to the semaphore. Never release more than
            # was granted.
            self.permit = False
            gate._pass_permit(self.cls)

    async def __aexit__(self, *exc_info):
        self._release_owned()
        return False


class LaneGate:
    """Request-class admission gate. Construct via :meth:`from_config`."""

    def __init__(self):
        self._enabled = False
        self._logger = logging.getLogger("admission")
        self._rules = {}  # name -> _ClassRule, declaration order
        self._default = "default"  # sentinel for the disabled gate
        self._hold_timeout = DEFAULT_HOLD_TIMEOUT
        self._max_batch_size = None
        self._sems = {}  # class -> asyncio.Semaphore (loop-bound)
        self._queues = {}  # class -> deque of waiter futures (FIFO)
        self._held = {}  # class -> int
        self._waiting = {}  # class -> int
        self._lane_loop = None  # loop the semaphores above belong to

    @classmethod
    def from_config(cls, raw, *, max_batch_size=None, logger=None):
        """Build a gate from the ``admission`` config block.

        ``None`` or ``{}`` yields a **disabled** gate. Raises ``ValueError``
        naming the offending key for malformed *config* — validation happens
        here, at config-load time only; nothing in this module raises at
        request time. ``logger`` defaults to ``logging.getLogger("admission")``
        so tabbyAPI can inject its own while this module stays stdlib-only.
        """
        gate = cls()
        gate._logger = logger if logger is not None else logging.getLogger("admission")
        if raw is None or (isinstance(raw, dict) and not raw):
            gate._enabled = False
            return gate
        if not _is_mapping(raw):
            raise ValueError(f"admission: config must be a mapping, got {raw!r}")
        for key in raw:
            if key not in _CONFIG_KEYS:
                raise ValueError(f"admission: unknown key {key!r}")

        classes = raw.get("classes")
        if classes is None or (isinstance(classes, dict) and not classes):
            raise ValueError(f"admission: classes must be a non-empty mapping, got {classes!r}")
        if not _is_mapping(classes):
            raise ValueError(f"admission: classes must be a mapping, got {classes!r}")
        # Ordered mapping: preserve declaration order — first match wins.
        for name, body in classes.items():
            gate._rules[name] = _ClassRule(name, body)

        default = raw.get("default")
        if default is None:
            # No explicit default: fall back to the first declared class.
            gate._default = next(iter(gate._rules))
        else:
            if default not in gate._rules:
                raise ValueError(
                    f"admission: default {default!r} is not a declared class "
                    f"(known: {list(gate._rules)})"
                )
            gate._default = default

        hold_timeout = raw.get("hold_timeout", DEFAULT_HOLD_TIMEOUT)
        if hold_timeout is None:
            hold_timeout = DEFAULT_HOLD_TIMEOUT
        if isinstance(hold_timeout, bool) or not isinstance(hold_timeout, (int, float)):
            raise ValueError(f"admission: hold_timeout must be a number, got {hold_timeout!r}")
        if hold_timeout < 0:
            raise ValueError(f"admission: hold_timeout must be >= 0, got {hold_timeout!r}")
        gate._hold_timeout = float(hold_timeout)

        gate._max_batch_size = max_batch_size
        # Oversubscription is advisory, never fatal: class capacities that
        # exceed the engine's max_batch_size are legal (the engine queues),
        # but usually a misconfiguration — say so once and keep working.
        if max_batch_size is not None:
            total = sum(rule.max_concurrent for rule in gate._rules.values())
            if total > max_batch_size:
                gate._logger.warning(
                    f"admission: sum of class max_concurrent ({total}) "
                    f"exceeds max_batch_size ({max_batch_size}); advisory "
                    f"only, gate stays enabled",
                )
        gate._enabled = True
        return gate

    @property
    def enabled(self) -> bool:
        return self._enabled

    def classify(self, prompt_tokens, max_new_tokens):
        """First class whose present predicates all match, else ``default``.

        Never raises: garbage request-side values (negative token counts,
        non-numbers) degrade to the default class with a warning. Bad
        *config* is rejected at :meth:`from_config`, not here.
        """
        if not isinstance(prompt_tokens, (int, float)) or isinstance(prompt_tokens, bool):
            self._logger.warning(
                f"admission: non-numeric prompt_tokens {prompt_tokens!r}; using default class",
            )
            prompt_tokens = -1
        if max_new_tokens is not None and (
            isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, (int, float))
        ):
            self._logger.warning(
                f"admission: non-numeric max_new_tokens {max_new_tokens!r}; treating as no limit",
            )
            max_new_tokens = None
        # NaN/inf defeat `<` comparisons (nan < x is False, which would
        # *match* a gte predicate): route them to the default class instead
        # of letting them masquerade as a long request.
        if isinstance(prompt_tokens, float) and not math.isfinite(prompt_tokens):
            self._logger.warning(
                f"admission: non-finite prompt_tokens {prompt_tokens!r}; using default class",
            )
            prompt_tokens = -1
        if isinstance(max_new_tokens, float) and not math.isfinite(max_new_tokens):
            self._logger.warning(
                f"admission: non-finite max_new_tokens {max_new_tokens!r}; treating as no limit",
            )
            max_new_tokens = None
        for rule in self._rules.values():
            if rule.matches(prompt_tokens, max_new_tokens):
                return rule.name
        return self._default

    def acquire(
        self, cls, request_id, *, prompt_tokens=None, max_new_tokens=None, abort_event=None
    ):
        """``async with gate.acquire(cls, request_id): ...``

        Awaits capacity for *one slot of class ``cls``*; holds it for the
        body of the ``async with``. Never raises on timeout (warns and
        proceeds); releases on every exit path, including cancellation.
        The disabled gate returns a no-op CM that never awaits.

        ``abort_event`` is the tabbyAPI ``DisconnectHandler.abort_event``
        (SPEC §3/§5): already set on entry or fired while queued → the hold is
        abandoned *without* taking a permit and ``asyncio.CancelledError`` is
        raised (the in-tree representation of a vanished client — no new
        exception class); a grant landing in the same tick is handed back
        through the FIFO, not consumed. It is ignored while the gate is
        disabled (SPEC §8). The three keyword args are additive optionals: the
        required call shape stays ``acquire(cls, request_id)``.
        """
        if not self._enabled:
            return _NOOP_CM
        if cls not in self._rules:
            # Request-time badness must never raise: park in the default
            # lane, warn once per unknown name is acceptable noise (rare).
            self._logger.warning(
                f"admission: request_id={request_id} asked for undeclared "
                f"class {cls!r}; using default {self._default!r}",
            )
            cls = self._default
        self._lanes_for(asyncio.get_running_loop())
        self._logger.info(
            f"admission: hold request_id={request_id} class={cls} "
            f"prompt_tokens={prompt_tokens} "
            f"max_new_tokens={max_new_tokens}",
        )
        return _Holder(
            self, cls, request_id, prompt_tokens, max_new_tokens, abort_event=abort_event
        )

    def held(self):
        """Snapshot of in-flight holders per class (``{}`` when idle)."""
        return dict(self._held)

    def waiting(self):
        """Snapshot of queued waiters per class (``{}`` when idle)."""
        return dict(self._waiting)

    def abort_all(self):
        """Release every waiter immediately, FIFO order (used by unload).

        Waiters are *not* rejected — each proceeds fail-open (with a warning)
        so a cancelling unload cannot turn held requests into client errors.
        No-op when nothing is waiting. Wakes at most one release per waiter
        already queued, in queue (arrival) order; wake-ups own no permit, so
        the semaphore count can never exceed its limit.
        """
        if not self._enabled:
            return
        try:
            self._lanes_for(asyncio.get_running_loop())
        except RuntimeError:
            pass  # no running loop: operate on whatever queues exist
        for queue in self._queues.values():
            while queue:
                fut = queue.popleft()
                if not fut.done():
                    # False = fail-open wake (admitted, owns no permit) —
                    # never release more than was granted.
                    fut.set_result(False)
                # A done future is a timed-out/cancelled waiter already on
                # its way out of the queue; it holds no queue position.

    # -- internals ---------------------------------------------------------

    def _lanes_for(self, loop):
        """Build the loop-bound semaphores/queues on first use.

        ``from_config`` may run at config-load time with no event loop, so
        the primitives are created lazily on the first request (and rebuilt
        if the loop changes — a test-only scenario; production binds one
        loop and never rebinds).
        """
        if self._lane_loop is loop:
            return
        self._sems = {
            name: asyncio.Semaphore(rule.max_concurrent) for name, rule in self._rules.items()
        }
        self._queues = {name: deque() for name in self._rules}
        self._held = {}
        self._waiting = {}
        self._lane_loop = loop

    def _clock(self):
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return time.monotonic()

    def _dec_waiting(self, cls):
        n = self._waiting.get(cls, 0) - 1
        if n <= 0:
            self._waiting.pop(cls, None)
        else:
            self._waiting[cls] = n

    def _dec_held(self, cls):
        n = self._held.get(cls, 0) - 1
        if n <= 0:
            self._held.pop(cls, None)
        else:
            self._held[cls] = n

    def _pass_permit(self, cls):
        """Hand the freed permit to the next live FIFO waiter, else return it.

        Never calls ``Semaphore.release`` for a permit that a waiter has not
        consumed, so the semaphore count can never exceed its limit.
        """
        queue = self._queues[cls]
        while queue:
            fut = queue.popleft()
            if not fut.done():
                fut.set_result(True)
                return
            # done == already timed out or cancelled; that holder releases
            # any permit it raced into itself.
        self._sems[cls].release()
