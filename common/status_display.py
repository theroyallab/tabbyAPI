"""
Live console status line: cache usage and in-flight generation jobs.

Rendered with rich's Live display below the regular log output, so log lines
keep scrolling above it. Only active on an interactive terminal.
"""

import asyncio
import contextlib
import time
from collections import deque
from io import StringIO
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from rich.console import Console, Group
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from common.logger import RICH_CONSOLE

REFRESH_INTERVAL = 0.25
SPEED_WINDOW = 2.0

HEIGHT_HOLD = 5.0

# Finished requests the status page lists, with their text
RECENT_JOBS = 20


@dataclass
class JobStatus:
    """Progress of one generation job, updated from generator events."""

    label: str
    prompt_tokens: int
    stage: str = "queued"
    cached_tokens: int = 0
    prefill_tokens: int = 0
    gen_tokens: int = 0
    created: float = field(default_factory=time.monotonic)
    samples: deque = field(default_factory=deque)
    last_generated: float = 0.0
    # Context restored when the engine takes the job back after a requeue
    resume_tokens: int = 0
    resume_total: int = 0
    # Called after every state change so the display can redraw promptly
    on_change: Optional[Callable[[], None]] = None
    # Whether the engine has put the job back in its queue. A long generation
    # gives up its cache pages every max_rq_tokens and waits there until there
    # is room to continue, which can take a while when the cache is full
    is_waiting: Optional[Callable[[], bool]] = None
    # For the status page: the prompt and the text generated so far, kept as
    # chunks and joined when read
    request_id: str = ""
    prompt: str = ""
    chunks: list = field(default_factory=list)
    text_length: int = 0
    finish_reason: Optional[str] = None
    finished: Optional[float] = None
    # Ends the job from outside its request (the status page). The backend
    # finishes the response with what was generated, as finish_reason "abort"
    cancel: Optional[Callable[[], Awaitable[None]]] = None
    cancel_requested: bool = False

    def _changed(self):
        if self.on_change is not None:
            self.on_change()

    def started(self, cached_tokens: int):
        if self.gen_tokens:
            # Taken back after a requeue: the context so far is restored from
            # the cache, or prefilled again where it was evicted
            self.stage = "resuming"
            self.resume_tokens = 0
            self.resume_total = 0
            self.samples.clear()
            self._changed()
            return

        self.stage = "prefill"
        self.cached_tokens = cached_tokens
        self.prefill_tokens = cached_tokens
        self._changed()

    def prefill(self, progress: int, total: int = 0):
        if self.stage == "resuming":
            self.resume_tokens = max(self.resume_tokens, progress)
            self.resume_total = max(self.resume_total, total)
            self._changed()
            return

        self.stage = "prefill"
        self.prefill_tokens = max(self.prefill_tokens, progress)
        self._changed()

    def generated(self, gen_tokens: int):
        self.stage = "generating"
        self.prefill_tokens = self.prompt_tokens
        self.gen_tokens = gen_tokens

        now = time.monotonic()
        self.last_generated = now
        self.samples.append((now, gen_tokens))
        while self.samples and now - self.samples[0][0] > SPEED_WINDOW:
            self.samples.popleft()
        self._changed()

    def append_text(self, text: str):
        self.chunks.append(text)
        self.text_length += len(text)

    def text(self, offset: int = 0) -> str:
        """The generated text from offset on."""

        if len(self.chunks) > 1:
            self.chunks[:] = ["".join(self.chunks)]
        return self.chunks[0][offset:] if self.chunks else ""

    def info(self, now: float) -> dict:
        """A summary for the status page."""

        waiting = self.waiting()
        end = self.finished if self.finished is not None else now
        return {
            "request_id": self.request_id,
            "label": self.label,
            "stage": "finished"
            if self.finished is not None
            else ("waiting" if waiting else self.stage),
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "prefill_tokens": self.prefill_tokens,
            "gen_tokens": self.gen_tokens,
            "resume_tokens": self.resume_tokens,
            "resume_total": self.resume_total,
            "tokens_per_second": None if self.finished is not None else self.tokens_per_second(),
            "duration": end - self.created,
            "waiting_for": now - self.last_generated if waiting else None,
            "text_length": self.text_length,
            "finish_reason": self.finish_reason,
            "can_cancel": self.finished is None
            and self.cancel is not None
            and not self.cancel_requested,
        }

    def waiting(self) -> bool:
        """Whether a job that has started generating is back in the engine's queue."""

        if self.stage != "generating" or self.is_waiting is None:
            return False
        try:
            return self.is_waiting()
        except Exception:
            return False

    def tokens_per_second(self) -> Optional[float]:
        # Samples are only pruned when a new one arrives, so a job that stopped
        # getting tokens would keep showing its last rate
        if len(self.samples) < 2 or time.monotonic() - self.samples[-1][0] > SPEED_WINDOW:
            return None

        (t0, n0), (t1, n1) = self.samples[0], self.samples[-1]
        return (n1 - n0) / (t1 - t0) if t1 > t0 else None


class StatusDisplay:
    def __init__(self):
        self.jobs: dict[str, JobStatus] = {}
        self.completed = 0
        # Finished jobs, newest first, and the tokens finished jobs processed
        self.recent: deque = deque(maxlen=RECENT_JOBS)
        self.done_gen_tokens = 0
        self.done_prefill_tokens = 0
        self._live = None
        self._task: Optional[asyncio.Task] = None
        self._last_frame: Optional[str] = None
        self._last_push = 0.0
        # (time, job rows) samples over the last HEIGHT_HOLD seconds
        self._heights: deque = deque()

    @property
    def active(self) -> bool:
        return self._live is not None

    def start(self):
        """Show the status line. No-op when the console isn't a terminal."""

        if self._live is not None or not RICH_CONSOLE.is_terminal:
            return

        # Deferred import: rich.live pulls in a fair amount at import time
        from rich.live import Live

        self._live = Live(
            self.render(),
            console=RICH_CONSOLE,
            auto_refresh=False,
            transient=True,
        )
        self._live.start()
        self._task = asyncio.create_task(self._refresh_loop())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

        if self._live is not None:
            self._live.stop()
            self._live = None

    @contextlib.asynccontextmanager
    async def suspended(self):
        """
        Hide the display while another live renderable runs, e.g. a loading or
        download progress bar. Rich nests a second live display inside the first
        and only repaints it when the outer one refreshes, which this display
        avoids doing while its own content is unchanged, so a nested bar would
        sit frozen. The display comes back once the block exits.
        """

        was_active = self.active
        if was_active:
            await self.stop()
        try:
            yield
        finally:
            if was_active:
                self.start()

    def refresh(self, rate_limited: bool = True):
        """
        Redraw if the content changed. Frames are only pushed when their text
        differs from the last one: the Windows console host scrolls the viewport
        to the cursor on every write, so an idle display that kept redrawing
        would make scrolling back through the log impossible there.

        Called from job state changes as well as the periodic loop. The generator
        blocks the event loop for a whole prefill chunk, and after each chunk the
        loop runs the consumer and then the generator again before any timer, so
        a timer-driven refresh always renders one event late. Drawing right when
        the consumer updates the job state shows the fresh state before the next
        chunk blocks the loop. Rate limited so per-token updates don't flood it.
        """

        if self._live is None:
            return

        now = time.monotonic()
        if rate_limited and now - self._last_push < REFRESH_INTERVAL:
            return

        try:
            renderable = self.render()
            frame = self._frame_text(renderable)
            if frame != self._last_frame:
                self._last_frame = frame
                self._last_push = now
                self._live.update(renderable, refresh=True)
        except Exception:
            pass

    async def _refresh_loop(self):
        # Refreshing from the event loop keeps every read of job state on the
        # same thread that writes it; this loop catches changes that don't come
        # through a job event, such as cache statistics after a job ends
        while True:
            await asyncio.sleep(REFRESH_INTERVAL)
            self.refresh(rate_limited=False)

    @staticmethod
    def _frame_text(renderable) -> str:
        """Plain-text rendering of a frame at the current console width, for change detection."""

        scratch = Console(
            file=StringIO(),
            width=RICH_CONSOLE.width,
            force_terminal=False,
            color_system=None,
        )
        scratch.print(renderable)
        return scratch.file.getvalue()

    # Job tracking, called from the generation loop

    def add_job(self, request_id: str, label: str, prompt_tokens: int) -> JobStatus:
        status = JobStatus(
            label=label, prompt_tokens=prompt_tokens, on_change=self.refresh, request_id=request_id
        )
        self.jobs[request_id] = status
        self.refresh()
        return status

    def remove_job(self, request_id: str):
        job = self.jobs.pop(request_id, None)
        if job is not None:
            self.completed += 1
            self.done_gen_tokens += job.gen_tokens
            self.done_prefill_tokens += max(job.prefill_tokens - job.cached_tokens, 0)
            job.finished = time.monotonic()
            job.cancel = None
            job.is_waiting = None
            self.recent.appendleft(job)
            self.refresh()

    def find_job(self, request_id: str) -> Optional[JobStatus]:
        """An active job, or a recently finished one."""

        job = self.jobs.get(request_id)
        if job is None:
            job = next((j for j in self.recent if j.request_id == request_id), None)
        return job

    def snapshot(self) -> dict:
        """Jobs, cache statistics and token counters, for the status page."""

        now = time.monotonic()
        active = list(self.jobs.values())
        return {
            "jobs": [job.info(now) for job in active],
            "recent": [job.info(now) for job in self.recent],
            "completed": self.completed,
            # Totals since startup. A client takes rates from the change between polls
            "generated_tokens": self.done_gen_tokens + sum(job.gen_tokens for job in active),
            "prefilled_tokens": self.done_prefill_tokens
            + sum(max(job.prefill_tokens - job.cached_tokens, 0) for job in active),
            "cache": self._cache_stats(),
        }

    # Rendering

    @staticmethod
    def _cache_stats() -> Optional[dict]:
        """Cache statistics from the backend, or None when it doesn't provide them."""

        from common import model

        container = model.container
        generator = getattr(getattr(container, "generator", None), "generator", None)
        if generator is None or not hasattr(generator, "get_cache_stats"):
            return None

        try:
            return generator.get_cache_stats()
        except Exception:
            return None

    def _summary_line(self) -> Text:
        from common import model

        line = Text()
        if not (model.container and model.container.loaded):
            line.append("No model loaded", style="dim")
            return line

        stats = self._cache_stats()
        queued = 0
        if stats:
            max_tokens = stats["max_tokens"]
            used = stats["used_tokens"]
            line.append("cache ", style="bold")
            line.append(f"{used:,}/{max_tokens:,} tokens in use")
            line.append(f" ({used / max_tokens * 100:.0f}%)", style="dim")
            line.append(f" · {stats['cached_tokens']:,} reusable")
            # Each tier's count is what it holds beyond the tiers above it: a
            # page restored from disk is in VRAM, so it no longer counts as
            # reusable *from* disk even though the file is still there
            has_tier = bool(stats.get("tier_max_tokens"))
            # The disk tier's entries are only present in engines that have one
            has_disk = bool(stats.get("disk_max_bytes"))
            if has_tier:
                line.append(f", +{stats['tier_cached_tokens']:,} more in sysmem")
            if has_disk:
                line.append(f", +{stats['disk_cached_tokens']:,} more on disk")
                line.append(
                    f" · disk {stats['disk_used_bytes'] / 1024**3:.1f}"
                    f"/{stats['disk_max_bytes'] / 1024**3:.0f} GB"
                )
            if stats["hit_rate"] is not None:
                line.append(f" · hit rate {stats['hit_rate'] * 100:.0f}%")
                # Where the hits came from; the rest was still in VRAM. The
                # engine's tier rate includes pages promoted from disk
                tier_rate = stats.get("tier_hit_rate") if has_tier else None
                disk_rate = stats.get("disk_hit_rate") if has_disk else None
                sources = []
                if tier_rate is not None:
                    sysmem_rate = max(tier_rate - (disk_rate or 0), 0)
                    sources.append(f"{sysmem_rate * 100:.0f}% from sysmem")
                if disk_rate is not None:
                    sources.append(f"{disk_rate * 100:.0f}% from disk")
                if sources:
                    line.append(" incl. " + ", ".join(sources))
            line.append(" · ")
            queued = stats["pending_jobs"]

        # Every request here is either active or still queued in the engine
        active = max(len(self.jobs) - queued, 0)
        line.append(f"{active} active")
        if queued:
            line.append(f", {queued} queued")

        total_tps = sum(j.tokens_per_second() or 0 for j in self.jobs.values())
        if total_tps:
            line.append(f" · {total_tps:,.0f} T/s")

        return line

    def _job_row(self, job: JobStatus):
        label = Text(job.label, style="cyan")

        if job.stage == "queued":
            return label, Text("queued", style="dim"), Text(""), Text("")

        if job.stage == "prefill":
            bar = ProgressBar(
                total=max(job.prompt_tokens, 1), completed=job.prefill_tokens, width=24
            )
            detail = Text(f"{job.prefill_tokens:,}/{job.prompt_tokens:,} tokens")
            if job.cached_tokens:
                detail.append(f" ({job.cached_tokens:,} cached)", style="dim")
            return label, Text("prefill"), bar, detail

        if job.stage == "resuming":
            bar = ProgressBar(total=max(job.resume_total, 1), completed=job.resume_tokens, width=24)
            detail = Text(f"{job.gen_tokens:,} tokens")
            if job.resume_total:
                detail.append(f" · context {job.resume_tokens:,}/{job.resume_total:,}", style="dim")
            return label, Text("resuming"), bar, detail

        if job.waiting():
            detail = Text(f"{job.gen_tokens:,} tokens")
            waited = int(time.monotonic() - job.last_generated)
            detail.append(
                f" · requeued, waiting {waited // 60}:{waited % 60:02d}",
                style="dim",
            )
            return label, Text("waiting", style="yellow"), Text(""), detail

        detail = Text(f"{job.gen_tokens:,} tokens")
        tps = job.tokens_per_second()
        if tps is not None:
            detail.append(f" · {tps:,.1f} T/s")
        return label, Text("generating"), Text(""), detail

    def _held_height(self, rows: int) -> int:
        """Number of job rows to show: the most seen in the last HEIGHT_HOLD seconds."""

        now = time.monotonic()
        self._heights.append((now, rows))
        while self._heights and now - self._heights[0][0] > HEIGHT_HOLD:
            self._heights.popleft()

        return max(height for _, height in self._heights)

    def render(self):
        table = Table.grid(padding=(0, 2))
        for _ in range(4):
            table.add_column(no_wrap=True)

        jobs = list(self.jobs.values())
        for job in jobs:
            table.add_row(*self._job_row(job))

        # Rows are anchored at the bottom of the terminal, so a finished job would
        # pull the rows above it down. Keep the height for a while instead
        for _ in range(self._held_height(len(jobs)) - len(jobs)):
            table.add_row(Text(""), Text(""), Text(""), Text(""))

        return Group(Rule(style="dim"), self._summary_line(), table)


status_display = StatusDisplay()
