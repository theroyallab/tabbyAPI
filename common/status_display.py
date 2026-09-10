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
from typing import Callable, Optional

from rich.console import Console, Group
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from common.logger import RICH_CONSOLE

REFRESH_INTERVAL = 0.25
SPEED_WINDOW = 2.0

HEIGHT_HOLD = 5.0


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
    # Called after every state change so the display can redraw promptly
    on_change: Optional[Callable[[], None]] = None

    def _changed(self):
        if self.on_change is not None:
            self.on_change()

    def started(self, cached_tokens: int):
        self.stage = "prefill"
        self.cached_tokens = cached_tokens
        self.prefill_tokens = cached_tokens
        self._changed()

    def prefill(self, progress: int):
        self.stage = "prefill"
        self.prefill_tokens = max(self.prefill_tokens, progress)
        self._changed()

    def generated(self, gen_tokens: int):
        self.stage = "generating"
        self.prefill_tokens = self.prompt_tokens
        self.gen_tokens = gen_tokens

        now = time.monotonic()
        self.samples.append((now, gen_tokens))
        while self.samples and now - self.samples[0][0] > SPEED_WINDOW:
            self.samples.popleft()
        self._changed()

    def tokens_per_second(self) -> Optional[float]:
        if len(self.samples) < 2:
            return None

        (t0, n0), (t1, n1) = self.samples[0], self.samples[-1]
        return (n1 - n0) / (t1 - t0) if t1 > t0 else None


class StatusDisplay:
    def __init__(self):
        self.jobs: dict[str, JobStatus] = {}
        self.completed = 0
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
        status = JobStatus(label=label, prompt_tokens=prompt_tokens, on_change=self.refresh)
        self.jobs[request_id] = status
        self.refresh()
        return status

    def remove_job(self, request_id: str):
        if self.jobs.pop(request_id, None) is not None:
            self.completed += 1
            self.refresh()

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
            if stats.get("tier_max_tokens"):
                line.append(f" + {stats['tier_cached_tokens']:,} in sysmem")
            if stats["hit_rate"] is not None:
                line.append(f" · hit rate {stats['hit_rate'] * 100:.0f}%")
                if stats.get("tier_max_tokens") and stats.get("tier_hit_rate") is not None:
                    line.append(f" incl. {stats['tier_hit_rate'] * 100:.0f}% from sysmem")
            line.append(" · ")
            queued = stats["pending_jobs"]

        line.append(f"{len(self.jobs)} active")
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
