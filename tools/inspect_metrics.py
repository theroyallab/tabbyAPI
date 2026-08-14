#!/usr/bin/env python3
"""Pretty-print TabbyAPI's Prometheus /metrics endpoint.

Stdlib only (no prometheus_client, no requests) — same constraint as the
launcher paths, so it runs on a bare community VM.

    python3 tools/inspect_metrics.py               # one shot, localhost:5000
    python3 tools/inspect_metrics.py --port 8010
    python3 tools/inspect_metrics.py --watch 2     # refresh every 2s
    python3 tools/inspect_metrics.py --json        # machine-readable digest

Requires network.enable_metrics to be true in config.yml.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

PREFIX = "tabbyapi:"
SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>.+)$")


# ---------------------------------------------------------------- scrape ----


def scrape(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse(text: str):
    """-> (scalars: name->float, histograms: name->{'buckets': [(le, count)], 'sum', 'count'})"""
    scalars: dict[str, float] = {}
    hists: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name.startswith(PREFIX):
            name = name[len(PREFIX) :]
        try:
            value = float(m.group("value").split()[0])
        except ValueError:
            continue

        if name.endswith("_bucket"):
            base = name[: -len("_bucket")]
            le = (m.group("labels") or "").partition("le=")[2].strip('"').strip('",')
            le = float("inf") if le in ("+Inf", "Inf") else _f(le)
            if le is not None:
                hists.setdefault(base, {"buckets": [], "sum": None, "count": None})[
                    "buckets"
                ].append((le, value))
        elif name.endswith("_sum") and name[: -len("_sum")] in hists or name.endswith("_sum"):
            hists.setdefault(name[: -len("_sum")], {"buckets": [], "sum": None, "count": None})[
                "sum"
            ] = value
        elif name.endswith("_count"):
            hists.setdefault(name[: -len("_count")], {"buckets": [], "sum": None, "count": None})[
                "count"
            ] = value
        else:
            scalars[name] = value

    for h in hists.values():
        h["buckets"].sort(key=lambda b: b[0])
    return scalars, hists


def _f(s: str):
    try:
        return float(s)
    except ValueError:
        return None


# ------------------------------------------------------------ histograms ----


def quantile(hist: dict, q: float, cap: float | None = None):
    """Prometheus-style linear interpolation inside the matched bucket.

    Interpolation assumes the samples are spread evenly across the bucket they
    landed in, which the 1-2-5 token buckets make wildly untrue at the top end:
    31 prompts of ~200k tokens put 14 samples in the (200k, 500k] bucket, all
    of them within a few thousand of its floor, and the p99 comes out at 493k.
    `cap` clamps the estimate to a separately reported maximum, since no
    percentile of a sample can exceed the largest value in it.
    """
    buckets = hist.get("buckets") or []
    total = hist.get("count")
    if total is None:
        total = buckets[-1][1] if buckets else 0
    if not buckets or not total:
        return None
    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if count == prev_count:
                return le
            frac = (target - prev_count) / (count - prev_count)
            value = prev_le + (le - prev_le) * frac
            return min(value, cap) if cap is not None else value
        prev_le, prev_count = le, count
    return buckets[-1][0]


def mean(hist: dict):
    c = hist.get("count") or 0
    s = hist.get("sum")
    return (s / c) if (s is not None and c) else None


def per_bucket(hist: dict):
    """Cumulative buckets -> [(le, non-cumulative count)]."""
    out, prev = [], 0.0
    for le, count in hist.get("buckets") or []:
        out.append((le, count - prev))
        prev = count
    return out


# --------------------------------------------------------------- display ----

BLOCKS = " ▁▂▃▄▅▆▇█"


def color_on() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class C:
    def __init__(self, enabled: bool):
        self.e = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.e else s

    def head(self, s):
        return self._w("1;36", s)

    def key(self, s):
        return self._w("38;5;245", s)

    def val(self, s):
        return self._w("1", s)

    def good(self, s):
        return self._w("32", s)

    def warn(self, s):
        return self._w("33", s)

    def bad(self, s):
        return self._w("31", s)

    def dim(self, s):
        return self._w("2", s)


def fmt_num(v, unit=""):
    if v is None:
        return "—"
    if unit == "s":
        if v < 1:
            return f"{v * 1000:.0f} ms"
        if v < 60:
            return f"{v:.2f} s"
        return f"{v / 60:.1f} min"
    if unit == "tok":
        if v >= 1e9:
            return f"{v / 1e9:.2f}B"
        if v >= 1e6:
            return f"{v / 1e6:.2f}M"
        if v >= 1e3:
            return f"{v / 1e3:.1f}K"
        return f"{v:.0f}"
    if unit == "%":
        return f"{v * 100:.1f}%"
    if unit == "B":
        for scale, suffix in ((1024**4, "TB"), (1024**3, "GB"), (1024**2, "MB"), (1024, "KB")):
            if v >= scale:
                return f"{v / scale:.2f} {suffix}"
        return f"{v:.0f} B"
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.2f}"


def fmt_le(le, unit):
    if le == float("inf"):
        return "+Inf"
    if unit == "s":
        return f"{le * 1000:.0f}ms" if le < 1 else f"{le:g}s"
    if unit == "tok":
        return fmt_num(le, "tok")
    return f"{le:g}"


def bar(frac: float, width: int, c: C) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    rem = frac * width - filled
    tail = BLOCKS[int(rem * 8)] if filled < width and rem > 0 else ""
    body = "█" * filled + tail
    return body.ljust(width)


# Lookback windows for the rate table, as seconds. Six columns keep the table
# inside the width the widest existing row already occupies; more still render,
# just wider. Windows shorter than the refresh interval are dropped outright,
# since nothing could ever fill them; the rest stay blank until the session has
# run long enough to reach back that far, so the table fills in as you watch.
#
# Windows are written as durations — 30s, 5m, 1h, 24h — rather than as raw
# seconds. It matches PromQL's rate(...[5m]) spelling, which is the idiom this
# table is a stand-in for, and each token doubles as its own column header, so
# a header always reads back exactly what was asked for. A bare number is taken
# as seconds.
RATE_WINDOWS_DEFAULT = "30s,1m,5m,15m,1h,24h"
RATE_COL = 8
DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "": 1}


def parse_windows(spec: str) -> list:
    """Parse a comma-separated duration list into (label, seconds) pairs."""

    out = {}
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", token)
        if not m:
            raise argparse.ArgumentTypeError(
                f"bad window {token!r}: want a duration like 30s, 5m, 1h, 24h"
            )
        seconds = float(m.group(1)) * DURATION_UNITS[m.group(2)]
        if seconds <= 0:
            raise argparse.ArgumentTypeError(f"window {token!r} must be positive")
        # Keep the first spelling seen for any given length, so 60s and 1m do
        # not both claim a column.
        out.setdefault(seconds, token)
    if not out:
        raise argparse.ArgumentTypeError("no windows given")
    return [(label, seconds) for seconds, label in sorted(out.items())]


# History retention. Full resolution for the recent past, thinned beyond that:
# a 24h window scraped every few seconds would otherwise hold tens of thousands
# of samples. Since a column divides by the true elapsed time, a baseline that
# is a little older than its nominal window stays correct — it just measures a
# slightly wider window than the header says.
HISTORY_FINE_AGE = 90.0
HISTORY_COARSE_SPACING = 30.0


def fmt_rate(v) -> str:
    """Compact fixed-width rate, at most six characters wide."""
    if v is None:
        return "—"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}K"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    # A true zero and a rate too small to render must not look alike: one
    # request in a 5m window is 0.0033/s, and printing that as "0.00" reads as
    # an idle server. Bare "0" is the only cell that means nothing happened.
    if v == 0:
        return "0"
    if v < 0.01:
        return "<0.01"
    return f"{v:.2f}"


def fmt_span(seconds: float) -> str:
    """Compact duration for a column header."""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"


def baseline_for(history, now: float, window: float):
    """The sample to difference against for a given lookback window, or None.

    Picks the newest sample at least `window` old, so the figure covers the
    whole window rather than part of it. Returns None when no sample is that
    old yet, and also when the best candidate is more than twice the window
    old, since labelling it with that window would misrepresent it. That second
    case is what keeps the short columns blank when the refresh interval is
    long, or when inference has held the event loop through several refreshes.
    """

    # Window 0 is the "now" column: difference against the previous scrape,
    # whatever interval that turned out to be.
    if window <= 0:
        return history[-1] if history else None

    for ts, scalars in reversed(history):
        age = now - ts
        if age >= window:
            return None if age > window * 2 else (ts, scalars)
    return None


# A window wider than the session so far would print as a permanent blank, even
# though the counters hold the answer: the session's own history *is* the widest
# honest lookback available. So the first window the session has not lived
# through yet overflows into the whole history, relabelled with the span it
# actually covers (`~19.6h` under a `24h` header), and the windows beyond it
# stay blank rather than repeating it. The overflow column shrinks back to its
# nominal label once the session is old enough to fill it for real.
#
# The relabelling only fires when history is genuinely shorter than the window.
# A window that history *does* span but whose baseline was rejected as stale --
# an event-loop stall pushing the last scrape past the `now` or `30s` column --
# is a transient gap in one narrow column, not an overflow, and must not blank
# the wider columns behind it.
OVERFLOW_MIN_RATIO = 1.25


def resolve_columns(history, now: float, columns):
    """-> [(header, baseline or None)], one per column, applying the overflow."""

    span = (now - history[0][0]) if history else 0.0
    out, overflowed, prev = [], False, 0.0
    for label, window in columns:
        base = baseline_for(history, now, window)
        if base is not None:
            # The leftmost column has no nominal width -- it is however long the
            # last refresh happened to take -- so it names its own interval.
            # Watching that number drift above the refresh interval is the
            # cheapest sign that inference is sitting on the event loop.
            out.append((fmt_span(now - base[0]) if window <= 0 else label, base))
        elif history and window > span and not overflowed:
            overflowed = True
            # Too close to the column on its left to be worth a column of its
            # own: it would read as two near-identical numbers.
            if span >= prev * OVERFLOW_MIN_RATIO:
                out.append(("~" + fmt_span(span), history[0]))
            else:
                out.append((label, None))
        else:
            out.append((label, None))
        prev = window
    return out


def prune_history(history, now: float, longest: float) -> None:
    """Thin old samples in place, keeping the recent past at full resolution."""

    kept, last_coarse = [], None
    for ts, scalars in history:
        age = now - ts
        if age > longest * 1.1:
            continue
        if age <= HISTORY_FINE_AGE:
            kept.append((ts, scalars))
        elif last_coarse is None or ts - last_coarse >= HISTORY_COARSE_SPACING:
            kept.append((ts, scalars))
            last_coarse = ts
    history[:] = kept


def row(c: C, label: str, value: str, note: str = "") -> str:
    line = f"  {c.key(label.ljust(26))} {c.val(value)}"
    if note:
        line += f"  {c.dim(note)}"
    return line


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
GUTTER = 3


def vlen(s: str) -> int:
    """Visible width, ignoring colour escapes — ljust() alone would pad by the byte count."""
    return len(ANSI_RE.sub("", s))


class Block:
    """One section, laid out only once its final column width is known.

    Rows are kept as (label, value, note) rather than formatted strings so the
    label column can be sized to the block's own longest label. In two-column
    mode that matters: a fixed 26-wide label field wastes a quarter of the
    available room on sections whose labels are half that.
    """

    def __init__(self, title: str, full_width: bool = False, new_row: bool = False):
        self.title = title
        self.items: list = []
        # Wide, self-aligning content (the rate table, histogram bars) cannot be
        # squeezed into a column and spans the terminal instead.
        self.full_width = full_width
        # Start a fresh row rather than filling the one in progress. Which
        # sections are present varies with the model and the config, so a
        # section that must sit beside the next one cannot rely on the count of
        # sections before it coming out even.
        self.new_row = new_row

    def row(self, label: str, value: str, note: str = ""):
        self.items.append((label, value, note))

    def raw(self, line: str):
        self.items.append((None, line, ""))

    def __bool__(self):
        return bool(self.items)

    def _labw(self) -> int:
        return max([len(label) for label, _, _ in self.items if label] or [0])

    def natural_width(self) -> int:
        labw = self._labw()
        widest = vlen(self.title) + 4
        for label, value, note in self.items:
            if label is None:
                widest = max(widest, vlen(value) + 2)
            else:
                w = 2 + labw + 2 + vlen(value) + (2 + vlen(note) if note else 0)
                widest = max(widest, w)
        return widest

    def lines(self, width: int, c: C) -> list:
        rule = "─" * max(3, width - vlen(self.title) - 1)
        out = [f"{c.head(self.title)} {c.dim(rule)}"]
        labw = self._labw()
        for label, value, note in self.items:
            if label is None:
                out.append(f"  {value}")
                continue
            line = f"  {c.key(label.ljust(labw))}  {c.val(value)}"
            if note:
                line += f"  {c.dim(note)}"
            out.append(line)
        return out


class Stack:
    """Several sections sharing one column, laid out top to bottom.

    Packing works in whole columns, so a short section beside a tall one leaves
    the space beneath it empty while the next section starts a row of its own.
    Grouping the short ones into a single unit fills that gap instead. It is a
    layout hint and nothing more: the sections keep their own titles, rules and
    label widths, and a stack whose members are all absent vanishes with them.
    """

    def __init__(self, *blocks):
        self.blocks = [b for b in blocks if b]
        # A stack is only ever built from column-sized sections; if one of them
        # turns out too wide for a cell, pack gives the whole stack its own row.
        self.full_width = False
        self.new_row = bool(self.blocks) and self.blocks[0].new_row

    def __bool__(self):
        return bool(self.blocks)

    def natural_width(self) -> int:
        return max(b.natural_width() for b in self.blocks)

    def lines(self, width: int, c: C) -> list:
        out: list = []
        for b in self.blocks:
            if out:
                out.append("")
            out.extend(b.lines(width, c))
        return out


def pack(blocks: list, width: int, max_cols: int) -> list:
    """Group blocks into rows of side-by-side sections that fit the terminal.

    Greedy and order preserving, so sections stay where they are expected to be
    rather than being reshuffled to fill space. A block that will not fit
    alongside its predecessor starts a new row rather than being truncated.
    """
    # Every column is the same width, so a block wider than its share would run
    # into its neighbour however well the pair sums. Test each block against the
    # cell, not the pair against the terminal.
    cell = (width - GUTTER * (max_cols - 1)) // max_cols if max_cols > 1 else width

    rows, cur = [], []
    for b in blocks:
        if not b:
            continue
        if b.full_width or max_cols == 1 or b.natural_width() > cell:
            if cur:
                rows.append(cur)
                cur = []
            rows.append([b])
            continue
        if b.new_row and cur:
            rows.append(cur)
            cur = []
        trial = cur + [b]
        if len(trial) <= max_cols:
            cur = trial
            if len(cur) == max_cols:
                rows.append(cur)
                cur = []
        else:
            if cur:
                rows.append(cur)
            cur = [b]
    if cur:
        rows.append(cur)
    return rows


def compose(rows: list, width: int, c: C) -> list:
    """Render packed rows, aligning every multi-column row on one column width."""
    paired = [b for r in rows if len(r) > 1 for b in r]
    col_w = 0
    if paired:
        widest = max(b.natural_width() for b in paired)
        col_w = min(widest, (width - GUTTER) // 2)

    out: list = []
    for r in rows:
        if len(r) == 1:
            # full_width governs packing, not the rule: a wide block still ends
            # its underline at its own content rather than trailing dashes to
            # the edge of a very wide terminal.
            b = r[0]
            out.extend(b.lines(min(b.natural_width(), width), c))
        else:
            cols = [b.lines(col_w, c) for b in r]
            height = max(map(len, cols))
            padded = [cl + [""] * (height - len(cl)) for cl in cols]
            for line_set in zip(*padded, strict=True):
                line = ""
                for i, cell in enumerate(line_set):
                    if i:
                        line += " " * (col_w + GUTTER - vlen(line)) + cell
                    else:
                        line = cell
                out.append(line.rstrip())
        out.append("")
    return out


def hist_block(
    c: C, title: str, hist: dict, unit: str, width: int = 78, cap: float | None = None
) -> Block:
    b = Block(title, full_width=True)
    p50, p90, p99 = (quantile(hist, q, cap) for q in (0.5, 0.9, 0.99))
    b.row(
        "mean / p50 / p90 / p99",
        " · ".join(fmt_num(v, unit) for v in (mean(hist), p50, p90, p99)),
        f"n={fmt_num(hist.get('count'))}",
    )
    counts = per_bucket(hist)
    counts = [(le, n) for le, n in counts if n > 0]
    if not counts:
        return b
    peak = max(n for _, n in counts)
    labw = max(len(fmt_le(le, unit)) for le, _ in counts)
    barw = max(10, width - labw - 12)
    for le, n in counts:
        b.raw(
            f"{c.dim(fmt_le(le, unit).rjust(labw))} {c.good(bar(n / peak, barw, c))} "
            f"{c.dim(str(int(n)))}"
        )
    return b


# ---------------------------------------------------------------- render ----


def render(
    scalars: dict,
    hists: dict,
    url: str,
    c: C,
    show_hists: bool,
    history: list | None = None,
    now: float = 0.0,
    windows: list | None = None,
    width: int = 100,
    max_cols: int = 1,
) -> str:
    g = scalars.get
    # A 24-wide bar reads well down a single column but eats a third of a
    # narrow one, and the number beside it carries the same information.
    barw = 24 if max_cols == 1 else 14
    blocks: list = []

    # --- live state
    live = Block("Live")
    blocks.append(live)
    processing, deferred = g("requests_processing"), g("requests_deferred")
    state = "idle" if not processing else f"{fmt_num(processing)} generating"
    if deferred:
        state += f", {fmt_num(deferred)} queued"
    live.row("state", state)

    kv_ratio, kv_tok, kv_max = (
        g("kv_cache_usage_ratio"),
        g("kv_cache_tokens"),
        g("kv_cache_max_tokens"),
    )
    if kv_ratio is not None:
        paint = c.good if kv_ratio < 0.75 else (c.warn if kv_ratio < 0.9 else c.bad)
        live.row(
            "KV cache",
            f"{paint(bar(kv_ratio, barw, c))} {fmt_num(kv_ratio, '%')}",
            f"{fmt_num(kv_tok, 'tok')} / {fmt_num(kv_max, 'tok')}",
        )

    # Offloaded pages are headroom, not load: they cost RAM rather than VRAM and
    # are what a displaced request comes back from. Shown next to the VRAM cache
    # because the two together are the context the server can still resume.
    off_max = g("kv_offload_max_tokens")
    if off_max:
        off_ratio = g("kv_offload_usage_ratio") or 0.0
        live.row(
            "KV offload",
            f"{c.good(bar(off_ratio, barw, c))} {fmt_num(off_ratio, '%')}",
            f"{fmt_num(g('kv_offload_tokens'), 'tok')} / {fmt_num(off_max, 'tok')} in RAM",
        )

    if g("n_tokens_max") is not None:
        live.row("largest ctx seen", fmt_num(g("n_tokens_max"), "tok"))
    # The peaks the size histograms cannot show: a lone big completion sits
    # above every percentile they report.
    if g("tokens_predicted_max") is not None:
        live.row("largest generation", fmt_num(g("tokens_predicted_max"), "tok"))

    # --- throughput
    # Work done and time spent, for both phases, and deliberately no rate for
    # either. Both times are summed per request, so under concurrency they cover
    # more wall clock than actually elapsed, and tokens over such a span is the
    # speed one stream saw while sharing the batch -- a figure that sags as load
    # rises on unchanged hardware and is invariably misread as the model getting
    # slower. Prefill would be worse still: a prefix cache hit takes tokens out
    # of the count while leaving its lookup time in the span, so the ratio decays
    # towards nonsense as reuse accumulates. Real rates are wall-clock ones, and
    # they live in the staggered windows under --watch; for a per-stream speed
    # worth comparing across configs, measure at concurrency 1 or use
    # exllamav3's eval/perf.py.
    thr = Block("Throughput (cumulative)")
    blocks.append(thr)
    thr.row(
        "prefill work",
        f"{fmt_num(g('prompt_tokens_total'), 'tok')} computed",
        f"over {fmt_num(g('prompt_seconds_total'), 's')}",
    )
    thr.row(
        "decode work",
        f"{fmt_num(g('tokens_predicted_total'), 'tok')} generated",
        f"over {fmt_num(g('tokens_predicted_seconds_total'), 's')}",
    )
    thr.row("requests finished", fmt_num(g("requests_total")))

    # --- where the time goes
    # The question this answers is which phase to buy speed in. A parallelism
    # layout trades prefill against decode -- tensor parallel favours decode,
    # pipeline parallel favours prefill -- and the right choice depends entirely
    # on which one this server actually spends its time in, which is a property
    # of the workload rather than of the hardware.
    #
    # The two phase counters are summed per request, so under concurrency they
    # add up to more than the wall clock they ran in. That does not affect the
    # split, which is what the decision turns on, but it does mean the share is
    # of engine time rather than of elapsed time, and it is labelled as such
    # when the two disagree.
    prefill_s = g("prompt_seconds_total") or 0.0
    decode_s = g("tokens_predicted_seconds_total") or 0.0
    engine_s = prefill_s + decode_s
    if engine_s > 0:
        # process_start_time_seconds is epoch-based, following the convention,
        # so it is compared against wall clock rather than the monotonic clock
        # the rate windows use.
        started = g("process_start_time_seconds")
        uptime = (time.time() - started) if started else None

        tb = Block("Where the time goes", new_row=True)
        blocks.append(tb)

        if uptime:
            # Clamped, because summed request time can exceed the interval it
            # ran in and a utilization above 100% reads as a bug rather than as
            # concurrency, which the note beside it names explicitly
            util = min(engine_s, uptime) / uptime
            tb.row(
                "engine busy",
                f"{fmt_num(engine_s, 's')} of {fmt_num(uptime, 's')} up",
                f"{fmt_num(util, '%')} utilization"
                + ("" if engine_s <= uptime else c.dim(" · overlapped")),
            )

        for label, value in (("prefill", prefill_s), ("decode", decode_s)):
            share = value / engine_s
            tb.row(
                label,
                f"{c.dim(bar(share, barw, c))} {fmt_num(share, '%')}",
                fmt_num(value, "s"),
            )

        # Queue time is not part of the split: a request waiting is the engine
        # working on another one, so it would double-count. It is worth seeing
        # because it is the part that more speed anywhere would relieve.
        queued = (hists.get("request_queue_time_seconds") or {}).get("sum")
        if queued:
            tb.row("queued behind others", fmt_num(queued, "s"), c.dim("not engine time"))

        # The trade a parallelism layout presents, as one number. Tensor
        # parallel buys decode and gives up prefill, pipeline parallel the
        # reverse, and no fixed speedup factor describes either, so what is
        # printed is the exchange rate between the two phases rather than the
        # result of some assumed swap.
        #
        # Cutting prefill time by a fraction x saves x*p of the engine's time;
        # inflating decode time by y costs y*d. Break-even is x*p == y*d, so one
        # percent off the winning phase pays for exactly (p/d) percent onto the
        # losing one. That ratio is the whole decision, and it is a property of
        # the workload rather than of any particular hardware layout.
        p, d = prefill_s / engine_s, decode_s / engine_s
        win, lose = ("prefill", "decode") if p >= d else ("decode", "prefill")
        hi, lo = (p, d) if p >= d else (d, p)

        if lo <= 0:
            tb.row("exchange rate", f"{win} is all of it", c.dim(f"{lose} costs nothing"))
        else:
            rate = hi / lo
            tb.row(
                "exchange rate",
                f"1% off {win} pays for {rate:,.0f}% onto {lose}"
                if rate >= 100
                else f"1% off {win} pays for {rate:.1f}% onto {lose}",
                c.dim(f"{fmt_num(hi, '%')} vs {fmt_num(lo, '%')}"),
            )

        # The marginal rate above is a linear reading and overstates the trade
        # for large moves, so the exact break-even is given for a concrete one:
        # after speeding the winning phase up by `factor`, the losing phase can
        # fall to lo / (1 - hi/factor) of its current speed before the swap
        # stops paying. Two factors, a cautious one and an ambitious one, bracket
        # what a layout change realistically buys.
        for factor in (1.5, 2.0) if lo > 0 else ():
            headroom = 1.0 - hi / factor
            if headroom <= 0:
                continue
            floor = lo / headroom
            if floor >= 1.0:
                verdict = c.bad(f"{lose} cannot give up anything")
            else:
                verdict = c.good(f"{lose} may fall {1 / floor:.0f}x")
            tb.row(
                f"  at {factor:g}x {win}",
                f"break-even at {fmt_num(floor, '%')} of {lose} speed",
                verdict,
            )

    # --- spec decode and prefix cache
    # Both are short, and "Where the time goes" beside them is not, so they are
    # stacked into one column under it rather than each claiming a row.
    sd = Block("Speculative decode")
    if g("spec_decode_requests_total"):
        acc = g("spec_decode_draft_acceptance_rate")
        if acc is not None:
            paint = c.good if acc > 0.7 else (c.warn if acc > 0.5 else c.bad)
            sd.row(
                "draft acceptance",
                f"{paint(bar(acc, barw, c))} {fmt_num(acc, '%')}",
                f"{fmt_num(g('spec_decode_num_accepted_tokens_total'), 'tok')} / "
                f"{fmt_num(g('spec_decode_num_draft_tokens_total'), 'tok')}",
            )
        tps = g("spec_decode_tokens_per_step")
        if tps is not None:
            sd.row("tokens / step", f"{tps:.3f}", f"+{(tps - 1) * 100:.0f}% vs. no drafter")
        sd.row(
            "decode steps",
            fmt_num(g("spec_decode_num_decode_steps_total")),
            f"{fmt_num(g('spec_decode_requests_total'))} req w/ drafter",
        )

    pc = Block("Prefix cache")
    queries, hits = g("prefix_cache_queries"), g("prefix_cache_hits")
    if queries:
        hr = (hits or 0) / queries
        paint = c.good if hr > 0.5 else (c.warn if hr > 0.2 else c.bad)
        pc.row(
            "hit rate",
            f"{paint(bar(hr, barw, c))} {fmt_num(hr, '%')}",
            f"{fmt_num(hits, 'tok')} / {fmt_num(queries, 'tok')}",
        )
        pc.row("prefilled", fmt_num(g("prompt_tokens_total"), "tok"), "= queried − cached")

    blocks.append(Stack(sd, pc))

    # --- KV offload
    # A restore is already counted as a prefix cache hit above, since the server
    # does not distinguish a page found in VRAM from one read back over PCIe.
    # This section is that breakdown, plus what the reuse is costing in RAM.
    if off_max:
        off = Block("KV offload", new_row=True)
        blocks.append(off)

        # The server does not count lookups that never reach RAM, so the reuse
        # figure is taken against the prefix cache hits above: what share of the
        # prompt the cache saved came back over PCIe rather than being found in
        # VRAM. A low share is not a fault on its own — it means VRAM is serving
        # the working set, which is the better outcome — so it is not painted as
        # one. What it does bound is how much the offload cache is contributing.
        restored = g("kv_offload_restored_tokens_total") or 0
        cache_hits = g("prefix_cache_hits") or 0
        share = restored / cache_hits if cache_hits else None
        if share is not None:
            off.row(
                "share of cache hits",
                f"{c.dim(bar(share, barw, c))} {fmt_num(share, '%')}",
                f"{fmt_num(restored, 'tok')} of {fmt_num(cache_hits, 'tok')} from RAM",
            )

        # The whole capacity is pinned up front in the background, so max is the
        # RAM this feature costs whatever the usage figure reads.
        note = ""
        cold = g("kv_offload_cold_allocs_total")
        if cold:
            note = c.warn(f"{fmt_num(cold)} stores pinned synchronously")
        held, pinned = g("kv_offload_bytes"), g("kv_offload_max_bytes")
        off.row("RAM in use", f"{fmt_num(held, 'B')} / {fmt_num(pinned, 'B')} pinned", note)

        # Once the cache is full every store has to evict something, so the
        # eviction count climbing towards the store count says only that it is
        # saturated -- which is the steady state of a cache that is being used,
        # not a fault. Whether saturation is costing anything is a question about
        # traffic: pages written out and never read back are the wasted ones, and
        # bytes in each direction are counted exactly.
        stores, evictions = g("kv_offload_stores_total"), g("kv_offload_evictions_total")
        written = g("kv_offload_written_bytes_total") or 0
        read = g("kv_offload_read_bytes_total") or 0
        payback = read / written if written else None
        if stores:
            note = ""
            if evictions and payback is not None and payback < 0.25:
                note = c.warn("thrashing — raise sysmem_kv_cache")
            deduped = g("kv_offload_deduped_stores_total") or 0
            detail = f"{fmt_num(stores)} stored, {fmt_num(evictions)} evicted"
            if deduped:
                detail += f", {fmt_num(deduped)} deduped"
            off.row("churn", detail, note)

        off.row(
            "traffic",
            f"{fmt_num(read, 'B')} read · {fmt_num(written, 'B')} written",
            f"{payback:.2f}x read back" if payback is not None else "",
        )

    # --- recurrent checkpoints (hybrid models only)
    # Prompt reuse is capped at the longest prefix with both valid K/V pages and
    # a matching recurrent checkpoint, so an undersized cache here silently
    # defeats the KV cache above — and with offloading on, wastes the PCIe read
    # that fetched the pages first. capped is that waste, in tokens.
    rec_max = g("recurrent_cache_max_bytes")
    capped = g("recurrent_capped_tokens_total") or 0
    if rec_max or capped:
        rec = Block("Recurrent state")
        blocks.append(rec)
        rr = g("recurrent_cache_usage_ratio") or 0.0
        # High usage is not a problem in itself; eviction under pressure is, and
        # that shows up in the capped row below.
        rec.row(
            "cache",
            f"{c.good(bar(rr, barw, c))} {fmt_num(rr, '%')}",
            f"{fmt_num(g('recurrent_cache_bytes'), 'B')} / {fmt_num(rec_max, 'B')}",
        )
        ckpt = g("recurrent_checkpoint_bytes")
        rec.row(
            "checkpoints",
            f"{fmt_num(g('recurrent_checkpoints'))} held",
            f"{fmt_num(ckpt, 'B')} each" if ckpt else "",
        )
        # Evictions on their own say nothing: a checkpoint whose anchor K/V page
        # is already gone could never have been resumed, so dropping it is free,
        # and the same for one reclaimed while idle. The costly ones are those
        # dropped while their anchor page was still cached — those are the drops
        # that become capped tokens below.
        ev = g("recurrent_cache_evictions_total") or 0
        costly = g("recurrent_cache_live_kv_evictions_total") or 0
        free = (g("recurrent_cache_stranded_evictions_total") or 0) + (
            g("recurrent_cache_pruned_total") or 0
        )
        if ev:
            note = c.warn(f"{fmt_num(costly)} dropped live") if costly else c.dim("all free")
            rec.row("evictions", f"{fmt_num(ev)}, {fmt_num(free)} free", note)

        # The other direction: the K/V cache is the one under pressure, and
        # evicting a page stranded the checkpoint anchored to it. Raising the
        # recurrent budget cannot help with these.
        by_kv = g("recurrent_stranded_by_kv_total")
        if by_kv:
            rec.row("stranded by KV eviction", fmt_num(by_kv), c.dim("KV cache is the constraint"))

        # The headline failure: valid K/V that was re-prefilled anyway. Whether a
        # bigger budget would have prevented it is a separate question, and the
        # answer is no unless something was actually dropped. Capping with an
        # empty, never-evicted cache means no checkpoint covered those pages in
        # the first place -- a prefix nothing ever generated past, or one whose
        # checkpoint predates this process -- and no amount of RAM fixes that.
        hits = g("prefix_cache_hits") or 0
        note = ""
        if capped:
            share = capped / (capped + hits) if (capped + hits) else 1.0
            under_pressure = bool(costly) or (bool(ev) and rr > 0.9)
            if share <= 0.05:
                note = c.dim("negligible")
            elif under_pressure:
                note = c.bad("raise sysmem_recurrent_cache")
            elif by_kv:
                note = c.warn("KV cache is the constraint, not this one")
            else:
                note = c.dim("no checkpoint covered them — not a budget problem")
        rec.row("KV wasted by cap", fmt_num(capped, "tok"), note)

    # --- latency and size summaries
    lat = [
        ("time to first token", "time_to_first_token_seconds", "s"),
        ("queue wait", "request_queue_time_seconds", "s"),
        ("prefill time", "request_prefill_time_seconds", "s"),
        ("decode time", "request_decode_time_seconds", "s"),
        ("end-to-end latency", "e2e_request_latency_seconds", "s"),
    ]
    # The size histograms have a companion peak counter, so their percentiles
    # can be held to a value that was actually observed. The latency ones have
    # no such counter and are left as the estimator reports them.
    sizes = [
        ("prompt size", "request_prompt_tokens", "tok", g("n_tokens_max")),
        ("generation size", "request_generation_tokens", "tok", g("tokens_predicted_max")),
    ]
    lat = [(lbl, key, unit, None) for lbl, key, unit in lat]
    for title, spec in (("Latency  (mean·p50·p90·p99)", lat), ("Size  (mean·p50·p90·p99)", sizes)):
        present = [(lbl, hists[k], u, cap) for lbl, k, u, cap in spec if k in hists]
        if not present:
            continue
        b = Block(title)
        blocks.append(b)
        for lbl, h, u, cap in present:
            qs = [quantile(h, q) for q in (0.5, 0.9, 0.99)]
            capped = cap is not None and any(q is not None and q > cap for q in qs)
            values = [mean(h)] + [quantile(h, q, cap) for q in (0.5, 0.9, 0.99)]
            b.row(
                lbl,
                " · ".join(fmt_num(v, u) for v in values),
                c.dim("capped at peak") if capped else "",
            )

    # --- windowed rates (--watch only)
    # Counter deltas over elapsed wall clock. Nothing here divides by a
    # per-request span, so prefix cache hits cannot skew it: a request that hits
    # the cache contributes few tokens and little wall clock alike. This is
    # throughput of the deployment, not speed of the model — an idle server
    # reads near zero however fast its prefill is.
    #
    # Each column differences the current counters against the newest sample at
    # least that old, and divides by the true elapsed time rather than the
    # nominal window, so a scrape delayed by a busy event loop still yields a
    # correct figure. Columns fill in from the left as the session gets longer;
    # the widest one the session has not outlived yet covers all of history so
    # far instead, under a `~`-prefixed header naming its true span.
    if history and windows:
        rates = Block("Rates over staggered windows", full_width=True)
        blocks.append(rates)

        # The leftmost column is the delta since the previous scrape, headed by
        # that interval. It is a liveness indicator rather than a throughput
        # figure: the counters only advance when a request finishes, so a long
        # prefill leaves every column at zero and then lands its whole prompt in
        # one tick. The wider windows are the ones to read for a rate.
        columns = resolve_columns(history, now, [("now", 0.0)] + list(windows))
        head = "".join(w.rjust(RATE_COL) for w, _ in columns)
        rates.raw(f"{c.dim('window'.ljust(26))} {c.dim(head)}")

        # Byte counters are scaled to MB before rating, since fmt_rate is fixed
        # width and PCIe traffic in raw bytes/s would render as "6000.0M".
        for label, key, scale in (
            # "computed" is prefill actually performed; "cache hits" is prefill
            # skipped. They sum to the rate prompt tokens are being ingested,
            # and only the first is work the server did. Naming the second
            # "cached" invited reading it as tokens going *into* the cache.
            ("prompt computed (tok/s)", "prompt_tokens_total", 1.0),
            ("prompt cache hits (tok/s)", "cached_tokens_total", 1.0),
            ("generation (tok/s)", "tokens_predicted_total", 1.0),
            ("requests (/s)", "requests_total", 1.0),
            # Restored against the read/written pair below separates a cache
            # that is idle from one that is being written to and never read
            # back. Both read zero restores; only the second is a fault.
            ("offload restored (tok/s)", "kv_offload_restored_tokens_total", 1.0),
            ("offload read (MB/s)", "kv_offload_read_bytes_total", 1 / 1024**2),
            ("offload written (MB/s)", "kv_offload_written_bytes_total", 1 / 1024**2),
        ):
            cur = g(key)
            if cur is None:
                continue
            if key.startswith("kv_offload_") and not off_max:
                continue
            cells = []
            for _, base in columns:
                if base is None or base[1].get(key) is None:
                    cells.append("—".rjust(RATE_COL))
                    continue
                ts, prev = base
                cells.append(fmt_rate((cur - prev[key]) * scale / (now - ts)).rjust(RATE_COL))
            rates.raw(f"{c.key(label.ljust(26))} {c.val(''.join(cells))}")

    if show_hists:
        for lbl, key, u, cap in lat + sizes:
            if key in hists:
                blocks.append(hist_block(c, f"Distribution — {lbl}", hists[key], u, cap=cap))
        if "spec_decode_acceptance_rate" in hists:
            blocks.append(
                hist_block(
                    c,
                    "Distribution — spec-decode acceptance rate",
                    hists["spec_decode_acceptance_rate"],
                    "",
                )
            )

    out = [c.head(f"TabbyAPI metrics  {c.dim(url)}"), ""]
    out.extend(compose(pack(blocks, width, max_cols), width, c))
    return "\n".join(out)


def digest(scalars: dict, hists: dict) -> dict:
    out = {"scalars": scalars, "histograms": {}}
    for name, h in hists.items():
        out["histograms"][name] = {
            "count": h.get("count"),
            "sum": h.get("sum"),
            "mean": mean(h),
            "p50": quantile(h, 0.5),
            "p90": quantile(h, 0.9),
            "p99": quantile(h, 0.99),
        }
    queries, hits = scalars.get("prefix_cache_queries"), scalars.get("prefix_cache_hits")
    out["derived"] = {"prefix_cache_hit_rate": (hits / queries) if queries else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Pretty-print TabbyAPI /metrics.")
    ap.add_argument("--url", help="full metrics URL (overrides --host/--port)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--watch", type=float, metavar="SEC", help="refresh every SEC seconds")
    ap.add_argument("--hist", action="store_true", help="also draw bucket distributions")
    ap.add_argument(
        "--columns",
        type=int,
        default=0,
        metavar="N",
        help="section columns; 0 auto-fits the terminal (default 0)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=0,
        metavar="COLS",
        help="assume this terminal width instead of detecting it",
    )
    ap.add_argument("--json", action="store_true", help="emit a JSON digest instead")
    # Generous by default: a scrape competing with a large prefill has been
    # measured waiting 8s, and waiting costs nothing when the server is idle.
    ap.add_argument(
        "--timeout", type=float, default=30.0, metavar="SEC", help="scrape timeout (default 30)"
    )
    ap.add_argument(
        "--windows",
        type=parse_windows,
        default=RATE_WINDOWS_DEFAULT,
        metavar="LIST",
        help="lookback windows for the --watch rate table, as a "
        f"comma-separated duration list (default {RATE_WINDOWS_DEFAULT})",
    )
    args = ap.parse_args()

    # argparse leaves a default string untouched, so parse it here when unset.
    windows = args.windows
    if isinstance(windows, str):
        windows = parse_windows(windows)

    # A window shorter than the refresh interval can never be filled, so drop
    # it rather than showing a permanently blank column. A window equal to the
    # interval is dropped too: it differences against the previous scrape, so it
    # only ever duplicates the "now" column. Keep the longest if that would
    # empty the table.
    if args.watch:
        kept = [w for w in windows if w[1] > args.watch]
        windows = kept or windows[-1:]

    if hasattr(sys.stdout, "reconfigure"):
        # Line buffering too, or --watch writes nothing at all until it is
        # killed when its output is a pipe rather than a terminal.
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    url = args.url or f"http://{args.host}:{args.port}/metrics"
    c = C(color_on() and not args.json)

    def geometry():
        # Re-read on every frame so --watch reflows when the window is resized.
        # Piped output has no size to detect, so fall back to a width that keeps
        # two columns viable rather than to the 80 the OS reports.
        width = args.width or shutil.get_terminal_size((160, 24)).columns
        if args.columns:
            return width, args.columns
        # Two columns need room for the widest pair plus the gutter. Below that
        # the sections stack, which is the old single-column layout.
        return width, 2 if width >= 150 else 1

    # Scrape history, so --watch can show rates over a range of lookbacks.
    history: list = []

    def once(tolerant: bool = False) -> int:
        # TabbyAPI renders /metrics on the same event loop that drives the
        # generator, and the generator holds the loop through each prefill
        # chunk, so a scrape landing mid-inference waits for the chunk to
        # finish. Measured on a 27B model at 4096 chunk size: 7ms idle, but a
        # median of 6.8s and a peak of 8.1s during a 70k-token prefill. Hence
        # the long default timeout, and hence --watch treating a failure as a
        # skipped frame rather than a reason to quit.
        started = time.monotonic()
        try:
            text = scrape(url, args.timeout)
        except urllib.error.HTTPError as e:
            print(
                f"error: {url} returned HTTP {e.code}\n"
                f"Fix: TabbyAPI serves /metrics only when network.enable_metrics is "
                f"true in config.yml.",
                file=sys.stderr,
            )
            return 2
        except (urllib.error.URLError, OSError) as e:
            if tolerant:
                waited = time.monotonic() - started
                print(c.dim(f"  scrape failed after {waited:.1f}s ({e}) — retrying"))
                return 0
            print(
                f"error: cannot reach {url} ({e})\n"
                f"Fix: check TabbyAPI is up — `curl -s {url} | head`",
                file=sys.stderr,
            )
            return 2

        scalars, hists = parse(text)
        if not scalars and not hists:
            print(f"error: no tabbyapi: metrics found at {url}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(digest(scalars, hists), indent=2))
        else:
            # Timestamp on receipt. The server renders its counters immediately
            # before responding, so this is when the snapshot was taken, and a
            # scrape delayed by a busy loop still yields a correct window.
            now = time.monotonic()
            width, cols = geometry()
            print(
                render(
                    scalars,
                    hists,
                    url,
                    c,
                    args.hist,
                    history,
                    now,
                    windows,
                    width=width,
                    max_cols=cols,
                )
            )
            history.append((now, scalars))
            prune_history(history, now, windows[-1][1])
            blocked = now - started
            if blocked > 1.0:
                print(
                    c.dim(
                        f"  this scrape waited {blocked:.1f}s for the server to "
                        f"come up for air (inference holds the event loop)"
                    )
                )
        return 0

    if args.watch:
        try:
            while True:
                sys.stdout.write("\033[H\033[J" if c.e else "")
                rc = once(tolerant=True)
                if rc:
                    return rc
                print(c.dim(f"  refreshing every {args.watch:g}s — Ctrl-C to stop"))
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0
    return once()


if __name__ == "__main__":
    sys.exit(main())
