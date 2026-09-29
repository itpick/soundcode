"""Slices: sections, 20 s windows, the whole song, and the activity gate.

Spec 2026-09-28-benchmark-scorer, Task 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..expand import build_grid
from ..model import Document


@dataclass
class Slice:
    """A span of the song to score: a section, a 20 s window, or the song."""

    kind: str       # "section", "window" or "song"
    label: str
    a: float        # seconds, inclusive
    b: float        # seconds, exclusive


def _mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=0) if y.ndim > 1 else y


def song(duration: float) -> Slice:
    """The whole song as a single slice."""
    return Slice("song", "song", 0.0, duration)


def windows(duration: float, length: float = 20.0, hop: float = 10.0) -> list[Slice]:
    """20 s windows every `hop` seconds, covering the song; the last is clamped."""
    out: list[Slice] = []
    start = 0.0
    while start < duration:
        end = min(start + length, duration)
        label = f"{int(start // 60)}:{int(start % 60):02d}"
        out.append(Slice("window", label, start, end))
        if end >= duration:
            break
        start += hop
    return out


def _struct_segments(doc: Document) -> list[tuple[int, int, str]]:
    """(first_bar, last_bar, label) triples from :struct, sorted by first bar."""
    struct = doc.stream("struct")
    if struct is None:
        return []
    segments: list[tuple[int, int, str]] = []
    for label, args in struct.statements:
        if not args or "-" not in args[0]:
            continue
        lo_s, hi_s = args[0].split("-", 1)
        try:
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            continue
        segments.append((lo, hi, label))
    segments.sort(key=lambda seg: seg[0])
    return segments


def _bar_windows(grid, duration: float, bars: int = 8) -> list[Slice]:
    """Fallback: cut `bars`-bar windows up to duration, clamping the last."""
    out: list[Slice] = []
    bar_lo = 1
    while True:
        bar_hi = bar_lo + bars - 1
        a = grid.time_of(bar_lo, 1.0)
        if a >= duration:
            break
        b = min(grid.time_of(bar_hi + 1, 1.0), duration)
        out.append(Slice("section", f"bars {bar_lo}-{bar_hi}", a, b))
        if b >= duration:
            break
        bar_lo += bars
    return out


def sections(doc: Document, duration: float) -> list[Slice]:
    """Sections from :struct when it names ≥ 2 distinct labels, else 8-bar windows.

    Each section's end is `min(this segment's own hi+1 bar time, the next
    segment's start time)` — a segment never stretches past its own declared
    range to close a gap against its neighbor. If :struct leaves a gap
    between two bar ranges (or before/after the whole set), that gap is left
    uncovered rather than absorbed into an adjacent section.
    """
    grid = build_grid(doc)
    segments = _struct_segments(doc)
    labels = {label for _, _, label in segments}
    if len(labels) < 2:
        return _bar_windows(grid, duration)

    out: list[Slice] = []
    for i, (lo, hi, label) in enumerate(segments):
        a = grid.time_of(lo, 1.0)
        b = grid.time_of(hi + 1, 1.0)
        if i + 1 < len(segments):
            b = min(b, grid.time_of(segments[i + 1][0], 1.0))
        b = min(b, duration)
        if b > a:
            out.append(Slice("section", label, a, b))
    return out


# Amendment 2 (2026-09-29): a single loud 100 ms frame in an otherwise
# silent slice ("keys intro, score 0" on full Discipline: the original
# keys stem was at -84 dBFS there, but one click made the whole 25 s slice
# count as active) is a false alarm, not activity. A slice is active only
# when a real fraction of it is loud, and the slice as a whole isn't
# effectively silent.
GATE_DB = -50.0
MIN_ACTIVE_FRACTION = 0.10
SLICE_FLOOR_DB = -60.0


def active(y: np.ndarray, sr: int, a: float, b: float, gate_db: float = GATE_DB,
          min_fraction: float = MIN_ACTIVE_FRACTION, floor_db: float = SLICE_FLOOR_DB) -> bool:
    """True iff the fraction of 100 ms frames in [a, b) with RMS >= gate_db
    dBFS is >= min_fraction AND the whole slice's RMS is >= floor_db.

    Both conditions are needed: the fraction rule alone would still call a
    slice active if one enormous click also happened to be loud enough to
    drag the whole-slice RMS up, and the floor rule alone would still call
    a slice active if it were sustained but too quiet to be real signal.
    """
    mono = _mono(y)
    i0 = max(int(a * sr), 0)
    i1 = min(int(b * sr), mono.shape[0])
    seg = mono[i0:i1]
    if seg.size == 0:
        return False
    slice_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    if slice_rms <= 0.0 or 20.0 * np.log10(slice_rms) < floor_db:
        return False
    frame = max(int(round(0.1 * sr)), 1)
    n_frames = 0
    n_active = 0
    for start in range(0, seg.size, frame):
        chunk = seg[start:start + frame]
        if chunk.size == 0:
            continue
        n_frames += 1
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms > 0.0 and 20.0 * np.log10(rms) >= gate_db:
            n_active += 1
    if n_frames == 0:
        return False
    return (n_active / n_frames) >= min_fraction
