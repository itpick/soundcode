"""tsumugi MIDI -> .sc stream lines.

Positions keep the performed timing (3-decimal beats), not a quantised grid:
the point of the file is a near-original rebuild. Notes before the first
downbeat are written in absolute seconds rather than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import gm
from .pitch import cents_to_name

BEATS_PER_BAR = 4


@dataclass
class Track:
    klass: str
    inst: str
    notes: list[tuple[float, float, int, int]] = field(default_factory=list)


def read_tracks(midi: Path) -> list[Track]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi))
    out = []
    for inst in pm.instruments:
        if not inst.notes:
            continue
        klass = "drums" if inst.is_drum else gm.tsumugi_class_for_program(inst.program)
        notes = sorted((n.start, n.end, n.pitch, n.velocity) for n in inst.notes)
        out.append(Track(klass, gm.TSUMUGI[klass]["inst"], notes))
    return out


def position(t: float, grid: dict) -> str:
    down, bar_dur = grid["downbeat"], grid["bar_dur"]
    if t < down - 1e-6:
        return f"@{t:.3f}"
    beats = (t - down) / (bar_dur / BEATS_PER_BAR)
    beats = round(beats, 3)
    bar, beat = divmod(beats, BEATS_PER_BAR)
    return f"{int(bar) + 1}:{beat + 1:.3f}"


def note_lines(track: Track, grid: dict) -> list[str]:
    beat_s = grid["bar_dur"] / BEATS_PER_BAR
    lines = []
    for start, end, pitch, vel in track.notes:
        pos = position(start, grid)
        if track.klass == "drums":
            lines.append(f"{pos} {gm.drum_voice(pitch)} {vel}")
        else:
            dur = max(end - start, 0.01) / beat_s
            lines.append(f"{pos}  {cents_to_name(pitch * 100)}  {dur:.3f}b {vel}")
    return lines


def drop_bleed(tracks: list[Track], min_notes: int = 3, min_share: float = 0.02,
               min_sound_s: float = 2.0) -> tuple[list[Track], list[str]]:
    """Drop tracks that are separation bleed. A track is kept when it has
    enough notes *or* enough sounding time: a held drone is one note of music."""
    def sounding(t: Track) -> float:
        return sum(max(e - s, 0.0) for s, e, _, _ in t.notes)

    total = sum(sounding(t) for t in tracks) or 1.0
    kept, why = [], []
    for t in tracks:
        n, dur = len(t.notes), sounding(t)
        few = n < min_notes and dur < min_sound_s
        if few or dur / total < min_share:
            why.append(f"{t.klass}: bleed ({n} notes, {dur:.1f}s, {dur / total:.0%} of stem)")
        else:
            kept.append(t)
    return kept, why


def merge_same_inst(tracks: list[Track]) -> list[Track]:
    """One track per instrument: tsumugi may split a stem into several tracks
    that refinement then relabels to the same class."""
    out: dict[str, Track] = {}
    for t in tracks:
        if t.inst in out:
            out[t.inst].notes = sorted(out[t.inst].notes + t.notes)
        else:
            out[t.inst] = Track(t.klass, t.inst, sorted(t.notes))
    return list(out.values())


def stream_name(inst: str, stem: str, taken: set[str]) -> str:
    short = inst.split(".")[1] if "." in inst else inst
    name = f"notes.{short}"
    if name in taken:
        name = f"notes.{short}.{stem}"
    n = 2
    base = name
    while name in taken:
        name, n = f"{base}{n}", n + 1
    taken.add(name)
    return name
