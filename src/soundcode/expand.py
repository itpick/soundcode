"""Expand a Document into flat, fully-resolved events.

Everything symbolic is resolved here and nowhere else: patterns and `like`
bindings are expanded to explicit events, bar:beat becomes seconds, note names
become cents, `.` becomes the current chord root. Decode and evaluation always
run on expanded form (spec §4.5), so pattern factoring can never change what
the decoder hears.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import Document, Event, Stream
from .pitch import cents_to_hz, parse_pitch
from .timebase import Grid

_CHORD_ROOT_RE = re.compile(r"^([A-G][#b]?)")
_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
       "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


@dataclass
class Note:
    """A fully resolved sounding event."""

    start: float          # seconds, dev applied
    dur: float            # seconds
    cents: float | None   # None for percussion
    voice: str | None     # percussion voice name
    vel: int
    stream: str
    inst: str
    env: dict[str, float] | None = None
    vib: dict[str, float] | None = None

    @property
    def hz(self) -> float:
        return cents_to_hz(self.cents) if self.cents is not None else 0.0


def build_grid(doc: Document) -> Grid:
    s = doc.stream("grid")
    if s is None:
        return Grid()
    # collect into plain lists: Grid.__post_init__ injects defaults into
    # whatever lists it is handed, which would shadow the real values
    tempo: list[tuple[float, float]] = []
    meter: list[tuple[float, int, int]] = []
    anchors: list[tuple[int, float]] = []
    for name, args in s.statements:
        if name == "tempo" and len(args) >= 2:
            tempo.append((float(args[0].lstrip("@")), float(args[1])))
        elif name == "meter" and len(args) >= 2:
            num, den = args[1].split("/")
            meter.append((float(args[0].lstrip("@")), int(num), int(den)))
        elif name == "anchor" and len(args) >= 3:
            anchors.append((int(args[1]), float(args[2].lstrip("@"))))
    return Grid(tempo=tempo, meter=meter, anchors=anchors)


def build_chord_map(doc: Document, max_bar: int) -> dict[int, str]:
    """bar -> chord symbol, expanding `like`/`xN` reuse in :harmony."""
    out: dict[int, str] = {}
    s = doc.stream("harmony")
    if s is None:
        return out
    for b in s.bindings:
        if b.chords:
            span = b.last - b.first + 1
            for i in range(span):
                out[b.first + i] = b.chords[i % len(b.chords)]
        elif b.like:
            src = list(range(b.like[0], b.like[1] + 1))
            for i in range(b.last - b.first + 1):
                out[b.first + i] = out.get(src[i % len(src)], "")
    return out


def chord_root_cents(symbol: str, floor_midi: int = 28) -> float | None:
    """Root of a chord symbol, voiced at the lowest octave above `floor_midi`."""
    m = _CHORD_ROOT_RE.match(symbol or "")
    if not m:
        return None
    pc = _PC.get(m.group(1))
    if pc is None:
        return None
    midi = pc + 12
    while midi < floor_midi:
        midi += 12
    return midi * 100.0


def _instantiate(events: list[Event], bar: int) -> list[Event]:
    """Give bare-beat pattern events a concrete bar."""
    out = []
    for e in events:
        c = Event(**{**e.__dict__})
        if c.time.bar is None:
            c.time = type(c.time)(bar=bar, beat=c.time.beat)
        out.append(c)
    return out


def _parse_env(text: str) -> dict[str, float]:
    """`a 21ms, d 110ms, s 0.68, r 240ms` -> seconds (s stays a ratio)."""
    out: dict[str, float] = {}
    for part in text.split(","):
        toks = part.split()
        if len(toks) != 2:
            continue
        k, v = toks
        if v.endswith("ms"):
            out[k] = float(v[:-2]) / 1000.0
        elif v.endswith("s"):
            out[k] = float(v[:-1])
        else:
            out[k] = float(v)
    return out


def _parse_vib(text: str) -> dict[str, float]:
    """`rate 5.4Hz, depth ±31c, onset 0.22s`"""
    out: dict[str, float] = {}
    for part in text.split(","):
        toks = part.split()
        if len(toks) != 2:
            continue
        k, v = toks
        v = v.replace("±", "").replace("Hz", "").replace("c", "").replace("s", "")
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _stream_events(stream: Stream, grid: Grid) -> list[Event]:
    """Explicit events plus everything the bindings expand to."""
    events = list(stream.events)
    by_bar: dict[int, list[Event]] = {}
    for e in stream.events:
        if e.time.bar is not None:
            by_bar.setdefault(e.time.bar, []).append(e)

    for b in stream.bindings:
        if b.pattern and b.pattern in stream.patterns:
            for bar in range(b.first, b.last + 1):
                events.extend(_instantiate(stream.patterns[b.pattern], bar))
        elif b.like:
            src = list(range(b.like[0], b.like[1] + 1))
            for i in range(b.last - b.first + 1):
                for e in by_bar.get(src[i % len(src)], []):
                    c = Event(**{**e.__dict__})
                    c.time = type(c.time)(bar=b.first + i, beat=e.time.beat)
                    if b.vel_delta and c.vel is not None:
                        c.vel = max(1, min(127, c.vel + b.vel_delta))
                    events.append(c)
    return events


def expand(doc: Document) -> list[Note]:
    grid = build_grid(doc)
    max_bar = 1
    for s in doc.streams:
        for b in s.bindings:
            max_bar = max(max_bar, b.last)
        for e in s.events:
            if e.time.bar:
                max_bar = max(max_bar, e.time.bar)
    chords = build_chord_map(doc, max_bar)

    # contour verbs, keyed by (track, rounded start) so notes can pick them up
    contours: dict[tuple[str, float], dict[str, float]] = {}
    for s in doc.of_kind("contour"):
        for e in s.events:
            t = _seconds(e, grid)
            if "vib" in e.blocks:
                contours[(s.track or "", round(t, 3))] = _parse_vib(e.blocks["vib"])

    notes: list[Note] = []
    for s in doc.streams:
        if s.kind not in ("notes", "perc"):
            continue
        inst = s.fields.get("inst", "unknown")
        for e in _stream_events(s, grid):
            if e.atom is None:
                continue
            start = _seconds(e, grid) + e.dev_ms / 1000.0
            dur = _duration(e, grid)

            cents: float | None = None
            voice: str | None = None
            if s.kind == "perc":
                voice = e.atom
            elif e.atom == ".":
                bar = e.time.bar or 1
                cents = chord_root_cents(chords.get(bar, ""))
                if cents is None:
                    continue
            else:
                cents = parse_pitch(e.atom)

            notes.append(Note(
                start=start,
                dur=max(dur, 0.02),
                cents=cents,
                voice=voice,
                vel=e.vel if e.vel is not None else 90,
                stream=s.name,
                inst=inst,
                env=_parse_env(e.blocks["env"]) if "env" in e.blocks else None,
                vib=contours.get((s.track or "", round(start - e.dev_ms / 1000.0, 3))),
            ))

    notes.sort(key=lambda n: n.start)
    return notes


def _seconds(e: Event, grid: Grid) -> float:
    if e.time.is_absolute:
        return e.time.seconds or 0.0
    return grid.time_of(e.time.bar or 1, e.time.beat or 1.0)


def _duration(e: Event, grid: Grid) -> float:
    if not e.dur:
        return 0.25
    if e.dur.endswith("s"):
        return float(e.dur[:-1])
    beats = float(e.dur[:-1])
    if e.time.is_absolute:
        return beats * 0.5
    return grid.duration_seconds(e.time.bar or 1, e.time.beat or 1.0, beats)
