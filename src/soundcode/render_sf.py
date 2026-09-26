"""Sampled renderer: .sc -> audio through a General MIDI SoundFont.

Each stream is rendered on its own (so `compare` can score it against its
source stem), then mixed. Pitch: chords round to the nearest semitone; a
monophonic line keeps its cents offset as pitch bend, so tuning survives.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import gm
from .expand import Note

BEND_RANGE_SEMITONES = 2
_ORDER = {"off": 0, "bend": 1, "on": 2}


@dataclass(frozen=True, order=True)
class MidiEvent:
    time: float
    order: int
    kind: str
    key: int = 0
    value: int = 0


def _event(time: float, kind: str, key: int = 0, value: int = 0) -> MidiEvent:
    return MidiEvent(time, _ORDER[kind], kind, key, value)


def stream_events(notes: list[Note], target: gm.Target) -> list[MidiEvent]:
    mono = target.family in gm.MONO_FAMILIES
    out: list[MidiEvent] = []
    for n in notes:
        if target.drums:
            key = gm.drum_note(n.voice or "")
        elif n.cents is None:
            continue
        else:
            key = int(round(n.cents / 100.0))
            if mono:
                offset = (n.cents - key * 100) / 100.0          # semitones
                value = 8192 + round(offset / BEND_RANGE_SEMITONES * 8192)
                out.append(_event(n.start, "bend", value=max(0, min(16383, value))))
        key = max(0, min(127, key))
        out.append(_event(n.start, "on", key, max(1, min(127, n.vel))))
        out.append(_event(n.start + max(n.dur, 0.02), "off", key))
    return out
