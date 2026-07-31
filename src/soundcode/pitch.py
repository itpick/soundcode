"""Pitch: cents are canonical, note names are sugar.

Canonical unit is `cents`, defined as 100 x MIDI note number, so A4 = 6900c
and A3+8c = 5708c. This keeps the integer readable next to a note name while
letting any microtonal value be expressed exactly.
"""

from __future__ import annotations

import math
import re

# semitone offset of each letter from C
_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# A3+8c / F1-6c / Bb2 / C#4 — octave is a single digit, so the optional
# cent offset that follows can never be mistaken for part of it.
_NOTE_RE = re.compile(r"^([A-G])([#b]?)(\d)([+-]\d+)?c?$")
_RAWCENT_RE = re.compile(r"^(-?\d+)c$")
_HZ_RE = re.compile(r"^~([\d.]+)Hz$", re.IGNORECASE)


class PitchError(ValueError):
    pass


def name_to_cents(text: str) -> int:
    """`A3` -> 5700, `A3+8c` -> 5708, `F1-6c` -> 2894."""
    m = _NOTE_RE.match(text)
    if not m:
        raise PitchError(f"not a note name: {text!r}")
    letter, accidental, octave, offset = m.groups()
    semis = _LETTER[letter]
    if accidental == "#":
        semis += 1
    elif accidental == "b":
        semis -= 1
    # MIDI: C-1 is 0, so C4 = 60 -> (octave + 1) * 12
    midi = (int(octave) + 1) * 12 + semis
    return midi * 100 + (int(offset) if offset else 0)


def hz_to_cents(hz: float, ref_hz: float = 440.0, ref_cents: int = 6900) -> float:
    if hz <= 0:
        raise PitchError(f"non-positive frequency: {hz}")
    return ref_cents + 1200.0 * math.log2(hz / ref_hz)


def cents_to_hz(cents: float, ref_hz: float = 440.0, ref_cents: int = 6900) -> float:
    return ref_hz * (2.0 ** ((cents - ref_cents) / 1200.0))


def parse_pitch(text: str, ref_hz: float = 440.0, ref_cents: int = 6900) -> float:
    """Accept any of the three notations and return canonical cents."""
    m = _RAWCENT_RE.match(text)
    if m:
        return float(m.group(1))
    m = _HZ_RE.match(text)
    if m:
        return hz_to_cents(float(m.group(1)), ref_hz, ref_cents)
    return float(name_to_cents(text))


def cents_to_name(cents: float) -> str:
    """Inverse of `name_to_cents`, preferring sharps. Used when re-emitting."""
    nearest = int(round(cents / 100.0))
    offset = int(round(cents - nearest * 100))
    octave, semis = divmod(nearest, 12)
    name = f"{_SHARP_NAMES[semis]}{octave - 1}"
    if offset:
        name += f"{offset:+d}c"
    return name
