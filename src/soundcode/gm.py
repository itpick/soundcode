"""General MIDI targets for .sc streams.

Plan 1 maps at family level (`bass.electric` -> the bass program). Plan 2
replaces FAMILY_PROGRAM with the full instrument taxonomy's render targets.
"""

from __future__ import annotations

from dataclasses import dataclass

# GM programs, 0-indexed, as GeneralUser GS names them
FAMILY_PROGRAM = {
    "keys": 0,        # Acoustic Grand Piano
    "mallet": 11,     # Vibraphone
    "organ": 16,      # Drawbar Organ
    "gtr": 27,        # Clean Guitar
    "bass": 33,       # Finger Bass
    "strings": 48,    # String Ensemble
    "voice": 53,      # Voice Oohs
    "brass": 61,      # Brass Section
    "winds": 65,      # Alto Sax
    "synth": 89,      # Warm Pad
    "unknown": 0,
}
# stream track name -> family, for files whose streams carry no inst=
STREAM_FAMILY = {"bass": "bass", "vox": "voice", "bvox": "voice",
                 "guitar": "gtr", "piano": "keys"}
DRUM_NOTES = {"kick": 36, "snare": 38, "hat": 42, "hat.open": 46, "crash": 49,
              "ride": 51, "tom.hi": 50, "tom.mid": 47, "tom.lo": 43}
DRUM_DEFAULT = 39   # hand clap: audible, and obviously "unmapped"
MONO_FAMILIES = frozenset({"voice"})


@dataclass(frozen=True)
class Target:
    bank: int
    preset: int
    drums: bool
    family: str


def target_for(stream_name: str, inst: str) -> Target:
    if stream_name.startswith("perc.") or inst.startswith(("drums", "perc")):
        return Target(128, 0, True, "drums")
    family = inst.split(".")[0] if inst and inst != "unknown" else \
        STREAM_FAMILY.get(stream_name.split(".")[-1], "unknown")
    if family not in FAMILY_PROGRAM:
        family = "unknown"
    return Target(0, FAMILY_PROGRAM[family], False, family)


def drum_note(voice: str) -> int:
    return DRUM_NOTES.get(voice, DRUM_DEFAULT)
