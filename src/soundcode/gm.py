"""General MIDI targets for .sc streams.

Plan 1 maps at family level (`bass.electric` -> the bass program). Plan 2
replaces FAMILY_PROGRAM with the full instrument taxonomy's render targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
    "other": 0,
    "unknown": 0,
}
# stream track name -> family, for files whose streams carry no inst=
STREAM_FAMILY = {"bass": "bass", "vox": "voice", "bvox": "voice",
                 "guitar": "gtr", "piano": "keys"}
DRUM_NOTES = {"kick": 36, "snare": 38, "hat": 42, "hat.open": 46, "crash": 49,
              "ride": 51, "tom.hi": 50, "tom.mid": 47, "tom.lo": 43}
DRUM_DEFAULT = 39   # hand clap: audible, and obviously "unmapped"

# tsumugi's 36 instrument classes -> our inst vocabulary + a GM render program
TSUMUGI: dict[str, dict] = json.loads(
    (Path(__file__).parent / "data" / "tsumugi_classes.json").read_text())
_INST_PROGRAM = {e["inst"]: e["program"] for e in TSUMUGI.values() if e["family"] != "drums"}
_PROGRAM_CLASS = {p: name for name, e in TSUMUGI.items() for p in e["programs"]}

DRUM_NOTES.update({"kick.acoustic": 35, "stick": 37, "clap": 39, "snare.electric": 40,
                   "tom.floor.lo": 41, "hat.pedal": 44, "tom.lo.mid": 45, "tom.hi.mid": 48,
                   "crash.2": 57, "ride.bell": 53, "tamb": 54, "splash": 55, "cowbell": 56,
                   "ride.2": 59, "china": 52, "shaker": 70})
_DRUM_VOICE = {v: k for k, v in DRUM_NOTES.items()}
_DRUM_VOICE[35] = "kick"                       # both GM kicks read as a kick


def drum_voice(pitch: int) -> str:
    return _DRUM_VOICE.get(pitch, f"gm{pitch}")


def tsumugi_class_for_program(program: int) -> str:
    return _PROGRAM_CLASS.get(program, "piano")
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
    if inst in _INST_PROGRAM:
        fam = inst.split(".")[0]
        return Target(0, _INST_PROGRAM[inst], False, fam if fam in FAMILY_PROGRAM else "unknown")
    family = inst.split(".")[0] if inst and inst != "unknown" else \
        STREAM_FAMILY.get(stream_name.split(".")[-1], "unknown")
    if family not in FAMILY_PROGRAM:
        family = "unknown"
    return Target(0, FAMILY_PROGRAM[family], False, family)


def drum_note(voice: str) -> int:
    if voice.startswith("gm") and voice[2:].isdigit():
        return int(voice[2:])
    return DRUM_NOTES.get(voice, DRUM_DEFAULT)
