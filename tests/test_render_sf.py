"""Sampled renderer (spec Part A2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import gm  # noqa: E402


def test_inst_family_picks_the_program():
    assert gm.target_for("notes.x", "bass.electric") == gm.Target(0, 33, False, "bass")
    assert gm.target_for("notes.x", "keys.piano").preset == 0
    assert gm.target_for("notes.x", "voice.lead").preset == 53


def test_stream_name_is_the_fallback_when_inst_is_missing_or_unknown():
    assert gm.target_for("notes.bass", "unknown").family == "bass"
    assert gm.target_for("notes.vox", "unknown").family == "voice"
    assert gm.target_for("notes.guitar", "unknown").preset == 27
    assert gm.target_for("notes.lead2", "unknown") == gm.Target(0, 0, False, "unknown")


def test_percussion_streams_use_the_drum_kit():
    t = gm.target_for("perc.drums", "unknown")
    assert t.drums and t.bank == 128 and t.family == "drums"


def test_drum_voice_map():
    assert [gm.drum_note(v) for v in ("kick", "snare", "hat", "crash")] == [36, 38, 42, 49]
    assert gm.drum_note("cowbell-ish") == 39
