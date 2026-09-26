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


# --- notes -> MIDI events ------------------------------------------------------

from soundcode import render_sf  # noqa: E402
from soundcode.expand import Note  # noqa: E402


def note(start, cents, dur=0.5, stream="notes.keys", inst="keys.piano", voice=None, vel=90):
    return Note(start=start, dur=dur, cents=cents, voice=voice, vel=vel,
                stream=stream, inst=inst)


def test_polyphonic_streams_round_to_the_nearest_semitone_without_bend():
    t = gm.target_for("notes.keys", "keys.piano")
    ev = render_sf.stream_events([note(0.0, 6040.0), note(0.0, 6360.0)], t)
    ons = [e for e in ev if e.kind == "on"]
    assert sorted(e.key for e in ons) == [60, 64]
    assert not [e for e in ev if e.kind == "bend"]


def test_monophonic_streams_send_the_cents_offset_as_pitch_bend():
    t = gm.target_for("notes.vox", "voice.lead")
    ev = render_sf.stream_events([note(1.0, 6930.0, stream="notes.vox", inst="voice.lead")], t)
    bend = next(e for e in ev if e.kind == "bend")
    on = next(e for e in ev if e.kind == "on")
    assert on.key == 69 and bend.time == on.time and bend < on
    # +30 cents of a +/-2 semitone range: 8192 + 0.30/2 * 8192
    assert bend.value == 8192 + round(0.15 * 8192)


def test_offs_follow_ons_by_duration_and_sort_before_ons_at_the_same_time():
    t = gm.target_for("notes.keys", "keys.piano")
    ev = sorted(render_sf.stream_events([note(0.0, 6000.0, dur=0.5),
                                        note(0.5, 6000.0, dur=0.5)], t))
    kinds = [(round(e.time, 3), e.kind) for e in ev]
    assert kinds == [(0.0, "on"), (0.5, "off"), (0.5, "on"), (1.0, "off")]


def test_drum_notes_use_the_voice_map():
    t = gm.target_for("perc.drums", "unknown")
    ev = render_sf.stream_events([note(0.0, None, stream="perc.drums", voice="snare")], t)
    assert next(e for e in ev if e.kind == "on").key == 38
