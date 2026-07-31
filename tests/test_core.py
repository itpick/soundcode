"""Core invariants for the .sc parser, timebase and expander.

The golden file examples/signal-lost.v3.sc is the reference for the grammar
(spec §4.6), so most assertions are anchored to values that file states.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode.expand import build_grid, chord_root_cents, expand   # noqa: E402
from soundcode.parser import parse, parse_file                      # noqa: E402
from soundcode.pitch import cents_to_name, name_to_cents, parse_pitch  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "examples" / "signal-lost.v3.sc"


@pytest.fixture(scope="module")
def doc():
    return parse_file(str(GOLDEN))


# --- pitch: cents are canonical ---------------------------------------------

def test_note_names_to_cents():
    assert name_to_cents("A4") == 6900        # reference pitch
    assert name_to_cents("C4") == 6000        # middle C
    assert name_to_cents("A3+8c") == 5708     # the spec's worked example
    assert name_to_cents("F1-6c") == 2894


def test_octave_digit_never_swallows_cent_offset():
    # F1-6c must be F1 minus 6 cents, not octave "1-6"
    assert name_to_cents("F1-6c") == name_to_cents("F1") - 6


def test_hz_and_raw_cents_notations_agree():
    assert parse_pitch("~440.0Hz") == pytest.approx(6900.0)
    assert parse_pitch("~220.0Hz") == pytest.approx(5700.0)
    assert parse_pitch("5708c") == 5708.0


def test_cents_to_name_round_trips():
    for name in ("A3", "C4", "F#2", "A3+8c"):
        assert cents_to_name(name_to_cents(name)) == name


# --- timebase: seconds are canonical, anchors win ---------------------------

def test_grid_reads_declared_tempo_not_a_default(doc):
    grid = build_grid(doc)
    assert len(grid.tempo) == 3
    assert grid.tempo[0][1] == pytest.approx(128.0)
    assert len(grid.anchors) == 2


def test_bar_times_hit_their_anchors_exactly(doc):
    grid = build_grid(doc)
    assert grid.time_of(1) == pytest.approx(0.0)
    assert grid.time_of(13) == pytest.approx(22.5, abs=1e-6)   # declared anchor
    assert grid.time_of(5) == pytest.approx(7.5, abs=1e-3)
    assert grid.time_of(13, 3.0) == pytest.approx(23.4375, abs=1e-4)


def test_final_ritard_stretches_the_last_bar(doc):
    grid = build_grid(doc)
    nominal = 4 * 60.0 / 128.0                       # 1.875s at 128bpm
    assert grid.time_of(17) - grid.time_of(16) > nominal


# --- parsing ----------------------------------------------------------------

def test_streams_and_namespacing(doc):
    assert doc.pragmas["sc"] == "0.3"
    names = {s.name for s in doc.streams}
    assert {"grid", "tuning", "struct", "harmony", "mix"} <= names
    assert {"notes.vox", "perc.drums", "text.vox", "contour.vox"} <= names
    assert doc.stream("notes.vox").track == "vox"
    assert doc.stream("notes.vox").kind == "notes"


def test_declaration_fields_blocks_and_gloss(doc):
    bass = doc.stream("notes.bass")
    assert bass.fields["inst"] == "bass.synth"
    assert "centroid 410Hz" in bass.blocks["timbre"]
    assert "synth bass" in bass.blocks["tags"]
    assert bass.gloss and "distorted saw" in bass.gloss


def test_micro_timing_is_preserved(doc):
    # the late hats are the feel; they must survive parsing
    devs = [e.dev_ms for e in doc.stream("perc.drums").patterns["basic"]
            if e.atom == "hat"]
    assert devs and all(d > 0 for d in devs)


def test_lossy_bindings_are_flagged_and_exact_ones_are_not(doc):
    drums = doc.stream("perc.drums")
    sims = {(b.first, b.last): b.similarity for b in drums.bindings}
    assert sims[(5, 11)] == pytest.approx(0.96)
    assert sims[(1, 2)] is None            # bare binding == exact


def test_text_stream_keeps_tokens_and_times(doc):
    text = doc.stream("text.vox")
    assert len(text.events) == 45
    assert all(e.text for e in text.events)


def test_unknown_stream_namespace_survives():
    d = parse(":future.thing  weird=1\n@1.0 blah 42\n")
    assert d.stream("future.thing") is not None


# --- expansion --------------------------------------------------------------

def test_patterns_and_like_expand(doc):
    notes = expand(doc)
    assert len(notes) == 261
    # :notes.pad has 12 explicit events, then `like 1-4 x2` and `like 1-4`
    assert sum(1 for n in notes if n.stream == "notes.pad") == 48


def test_chord_root_placement():
    # `.` resolves to the chord root voiced above the bass floor
    assert chord_root_cents("Am") == 3300.0    # A1
    assert chord_root_cents("C") == 3600.0     # C2, bumped above the floor
    assert chord_root_cents("F") == 2900.0     # F1


def test_expanded_events_stay_inside_the_declared_duration(doc):
    notes = expand(doc)
    assert max(n.start for n in notes) < doc.duration
    assert all(n.dur > 0 for n in notes)


def test_contour_vibrato_reaches_the_notes(doc):
    notes = expand(doc)
    assert any(n.vib for n in notes if n.stream == "notes.vox")
