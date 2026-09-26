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


# --- SoundFont rendering -------------------------------------------------------------

from soundcode.parser import parse  # noqa: E402

SF2 = Path(__file__).resolve().parents[1] / "models" / "soundfonts" / "GeneralUser-GS.sf2"
needs_sf2 = pytest.mark.skipif(not SF2.exists(), reason="SoundFont not downloaded")

SCALE_SC = """%sc 0.3
@duration 5.0
@sr 44100

:notes.keys inst=keys.piano
@0.0 C4 0.4s 100
@0.5 D4 0.4s 100
@1.0 E4 0.4s 100
@1.5 F4 0.4s 100
@2.0 G4 0.4s 100
"""


def _f0(x, sr):
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo = int(sr / 1000)
    return sr / (np.argmax(ac[lo:]) + lo)


@needs_sf2
def test_scale_renders_in_tune_and_to_length():
    doc = parse(SCALE_SC)
    streams = render_sf.render_streams(doc, sf2=SF2)
    y = streams["notes.keys"]
    assert y.shape[1] == 2 and y.dtype == np.float32
    assert abs(y.shape[0] / 44100 - 6.5) < 0.01          # @duration + 1.5 s release tail
    for i, hz in enumerate([261.63, 293.66, 329.63, 349.23, 392.0]):
        seg = y[int((i * 0.5 + 0.05) * 44100): int((i * 0.5 + 0.30) * 44100), 0]
        cents = 1200 * np.log2(_f0(seg, 44100) / hz)
        assert abs(cents) < 20, (i, cents)


@needs_sf2
def test_meta_level_sets_the_stream_rms():
    doc = parse(SCALE_SC.replace(":notes.keys inst=keys.piano",
                                 ":notes.keys inst=keys.piano\nmeta level=-30.0"))
    y = render_sf.render_streams(doc, sf2=SF2)["notes.keys"]
    active = y[np.abs(y).max(1) > 1e-4]
    rms_db = 20 * np.log10(np.sqrt(np.mean(active ** 2)))
    assert abs(rms_db - (-30.0)) < 0.5


@needs_sf2
def test_mix_is_normalised_and_honours_sample_rate():
    doc = parse(SCALE_SC.replace("@sr 44100", "@sr 22050"))
    y = render_sf.render(doc, sf2=SF2)
    assert 0.8 < float(np.abs(y).max()) <= 0.9
    assert abs(y.shape[0] / 22050 - 6.5) < 0.01


def test_missing_soundfont_offline_is_one_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCODE_SOUNDFONT", str(tmp_path / "nope.sf2"))
    with pytest.raises(render_sf.SoundFontError, match="SOUNDCODE_SOUNDFONT"):
        render_sf.soundfont_path()


def test_bad_download_is_not_left_behind(tmp_path, monkeypatch):
    monkeypatch.delenv("SOUNDCODE_SOUNDFONT", raising=False)
    monkeypatch.setattr(render_sf, "DEFAULT_SF2", tmp_path / "sf" / "GeneralUser-GS.sf2")
    monkeypatch.setattr(render_sf, "_download", lambda url, dest: dest.write_bytes(b"junk"))
    with pytest.raises(render_sf.SoundFontError, match="checksum"):
        render_sf.soundfont_path()
    assert not (tmp_path / "sf" / "GeneralUser-GS.sf2").exists()


# --- CLI + server ----------------------------------------------------------------

from soundcode import cli, server  # noqa: E402


def test_render_defaults_to_sf2_and_names_the_output(tmp_path, monkeypatch):
    sc = tmp_path / "song.sc"
    sc.write_text(SCALE_SC)
    calls = {}
    monkeypatch.setattr(render_sf, "render_to_file",
                        lambda doc, path, sr=None: calls.setdefault("path", path) and (5, 5.0))
    assert cli.main(["render", str(sc)]) == 0
    assert calls["path"].endswith("song.render.wav")


def test_render_reports_missing_soundfont_and_exits_two(tmp_path, monkeypatch, capsys):
    sc = tmp_path / "song.sc"
    sc.write_text(SCALE_SC)

    def boom(*a, **k):
        raise render_sf.SoundFontError("could not download the SoundFont; set SOUNDCODE_SOUNDFONT")
    monkeypatch.setattr(render_sf, "render_to_file", boom)
    assert cli.main(["render", str(sc)]) == 2
    assert "SOUNDCODE_SOUNDFONT" in capsys.readouterr().err


def test_server_tags_render_files(tmp_path, monkeypatch):
    (tmp_path / "out" / "sc").mkdir(parents=True)
    (tmp_path / "out" / "sc" / "song.render.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("SOUNDCODE_ROOT", str(tmp_path))
    assert server._discover_tracks()[0]["kind"] == "render"
