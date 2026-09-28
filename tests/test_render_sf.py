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
    assert abs(render_sf._rms_db(y, 44100) - (-30.0)) < 0.5


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
                        lambda doc, path, sr=None, **k: calls.setdefault("path", path) and (5, 5.0))
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


# --- instrumental focus: vocals left out of renders unless asked for -------------------

def test_mix_leaves_out_vocal_streams_by_default():
    doc = parse("%sc 0.3\n@duration 1.0\n\n:notes.vox\n@0.0 C4 0.5s 90\n"
                "\n:notes.keys inst=keys.piano\n@0.0 E4 0.5s 90\n")
    t = np.arange(44100) / 44100
    keys = np.stack([0.1 * np.sin(2 * np.pi * 330 * t)] * 2, 1).astype(np.float32)
    vox = np.stack([0.5 * np.sin(2 * np.pi * 262 * t)] * 2, 1).astype(np.float32)
    only_keys = render_sf.mix(doc, {"notes.keys": keys}, 44100)
    default = render_sf.mix(doc, {"notes.keys": keys, "notes.vox": vox}, 44100)
    with_vox = render_sf.mix(doc, {"notes.keys": keys, "notes.vox": vox}, 44100,
                             with_vocals=True)
    np.testing.assert_allclose(default, only_keys)
    assert not np.allclose(with_vox, only_keys)


def test_cli_render_with_vocals_flag(tmp_path, monkeypatch):
    sc = tmp_path / "song.sc"
    sc.write_text(SCALE_SC)
    seen = {}
    monkeypatch.setattr(render_sf, "render_to_file",
                        lambda doc, path, sr=None, with_vocals=False, **k:
                        seen.setdefault("v", with_vocals) is not None and (1, 1.0))
    cli.main(["render", str(sc), "--with-vocals"])
    assert seen["v"] is True


# --- final-review fixes ---------------------------------------------------------------

SPARSE_SC = """%sc 0.3
@duration 8.0
@sr 44100

:notes.keys inst=keys.piano
meta level=-30.0
@0.0 C4 0.3s 100
@2.0 E4 0.3s 100
@4.0 G4 0.3s 100
@6.0 C5 0.3s 100
"""


@needs_sf2
def test_level_matching_measures_like_the_encoder_over_whole_blocks():
    """meta level is RMS over 2 s blocks where the part plays, gaps included;
    a sparse part must come out at that level measured the same way."""
    y = render_sf.render_streams(parse(SPARSE_SC), sf2=SF2)["notes.keys"]
    b = int(render_sf.LEVEL_BLOCK_S * 44100)
    blocks = [y[i:i + b] for i in range(0, len(y), b)]
    active = np.concatenate([x for x in blocks if np.abs(x).max() > 1e-4])
    rms_db = 20 * np.log10(np.sqrt(np.mean(active.astype(np.float64) ** 2)))
    assert abs(rms_db - (-30.0)) < 0.5


def test_level_block_matches_the_encoder_gate_block():
    from soundcode import encode as enc
    assert render_sf.LEVEL_BLOCK_S == enc.GATE_BLOCK_S


def test_server_mock_buttons_pin_the_mock_engine():
    src = (Path(__file__).resolve().parents[1] / "src" / "soundcode" / "server.py").read_text()
    for line in src.splitlines():
        if '"render"' in line and ".mock" not in line and ("-o" in line):
            assert '"--engine", "mock"' in line, line


# --- tsumugi classes + full drum kit -------------------------------------------------

def test_every_tsumugi_class_has_an_inst_program_and_family():
    assert len(gm.TSUMUGI) == 36
    for name, e in gm.TSUMUGI.items():
        assert set(e) >= {"inst", "program", "family", "programs"}, name
        assert 0 <= e["program"] <= 127


def test_fine_grained_inst_picks_its_own_program():
    assert gm.target_for("notes.x", "keys.ep").preset == 4
    assert gm.target_for("notes.x", "gtr.electric.distortion").preset == 30
    assert gm.target_for("notes.x", "bass.electric").preset == 33      # family table still works


def test_programs_map_back_to_tsumugi_classes():
    assert gm.tsumugi_class_for_program(5) == "electric_piano"
    assert gm.tsumugi_class_for_program(30) == "distorted_guitar"


def test_drum_voices_round_trip_including_clap():
    for pitch in (35, 36, 37, 38, 39, 42, 44, 46, 49, 51, 56):
        assert gm.drum_note(gm.drum_voice(pitch)) == (36 if pitch == 35 else pitch)
    assert gm.drum_voice(39) == "clap"
    assert gm.drum_voice(81) == "gm81" and gm.drum_note("gm81") == 81


# --- part toggles: per-part renders for the demo page ---------------------------------

def test_parts_sum_to_the_mix_and_are_keyed_by_stem():
    import numpy as np
    from soundcode.parser import parse
    doc = parse("""%sc 0.3
@duration 2.0

:notes.piano inst=keys.piano
meta stem=piano
@0.0 C4 1.0s 90

:notes.bass inst=bass.electric
@0.5 C2 1.0s 90
""")
    sr = 8000
    streams = {"notes.piano": np.full((sr * 2, 2), 0.1, np.float32),
               "notes.bass": np.full((sr * 2, 2), 0.2, np.float32)}
    buf, parts = render_sf.mix_parts(doc, streams, sr)
    assert set(parts) == {"piano", "bass"}                 # meta stem, else GM family
    assert np.allclose(sum(parts.values()), buf, atol=1e-6)
    assert np.array_equal(buf, render_sf.mix(doc, streams, sr))   # mix() is unchanged


def test_part_key_falls_back_to_family():
    from soundcode.parser import parse
    doc = parse("%sc 0.3\n\n:notes.lead inst=voice.lead\n@0.0 C4 1.0s 90\n\n:perc.drums\n@0.0 kick 0.1s 100\n")
    assert render_sf.part_key(doc, "notes.lead") == "lead_vocals"
    assert render_sf.part_key(doc, "perc.drums") == "drums"


@needs_sf2
def test_cli_render_parts_writes_each_sounding_part(tmp_path):
    import soundfile as sf
    sc = tmp_path / "song.sc"
    sc.write_text(SCALE_SC + "\n:notes.bass inst=bass.electric\n@0.0 C2 1.0s 100\n")
    out, parts = tmp_path / "song.wav", tmp_path / "parts"
    assert cli.main(["render", str(sc), "-o", str(out), "--parts", str(parts)]) == 0
    assert out.exists()
    assert sorted(p.name for p in parts.iterdir()) == ["bass.wav", "piano.wav"]
    info = sf.info(str(parts / "piano.wav"))
    assert info.subtype == "PCM_16" and info.frames == sf.info(str(out)).frames


def test_cli_render_parts_needs_the_sf2_engine(tmp_path, capsys):
    sc = tmp_path / "song.sc"
    sc.write_text(SCALE_SC)
    assert cli.main(["render", str(sc), "--engine", "mock", "--parts", str(tmp_path / "p")]) == 2
    assert "--parts" in capsys.readouterr().err
