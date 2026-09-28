"""soundcode compare (spec Part A1)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import compare as cmp  # noqa: E402

SR = cmp.SR


def tone(freq, secs=2.0, amp=0.3, start=0.0, total=None):
    total = total or secs + start
    y = np.zeros(int(total * SR), np.float32)
    t = np.arange(int(secs * SR)) / SR
    y[int(start * SR):int(start * SR) + t.size] = amp * np.sin(2 * np.pi * freq * t)
    return y


def clicks(times, total=4.0):
    y = np.zeros(int(total * SR), np.float32)
    for t in times:
        i = int(t * SR)
        y[i:i + 200] = np.hanning(200)
    return y


def test_level_of_silence_is_none():
    assert cmp.level_db(np.zeros(100, np.float32)) is None
    assert cmp.level_db(np.ones(100, np.float32)) == pytest.approx(0.0)


def test_note_f1_identical_is_one_and_octave_agnostic_forgives_octaves():
    iv = np.array([[0.0, 0.5], [0.5, 1.0]])
    hz = np.array([440.0, 523.25])
    assert cmp.note_f1(iv, hz, iv, hz) == pytest.approx(1.0)
    assert cmp.note_f1(iv, hz, iv, hz * 2) == pytest.approx(0.0)
    assert cmp.note_f1(iv, hz, iv, hz * 2, octave_agnostic=True) == pytest.approx(1.0)


def test_note_f1_empty_sides():
    e = np.zeros((0, 2))
    assert cmp.note_f1(e, np.zeros(0), e, np.zeros(0)) is None
    assert cmp.note_f1(np.array([[0.0, 0.5]]), np.array([440.0]), e, np.zeros(0)) == 0.0


def test_chroma_blocks_same_pitch_high_different_low_silent_none():
    a, b = tone(440, 4.0), tone(440, 4.0)
    c = tone(311.13, 4.0)
    blocks = [(0.0, 2.0), (2.0, 4.0)]
    assert min(cmp.chroma_blocks(a, b, blocks)) > 0.95
    assert max(cmp.chroma_blocks(a, c, blocks)) < 0.6
    z = np.zeros_like(a)
    assert cmp.chroma_blocks(a, z, blocks) == [None, None]


def test_onset_f1_matches_same_clicks():
    ref = clicks([0.5, 1.0, 1.5, 2.0, 2.5])
    assert cmp.onset_f1(ref, ref) == pytest.approx(1.0)
    assert cmp.onset_f1(ref, clicks([3.5])) < 0.3


def test_energy_corr_identical_is_one_and_silence_is_none():
    y = tone(440, 1.0, start=1.0, total=3.0)
    assert cmp.energy_corr(y, y) == pytest.approx(1.0)
    assert cmp.energy_corr(y, np.zeros_like(y)) is None


def test_blocks_fall_back_to_two_seconds_without_grid():
    from soundcode.parser import parse
    doc = parse("%sc 0.3\n@duration 5.0\n")
    assert cmp.blocks_from_grid(doc, 5.0) == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]


def test_json_safe_strips_non_finite():
    out = cmp.json_safe({"a": math.inf, "b": [math.nan, 1.0], "c": {"d": -math.inf}})
    assert json.dumps(out, allow_nan=False) == '{"a": null, "b": [null, 1.0], "c": {"d": null}}'


# --- pipeline ------------------------------------------------------------------------

from soundcode.parser import parse  # noqa: E402


def test_stream_to_stem_pairing():
    doc = parse("%sc 0.3\n\n:notes.pad inst=synth.pad\nmeta stem=other\n@0.0 C4 1s 90\n"
                "\n:notes.vox\n@0.0 C4 1s 90\n\n:notes.lead2\n@0.0 C4 1s 90\n")
    assert cmp.stem_for_stream(doc, "notes.pad") == "other"
    assert cmp.stem_for_stream(doc, "notes.vox") == "lead_vocals"
    assert cmp.stem_for_stream(doc, "perc.drums") == "drums"
    assert cmp.stem_for_stream(doc, "notes.piano") == "piano"
    assert cmp.stem_for_stream(doc, "notes.lead2") == "other"


def test_run_scores_a_perfect_render_highly_and_writes_the_report(tmp_path, monkeypatch):
    import soundfile as sf
    from soundcode import separate as sep

    # original stems: piano plays A4 then C5; everything else silent
    sr = 44100
    t = np.arange(sr) / sr
    a4 = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    c5 = (0.3 * np.sin(2 * np.pi * 523.25 * t)).astype(np.float32)
    piano = np.concatenate([a4, c5, np.zeros(sr, np.float32)])
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    for s in sep.STEMS:
        y = piano if s == "piano" else np.zeros_like(piano)
        sf.write(str(stems_dir / f"{s}.wav"), np.stack([y, y]).T, sr, subtype="FLOAT")
    orig = tmp_path / "orig.wav"
    sf.write(str(orig), np.stack([piano, piano]).T, sr)
    sc = tmp_path / "song.sc"
    sc.write_text("%sc 0.3\n@duration 3.0\n\n:notes.piano inst=keys.piano\n"
                  "@0.0 A4 1s 90\n@1.0 C5 1s 90\n")

    fake = lambda doc, sr=None, sf2=None: {"notes.piano": np.stack([piano, piano]).T}  # noqa: E731
    fake_notes = {  # transcriptions: identical on both sides
        "piano": (np.array([[0.0, 1.0], [1.0, 2.0]]), np.array([440.0, 523.25])),
    }
    monkeypatch.setattr(cmp, "transcribe",
                        lambda p: fake_notes["piano"] if "piano" in Path(p).name
                        else (np.zeros((0, 2)), np.zeros(0)))
    report = cmp.run(orig, sc, tmp_path / "cmp", render_streams=fake, stems_dir=stems_dir)

    piano_row = report["stems"]["piano"]
    assert piano_row["notes_f1"] == pytest.approx(1.0)
    assert piano_row["level_diff_db"] == pytest.approx(0.0, abs=0.1)
    assert report["stems"]["bass"]["notes_f1"] is None           # silent both sides
    for f in ("report.json", "report.html", "piano.png", "bars.png"):
        assert (tmp_path / "cmp" / f).exists()
    json.loads((tmp_path / "cmp" / "report.json").read_text())


def test_cli_compare_prints_a_row_per_stem(tmp_path, monkeypatch, capsys):
    from soundcode import cli
    monkeypatch.setattr(cmp, "run", lambda *a, **k: {"stems": {
        "piano": {"level_orig_db": -20.0, "level_render_db": -21.0, "level_diff_db": -1.0,
                  "notes_f1": 0.5, "notes_f1_octave": 0.6, "chroma": 0.9,
                  "onset_f1": 0.7, "energy_corr": 0.8}}, "report": "x/report.html"})
    assert cli.main(["compare", "a.wav", "b.sc", "-o", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "piano" in out and "0.50" in out and "report.html" in out


def test_server_serves_compare_reports(tmp_path, monkeypatch):
    from soundcode import server
    d = tmp_path / "out" / "compare" / "song"
    d.mkdir(parents=True)
    (d / "report.html").write_text("<html>ok</html>")
    monkeypatch.setenv("SOUNDCODE_ROOT", str(tmp_path))
    assert server._compare_file("song/report.html") == d / "report.html"
    assert server._compare_file("../../etc/passwd") is None


# --- vocal metrics --------------------------------------------------------------------------

def _sweep(freq, cents_off=0.0, secs=2.0, sr=16000):
    t = np.arange(int(secs * sr)) / sr
    f = freq * 2 ** (cents_off / 1200) * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    return (0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)).astype(np.float32)


def test_pitch_error_is_zero_for_the_same_line_and_measures_detune():
    a = _sweep(220.0)
    assert cmp.pitch_error_cents(a, a, 16000) < 5
    assert 35 < cmp.pitch_error_cents(a, _sweep(220.0, 50.0), 16000) < 65      # crepe bins are 20 c


def test_pitch_error_of_silence_is_none():
    assert cmp.pitch_error_cents(_sweep(220.0), np.zeros(32000, np.float32), 16000) is None


def test_voice_similarity_is_none_on_silence():
    assert cmp.voice_similarity(_sweep(220.0), np.zeros(32000, np.float32), 16000) is None


# --- spectral match ----------------------------------------------------------------------------

def test_spectral_db_is_zero_for_the_same_sound_and_grows_with_a_tilt():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(SR * 3) * 0.1).astype(np.float32)
    assert cmp.spectral_db(a, a, SR) < 0.5
    Y = np.fft.rfft(a); f = np.fft.rfftfreq(len(a), 1 / SR)
    dark = np.fft.irfft(Y * 10 ** (-6 * np.log2(np.maximum(f, 20) / 1000) / 20), len(a)).astype(np.float32)
    assert cmp.spectral_db(a, dark, SR) > 6
    assert cmp.spectral_db(a, np.zeros_like(a), SR) is None


def test_compare_passes_no_fx_to_the_renderer(tmp_path, monkeypatch):
    seen = {}
    def fake_render_streams(doc, sr=None, sf2=None, no_fx=False):
        seen["no_fx"] = no_fx
        return {}
    import soundcode.render_sf as rsf
    monkeypatch.setattr(rsf, "render_streams", fake_render_streams)
    import soundfile as sf
    from soundcode import separate as sep
    d = tmp_path / "stems"; d.mkdir()
    for s in sep.STEMS:
        sf.write(str(d / f"{s}.wav"), np.zeros((22050, 2), np.float32), 22050)
    sc = tmp_path / "s.sc"; sc.write_text("%sc 0.3\n@duration 1.0\n")
    monkeypatch.setattr(cmp, "transcribe", lambda p: (np.zeros((0, 2)), np.zeros(0)))
    cmp.run(tmp_path / "o.wav", sc, tmp_path / "c", stems_dir=d, no_fx=True)
    assert seen["no_fx"] is True


def test_wer():
    assert cmp.wer("a b c d".split(), "a b c d".split()) == 0.0
    assert cmp.wer("a b c d".split(), "a x c".split()) == pytest.approx(0.5)
    assert cmp.wer([], ["a"]) == 1.0


def test_lyric_wer_uses_only_the_cached_reference(tmp_path, monkeypatch):
    import json
    from soundcode import lyrics as ly
    from soundcode.parser import parse
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "lyrics").mkdir(parents=True)
    (tmp_path / "out" / "lyrics" / "nobody-harbor.json").write_text(json.dumps(
        {"syncedLyrics": "[00:00.50] walk me down to the harbor"}))
    doc = parse('%sc 0.3\n@title "harbor"\n@source Nobody - harbor.wav\n@duration 10.0\n\n'
                ':text.vox\n@0.5 "walk" | @0.8 "me" | @1.0 "down" | @1.2 "to" | @1.4 "the" | @1.6 "harder"\n')
    monkeypatch.setattr(ly.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert cmp.lyric_wer(doc) == pytest.approx(1 / 6)


def test_sung_words_are_only_those_inside_the_song():
    from soundcode.parser import parse
    doc = parse('%sc 0.3\n@duration 2.0\n\n:text.vox\n@0.5 "walk" | @1.0 "down-town" | @1.97 "oh" | @2.5 "oh"\n')
    assert cmp._sc_words(doc) == ["walk", "down", "town", "oh", "oh"]
    assert cmp._sung_words(doc) == ["walk", "down", "town"]   # the singer drops words past the end
