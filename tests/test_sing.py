"""Singing thin slice (spec 2026-09-27)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import contour as ct  # noqa: E402
from soundcode.parser import parse  # noqa: E402


def test_phrases_resample_to_20ms_and_split_on_long_gaps():
    t = np.arange(0, 2.0, 0.01)                       # 10 ms frames
    cents = np.full_like(t, 6900.0)
    voiced = np.ones_like(t, bool)
    voiced[(t >= 0.50) & (t < 0.53)] = False          # 30 ms: filled
    voiced[(t >= 1.00) & (t < 1.20)] = False          # 200 ms: split
    ph = ct.phrases(t, cents, voiced)
    assert [round(s, 2) for s, _ in ph] == [0.0, 1.2]
    assert len(ph[0][1]) == 50 and all(v == 6900 for v in ph[0][1])


def test_vibrato_survives_at_50hz():
    t = np.arange(0, 1.0, 0.01)
    cents = 6900 + 50 * np.sin(2 * np.pi * 5.5 * t)   # 5.5 Hz, +/-50 c
    (start, vals), = ct.phrases(t, cents, np.ones_like(t, bool))
    assert max(vals) - min(vals) >= 90


def test_contour_lines_round_trip_through_the_parser():
    ph = [(12.34, [6912, 6915, 6920]), (14.02, list(range(6700, 6760)))]
    text = "%sc 0.3\n\n:contour.vox rate=50\n" + "\n".join(ct.contour_lines(ph)) + "\n"
    segs = ct.read_contour(parse(text))
    assert [round(s, 2) for s, _ in segs] == [12.34, 14.02, 15.02]       # 60 values -> 50 + 10
    assert list(segs[0][1]) == [6912, 6915, 6920]
    assert len(segs[1][1]) == 50 and len(segs[2][1]) == 10


def test_no_voiced_frames_gives_no_phrases():
    t = np.arange(0, 1.0, 0.01)
    assert ct.phrases(t, np.zeros_like(t), np.zeros_like(t, bool)) == []


def test_stage_contour_on_a_gated_stem_writes_nothing(tmp_path):
    import soundfile as sf
    from soundcode import encode as enc
    sr = 16000
    p = tmp_path / "lead_vocals.wav"
    sf.write(str(p), np.zeros(sr * 4, np.float32), sr)
    st = enc.stage_contour(p, "lead_vocals", np.random.default_rng(0).standard_normal(sr * 4) * 0.1, sr)
    assert not st.ok


# --- lyrics as performed ------------------------------------------------------------------

from soundcode import encode as enc  # noqa: E402

GRID = {"downbeat": 1.0, "bar_dur": 2.0}


def test_text_events_keep_their_duration():
    doc = parse('%sc 0.3\n\n:text.vox\n1:2.125 "river" 0.500b ?0.71\n@0.400 "I" 0.200s\n')
    evs = doc.stream("text.vox").events
    assert [(e.text, e.dur) for e in evs] == [("river", "0.500b"), ("I", "0.200s")]


def test_lyric_cells_use_performed_timing_and_durations():
    words = [(1.5625, 1.9375, "river", 0.9), (0.4, 0.6, "I", 0.5)]
    cells = enc.lyric_cells(words, GRID)
    assert cells == ['@0.400 "I" 0.200s ?0.50', '1:2.125 "river" 0.750b']   # chronological


def test_colliding_words_are_nudged_20ms():
    words = [(1.5, 1.8, "a", 0.9), (1.5, 1.9, "b", 0.9)]
    cells = enc.lyric_cells(words, GRID)
    assert cells[1].startswith("1:2.040")                         # +20 ms at 0.5 s per beat


# --- vocal score --------------------------------------------------------------------------

from soundcode import sing_score as ss  # noqa: E402

SONG = """%sc 0.3
@duration 4.0

:grid
meter @0.000 4/4
anchor bar 1 @0.000
tempo @0.000 120

:notes.lead inst=voice.lead
meta stem=lead_vocals
1:1.000  C4  1.000b 90
1:2.000  D4  1.000b 90
1:3.000  E4  2.000b 90

:text.vox
1:1.000 "hello" 1.000b | 1:3.000 "river" 2.000b
"""


def test_vocal_stream_and_notes():
    doc = parse(SONG)
    assert ss.vocal_stream(doc) == "notes.lead"
    assert ss.vocal_notes(doc, "notes.lead") == [(0.0, 0.5, 60), (0.5, 1.0, 62), (1.0, 2.0, 64)]


def test_words_and_g2p():
    doc = parse(SONG)
    assert [(round(a, 2), round(b, 2), w) for a, b, w in ss.words(doc)] == [(0.0, 0.5, "hello"), (1.0, 2.0, "river")]
    ph, hit = ss.g2p("river")
    assert hit and ph[0] == "en/r" and "en/er" in ph


def test_unknown_word_falls_back_and_warns():
    ph, hit = ss.g2p("zzxqv")
    assert not hit and ph                                   # letter-based guess, never empty


def test_score_frames_cover_the_song_and_f0_follows_notes():
    sc = ss.build(parse(SONG))
    assert sum(sc.frames) == sc.n_frames == int(round(4.0 * ss.SR / ss.HOP))
    t = (np.arange(sc.n_frames) + 0.5) * ss.HOP / ss.SR
    mid_c4 = sc.f0_hz[(t > 0.1) & (t < 0.4)]
    assert np.all(np.abs(1200 * np.log2(mid_c4 / 261.63)) < 5)
    assert "SP" in sc.phonemes and sc.phonemes.count("en/ow") == 1


def test_contour_overrides_notes_where_present():
    doc = parse(SONG + "\n:contour.vox rate=50\nf0  @0.100  " + " ".join(["6030"] * 10) + "\n")
    sc = ss.build(doc)
    t = (np.arange(sc.n_frames) + 0.5) * ss.HOP / ss.SR
    seg = sc.f0_hz[(t > 0.12) & (t < 0.28)]
    assert np.all(np.abs(1200 * np.log2(seg / 261.63) - 30) < 3)        # C4 + 30 c from contour


def test_melisma_and_extra_words():
    doc = parse(SONG.replace('| 1:3.000 "river" 2.000b', '| 1:2.500 "river" 0.500b | 1:2.750 "run" 0.250b'))
    sc = ss.build(doc)
    assert sum(sc.frames) == sc.n_frames
    assert not any("crash" in w for w in sc.warnings)


# --- DiffSinger engine -------------------------------------------------------------------

from soundcode import diffsinger as ds  # noqa: E402


def test_missing_bank_is_one_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCODE_DIFFSINGER", str(tmp_path / "nope"))
    with pytest.raises(ss.SingError, match="nope"):
        ds.bank_dir()


def _bank_present():
    try:
        ds.bank_dir(), ds.vocoder_path()
        return True
    except ss.SingError:
        return False


@pytest.mark.skipif(not _bank_present(), reason="DiffSinger bank not installed")
def test_diffsinger_sings_two_bars_at_the_requested_pitch():
    import librosa
    sc = ss.build(parse(SONG))
    y = ds.render(sc)
    assert abs(len(y) / ss.SR - 4.0) < 0.1 and np.abs(y).max() > 0.1
    f0, v, _ = librosa.pyin(y, fmin=100, fmax=600, sr=ss.SR, frame_length=2048)
    t = librosa.times_like(f0, sr=ss.SR)
    seg = f0[(t > 1.3) & (t < 1.8) & v]                    # the E4 note
    assert len(seg) and abs(np.median(1200 * np.log2(seg / 329.63))) < 50
