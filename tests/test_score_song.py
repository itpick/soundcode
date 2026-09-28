"""score_song and the report (spec 2026-09-28-benchmark-scorer, Task 5).

Every model is faked: MERT (embed.frames), tsumugi (drums.voice_onsets),
basic-pitch / torchcrepe (metrics._notes / metrics._f0), Whisper
(scorer._words) and resemblyzer (compare.voice_similarity).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundcode import compare  # noqa: E402
from soundcode.score import drums, embed, metrics, report, scorer  # noqa: E402

SR = 44100
DUR = 12.0
N = int(SR * DUR)

# 120 bpm 4/4: a bar is 2 s, so three 2-bar sections cover the 12 s song
SC = ("%sc 0.3\n@duration 12.0\n\n:grid\nmeter @0.000 4/4\nanchor bar 1 @0.000\n"
      "tempo @0.000 120\n\n:struct\nintro 1-2 inst energy=0.2\n"
      "verse 3-4 inst energy=0.5\nchorus 5-6 inst energy=0.8\n")

_rng = np.random.default_rng(7)
# irregular onset times so the lag search has one clear peak
PIANO_T = np.sort(_rng.uniform(0.2, DUR - 0.8, 24))
DRUM_T = np.sort(_rng.uniform(0.2, DUR - 0.5, 30))


def _piano(shift=0.0, gain=1.0):
    y = np.zeros(N, np.float32)
    t = np.arange(int(0.4 * SR)) / SR
    for k, t0 in enumerate(PIANO_T):
        f = (440.0, 523.25, 659.25)[k % 3]
        note = 0.3 * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.12)
        i = int((t0 + shift) * SR)
        seg = note[: max(0, min(note.size, N - i))]
        y[i:i + seg.size] += seg
    return y * gain


def _drums():
    y = np.zeros(N, np.float32)
    rng = np.random.default_rng(3)
    t = np.arange(int(0.15 * SR)) / SR
    for t0 in DRUM_T:
        hit = 0.5 * rng.standard_normal(t.size) * np.exp(-t / 0.02)
        i = int(t0 * SR)
        seg = hit[: max(0, min(hit.size, N - i))]
        y[i:i + seg.size] += seg
    return y


def _write(path: Path, y: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, SR)


def _fake_notes(y, sr):
    import librosa
    on = librosa.onset.onset_detect(y=np.asarray(y, np.float32), sr=sr, units="time")
    iv = np.array([[t, t + 0.2] for t in on]).reshape(-1, 2)
    return iv, np.full(len(on), 440.0)


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(embed, "frames", lambda wav: np.ones((int(DUR * 75) + 1, 8)))
    monkeypatch.setattr(drums, "voice_onsets",
                        lambda wav, work: {"kick": DRUM_T[::2], "snare": DRUM_T[1::2],
                                           "hat": np.zeros(0), "clap": np.zeros(0),
                                           "other": np.zeros(0)})
    monkeypatch.setattr(metrics, "_notes", _fake_notes)
    monkeypatch.setattr(metrics, "_f0",
                        lambda y, sr, fmin=50.0: (np.full(int(len(y) / sr * 100), 440.0),
                                                  np.ones(int(len(y) / sr * 100), bool)))
    monkeypatch.setattr(compare, "voice_similarity", lambda a, b, sr: 0.9)

    def no_whisper(wav):
        raise AssertionError("Whisper must be faked")
    monkeypatch.setattr(scorer, "_words", no_whisper)


def _song(tmp_path, stems: dict, parts: dict, sc=SC):
    stems_dir, parts_dir = tmp_path / "stems", tmp_path / "parts"
    for k, y in stems.items():
        _write(stems_dir / f"{k}.wav", y)
    for k, y in parts.items():
        _write(parts_dir / f"{k}.wav", y)
    parts_dir.mkdir(parents=True, exist_ok=True)
    orig = sum(stems.values())
    reb = sum(parts.values()) if parts else np.zeros(N, np.float32)
    _write(tmp_path / "song.wav", np.stack([orig, 0.8 * orig], 1))
    _write(tmp_path / "rebuild.wav", np.stack([reb, 0.8 * reb], 1))
    (tmp_path / "song.sc").write_text(sc)
    return scorer.score_song(tmp_path / "song.wav", stems_dir, tmp_path / "song.sc",
                             parts_dir, tmp_path / "rebuild.wav", tmp_path / "out")


def test_score_song_flags_a_late_quiet_part_and_writes_the_report(tmp_path, fakes):
    res = _song(tmp_path,
                {"piano": _piano(), "drums": _drums()},
                {"piano": _piano(shift=0.2, gain=0.5), "drums": _drums()})

    assert res["song"] == "song"
    assert res["duration"] == pytest.approx(DUR, abs=0.01)
    assert (tmp_path / "out" / "scores.json").exists()
    assert json.loads((tmp_path / "out" / "scores.json").read_text())["song"] == "song"

    piano, dr = res["parts"]["piano"], res["parts"]["drums"]
    assert dr["song"]["score"] > 90
    assert res["drift"]["piano"] and all(w["drift"] for w in res["drift"]["piano"])
    assert not any(w["drift"] for w in res["drift"]["drums"])
    assert piano["song"]["axes"]["what"] <= dr["song"]["axes"]["what"] - 30
    assert [s["label"] for s in piano["sections"]] == ["intro", "verse", "chorus"]
    assert res["worst"]["part"] == "piano"
    assert res["mix"]["song"]["score"] is not None
    assert res["score"] is not None

    table = report.table(res)
    rows = [ln for ln in table.splitlines() if ln.split("|")[0].strip() in ("piano", "drums")]
    assert rows[0].split("|")[0].strip() == "piano"

    path = report.html(res, tmp_path / "out")
    text = Path(path).read_text()
    assert "Worst slices" in text
    assert "<script src" not in text and "http" not in text.replace("http-equiv", "")
    for n in (1, 2, 3):
        for side in ("original", "rebuild"):
            ex = tmp_path / "out" / "excerpts" / f"{n}-{side}.wav"
            assert ex.exists()
            assert sf.info(str(ex)).duration == pytest.approx(DUR, abs=0.05)  # clamped to the file
            assert f"excerpts/{n}-{side}.wav" in text


def test_missing_and_silent_parts(tmp_path, fakes):
    res = _song(tmp_path,
                {"piano": _piano(), "guitar": _piano(), "bass": np.zeros(N, np.float32)},
                {"piano": _piano()})
    assert res["parts"]["bass"]["silent"] and res["parts"]["bass"]["song"] is None
    g = res["parts"]["guitar"]
    assert g["missing"] and g["song"]["score"] == 0.0
    assert res["parts"]["piano"]["song"]["score"] > 90
    assert "residual" not in res["parts"]
    table = report.table(res)
    lines = {ln.split("|")[0].strip(): ln for ln in table.splitlines() if "|" in ln}
    assert "not rebuilt" in lines["guitar"]
    assert "—" in lines["bass"]
    rows = [ln.split("|")[0].strip() for ln in table.splitlines()
            if ln.split("|")[0].strip() in ("piano", "guitar", "bass")]
    assert rows == ["guitar", "piano", "bass"]
    assert res["score"] < res["parts"]["piano"]["song"]["score"]  # the missing part drags it down
    report.html(res, tmp_path / "out")                            # renders with missing/silent


def test_vocal_part_words_are_transcribed_once_per_file(tmp_path, fakes, monkeypatch):
    calls = []

    def fake_words(wav):
        calls.append(Path(wav))
        if Path(wav).parent.name == "stems":
            return [(" Walk", 1.0, 1.3), (" down", 5.0, 5.3), (" town,", 9.0, 9.3)]
        return [("walk", 1.1, 1.4), ("down", 5.1, 5.4)]
    monkeypatch.setattr(scorer, "_words", fake_words)

    sc = SC + '\n:text.vox\n@1.0 "walk" | @5.0 "down" | @9.0 "town"\n'
    vox = _piano()
    res = _song(tmp_path, {"lead_vocals": vox}, {"lead_vocals": vox}, sc=sc)
    lv = res["parts"]["lead_vocals"]
    m = lv["song"]["metrics"]
    assert m["word_mae_s"]["raw"] == pytest.approx(0.1)
    assert m["sung_wer_excess"]["raw"] == pytest.approx(1 / 3)
    assert m["voice_sim"]["raw"] == pytest.approx(0.9)
    assert m["f0_cents"]["raw"] == pytest.approx(0.0)
    assert sorted(p.parent.name for p in calls) == ["parts", "stems"]
    by_label = {s["label"]: s for s in lv["sections"]}
    assert by_label["intro"]["metrics"]["word_mae_s"]["raw"] == pytest.approx(0.1)
    assert "word_mae_s" not in by_label["chorus"]["metrics"] or \
        by_label["chorus"]["metrics"]["word_mae_s"]["raw"] is None  # "town" never matched


def test_sung_wer_accepts_precomputed_words():
    from soundcode.parser import parse
    doc = parse('%sc 0.3\n@duration 4.0\n\n:text.vox\n@0.5 "walk" | @1.0 "down-town"\n')
    assert compare.sung_wer(doc, None, words=[(" Walk,", 0.5, 0.8), ("down", 1.0, 1.2)]) \
        == pytest.approx(1 / 3)


def test_repeated_section_labels_keep_their_own_spans(tmp_path, fakes):
    sc = SC.replace("chorus 5-6", "verse 5-6")
    res = _song(tmp_path, {"piano": _piano()}, {"piano": _piano(shift=0.2, gain=0.5)}, sc=sc)
    p = res["parts"]["piano"]
    assert [(s["label"], s["a"]) for s in p["sections"]] == [("intro", 0.0), ("verse", 4.0),
                                                             ("verse", 8.0)]
    w = p["worst"]
    match = [s for s in p["sections"] if (s["a"], s["b"]) == (w["a"], w["b"])]
    assert match and match[0]["score"] == w["score"] and match[0]["label"] == w["label"]
    text = Path(report.html(res, tmp_path / "out")).read_text()
    assert text.count("title='piano · verse'") == 2
