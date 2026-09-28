"""SoulX-Singer adapter (spec 2026-09-27-lyrics-and-voice §2).

Format pinned from SoulX-Singer@81aeb3a preprocess/tools/midi_parser.py:
note_type 1 = rest <SP>, 2 = a word's first note, 3 = continuation note of the
previous word (melisma); f0 at 24 kHz / hop 480 = 50 Hz; time in ms; durations in s."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from soundcode import sing, soulx  # noqa: E402
from soundcode import sing_score as ss  # noqa: E402
from soundcode.parser import parse  # noqa: E402
from test_sing import SONG  # noqa: E402


def test_metadata_matches_the_pinned_format():
    segs = soulx.metadata(parse(SONG), 0.0, 4.0)
    assert len(segs) == 1
    s = segs[0]
    n = len(s["note_pitch"].split())
    assert len(s["text"].split()) == len(s["duration"].split()) == len(s["note_type"].split()) == n
    types, pitches, text = s["note_type"].split(), s["note_pitch"].split(), s["text"].split()
    assert types[text.index("river")] == "2" and pitches[text.index("river")] == "64"
    assert all(p == "0" for t, p in zip(text, pitches) if t == "<SP>")
    assert "en_R-IH1-V-ER0" in s["phoneme"].split()
    assert s["time"] == [0, 4000] and s["language"] == "English"
    assert abs(sum(float(d) for d in s["duration"].split()) - 4.0) < 0.02
    assert len(s["f0"].split()) == round(4.0 * soulx.F0_RATE)


def test_melisma_notes_are_continuations():
    doc = parse(SONG.replace('1:1.000 "hello" 1.000b | 1:3.000 "river" 2.000b', '1:1.000 "hello" 2.000b'))
    s = soulx.metadata(doc, 0.0, 4.0)[0]
    types, text = s["note_type"].split(), s["text"].split()
    assert types[:2] == ["2", "3"] and text[:2] == ["hello", "hello"]


def test_long_songs_split_at_rests_under_15s():
    notes = "\n".join(f"@{t:.2f} C4 0.5s 90" for k in range(5) for t in np.arange(k * 8.0, k * 8.0 + 6.0, 0.5))
    words = " | ".join(f'@{t:.2f} "la" 0.5s' for k in range(5) for t in np.arange(k * 8.0, k * 8.0 + 6.0, 0.5))
    doc = parse(f"%sc 0.3\n@duration 40.0\n\n:notes.lead inst=voice.lead\n{notes}\n\n:text.vox\n{words}\n")
    segs = soulx.metadata(doc, 0.0, 40.0)
    assert len(segs) >= 3 and all(s["time"][1] - s["time"][0] <= 15000 for s in segs)
    covered = sum(sum(1 for t in s["note_type"].split() if t != "1") for s in segs)
    assert covered == 60


def test_soulx_failure_falls_back_to_diffsinger(tmp_path, monkeypatch):
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(44100, np.float32), 44100)
    monkeypatch.setattr(soulx, "render", lambda *a, **k: (_ for _ in ()).throw(ss.SingError("CUDA OOM")))
    monkeypatch.setattr(sing.diffsinger, "render", lambda score, **k: np.zeros(44100, np.float32))
    def fake_convert(src, r, out, steps=30):
        sf.write(str(out), np.zeros(44100, np.float32), 44100); return out
    monkeypatch.setattr(sing.seedvc, "convert", fake_convert)
    p, warns = sing.sing(parse(SONG), ref, cache=tmp_path / "c", singer="soulx")
    assert p.exists() and any("soulx" in w.lower() and "CUDA OOM" in w for w in warns)


def test_soulx_does_not_need_the_diffsinger_bank(tmp_path, monkeypatch):
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(44100, np.float32), 44100)
    monkeypatch.setenv("SOUNDCODE_DIFFSINGER", str(tmp_path / "nope"))
    monkeypatch.setattr(soulx, "render", lambda *a, **k: np.zeros(44100, np.float32))
    p, _ = sing.sing(parse(SONG), ref, cache=tmp_path / "c", singer="soulx")
    assert p.exists()


def test_long_unbroken_singing_is_cut_at_a_rest_not_mid_note():
    notes = "\n".join(f"@{t:.2f} C4 0.45s 90" for t in np.arange(0.0, 20.0, 0.5))    # 50 ms gaps only
    words = " | ".join(f'@{t:.2f} "la" 0.45s' for t in np.arange(0.0, 20.0, 0.5))
    doc = parse(f"%sc 0.3\n@duration 20.0\n\n:notes.lead inst=voice.lead\n{notes}\n\n:text.vox\n{words}\n")
    segs = soulx.metadata(doc, 0.0, 20.0)
    assert all(s["time"][1] - s["time"][0] <= 15000 for s in segs)
    for s in segs:
        assert all(float(d) >= 0.01 for d in s["duration"].split())
    total_notes = sum(1 for s in segs for t in s["note_type"].split() if t != "1")
    assert total_notes == 40                                                      # no note split in two
