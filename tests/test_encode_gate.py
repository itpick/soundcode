"""Loudness gate and note filtering (spec Part B1-B3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import encode as enc  # noqa: E402

SR = 22050


def noise(secs, db):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(secs * SR)) * 10 ** (db / 20)).astype(np.float32)


def test_gate_closes_on_silence_and_bleed_opens_on_real_parts():
    mix = noise(8, -12)
    stem = np.concatenate([noise(4, -70), noise(4, -20)])     # silent-ish, then playing
    mask = enc.active_blocks(stem, mix, SR)
    assert mask.tolist() == [False, False, True, True]


def test_gate_keeps_a_quiet_but_real_part():
    mix = noise(4, -30)                                       # a soft passage
    stem = noise(4, -45)                                      # 15 dB under a soft mix
    assert enc.active_blocks(stem, mix, SR).all()


def test_gate_closes_on_bleed_far_below_a_loud_mix():
    mix = noise(4, -10)
    stem = noise(4, -48)                                      # 38 dB under the mix
    assert not enc.active_blocks(stem, mix, SR).any()


def test_same_pitch_overlaps_merge():
    ev = [(0.0, 0.5, 60, 0.9, [1]), (0.48, 1.0, 60, 0.8, [1]), (0.5, 1.0, 64, 0.7, [1])]
    merged = enc.merge_same_pitch(ev)
    assert [(e[0], e[1], e[2]) for e in merged] == [(0.0, 1.0, 60), (0.5, 1.0, 64)]


def test_active_level_uses_only_active_blocks():
    stem = np.concatenate([np.zeros(int(2 * SR), np.float32), noise(2, -20)])
    mask = np.array([False, True])
    assert abs(enc.active_level_db(stem, mask, SR) - (-20)) < 0.5
    assert enc.active_level_db(stem, np.array([False, False]), SR) is None


def test_header_writes_stem_and_level_meta():
    st = enc.Stage("notes.bass", src="basic-pitch:onnx", ok=True, stem="bass", level_db=-31.24)
    st.lines = ["1:1.0  E1  1.0b 90"]
    text = "\n".join(enc._stage_lines(st))
    assert "meta    stem=bass  level=-31.2dB" in text


# --- rulings from the baseline (Task 7) ----------------------------------------

def _clicks(secs, db, every=0.5):
    y = np.zeros(int(secs * SR), np.float32)
    for t in np.arange(0.25, secs, every):
        i = int(t * SR)
        y[i:i + 300] = np.hanning(300) * 10 ** (db / 20) * 4
    return y


def test_percussion_is_gated_on_a_silent_drum_stem(tmp_path):
    import soundfile as sf
    stem = tmp_path / "drums.wav"
    sf.write(str(stem), _clicks(8, -75), SR)
    mix = np.stack([noise(8, -12)] * 2)
    grid = {"bar_dur": 2.0, "downbeat": 0.0}
    st = enc.stage_percussion(stem, mix, SR, grid)
    assert not st.ok
    assert any("silent" in w for w in st.warns)


def test_percussion_keeps_a_real_drum_part_and_records_level(tmp_path):
    import soundfile as sf
    stem = tmp_path / "drums.wav"
    sf.write(str(stem), _clicks(8, -20), SR)
    mix = np.stack([noise(8, -18)] * 2)
    grid = {"bar_dur": 2.0, "downbeat": 0.0}
    st = enc.stage_percussion(stem, mix, SR, grid)
    assert st.ok and st.stem == "drums" and st.level_db is not None


def test_vocals_fallback_is_marked_for_compare(tmp_path):
    from soundcode import compare as cmp
    from soundcode.parser import parse
    doc = parse("%sc 0.3\n\n:notes.vox\nmeta stem=vocals\n@0.0 C4 1s 90\n")
    assert cmp.stem_for_stream(doc, "notes.vox") == "vocals"
    assert enc.vocal_stem_name(Path("x/out/stems/s/vocals.wav")) == "vocals"
    assert enc.vocal_stem_name(Path("x/out/stems/s/lead_vocals.wav")) == "lead_vocals"


# --- Task 10 findings: grid consistency and the ungated pyin fallback ----------------

def test_grid_agrees_with_how_notes_are_positioned(monkeypatch):
    """Notes are placed at downbeat + (bar-1)*bar_dur; the :grid the renderer
    reads must put every bar exactly there, whatever local tempo estimates say."""
    import librosa
    from soundcode.expand import build_grid
    from soundcode.parser import parse

    sr = 22050
    y = np.zeros(sr * 16, np.float32)                     # short: a single anchor
    for t in np.arange(0.5, 16, 0.5):                     # 120 bpm clicks
        y[int(t * sr):int(t * sr) + 200] = np.hanning(200)
    real_tempo = librosa.feature.tempo

    def local_is_wrong(**k):                              # only the per-window call
        return np.array([90.0]) if "aggregate" in k and k["aggregate"] is None \
            else real_tempo(**k)
    monkeypatch.setattr(librosa.feature, "tempo", local_is_wrong)
    st, grid = enc.stage_grid(np.stack([y, y]), sr, 16.0)
    doc = parse("%sc 0.3\n\n" + "\n".join(enc._stage_lines(st)))
    g = build_grid(doc)
    for bar in range(1, 8):
        want = grid["downbeat"] + (bar - 1) * grid["bar_dur"]
        assert abs(g.time_of(bar, 1.0) - want) < 0.005, bar


def test_pyin_fallback_does_not_undo_the_gate():
    gated = enc.Stage("notes.bass", warns=["stem silent (below the loudness gate throughout)"])
    gated.gated = True
    failed = enc.Stage("notes.bass", warns=["basic-pitch unavailable (ImportError)"])
    assert not enc.needs_fallback(gated)
    assert enc.needs_fallback(failed)


def test_gate_keeps_a_sparse_soft_part():
    """One short -42 dB note per 2 s block (a sparse soft piano) is real music:
    its 2 s mean is ~-51 dB, but its loudest 100 ms clears the floor."""
    t = np.arange(int(0.25 * SR)) / SR
    note = (10 ** (-42 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stem = np.zeros(int(8 * SR), np.float32)
    for k in range(4):
        stem[int(k * 2 * SR):int(k * 2 * SR) + note.size] = note
    mix = noise(8, -30)
    assert enc.active_blocks(stem, mix, SR).all()
