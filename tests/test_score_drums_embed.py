"""Per-voice drum onsets/decay, and MERT frame embeddings (spec 2026-09-28-benchmark-scorer, Task 3)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundcode.score import drums, embed  # noqa: E402


# --- drums.voice_f1 -----------------------------------------------------

REF_ONSETS = [(0.5, 36), (1.0, 38), (1.5, 42), (2.0, 36)]          # kick snare hat kick
EST_ONSETS = [(0.5, 36), (1.5, 42), (2.0, 36)]                     # snare missing


def _fake_transcribe(audio, out_midi, model):
    Path(out_midi).parent.mkdir(parents=True, exist_ok=True)
    Path(out_midi).write_bytes(b"")     # never read: _midi_onsets is monkeypatched
    return Path(out_midi)


def test_voice_f1_per_voice(tmp_path, monkeypatch):
    wav_ref = tmp_path / "ref.wav"
    wav_est = tmp_path / "est.wav"
    wav_ref.write_bytes(b"reference-audio-bytes")
    wav_est.write_bytes(b"estimate-audio-bytes")
    monkeypatch.setattr(drums.tsumugi, "transcribe", _fake_transcribe)

    monkeypatch.setattr(drums, "_midi_onsets", lambda midi: REF_ONSETS)
    ref = drums.voice_onsets(wav_ref, tmp_path / "work_ref")

    monkeypatch.setattr(drums, "_midi_onsets", lambda midi: EST_ONSETS)
    est = drums.voice_onsets(wav_est, tmp_path / "work_est")

    result = drums.voice_f1(ref, est, 0.0, 3.0)
    assert result["kick"] == pytest.approx(1.0)
    assert result["snare"] == pytest.approx(0.0)
    assert result["hat"] == pytest.approx(1.0)
    assert result["clap"] is None


def test_voice_onsets_groups_by_gm_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(drums.tsumugi, "transcribe", _fake_transcribe)
    monkeypatch.setattr(drums, "_midi_onsets",
                         lambda midi: [(0.1, 36), (0.2, 35), (0.3, 39), (0.4, 49), (0.5, 70)])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"some-bytes")
    voices = drums.voice_onsets(wav, tmp_path / "work")
    assert sorted(voices["kick"]) == [0.1, 0.2]      # 36 and 35 (kick.acoustic) both -> kick
    assert list(voices["clap"]) == [0.3]             # 39
    assert list(voices["other"]) == [0.4, 0.5]        # crash (49), shaker (gm70)
    assert list(voices["snare"]) == []
    assert list(voices["hat"]) == []


def test_voice_onsets_is_cached_by_sha1(tmp_path, monkeypatch):
    calls = []

    def counting_transcribe(audio, out_midi, model):
        calls.append(audio)
        return _fake_transcribe(audio, out_midi, model)

    monkeypatch.setattr(drums.tsumugi, "transcribe", counting_transcribe)
    monkeypatch.setattr(drums, "_midi_onsets", lambda midi: [(0.5, 36)])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"identical-bytes")
    work = tmp_path / "work"
    drums.voice_onsets(wav, work)
    drums.voice_onsets(wav, work)          # same file, same work dir: cache hit
    assert len(calls) == 1


def test_voice_f1_one_sided_is_zero_not_none(tmp_path, monkeypatch):
    monkeypatch.setattr(drums.tsumugi, "transcribe", _fake_transcribe)
    monkeypatch.setattr(drums, "_midi_onsets", lambda midi: [(0.5, 36)])
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    ref = drums.voice_onsets(wav, tmp_path / "work")
    est = {k: np.array([]) for k in ref}
    result = drums.voice_f1(ref, est, 0.0, 3.0)
    assert result["kick"] == 0.0


# --- drums.decay_s -------------------------------------------------------

def test_decay_s_synthetic_exponential():
    sr = 22050
    dur = 1.0
    n = int(dur * sr)
    t = np.arange(n) / sr
    tau = 0.3 / np.log(10)        # -20 dB (ratio 0.1) reached at t = 0.3 s
    rng = np.random.default_rng(0)
    carrier = rng.standard_normal(n).astype(np.float64)
    y = (carrier * np.exp(-t / tau)).astype(np.float32)
    d = drums.decay_s(y, sr, np.array([0.0]))
    assert d == pytest.approx(0.3, abs=0.03)


def test_decay_s_no_onsets_is_none():
    y = np.zeros(1000, np.float32)
    assert drums.decay_s(y, 22050, np.array([])) is None


def test_decay_s_medians_multiple_isolated_hits():
    sr = 22050
    tau1 = 0.3 / np.log(10)
    tau2 = 0.5 / np.log(10)
    rng = np.random.default_rng(1)

    def hit(tau, secs=0.8):
        n = int(secs * sr)
        t = np.arange(n) / sr
        return (rng.standard_normal(n) * np.exp(-t / tau)).astype(np.float32)

    y = np.concatenate([hit(tau1), hit(tau2)])
    onsets = np.array([0.0, 0.8])
    d = drums.decay_s(y, sr, onsets)
    assert d == pytest.approx(np.median([0.3, 0.5]), abs=0.03)


# --- embed.cosine (embed.frames monkeypatched) ---------------------------

def test_cosine_identical_array_is_one(monkeypatch):
    rng = np.random.default_rng(2)
    fr = rng.standard_normal((300, 768)).astype(np.float32)
    monkeypatch.setattr(embed, "frames", lambda wav: fr)
    c = embed.cosine(embed.frames("a"), embed.frames("a"), 0.0, 4.0)
    assert c == pytest.approx(1.0, abs=1e-5)


def test_cosine_orthogonal_pair_is_near_zero(monkeypatch):
    t = 300
    fr_ref = np.zeros((t, 768), np.float32)
    fr_ref[:, 0] = 1.0
    fr_est = np.zeros((t, 768), np.float32)
    fr_est[:, 1] = 1.0
    c = embed.cosine(fr_ref, fr_est, 0.0, 4.0)
    assert c == pytest.approx(0.0, abs=1e-6)


def test_cosine_slice_past_the_end_is_none():
    fr = np.zeros((75, 768), np.float32)   # 1 s of frames at 75 Hz
    assert embed.cosine(fr, fr, 5.0, 6.0) is None


# --- embed real MERT smoke test ------------------------------------------

def _mert_available() -> bool:
    return embed.MODEL_DIR is not None and (embed.MODEL_DIR / "config.json").exists()


@pytest.mark.skipif(not _mert_available(), reason="MERT model not installed; run scripts/install_mert.sh")
def test_mert_self_cosine_is_near_one(tmp_path):
    import soundfile as sf

    sr = 24000
    t = np.arange(int(3.0 * sr)) / sr
    y = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(str(path), y, sr)

    fr = embed.frames(path)
    assert fr.shape[1] == 768
    c = embed.cosine(fr, fr, 0.0, 3.0)
    assert c > 0.99
