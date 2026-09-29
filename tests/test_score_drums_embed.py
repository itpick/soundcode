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


# --- embed chunk stitching (fake model: no real MERT needed) -------------

def test_chunk_stitching_has_no_boundary_drift(monkeypatch):
    """A fake model that drops ~1 frame per independent call (mimicking a
    valid-convolution feature encoder run on an isolated chunk) must still
    stitch to round(duration * FRAME_RATE) +/- 1 frames, with frame i at
    time i / FRAME_RATE throughout — not drifting by ~1 frame per chunk
    boundary, which is what naive (no-overlap) concatenation used to do.
    """
    def fake_run_chunk(chunk):
        t = max(chunk.shape[0] // 320 - 1, 0)          # the lost edge frame
        if t == 0:
            return np.zeros((0, 768), np.float32)
        # content encodes each frame's true position in `chunk` (its own
        # array, not the whole song) as a "position marker" in every column,
        # the way a real feature vector is a deterministic function of a
        # window of input samples around that position
        idx = (np.arange(t) * 320).astype(np.int64)
        marker = chunk[idx].astype(np.float32)
        return np.tile(marker[:, None], (1, 768))

    monkeypatch.setattr(embed, "_run_chunk", fake_run_chunk)

    dur = 65.0                                          # 2 chunk boundaries at 30 s, 60 s
    n = int(round(dur * embed.TARGET_SR))
    y = (np.arange(n, dtype=np.float64) / embed.TARGET_SR).astype(np.float32)  # y[i] = its own time

    out = embed._frames_for_audio(y)

    expected = round(dur * embed.FRAME_RATE)
    assert abs(out.shape[0] - expected) <= 1              # length: no cumulative drift

    # alignment: frame i's content-derived marker must equal the (globally
    # normalized) sample at i / FRAME_RATE seconds in.
    # (naive, no-overlap concatenation would have failed BOTH assertions: it
    # loses ~1 frame at every boundary, so length would be off by ~2 here,
    # and the position markers after the first boundary would run ~1 frame
    # behind their row index.)
    y_norm = embed._normalize(y)                          # _frames_for_audio normalizes once, up front
    expected_idx = (np.arange(out.shape[0]) * 320).astype(np.int64)
    assert np.allclose(out[:, 0], y_norm[expected_idx], atol=1e-6)


def test_chunk_stitching_length_matches_a_single_pass(monkeypatch):
    """Same fake model, run once continuously vs. chunked: same length +/- 1."""
    def fake_run_chunk(chunk):
        t = max(chunk.shape[0] // 320 - 1, 0)
        return np.zeros((t, 768), np.float32)

    monkeypatch.setattr(embed, "_run_chunk", fake_run_chunk)
    dur = 65.0
    n = int(round(dur * embed.TARGET_SR))
    y = np.zeros(n, np.float32)

    continuous = fake_run_chunk(y)
    chunked = embed._frames_for_audio(y)
    assert abs(chunked.shape[0] - continuous.shape[0]) <= 1


# --- embed cache key includes model identity ------------------------------

def test_cache_path_changes_with_model_id(monkeypatch):
    monkeypatch.setattr(embed, "MODEL_ID", "m-a-p/MERT-v1-95M")
    p1 = embed._cache_path("deadbeef")
    monkeypatch.setattr(embed, "MODEL_ID", "some-other/model-v2")
    p2 = embed._cache_path("deadbeef")
    assert p1 != p2
    assert p1.parent == p2.parent == embed.CACHE_DIR
    assert p1.name.startswith("deadbeef.") and p2.name.startswith("deadbeef.")


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


def _naive_chunks(y: np.ndarray) -> np.ndarray:
    """The pre-fix behavior: independent, non-overlapping chunks straight
    concatenated (no context, no time-based frame selection). Used only as
    a "before" baseline in the regression test below."""
    y = embed._normalize(y)
    chunk_len = int(round(embed.CHUNK_S * embed.TARGET_SR))
    parts = [embed._run_chunk(y[s:s + chunk_len]) for s in range(0, y.size, chunk_len)
             if y[s:s + chunk_len].size]
    return np.concatenate(parts, axis=0)


@pytest.mark.skipif(not _mert_available(), reason="MERT model not installed; run scripts/install_mert.sh")
def test_mert_chunking_matches_a_continuous_pass():
    """65 s of a non-stationary signal (a linear chirp, tone changing
    throughout): the real model, chunked (`_frames_for_audio`, 30 s chunks +
    1 s overlap-and-discard) vs. run once continuously, must (a) have the
    same length within 1 frame, and (b) be markedly closer to the
    continuous pass at every checkpoint than the pre-fix, no-overlap,
    naive chunking was — proving the fix is real, not just plausible.

    NOTE on thresholds: MERT-v1-95M's transformer layers use full
    self-attention over its ENTIRE input in one forward call, so a chunk's
    embeddings differ from a continuous pass's by more than "the CNN
    feature encoder's ~480-sample (20 ms) receptive field lost at each
    edge" — every frame's embedding is shaped by attention to every OTHER
    frame the model saw in that call, which for a 30 s (+ context) chunk is
    a much smaller window than the full 65 s continuous pass sees. No
    amount of reasonably-sized overlap-and-discard padding closes that gap
    (measured: even +-15 s of context on both sides only reaches ~0.88-0.92
    cosine at a chunk boundary here, at several times the compute cost) —
    it is an inherent property of chunking a full-attention transformer,
    not a bug this fix can remove. The absolute floor below is therefore
    set from what +-1 s of context actually, reliably achieves (all
    measured checkpoints landed >= 0.75; the floor of 0.7 below is the
    measured worst case with 0.05 of margin), not the higher bar an
    idealized "just add overlap" model would allow.
    """
    from scipy.signal import chirp

    sr = embed.TARGET_SR
    dur = 65.0
    t = np.arange(int(dur * sr)) / sr
    y = (0.2 * chirp(t, f0=100.0, f1=2000.0, t1=dur, method="linear")).astype(np.float32)

    continuous = embed._run_chunk(embed._normalize(y))   # one pass, no chunking: the reference
    naive = _naive_chunks(y)                              # the pre-fix "before"
    chunked = embed._frames_for_audio(y)                  # the code under test (normalizes internally)

    expected = round(dur * embed.FRAME_RATE)
    assert abs(chunked.shape[0] - expected) <= 1
    assert abs(continuous.shape[0] - expected) <= 2
    # the pre-fix bug, reproduced: naive drops about 1 frame per boundary
    # (2 boundaries in 65 s / 30 s chunks), so it comes up short here too.
    assert naive.shape[0] < chunked.shape[0]

    def frame_cosine(arr, i):
        u, v = arr[i], continuous[i]
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        return float(np.dot(u, v) / denom) if denom > 1e-12 else None

    for label, check_s in (("start", 0.5), ("~30s", 30.0), ("~60s", 60.0), ("end", dur - 0.5)):
        idx_c = min(int(round(check_s * embed.FRAME_RATE)), chunked.shape[0] - 1)
        idx_n = min(idx_c, naive.shape[0] - 1)
        fixed_cos = frame_cosine(chunked, idx_c)
        naive_cos = frame_cosine(naive, idx_n)
        assert fixed_cos is not None and naive_cos is not None
        assert fixed_cos >= 0.7, f"{label} (frame {idx_c}): fixed cosine {fixed_cos} below the floor"
        assert fixed_cos > naive_cos, \
            f"{label}: fixed ({fixed_cos}) should beat naive ({naive_cos})"


def test_chunk_and_context_are_multiples_of_merts_320_sample_stride():
    assert embed.MERT_STRIDE == 320
    for s in (embed.CHUNK_S, embed.CONTEXT_S):
        assert round(s * embed.TARGET_SR) % embed.MERT_STRIDE == 0
    embed._check_stride(30.0, 1.0)                   # the shipped values pass
    with pytest.raises(ValueError, match="320"):
        embed._check_stride(30.0, 0.99)              # 23760 samples: not a stride multiple
    with pytest.raises(ValueError, match="320"):
        embed._check_stride(30.005, 1.0)
