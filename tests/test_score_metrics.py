import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundcode.score import metrics  # noqa: E402
from soundcode.score.slices import Slice  # noqa: E402

SR = 22050


def tone(freq, secs, sr=SR, amp=0.3):
    t = np.arange(int(secs * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr(metrics, "_notes", lambda y, sr: (np.array([[0.5, 1.5], [2.0, 3.0]]), np.array([440.0, 494.0])))
    monkeypatch.setattr(metrics, "_f0", lambda y, sr, **kwargs: (np.full(400, 440.0), np.ones(400, bool)))


def test_identical_parts_are_perfect():
    y = tone(440, 4.0)
    f = metrics.Features.of(y, SR, pitched=True, f0=True)
    m = metrics.slice_metrics(f, f, Slice("song", "song", 0.0, 4.0), "piano")
    assert m["chroma"] == pytest.approx(1.0) and m["note_f1"] == pytest.approx(1.0)
    assert m["level_diff_db"] == pytest.approx(0.0, abs=0.01) and m["logspec_db"] == pytest.approx(0.0, abs=0.01)


def test_quieter_and_silent_parts():
    y = tone(440, 4.0)
    ref = metrics.Features.of(y, SR, pitched=False, f0=False)
    est = metrics.Features.of(y * 0.5, SR, pitched=False, f0=False)
    m = metrics.slice_metrics(ref, est, Slice("song", "song", 0.0, 4.0), "other")
    assert m["level_diff_db"] == pytest.approx(-6.02, abs=0.1)
    z = metrics.Features.of(np.zeros_like(y), SR, pitched=False, f0=False)
    m0 = metrics.slice_metrics(ref, z, Slice("song", "song", 0.0, 4.0), "other")
    assert m0["env_corr"] is None and m0["chroma"] is None


def test_slice_past_the_end_is_none_not_an_error():
    f = metrics.Features.of(tone(440, 4.0), SR, pitched=False, f0=False)
    short = metrics.Features.of(tone(440, 2.0), SR, pitched=False, f0=False)
    m = metrics.slice_metrics(f, short, Slice("window", "0:03", 3.0, 4.0), "other")
    assert all(v is None for v in m.values())


def test_cover_tolerance_note_f1_forgives_60ms_and_60_cents():
    from soundcode import compare
    ref_iv, ref_hz = np.array([[1.0, 2.0]]), np.array([440.0])
    est_iv, est_hz = np.array([[1.06, 2.0]]), np.array([440.0 * 2 ** (60 / 1200)])
    assert compare.note_f1(ref_iv, ref_hz, est_iv, est_hz) == 0.0            # default ±50 ms/±50 c unchanged
    assert compare.note_f1(ref_iv, ref_hz, est_iv, est_hz, onset_tolerance=0.1, pitch_tolerance=100.0) == 1.0


# --- Supplementary tests (not from the brief's Step 1) -----------------------


def test_note_f1_and_f0_cents_are_none_for_non_matching_part_types():
    y = tone(440, 4.0)
    f = metrics.Features.of(y, SR, pitched=True, f0=True)
    m = metrics.slice_metrics(f, f, Slice("song", "song", 0.0, 4.0), "drums")
    assert m["note_f1"] is None and m["f0_cents"] is None


def test_f0_cents_computed_for_bass_and_vocal_parts():
    y = tone(440, 4.0)
    f = metrics.Features.of(y, SR, pitched=False, f0=True)
    for part in ("bass", "lead_vocals", "backing_vocals"):
        m = metrics.slice_metrics(f, f, Slice("song", "song", 0.0, 4.0), part)
        assert m["f0_cents"] == pytest.approx(0.0)
    assert metrics.slice_metrics(f, f, Slice("song", "song", 0.0, 4.0), "piano")["f0_cents"] is None


def test_flat_rms_env_corr_is_none_not_nan():
    # a real (non-silent) part whose RMS is exactly constant across frames:
    # the correlation denominator is 0, and the guard must return None, not NaN.
    y = tone(440, 4.0)
    f = metrics.Features.of(y, SR, pitched=False, f0=False)
    flat = metrics.Features(f.y, f.sr, f.onsets, f.chroma, np.full_like(f.rms, 0.1),
                            f.logspec, None, None, None)
    m = metrics.slice_metrics(flat, flat, Slice("song", "song", 0.0, 4.0), "other")
    assert m["env_corr"] is None
    assert all(v is None or np.isfinite(v) for v in m.values())


def test_logspec_db_ignores_a_pure_level_difference():
    # a signal with energy in several bands (not just one), so the fix's
    # "normalise by the mean over live bands only" actually gets exercised.
    t = np.arange(int(4.0 * SR)) / SR
    y = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 880 * t)
         + 0.075 * np.sin(2 * np.pi * 3300 * t)).astype(np.float32)
    est_y = (y * 10 ** (-9.5 / 20)).astype(np.float32)   # same shape, -9.5 dB
    ref = metrics.Features.of(y, SR, pitched=False, f0=False)
    est = metrics.Features.of(est_y, SR, pitched=False, f0=False)
    m = metrics.slice_metrics(ref, est, Slice("song", "song", 0.0, 4.0), "other")
    assert m["logspec_db"] == pytest.approx(0.0, abs=0.05)


def test_features_of_passes_f0_fmin_through_to_f0(monkeypatch):
    calls = []

    def fake_f0(y, sr, **kwargs):
        calls.append(kwargs.get("fmin"))
        return np.full(400, 440.0), np.ones(400, bool)

    monkeypatch.setattr(metrics, "_f0", fake_f0)
    metrics.Features.of(tone(440, 4.0), SR, pitched=False, f0=True, f0_fmin=30.0)
    assert calls == [30.0]


def test_onset_f1_uses_a_narrower_window_than_compare_onset_f1():
    from soundcode import compare

    def clicks(times, total=4.0):
        y = np.zeros(int(total * SR), np.float32)
        for t in times:
            i = int(t * SR)
            y[i:i + 200] = np.hanning(200)
        return y

    ref = clicks([0.5, 1.5, 2.5])
    est = clicks([0.55, 1.55, 2.55])  # 50 ms late: inside compare's 0.07 s window, outside ours (0.05 s)
    f_ref = metrics.Features.of(ref, SR, pitched=False, f0=False)
    f_est = metrics.Features.of(est, SR, pitched=False, f0=False)
    m = metrics.slice_metrics(f_ref, f_est, Slice("song", "song", 0.0, 4.0), "other")
    assert m["onset_f1"] is not None
    assert m["onset_f1"] <= compare.onset_f1(ref, est)
