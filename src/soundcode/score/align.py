"""Alignment: lag between reference and rebuild, and per-window drift.

Spec 2026-09-28-benchmark-scorer, Task 1.
"""

from __future__ import annotations

import numpy as np

from .slices import Slice, _mono

DRIFT_THRESHOLD_MS = 30.0


def _slice_audio(y: np.ndarray, sr: int, a: float, b: float) -> np.ndarray:
    mono = _mono(y)
    i0 = max(int(a * sr), 0)
    i1 = min(int(b * sr), mono.shape[0])
    return np.asarray(mono[i0:i1], dtype=np.float32)


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))


def lag_ms(
    y_ref: np.ndarray,
    y_est: np.ndarray,
    sr: int,
    a: float,
    b: float,
    search_s: float = 1.5,
) -> float | None:
    """Lag of `y_est` behind `y_ref` in [a, b), in ms; positive = est is late.

    Computed from the onset-strength envelopes (10 ms hop): the lag in
    [-search_s, search_s] with the best Pearson correlation. None if either
    slice is silent.
    """
    import librosa

    ref_seg = _slice_audio(y_ref, sr, a, b)
    est_seg = _slice_audio(y_est, sr, a, b)
    n = min(ref_seg.size, est_seg.size)
    if n == 0:
        return None
    ref_seg, est_seg = ref_seg[:n], est_seg[:n]

    if _rms(ref_seg) < 1e-5 or _rms(est_seg) < 1e-5:
        return None

    hop = max(sr // 100, 1)
    o_ref = librosa.onset.onset_strength(y=ref_seg, sr=sr, hop_length=hop)
    o_est = librosa.onset.onset_strength(y=est_seg, sr=sr, hop_length=hop)
    m = min(o_ref.size, o_est.size)
    if m < 2:
        return None
    o_ref, o_est = o_ref[:m], o_est[:m]

    if np.std(o_ref) == 0.0 or np.std(o_est) == 0.0:
        return None

    max_lag = int(round(search_s * 100))
    best_lag, best_corr = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = o_ref[: m - lag] if lag else o_ref
            e = o_est[lag:] if lag else o_est
        else:
            r = o_ref[-lag:]
            e = o_est[: m + lag]
        if r.size < 2 or np.std(r) == 0.0 or np.std(e) == 0.0:
            continue
        corr = float(np.corrcoef(r, e)[0, 1])
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag * 10.0


def drift(y_ref: np.ndarray, y_est: np.ndarray, sr: int, wins: list[Slice]) -> list[dict]:
    """Per-window lag, with drift flagged when |lag_ms| > 30."""
    out = []
    for s in wins:
        lag = lag_ms(y_ref, y_est, sr, s.a, s.b)
        out.append({
            "label": s.label,
            "a": s.a,
            "b": s.b,
            "lag_ms": lag,
            "drift": lag is not None and abs(lag) > DRIFT_THRESHOLD_MS,
        })
    return out
