"""Alignment: lag between reference and rebuild, and per-window drift.

Spec 2026-09-28-benchmark-scorer, Task 1.
"""

from __future__ import annotations

import numpy as np

from .slices import Slice, _mono

DRIFT_THRESHOLD_MS = 30.0
SILENCE_RMS = 1e-5


def _slice_audio(y: np.ndarray, sr: int, a: float, b: float) -> np.ndarray:
    mono = _mono(y)
    i0 = max(int(a * sr), 0)
    i1 = min(int(b * sr), mono.shape[0])
    return np.asarray(mono[i0:i1], dtype=np.float32)


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))


def _onset_envelope(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    import librosa

    mono = np.asarray(_mono(y), dtype=np.float32)
    return librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)


def _search_lag(
    o_ref: np.ndarray,
    o_est: np.ndarray,
    ms_per_frame: float,
    max_lag: int,
) -> float | None:
    """Best lag (in ms) by Pearson correlation of the onset envelopes.

    Searches lags of -max_lag..+max_lag frames, taking the overlapping part
    of each envelope at that lag. Returns None if there is no candidate lag
    with at least 2 overlapping frames of non-zero-variance signal on both
    sides (rather than defaulting to a lag of 0, which would misreport "in
    sync" as a real finding).
    """
    m = min(o_ref.size, o_est.size)
    if m < 2:
        return None
    o_ref, o_est = o_ref[:m], o_est[:m]
    if np.std(o_ref) == 0.0 or np.std(o_est) == 0.0:
        return None

    best_lag: int | None = None
    best_corr = -np.inf
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

    if best_lag is None:
        return None
    return best_lag * ms_per_frame


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
    ref_seg = _slice_audio(y_ref, sr, a, b)
    est_seg = _slice_audio(y_est, sr, a, b)
    n = min(ref_seg.size, est_seg.size)
    if n == 0:
        return None
    ref_seg, est_seg = ref_seg[:n], est_seg[:n]

    if _rms(ref_seg) < SILENCE_RMS or _rms(est_seg) < SILENCE_RMS:
        return None

    hop = max(sr // 100, 1)
    o_ref = _onset_envelope(ref_seg, sr, hop)
    o_est = _onset_envelope(est_seg, sr, hop)
    frames_per_sec = sr / hop
    ms_per_frame = 1000.0 / frames_per_sec
    max_lag = int(round(search_s * frames_per_sec))
    return _search_lag(o_ref, o_est, ms_per_frame, max_lag)


def drift(y_ref: np.ndarray, y_est: np.ndarray, sr: int, wins: list[Slice]) -> list[dict]:
    """Per-window lag, with drift flagged when |lag_ms| > 30.

    Computes each side's onset-strength envelope ONCE over the whole signal
    (the expensive step), then slices that envelope per window by frame
    index and runs the same lag search `lag_ms` uses — instead of
    recomputing the envelope from scratch for every window.
    """
    search_s = 1.5
    hop = max(sr // 100, 1)
    o_ref_full = _onset_envelope(y_ref, sr, hop)
    o_est_full = _onset_envelope(y_est, sr, hop)
    frames_per_sec = sr / hop
    ms_per_frame = 1000.0 / frames_per_sec
    max_lag = int(round(search_s * frames_per_sec))

    out = []
    for s in wins:
        lag = None
        ref_seg = _slice_audio(y_ref, sr, s.a, s.b)
        est_seg = _slice_audio(y_est, sr, s.a, s.b)
        n = min(ref_seg.size, est_seg.size)
        if n > 0:
            ref_seg, est_seg = ref_seg[:n], est_seg[:n]
            if _rms(ref_seg) >= SILENCE_RMS and _rms(est_seg) >= SILENCE_RMS:
                fa = int(round(s.a * frames_per_sec))
                fb = int(round(s.b * frames_per_sec))
                lag = _search_lag(o_ref_full[fa:fb], o_est_full[fa:fb], ms_per_frame, max_lag)
        out.append({
            "label": s.label,
            "a": s.a,
            "b": s.b,
            "lag_ms": lag,
            "drift": lag is not None and abs(lag) > DRIFT_THRESHOLD_MS,
        })
    return out
