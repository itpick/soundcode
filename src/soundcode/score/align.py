"""Alignment: lag between reference and rebuild, and per-window drift.

Spec 2026-09-28-benchmark-scorer, Task 1.
"""

from __future__ import annotations

import numpy as np

from .slices import Slice, _mono

DRIFT_THRESHOLD_MS = 30.0
SILENCE_RMS = 1e-5
LAG_PEAK_TOLERANCE = 0.05   # Amendment 3: nearest peak within 0.05 of the max


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


def _xcorr(o_ref: np.ndarray, o_est: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation of the two envelopes at every lag in
    -max_lag..+max_lag frames, over just the part that overlaps at that lag
    (the envelopes are first cut to the shorter one's length).

    Exactly what a per-lag `np.corrcoef` loop gives, but in one FFT: the
    cross-products come from an FFT cross-correlation, and each overlap's
    sums and sums of squares from prefix sums. A lag with < 2 overlapping
    frames, or a (numerically) flat overlap on either side, is NaN.
    Returns `(lags, corr)`; positive lag = est is late."""
    m = min(o_ref.size, o_est.size)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.full(lags.size, np.nan)
    if m < 2:
        return lags, corr
    r = np.asarray(o_ref[:m], dtype=np.float64)
    e = np.asarray(o_est[:m], dtype=np.float64)
    # centre first: Pearson is shift-invariant, and it keeps the variance
    # sums below free of catastrophic cancellation
    r = r - r.mean()
    e = e - e.mean()

    nfft = 1 << int(np.ceil(np.log2(2 * m)))
    cc = np.fft.irfft(np.conj(np.fft.rfft(r, nfft)) * np.fft.rfft(e, nfft), nfft)
    # cc[L] = sum_i r[i] e[i+L] for L >= 0; cc[nfft+L] likewise for L < 0
    c_r, c_e = np.concatenate([[0.0], np.cumsum(r)]), np.concatenate([[0.0], np.cumsum(e)])
    q_r, q_e = np.concatenate([[0.0], np.cumsum(r * r)]), np.concatenate([[0.0], np.cumsum(e * e)])
    scale_r, scale_e = q_r[-1], q_e[-1]

    for k, lag in enumerate(lags):
        n = m - abs(lag)
        if n < 2:
            continue
        if lag >= 0:
            r0, r1, e0, e1 = 0, m - lag, lag, m
            sxy = cc[lag]
        else:
            r0, r1, e0, e1 = -lag, m, 0, m + lag
            sxy = cc[nfft + lag]
        sx, sy = c_r[r1] - c_r[r0], c_e[e1] - c_e[e0]
        vx = (q_r[r1] - q_r[r0]) - sx * sx / n
        vy = (q_e[e1] - q_e[e0]) - sy * sy / n
        if vx <= 1e-12 * max(scale_r, 1e-300) or vy <= 1e-12 * max(scale_e, 1e-300):
            continue
        corr[k] = min(max((sxy - sx * sy / n) / np.sqrt(vx * vy), -1.0), 1.0)
    return lags, corr


def _search_lag(
    o_ref: np.ndarray,
    o_est: np.ndarray,
    ms_per_frame: float,
    max_lag: int,
) -> float | None:
    """The lag (in ms) of `o_est` behind `o_ref`, from the onset envelopes.

    Amendment 3 (2026-09-29): a periodic envelope correlates at every beat
    multiple, and a far peak can win by a hair (Discipline drums: median
    |lag| 10 ms, yet three windows at +987/-1227 ms). So the answer is not
    the global maximum but the **nearest peak**: of the local maxima of the
    correlation curve (`_xcorr`, lags -max_lag..+max_lag), the one with the
    smallest |lag| whose correlation is within `LAG_PEAK_TOLERANCE` (0.05)
    of the maximum; on an |lag| tie, the higher correlation.

    Returns None if either envelope is flat or no lag has >= 2 overlapping
    frames of non-zero-variance signal on both sides (rather than defaulting
    to a lag of 0, which would misreport "in sync" as a real finding).
    """
    m = min(o_ref.size, o_est.size)
    if m < 2:
        return None
    o_ref, o_est = o_ref[:m], o_est[:m]
    if np.std(o_ref) == 0.0 or np.std(o_est) == 0.0:
        return None

    lags, corr = _xcorr(o_ref, o_est, max_lag)
    ok = np.isfinite(corr)
    if not ok.any():
        return None
    best = float(np.max(corr[ok]))
    c = np.where(ok, corr, -np.inf)
    left = np.concatenate([[-np.inf], c[:-1]])
    right = np.concatenate([c[1:], [-np.inf]])
    peak = ok & (c >= left) & (c >= right) & (c >= best - LAG_PEAK_TOLERANCE)
    idx = np.flatnonzero(peak)
    k = min(idx, key=lambda i: (abs(int(lags[i])), -c[i]))
    return int(lags[k]) * ms_per_frame


def lag_ms(
    y_ref: np.ndarray,
    y_est: np.ndarray,
    sr: int,
    a: float,
    b: float,
    search_s: float = 1.5,
) -> float | None:
    """Lag of `y_est` behind `y_ref` in [a, b), in ms; positive = est is late.

    Computed from the onset-strength envelopes (10 ms hop): the nearest
    correlation peak in [-search_s, search_s] within 0.05 of the best
    (`_search_lag`). None if either slice is silent.
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
