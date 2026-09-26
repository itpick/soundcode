"""Compare a render against the original, stem by stem.

Every metric returns None when it has nothing to measure (a silent side),
never NaN: a report must say "no evidence", not print a number that lies.
"""

from __future__ import annotations

import math

import numpy as np

SR = 22050
_HOP = 512
_SILENT_DB = -70.0


def json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def level_db(y: np.ndarray) -> float | None:
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    return 20 * math.log10(r) if r > 0 else None


def _silent(y: np.ndarray) -> bool:
    lv = level_db(y)
    return lv is None or lv < _SILENT_DB


def note_f1(ref_iv, ref_hz, est_iv, est_hz, octave_agnostic: bool = False) -> float | None:
    import mir_eval

    if len(ref_iv) == 0 and len(est_iv) == 0:
        return None
    if len(ref_iv) == 0 or len(est_iv) == 0:
        return 0.0
    ref_hz, est_hz = np.asarray(ref_hz, float), np.asarray(est_hz, float)
    if octave_agnostic:
        fold = lambda h: 440.0 * 2 ** (((12 * np.log2(h / 440.0)) % 12) / 12)  # noqa: E731
        ref_hz, est_hz = fold(ref_hz), fold(est_hz)
    return float(mir_eval.transcription.precision_recall_f1_overlap(
        np.asarray(ref_iv, float), ref_hz, np.asarray(est_iv, float), est_hz,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)[2])


def chroma_blocks(y_ref, y_est, blocks) -> list[float | None]:
    import librosa

    n = min(len(y_ref), len(y_est))
    y_ref, y_est = y_ref[:n], y_est[:n]
    cr = librosa.feature.chroma_cqt(y=y_ref, sr=SR, hop_length=_HOP)
    ce = librosa.feature.chroma_cqt(y=y_est, sr=SR, hop_length=_HOP)
    out: list[float | None] = []
    for a, b in blocks:
        i, j = int(a * SR), int(b * SR)
        if _silent(y_ref[i:j]) or _silent(y_est[i:j]):
            out.append(None)
            continue
        fa, fb = int(a * SR / _HOP), max(int(b * SR / _HOP), int(a * SR / _HOP) + 1)
        u, v = cr[:, fa:fb].mean(1), ce[:, fa:fb].mean(1)
        out.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
    return out


def onset_f1(y_ref, y_est) -> float | None:
    import librosa
    import mir_eval

    if _silent(y_ref) and _silent(y_est):
        return None
    on_r = librosa.onset.onset_detect(y=y_ref, sr=SR, units="time")
    on_e = librosa.onset.onset_detect(y=y_est, sr=SR, units="time")
    if len(on_r) == 0 or len(on_e) == 0:
        return 0.0
    return float(mir_eval.onset.f_measure(on_r, on_e, window=0.07)[0])


def energy_corr(y_ref, y_est) -> float | None:
    import librosa

    n = min(len(y_ref), len(y_est))
    if _silent(y_ref[:n]) or _silent(y_est[:n]):
        return None
    a = librosa.feature.rms(y=y_ref[:n], hop_length=_HOP)[0]
    b = librosa.feature.rms(y=y_est[:n], hop_length=_HOP)[0]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def blocks_from_grid(doc, duration: float) -> list[tuple[float, float]]:
    from .expand import build_grid

    if doc.stream("grid") is not None:
        grid = build_grid(doc)
        edges, bar = [], 1
        while True:
            t = grid.time_of(bar, 1.0)
            if t >= duration or bar > 10000:
                break
            edges.append(t)
            bar += 1
        edges.append(duration)
        blocks = [(a, b) for a, b in zip(edges, edges[1:]) if b > a]
        if blocks:
            return blocks
    edges = list(np.arange(0.0, duration, 2.0)) + [duration]
    return [(float(a), float(b)) for a, b in zip(edges, edges[1:]) if b > a]
