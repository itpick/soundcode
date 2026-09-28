"""Metrics on precomputed per-file features.

Spec 2026-09-28-benchmark-scorer, Task 2.

`Features.of` runs every heavy step (librosa, basic-pitch, torchcrepe) ONCE
per file. `slice_metrics` only slices those precomputed arrays by time — it
never re-runs feature extraction, so it is cheap to call once per (window or
section) x part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import compare
from .slices import Slice
from .slices import _mono as _mono_mix

SR = compare.SR
_HOP = compare._HOP
_LOGSPEC_N_FFT = 4096
_LOGSPEC_HOP = 2048
_LOGSPEC_LO = 40.0
_LOGSPEC_HI = 16000.0
_LOGSPEC_LIVE_DB = 50.0
_F0_RATE = 100.0

# Which `part` strings get which part-restricted metrics (design doc: keys,
# guitar, other get note F1; bass and the two vocal parts get f0 agreement).
NOTE_F1_PARTS = {"piano", "guitar", "other"}
F0_PARTS = {"bass", "lead_vocals", "backing_vocals"}


def _notes(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """basic-pitch transcription of `y`, via `compare.transcribe` on a temp WAV."""
    import os
    import tempfile

    import soundfile as sf

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, y, sr, subtype="FLOAT")
        return compare.transcribe(path)
    finally:
        os.unlink(path)


def _f0(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """torchcrepe f0 at 16 kHz, 10 ms hop, weighted_argmax; voiced = periodicity >= 0.5."""
    import librosa
    import torch
    import torchcrepe

    y16 = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=16000) \
        if sr != 16000 else np.asarray(y, np.float32)
    hop = int(round(16000 * 0.01))
    hz, periodicity = torchcrepe.predict(
        torch.tensor(y16, dtype=torch.float32)[None], 16000, hop_length=hop,
        fmin=50.0, fmax=1100.0, model="full", return_periodicity=True,
        batch_size=512, device="cpu", decoder=torchcrepe.decode.weighted_argmax)
    return hz[0].numpy(), (periodicity[0].numpy() >= 0.5)


def _logspec(y: np.ndarray, sr: int) -> np.ndarray:
    """1/12-octave log-spectrum per frame: |STFT|^2 pooled into 12-per-octave
    bands from 40 Hz to 16 kHz, in dB. Shape (n_bands, n_frames)."""
    import librosa

    n_bands = int(np.floor(12 * np.log2(_LOGSPEC_HI / _LOGSPEC_LO))) + 1
    bands = _LOGSPEC_LO * 2.0 ** (np.arange(n_bands) / 12.0)
    if y.size == 0:
        return np.zeros((n_bands, 0))
    power = np.abs(librosa.stft(y, n_fft=_LOGSPEC_N_FFT, hop_length=_LOGSPEC_HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_LOGSPEC_N_FFT)
    edge = 2.0 ** (1.0 / 24.0)
    out = np.full((n_bands, power.shape[1]), -200.0)
    for i, fc in enumerate(bands):
        mask = (freqs >= fc / edge) & (freqs < fc * edge)
        if mask.any():
            out[i] = 10.0 * np.log10(power[mask].sum(axis=0) + 1e-20)
    return out


@dataclass
class Features:
    """Everything `slice_metrics` needs, computed once for a whole file."""

    y: np.ndarray                              # mono, at SR
    sr: int
    onsets: np.ndarray                          # onset times, seconds
    chroma: np.ndarray                          # (12, n_frames), hop 512
    rms: np.ndarray                             # (n_frames,), hop 512
    logspec: np.ndarray                         # (n_bands, n_frames), hop 2048
    notes: tuple[np.ndarray, np.ndarray] | None  # (iv, hz), only when pitched
    f0: np.ndarray | None                        # hz, 100 Hz, only when f0
    voiced: np.ndarray | None                    # bool, 100 Hz, only when f0

    @staticmethod
    def of(y: np.ndarray, sr: int, pitched: bool, f0: bool) -> "Features":
        import librosa

        mono = np.asarray(_mono_mix(y), dtype=np.float32)
        if sr != SR:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)

        onsets = librosa.onset.onset_detect(y=mono, sr=SR, units="time")
        chroma = librosa.feature.chroma_cqt(y=mono, sr=SR, hop_length=_HOP)
        rms = librosa.feature.rms(y=mono, hop_length=_HOP)[0]
        logspec = _logspec(mono, SR)
        notes = _notes(mono, SR) if pitched else None
        hz, voiced = _f0(mono, SR) if f0 else (None, None)
        return Features(mono, SR, onsets, chroma, rms, logspec, notes, hz, voiced)


def _slice_y(f: Features, a: float, b: float) -> np.ndarray:
    i0 = max(int(a * f.sr), 0)
    i1 = min(int(b * f.sr), f.y.shape[0])
    return f.y[i0:i1] if i1 > i0 else f.y[:0]


def _frame_idx(a: float, b: float, hop: int, sr: int, n: int) -> tuple[int, int]:
    fa = max(int(round(a * sr / hop)), 0)
    fb = min(int(round(b * sr / hop)), n)
    return fa, fb


def _level_diff(yr: np.ndarray, ye: np.ndarray) -> float | None:
    lr, le = compare.level_db(yr), compare.level_db(ye)
    return (le - lr) if lr is not None and le is not None else None


def _env_corr(ref: Features, est: Features, a: float, b: float) -> float | None:
    fa_r, fb_r = _frame_idx(a, b, _HOP, ref.sr, ref.rms.shape[0])
    fa_e, fb_e = _frame_idx(a, b, _HOP, est.sr, est.rms.shape[0])
    n = min(fb_r - fa_r, fb_e - fa_e)
    if n < 2:
        return None
    x, y = ref.rms[fa_r:fa_r + n], est.rms[fa_e:fa_e + n]
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    # float accumulation can leave a near-zero-but-not-exactly-zero std (e.g. a
    # constant array whose repeated sum drifts a hair in float32), which the
    # check above misses and corrcoef turns into NaN: never surface that.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else None


def _chroma(ref: Features, est: Features, a: float, b: float) -> float | None:
    fa_r, fb_r = _frame_idx(a, b, _HOP, ref.sr, ref.chroma.shape[1])
    fa_e, fb_e = _frame_idx(a, b, _HOP, est.sr, est.chroma.shape[1])
    n = min(fb_r - fa_r, fb_e - fa_e)
    if n <= 0:
        return None
    u = ref.chroma[:, fa_r:fa_r + n].mean(1)
    v = est.chroma[:, fa_e:fa_e + n].mean(1)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / denom) if denom > 1e-12 else None


def _logspec_db(ref: Features, est: Features, a: float, b: float) -> float | None:
    fa_r, fb_r = _frame_idx(a, b, _LOGSPEC_HOP, ref.sr, ref.logspec.shape[1])
    fa_e, fb_e = _frame_idx(a, b, _LOGSPEC_HOP, est.sr, est.logspec.shape[1])
    n = min(fb_r - fa_r, fb_e - fa_e)
    if n <= 0:
        return None
    r = ref.logspec[:, fa_r:fa_r + n].mean(1)
    e = est.logspec[:, fa_e:fa_e + n].mean(1)
    live = r >= (r.max() - _LOGSPEC_LIVE_DB)
    if not live.any():
        return None
    r_norm, e_norm = r - r.mean(), e - e.mean()
    return float(np.mean(np.abs(r_norm[live] - e_norm[live])))


def _onset_f1(ref: Features, est: Features, a: float, b: float,
              yr: np.ndarray, ye: np.ndarray) -> float | None:
    import mir_eval

    if compare._silent(yr) and compare._silent(ye):
        return None
    on_r = ref.onsets[(ref.onsets >= a) & (ref.onsets < b)]
    on_e = est.onsets[(est.onsets >= a) & (est.onsets < b)]
    if len(on_r) == 0 or len(on_e) == 0:
        return 0.0
    return float(mir_eval.onset.f_measure(on_r, on_e, window=0.05)[0])


def _note_f1(ref: Features, est: Features, a: float, b: float) -> float | None:
    r_iv, r_hz = ref.notes
    e_iv, e_hz = est.notes
    r_iv, r_hz = np.asarray(r_iv, float), np.asarray(r_hz, float)
    e_iv, e_hz = np.asarray(e_iv, float), np.asarray(e_hz, float)
    r_on = r_iv[:, 0] if r_iv.size else np.zeros(0)
    e_on = e_iv[:, 0] if e_iv.size else np.zeros(0)
    r_mask = (r_on >= a) & (r_on < b)
    e_mask = (e_on >= a) & (e_on < b)
    return compare.note_f1(r_iv[r_mask], r_hz[r_mask], e_iv[e_mask], e_hz[e_mask],
                           onset_tolerance=0.1, pitch_tolerance=100.0)


def _f0_cents(ref: Features, est: Features, a: float, b: float) -> float | None:
    fa_r, fb_r = _frame_idx(a, b, 1, int(_F0_RATE), ref.f0.shape[0])
    fa_e, fb_e = _frame_idx(a, b, 1, int(_F0_RATE), est.f0.shape[0])
    n = min(fb_r - fa_r, fb_e - fa_e)
    if n <= 0:
        return None
    hr, vr = ref.f0[fa_r:fa_r + n], ref.voiced[fa_r:fa_r + n]
    he, ve = est.f0[fa_e:fa_e + n], est.voiced[fa_e:fa_e + n]
    both = vr & ve
    if both.sum() < 10:
        return None
    cents = 1200.0 * np.log2(he[both] / hr[both])
    return float(np.median(np.abs(cents)))


def slice_metrics(ref: Features, est: Features, s: Slice, part: str) -> dict[str, float | None]:
    """Slice both files' precomputed features to `[s.a, s.b)` and score `part`.

    Never re-runs librosa/basic-pitch/torchcrepe: only slices arrays that
    `Features.of` already computed. Every value is None (not an error) on
    silence, a flat signal, a too-short slice, or a slice past the end of
    either signal.
    """
    a, b = s.a, s.b
    out: dict[str, float | None] = {k: None for k in (
        "note_f1", "chroma", "onset_f1", "f0_cents", "env_corr",
        "level_diff_db", "logspec_db")}

    yr, ye = _slice_y(ref, a, b), _slice_y(est, a, b)
    if yr.size == 0 or ye.size == 0:
        return out

    out["onset_f1"] = _onset_f1(ref, est, a, b, yr, ye)

    if not (compare._silent(yr) or compare._silent(ye)):
        out["level_diff_db"] = _level_diff(yr, ye)
        out["env_corr"] = _env_corr(ref, est, a, b)
        out["chroma"] = _chroma(ref, est, a, b)
        out["logspec_db"] = _logspec_db(ref, est, a, b)

    if part in NOTE_F1_PARTS and ref.notes is not None and est.notes is not None:
        out["note_f1"] = _note_f1(ref, est, a, b)

    if part in F0_PARTS and ref.f0 is not None and est.f0 is not None:
        out["f0_cents"] = _f0_cents(ref, est, a, b)

    return out
