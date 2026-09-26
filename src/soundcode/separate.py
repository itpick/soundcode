"""Stem separation: split a mix into voice and instrument stems.

Three passes, each a pretrained model run through `audio-separator`:

    1. BS-RoFormer        mix          -> vocals, instrumental
    2. Karaoke RoFormer   vocals       -> lead_vocals, backing_vocals
    3. HTDemucs 6-stem    instrumental -> drums, bass, guitar, piano, other

Whatever the models leave behind (original minus the sum of the stems) is
written as `residual.wav`, so the stems plus the residual always rebuild the
mix exactly, and the level of the residual says how much separation lost.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import soundfile as sf

SR = 44100
STEMS = ("lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other")

_LABEL_RE = re.compile(r"_\(([^()]+)\)")


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------

def read_stereo(path: Path | str, sr: int = SR, length: int | None = None) -> np.ndarray:
    """Any audio file -> (2, n) float32 at `sr`; mono is duplicated."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    y = y[:2].astype(np.float32)
    return fit_length(y, length) if length is not None else y


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad or trim to exactly n samples; models are often off by a few."""
    if y.shape[1] >= n:
        return y[:, :n]
    return np.pad(y, ((0, 0), (0, n - y.shape[1])))


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    # float WAV: separated stems can exceed full scale and must not clip
    sf.write(str(path), y.T, sr, subtype="FLOAT")


def rms_db(y: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20 * math.log10(r) if r > 0 else -math.inf


def label_outputs(paths: Iterable[Path]) -> dict[str, Path]:
    """audio-separator names outputs `<input>_(<Stem>)_<model>.wav`; the stem
    label is the last parenthesised group, lowercased."""
    out: dict[str, Path] = {}
    for p in paths:
        groups = _LABEL_RE.findall(Path(p).name)
        if groups:
            out[groups[-1].strip().lower()] = Path(p)
    return out


# --------------------------------------------------------------------------
# sum check
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SumReport:
    mix_db: float          # RMS of the original, dBFS
    sum_db: float          # RMS of all stems summed, dBFS
    level_diff_db: float   # sum_db - mix_db
    residual_db: float     # RMS of (mix - sum) relative to the mix, dB
    ok: bool


def sum_check(mix: np.ndarray, stems: Mapping[str, np.ndarray],
              max_level_diff_db: float = 1.0,
              max_residual_db: float = -15.0) -> SumReport:
    total = np.zeros_like(mix)
    for y in stems.values():
        total += y
    mix_db, sum_db = rms_db(mix), rms_db(total)
    res_db = rms_db(mix - total)

    if mix_db == -math.inf:
        # a silent mix: fine only if the stems are silent too
        diff = 0.0 if sum_db == -math.inf else math.inf
        rel = -math.inf if res_db == -math.inf else math.inf
    else:
        diff = sum_db - mix_db
        rel = res_db - mix_db
    ok = abs(diff) <= max_level_diff_db and rel <= max_residual_db
    return SumReport(mix_db, sum_db, diff, rel, ok)
