""":contour.vox — the lead vocal's f0 curve, 20 ms resolution, in cents.

The note list alone caps a rebuilt vocal at ~30 cents from the original
(singing spike, 2026-09-26): scoops, slides and vibrato live between the
notes. This stream carries them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RATE_HZ = 50
STEP_S = 1.0 / RATE_HZ
GAP_S = 0.06


def phrases(times: np.ndarray, cents: np.ndarray, voiced: np.ndarray) -> list[tuple[float, list[int]]]:
    if not np.any(voiced):
        return []
    grid = np.arange(times[0], times[-1] + 1e-9, STEP_S)
    v = np.interp(grid, times, voiced.astype(float)) > 0.5
    c = np.interp(grid, times[voiced], cents[voiced]) if voiced.sum() > 1 else \
        np.full_like(grid, float(cents[voiced][0]))
    out: list[tuple[float, list[int]]] = []
    i, n = 0, len(grid)
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i
        while j < n:
            if v[j]:
                j += 1
                continue
            k = j
            while k < n and not v[k]:
                k += 1
            if (k - j) * STEP_S < GAP_S and k < n:
                j = k                               # short gap: keep going (interpolated)
            else:
                break
        out.append((float(grid[i]), [int(round(x)) for x in c[i:j]]))
        i = j
    return out


def contour_lines(ph: list[tuple[float, list[int]]], per_line: int = 50) -> list[str]:
    lines = []
    for start, vals in ph:
        for k in range(0, len(vals), per_line):
            chunk = vals[k:k + per_line]
            lines.append(f"f0  @{start + k * STEP_S:.3f}  " + " ".join(str(v) for v in chunk))
    return lines


def read_contour(doc, stream: str = "contour.vox") -> list[tuple[float, np.ndarray]]:
    s = doc.stream(stream)
    if s is None:
        return []
    out = []
    for name, args in s.statements:
        if name != "f0" or not args or not args[0].startswith("@"):
            continue
        out.append((float(args[0][1:]), np.array([int(a) for a in args[1:]], dtype=float)))
    return out


def extract(stem: Path, sr: int = 16000) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    import librosa
    import torch
    import torchcrepe

    y, _ = librosa.load(str(stem), sr=sr, mono=True)
    audio = torch.tensor(y)[None]
    hz, per = torchcrepe.predict(audio, sr, hop_length=sr // 100, fmin=50.0, fmax=1100.0,
                                 model="full", return_periodicity=True, batch_size=512,
                                 decoder=torchcrepe.decode.weighted_argmax,
                                 device="cpu")
    per = torchcrepe.filter.median(per, 3)
    hz = torchcrepe.filter.median(hz, 3)
    hz, per = hz[0].numpy(), per[0].numpy()
    voiced = per >= 0.5
    cents = np.where(voiced, 1200 * np.log2(np.maximum(hz, 1e-6) / 440.0) + 6900, 0.0)
    times = np.arange(len(hz)) * 0.01
    return times, cents, voiced, float(per[voiced].mean()) if voiced.any() else 0.0
