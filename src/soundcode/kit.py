"""The song's own drum kit: one-shots cut from the separated drum stem at the
transcribed hit times, played back round-robin at the rebuilt pattern.

These samples are audio from the original recording (the faithful-rebuild
path); the .sc still carries every hit as code, and without `meta kit` the
render falls back to the General MIDI kit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import fsutil

MAX_PER_VOICE = 4
PRE_S, MAX_S, FADE_IN_S, FADE_OUT_S = 0.005, 0.6, 0.005, 0.02


def isolated(hits: list[tuple[float, str, int]], before: float = 0.08,
             after: float = 0.15) -> dict[str, list[float]]:
    times = [t for t, _, _ in hits]
    by_voice: dict[str, list[tuple[float, int]]] = {}
    for i, (t, v, vel) in enumerate(hits):
        # any *other* hit, including one at the very same instant, crowds this one
        crowded = any(0 <= t - u <= before or 0 <= u - t <= after
                      for j, u in enumerate(times) if j != i)
        if not crowded:
            by_voice.setdefault(v, []).append((t, vel))
    out = {}
    for v, lst in by_voice.items():
        med = float(np.median([vel for _, vel in lst]))
        out[v] = [t for t, _ in sorted(lst, key=lambda x: abs(x[1] - med))][:MAX_PER_VOICE]
    return out


def build(drum_stem: Path, hits: list[tuple[float, str, int]], out_dir: Path) -> dict[str, list[Path]]:
    import soundfile as sf

    y, sr = sf.read(str(drum_stem), always_2d=True)
    y = y.astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_t = sorted(t for t, _, _ in hits)
    made: dict[str, list[Path]] = {}
    for v, ts in isolated(hits).items():
        for k, t in enumerate(ts):
            nxt = next((u for u in all_t if u > t + 1e-6), t + MAX_S)
            a, b = int(max(0, t - PRE_S) * sr), int(min(nxt - PRE_S, t + MAX_S) * sr)
            seg = y[a:b].copy()
            if len(seg) < int(0.02 * sr):
                continue
            fi, fo = int(FADE_IN_S * sr), int(FADE_OUT_S * sr)
            seg[:fi] *= np.linspace(0, 1, fi)[:, None]
            seg[-fo:] *= np.linspace(1, 0, fo)[:, None]
            p = out_dir / f"{v}_{k}.wav"
            sf.write(str(p), seg, sr, subtype="FLOAT")
            made.setdefault(v, []).append(p)
    return made


def load(kit_dir: Path, sr: int) -> dict[str, list[np.ndarray]]:
    import librosa
    import soundfile as sf

    out: dict[str, list[np.ndarray]] = {}
    for p in fsutil.wavs(kit_dir):
        voice = p.stem.rsplit("_", 1)[0]
        y, s = sf.read(str(p), always_2d=True)
        y = y.astype(np.float32)
        if s != sr:
            y = np.stack([librosa.resample(y[:, c], orig_sr=s, target_sr=sr) for c in range(y.shape[1])], 1)
        if y.shape[1] == 1:
            y = np.repeat(y, 2, 1)
        out.setdefault(voice, []).append(y[:, :2])
    return out


def play(hits: list[tuple[float, str, int]], samples: dict[str, list[np.ndarray]],
         n: int, sr: int = 44100) -> tuple[np.ndarray, list[tuple[float, str, int]]]:
    out = np.zeros((n, 2), np.float32)
    missing, rr = [], {}
    for t, v, vel in sorted(hits):
        if v not in samples:
            missing.append((t, v, vel))
            continue
        k = rr.get(v, 0)
        rr[v] = k + 1
        s = samples[v][k % len(samples[v])]
        i = int(t * sr)
        if i >= n:
            continue
        m = min(len(s), n - i)
        out[i:i + m] += s[:m] * (max(vel, 1) / 127) ** 1.5
    return out, missing
