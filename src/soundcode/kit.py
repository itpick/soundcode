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
PRE_S, FADE_IN_S, FADE_OUT_S = 0.005, 0.005, 0.02
TAIL_FLOOR_DB = -40.0          # below this, a chopped tail isn't worth extending

# How long each voice should ring before any extension, matching the voice
# names `gm.drum_voice`/`gm.DRUM_NOTES` use. Anything not listed here (an
# unmapped `gmNN`, a shaker, a cowbell, ...) gets the default.
VOICE_TARGET_S = {
    "kick": 0.5, "snare": 0.6, "clap": 0.6,
    "hat": 0.25,                        # closed hat
    "hat.open": 1.5, "crash": 1.5, "ride": 1.5,
}
DEFAULT_TARGET_S = 0.6


def target_length_s(voice: str) -> float:
    """How long a one-shot for `voice` should ring before any extension."""
    if voice in VOICE_TARGET_S:
        return VOICE_TARGET_S[voice]
    if voice.startswith("tom"):
        return 0.8
    return DEFAULT_TARGET_S


def isolated(hits: list[tuple[float, str, int]], before: float = 0.08,
             after: float = 0.15) -> dict[str, list[float]]:
    raw_times = [t for t, _, _ in hits]
    sorted_times = sorted(raw_times)

    def gap_after(t: float) -> float:
        import bisect
        i = bisect.bisect_right(sorted_times, t)
        return sorted_times[i] - t if i < len(sorted_times) else float("inf")

    by_voice: dict[str, list[tuple[float, int, float]]] = {}
    for i, (t, v, vel) in enumerate(hits):
        # any *other* hit, including one at the very same instant, crowds this one
        crowded = any(0 <= t - u <= before or 0 <= u - t <= after
                      for j, u in enumerate(raw_times) if j != i)
        if not crowded:
            by_voice.setdefault(v, []).append((t, vel, gap_after(t)))
    out = {}
    for v, lst in by_voice.items():
        target = target_length_s(v)
        # prefer hits with room to ring to their target length uninterrupted;
        # among those, the ones nearest the voice's typical (median) velocity
        rung = [x for x in lst if x[2] >= target]
        if rung:
            med = float(np.median([vel for _, vel, _ in rung]))
            chosen = sorted(rung, key=lambda x: abs(x[1] - med))[:MAX_PER_VOICE]
        else:
            # no hit has a clean gap (e.g. a hat grid with no pause anywhere):
            # take whatever gives the sample the most room before extending
            chosen = sorted(lst, key=lambda x: -x[2])[:MAX_PER_VOICE]
        out[v] = [t for t, _, _ in chosen]
    return out


def _fit_decay_rate(tail: np.ndarray, sr: int) -> float:
    """Decay rate k (per second, >= 0) for an envelope ~ exp(-k*t) fitted to
    the amplitude trend across `tail` (n, ch). 0.0 if there isn't enough
    signal, or it isn't decaying, to tell."""
    n = len(tail)
    frames = 8
    frame_n = max(1, n // frames)
    mono = tail.mean(axis=1)
    env, t = [], []
    for i in range(0, n - frame_n + 1, frame_n):
        chunk = mono[i:i + frame_n].astype(np.float64)
        env.append(np.sqrt(np.mean(chunk ** 2)) + 1e-9)
        t.append((i + frame_n / 2) / sr)
    if len(env) < 2:
        return 0.0
    slope, _ = np.polyfit(t, np.log(env), 1)
    return max(0.0, -float(slope))


def _loop_tail(tail: np.ndarray, length: int) -> np.ndarray:
    """A seamlessly repeatable version of `tail` (n, ch), tiled to `length`
    samples: crossfades the tail's end into its own start (equal-power) so
    the repeat has no seam, then tiles."""
    n = len(tail)
    cross = max(1, n // 4)
    looped = tail.copy()
    fade_out = (np.cos(np.linspace(0, np.pi / 2, cross)) ** 2)[:, None]
    fade_in = (np.sin(np.linspace(0, np.pi / 2, cross)) ** 2)[:, None]
    looped[-cross:] = looped[-cross:] * fade_out + tail[:cross] * fade_in
    reps = int(np.ceil(length / n)) + 1
    return np.tile(looped, (reps, 1))[:length]


def _extend_tail(seg: np.ndarray, sr: int, target_n: int,
                 floor_db: float = TAIL_FLOOR_DB) -> np.ndarray:
    """Extend `seg` (n, ch) to `target_n` samples when it was cut short while
    still ringing, instead of chopping it. Fits an exponential decay to the
    last ~50 ms of the cut, then continues the sound past the cut by looping
    that window (crossfaded so the loop itself has no seam) and shaping it
    with the fitted decay -- so the tail keeps ringing down rather than
    sustaining or clicking. The join to the real audio is itself crossfaded.
    No-op if the cut is already long enough, or already quiet (<= `floor_db`
    relative to the sample's peak) -- chopping silence isn't audible."""
    if len(seg) >= target_n:
        return seg
    tail_n = min(int(0.05 * sr), len(seg) // 2)
    if tail_n < 16:
        return seg
    peak = float(np.abs(seg).max())
    if peak <= 1e-9:
        return seg
    tail = seg[-tail_n:]
    tail_rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
    if 20 * np.log10(max(tail_rms, 1e-12) / peak) <= floor_db:
        return seg

    edge = max(1, tail_n // 8)
    join_level = float(np.sqrt(np.mean(tail[-edge:].astype(np.float64) ** 2)) + 1e-9)
    k = _fit_decay_rate(tail, sr)

    need = target_n - len(seg)
    cross_n = min(tail_n // 2, need)
    loop = _loop_tail(tail, need + cross_n)
    t = np.arange(len(loop)) / sr
    ext = loop * np.exp(-k * t)[:, None]
    # Anchor the continuation to the real audio's own level right at the
    # join (not to an a-priori guess), so it is level-continuous by
    # construction; the fitted `k` still governs the decay from there on.
    at_join = float(np.sqrt(np.mean(ext[cross_n:cross_n + edge].astype(np.float64) ** 2)) + 1e-9)
    ext *= join_level / at_join

    out = seg.copy()
    fade_out = (np.cos(np.linspace(0, np.pi / 2, cross_n)) ** 2)[:, None]
    fade_in = (np.sin(np.linspace(0, np.pi / 2, cross_n)) ** 2)[:, None]
    out[-cross_n:] = out[-cross_n:] * fade_out + ext[:cross_n] * fade_in
    out = np.concatenate([out, ext[cross_n:cross_n + need]], axis=0)
    return out[:target_n]


def build(drum_stem: Path, hits: list[tuple[float, str, int]], out_dir: Path) -> dict[str, list[Path]]:
    import soundfile as sf

    y, sr = sf.read(str(drum_stem), always_2d=True)
    y = y.astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_t = sorted(t for t, _, _ in hits)
    made: dict[str, list[Path]] = {}
    for v, ts in isolated(hits).items():
        target_s = target_length_s(v)
        target_n = int(target_s * sr)
        for k, t in enumerate(ts):
            nxt = next((u for u in all_t if u > t + 1e-6), t + target_s)
            a, b = int(max(0, t - PRE_S) * sr), int(min(nxt - PRE_S, t + target_s) * sr)
            seg = y[a:b].copy()
            if len(seg) < int(0.02 * sr):
                continue
            seg = _extend_tail(seg, sr, target_n)
            fi, fo = int(FADE_IN_S * sr), min(int(FADE_OUT_S * sr), len(seg))
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
