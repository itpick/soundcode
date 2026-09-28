"""Production profile of a stem, and how to put it back on a rendered part.

Measured by the encoder (fx line in each note/perc stream), applied by the
renderer: tone (31-band EQ curve), room (rt60 + wet), stereo width and pan,
dynamics (crest). Old files without an fx line render exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BANDS = 20.0 * 2 ** (np.arange(31) / 3)
_EDGE = 2 ** (1 / 6)


@dataclass
class Fx:
    eq: list[int] = field(default_factory=list)
    rt60: float = 0.3
    wet: float = 0.1
    width: float = 0.0
    pan: float = 0.0
    crest: float = 12.0


def band_db(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, np.float64)
    if y.ndim == 2:
        y = y.mean(0)
    if not y.size or not np.any(y):
        return np.zeros(len(BANDS))
    n_fft = 8192
    frames = [y[i:i + n_fft] for i in range(0, max(1, len(y) - n_fft + 1), n_fft // 2)]
    win = np.hanning(n_fft)
    spec = np.mean([np.abs(np.fft.rfft(np.pad(f, (0, n_fft - len(f))) * win)) ** 2
                    for f in frames], axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    e = np.array([spec[(freqs >= fc / _EDGE) & (freqs < fc * _EDGE)].sum() for fc in BANDS])
    db = 10 * np.log10(e + 1e-20)
    ref = db.max()
    live = db >= ref - 60
    out = db - db[live].mean()
    out[~live] = -60.0
    return out


def _rms_db(x: np.ndarray) -> float:
    return 10 * float(np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20))


def measure(stereo: np.ndarray, sr: int, offsets: list[float]) -> Fx:
    st = np.asarray(stereo, np.float32)
    if st.ndim == 1:
        st = np.stack([st, st])
    mono = st.mean(0)
    f = Fx(eq=[int(round(v)) for v in band_db(mono, sr)])
    if not np.any(mono):
        return f
    # stereo image
    l, r = st[0].astype(np.float64), st[1].astype(np.float64)
    # width = how different the channels are (a panned mono source is 0 wide)
    corr = np.sum(l * r) / (np.sqrt(np.sum(l ** 2) * np.sum(r ** 2)) + 1e-20)
    f.width = float(np.clip(1 - abs(corr), 0, 1))
    if min(np.mean(l ** 2), np.mean(r ** 2)) < 1e-3 * max(np.mean(l ** 2), np.mean(r ** 2)):
        f.width = 0.0                     # one channel ~silent: a hard-panned mono source
    el, er = np.mean(l ** 2), np.mean(r ** 2)
    f.pan = float(np.clip((er - el) / (er + el + 1e-20), -1, 1))
    f.crest = float(20 * np.log10(np.abs(mono).max() / (np.sqrt(np.mean(mono.astype(np.float64) ** 2)) + 1e-20)))
    # room: how the energy decays after an isolated note (one followed by space).
    # Busy passages have no measurable tail; they keep the gentle defaults.
    hop = int(0.01 * sr)
    env = np.array([_rms_db(mono[i:i + hop]) for i in range(0, len(mono) - hop, hop)])
    slopes = []
    for t0 in offsets:
        k0 = int(t0 / 0.01)
        head = env[k0:k0 + 30]
        if len(head) < 10:
            continue
        start = k0 + int(np.argmax(head)) + 10
        tail = env[start:start + 100]
        if len(tail) < 20 or tail.max() < env.max() - 60:
            continue
        slope = np.polyfit(np.arange(len(tail)) * 0.01, tail, 1)[0]      # dB per second
        if slope < -20:
            slopes.append(slope)
    if slopes:
        f.rt60 = float(np.clip(-60.0 / np.median(slopes), 0.1, 4.0))
        f.wet = float(np.clip((f.rt60 - 0.2) / 2, 0.05, 0.35))
    return f


def isolated_offsets(onsets: list[float], gap: float = 0.5) -> list[float]:
    """Onsets of notes followed by at least `gap` seconds before the next one:
    the only places a room's decay can be heard."""
    on = sorted(float(t) for t in onsets)
    return [a for a, b in zip(on, on[1:]) if b - a >= gap]


def fx_line(f: Fx) -> str:
    eq = ",".join(str(int(v)) for v in f.eq)
    return (f"fx      eq={eq}  rt60={f.rt60:.2f}s  wet={f.wet:.2f}  width={f.width:.2f}  "
            f"pan={f.pan:.2f}  crest={f.crest:.1f}dB")


# hand-edited .sc files are allowed: anything outside these ranges is clamped
_RANGES = {"rt60": (0.1, 4.0), "wet": (0.0, 1.0), "width": (0.0, 1.0), "pan": (-1.0, 1.0),
           "crest": (0.0, 60.0)}


def parse_fx(stream) -> Fx | None:
    if stream is None:
        return None
    for name, args in stream.statements:
        if name != "fx":
            continue
        f = Fx()
        for tok in args:
            k, _, v = tok.partition("=")
            try:
                if k == "eq":
                    vals = [int(np.clip(int(x), -60, 60)) for x in v.split(",") if x.strip()]
                    f.eq = vals if len(vals) == len(BANDS) else []
                elif k in _RANGES:
                    x = float(v.rstrip("sdB"))
                    if np.isfinite(x):
                        lo, hi = _RANGES[k]
                        setattr(f, k, float(np.clip(x, lo, hi)))
            except ValueError:
                continue
        return f
    return None


EQ_LIMIT_DB = 15.0
EQ_FLOOR_DB = 40.0          # no boost where the render is this far under its own peak band


def eq_match(stereo: np.ndarray, sr: int, target_eq: list[int]) -> np.ndarray:
    import librosa

    if len(target_eq) != len(BANDS) or not np.any(stereo):
        return stereo
    have = band_db(stereo.T, sr)
    gain = np.clip(np.asarray(target_eq, float) - have, -EQ_LIMIT_DB, EQ_LIMIT_DB)
    # never lift bands the render has (almost) nothing in
    gain = np.where(have <= -EQ_FLOOR_DB, np.minimum(gain, 0.0), gain)
    n_fft = 4096
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    g = np.interp(np.log2(np.maximum(freqs, BANDS[0])), np.log2(BANDS), gain)
    lin = 10 ** (g / 20)
    out = np.empty_like(stereo)
    for ch in range(stereo.shape[1]):
        S = librosa.stft(stereo[:, ch], n_fft=n_fft, hop_length=n_fft // 4)
        out[:, ch] = librosa.istft(S * lin[:, None], hop_length=n_fft // 4, length=stereo.shape[0])
    return out.astype(np.float32)


def _room_size(rt60: float) -> float:
    return float(np.clip((rt60 - 0.1) / 3.0, 0.05, 0.98))


def apply(stereo: np.ndarray, sr: int, f: Fx) -> np.ndarray:
    import pedalboard as pb

    y = eq_match(stereo, sr, f.eq) if f.eq else stereo
    mono = y.mean(1).astype(np.float64)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    have_crest = 20 * np.log10(np.abs(mono).max() / (rms + 1e-20) + 1e-20) if rms > 0 else 0.0
    if rms > 0 and have_crest - f.crest > 3:
        # limit peaks to (RMS + target crest) at a fixed working level, so the
        # threshold means the same on every part whatever the synth's level
        g = 10 ** (-20 / 20) / rms
        lim = pb.Pedalboard([pb.Limiter(threshold_db=float(-20 + max(f.crest, 3.0)),
                                        release_ms=80)])
        y = lim((y * g).T.astype(np.float32), sr).T / g
    if f.wet > 0.02:
        verb = pb.Pedalboard([pb.Reverb(room_size=_room_size(f.rt60), wet_level=f.wet,
                                        dry_level=1 - f.wet / 2, width=1.0)])
        y = verb(y.T.astype(np.float32), sr).T
    mid, side = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
    side_now = np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12)
    want = f.width                                                 # side/mid ~ 1 - |corr|
    side = side * (np.clip(want / side_now, 0, 4) if side_now > 1e-4 else 0.0)
    if side_now <= 1e-4 and f.width > 0.05:                        # mono render: decorrelate
        side = np.roll(mid, int(0.011 * sr)) * np.sqrt(want)
    l, r = mid + side, mid - side
    theta = (f.pan + 1) * np.pi / 4
    out = np.stack([l * np.cos(theta) * np.sqrt(2), r * np.sin(theta) * np.sqrt(2)], 1)
    return out.astype(np.float32)
