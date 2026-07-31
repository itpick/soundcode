"""Mock renderer: .sc -> audio.

This is deliberately crude. Its job is not to sound good — it is to be
*correct* in pitch, timing and structure, because it is the only channel by
which melodic and rhythmic detail reaches the decoder (spec §7.3, Path B).
ACE-Step Remix supplies the timbre and production; this supplies the tune.

Crude but correct beats pretty but wrong, every time.
"""

from __future__ import annotations

import numpy as np

from .expand import Note, expand
from .model import Document

# stream-kind mix levels and stereo placement (-1 left .. +1 right)
_GAIN = {
    "drums": 0.90, "bass": 0.85, "keys": 0.42, "synth": 0.45,
    "voice": 0.80, "gtr": 0.55, "unknown": 0.50,
}
_PAN = {"keys": 0.0, "synth": 0.0, "voice": 0.0}

_DEFAULT_ENV = {
    "bass": {"a": 0.006, "d": 0.09, "s": 0.75, "r": 0.06},
    "keys": {"a": 0.018, "d": 0.35, "s": 0.45, "r": 0.55},
    "voice": {"a": 0.020, "d": 0.12, "s": 0.72, "r": 0.22},
    "synth": {"a": 0.030, "d": 0.30, "s": 0.60, "r": 0.40},
}


def _adsr(n: int, sr: int, env: dict[str, float], dur: float) -> np.ndarray:
    a = max(int(env.get("a", 0.01) * sr), 1)
    d = max(int(env.get("d", 0.1) * sr), 1)
    s = float(env.get("s", 0.7))
    r = max(int(env.get("r", 0.15) * sr), 1)

    out = np.zeros(n, dtype=np.float32)
    body = max(n - r, 1)
    i = 0
    seg = min(a, body)
    out[:seg] = np.linspace(0.0, 1.0, seg, dtype=np.float32)
    i += seg
    if i < body:
        seg = min(d, body - i)
        out[i:i + seg] = np.linspace(1.0, s, seg, dtype=np.float32)
        i += seg
    if i < body:
        out[i:body] = s
    tail = n - body
    if tail > 0:
        out[body:] = np.linspace(out[body - 1] if body else s, 0.0, tail, dtype=np.float32)
    return out


def _lowpass(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """One-pole. Cheap, and the character it lends is close enough for a mock."""
    if cutoff >= sr / 2:
        return x
    alpha = float(np.exp(-2.0 * np.pi * cutoff / sr))
    out = np.empty_like(x)
    acc = 0.0
    for i in range(x.size):
        acc = (1.0 - alpha) * x[i] + alpha * acc
        out[i] = acc
    return out


def _highpass(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    return x - _lowpass(x, cutoff, sr)


def _phase(hz: float, n: int, sr: int, vib: dict[str, float] | None) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    f = np.full(n, hz, dtype=np.float32)
    if vib:
        rate = vib.get("rate", 5.0)
        depth_c = vib.get("depth", 0.0)
        onset = vib.get("onset", 0.0)
        ramp = np.clip((t - onset) / max(onset, 0.05), 0.0, 1.0)
        f = f * (2.0 ** (depth_c * ramp * np.sin(2 * np.pi * rate * t) / 1200.0))
    return 2.0 * np.pi * np.cumsum(f) / sr


def _pitched(note: Note, sr: int) -> np.ndarray:
    n = max(int((note.dur + 0.25) * sr), 8)
    family = note.inst.split(".")[0]
    hz = max(note.hz, 20.0)
    ph = _phase(hz, n, sr, note.vib)

    if family == "bass":
        # saw, then lowpassed hard — the mock only needs the fundamental to read
        sig = 2.0 * (ph / (2 * np.pi) % 1.0) - 1.0
        sig = _lowpass(sig.astype(np.float32), 900.0, sr)
        sig = np.tanh(sig * 2.2)
    elif family == "keys":
        sig = (np.sin(ph) + 0.30 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)).astype(np.float32)
    elif family == "voice":
        # harmonic stack with a broad formant tilt; enough for pitch to read
        sig = np.zeros(n, dtype=np.float32)
        for k in range(1, 9):
            if hz * k > sr / 2:
                break
            weight = (1.0 / k) * (1.4 if 2 <= k <= 4 else 1.0)
            sig += weight * np.sin(k * ph).astype(np.float32)
        sig *= 0.45
    else:
        sig = (2.0 * (ph / (2 * np.pi) % 1.0) - 1.0).astype(np.float32)
        sig = _lowpass(sig, 3000.0, sr)

    env = note.env or _DEFAULT_ENV.get(family, _DEFAULT_ENV["synth"])
    return sig.astype(np.float32) * _adsr(n, sr, env, note.dur)


def _percussive(note: Note, sr: int) -> np.ndarray:
    v = (note.voice or "").lower()
    rng = np.random.default_rng(abs(hash(v)) % (2**32))

    def noise(dur: float) -> tuple[np.ndarray, np.ndarray]:
        n = max(int(dur * sr), 8)
        t = np.arange(n, dtype=np.float32) / sr
        return rng.standard_normal(n).astype(np.float32), t

    if v == "kick":
        n = int(0.34 * sr)
        t = np.arange(n, dtype=np.float32) / sr
        f = 110.0 * np.exp(-t * 34.0) + 45.0
        sig = np.sin(2 * np.pi * np.cumsum(f) / sr).astype(np.float32)
        sig *= np.exp(-t * 11.0)
        return np.tanh(sig * 1.6)
    if v in ("snare", "rim", "clap"):
        x, t = noise(0.20)
        tone = np.sin(2 * np.pi * 190.0 * t).astype(np.float32) * 0.5
        sig = (_highpass(x, 900.0, sr) * 0.9 + tone) * np.exp(-t * 22.0)
        return sig.astype(np.float32)
    if v in ("hat", "ohat"):
        decay = 55.0 if v == "hat" else 12.0
        x, t = noise(0.30 if v == "ohat" else 0.09)
        return (_highpass(x, 6500.0, sr) * np.exp(-t * decay)).astype(np.float32)
    if v == "crash":
        x, t = noise(1.60)
        return (_highpass(x, 3500.0, sr) * np.exp(-t * 2.4) * 0.8).astype(np.float32)
    if v.startswith("tom"):
        base = 165.0 if v.endswith("1") else 110.0
        n = int(0.34 * sr)
        t = np.arange(n, dtype=np.float32) / sr
        f = base * np.exp(-t * 8.0) + base * 0.55
        sig = np.sin(2 * np.pi * np.cumsum(f) / sr).astype(np.float32)
        return sig * np.exp(-t * 8.5)
    x, t = noise(0.15)
    return (x * np.exp(-t * 26.0)).astype(np.float32)


def section_gains(doc: Document) -> list[tuple[float, float, float]]:
    """(start_s, end_s, linear_gain) from :struct ranges and :mix by_section.

    Without this the mock is dynamically flat, and the arc is exactly what the
    decoder needs in order to place a chorus where the song has one.
    """
    from .expand import build_grid

    struct = doc.stream("struct")
    if struct is None:
        return []
    grid = build_grid(doc)

    levels: dict[str, float] = {}
    mix = doc.stream("mix")
    if mix is not None:
        for name, args in mix.statements:
            if name != "by_section":
                continue
            toks = [t.strip(",") for t in args]
            for label, value in zip(toks[0::2], toks[1::2]):
                try:
                    levels[label] = float(value)
                except ValueError:
                    continue

    spans: list[tuple[float, float, float]] = []
    loudest = max(levels.values(), default=0.0)
    for label, args in struct.statements:
        if not args or "-" not in args[0]:
            continue
        first, last = (int(x) for x in args[0].split("-"))
        start = grid.time_of(first, 1.0)
        end = grid.time_of(last + 1, 1.0)
        if label in levels:                     # LUFS relative to the loudest
            gain = 10.0 ** ((levels[label] - loudest) / 20.0)
        else:                                   # fall back to :struct energy
            energy = next((float(a.split("=")[1]) for a in args
                           if a.startswith("energy=")), 1.0)
            gain = max(energy, 0.05)
        spans.append((start, end, gain))
    return spans


def render(doc: Document, sr: int | None = None) -> np.ndarray:
    """Render a parsed document to a stereo float32 array."""
    sr = sr or doc.sample_rate
    notes: list[Note] = expand(doc)
    total = doc.duration or (max((n.start + n.dur for n in notes), default=1.0) + 1.0)
    length = int(total * sr) + sr  # a second of tail for releases
    buf = np.zeros((length, 2), dtype=np.float32)

    for note in notes:
        sig = _percussive(note, sr) if note.voice else _pitched(note, sr)
        gain = _GAIN.get(note.inst.split(".")[0], 0.5) * (note.vel / 127.0) ** 1.4
        pan = _PAN.get(note.inst.split(".")[0], 0.0)
        if note.stream.endswith("bvox"):
            pan, gain = 0.0, gain * 0.55

        start = int(note.start * sr)
        end = min(start + sig.size, length)
        if start >= length or end <= start:
            continue
        chunk = sig[: end - start] * gain
        buf[start:end, 0] += chunk * (1.0 - max(pan, 0.0))
        buf[start:end, 1] += chunk * (1.0 + min(pan, 0.0))

    for start_s, end_s, gain in section_gains(doc):
        a, b = int(start_s * sr), min(int(end_s * sr), length)
        if b > a:
            buf[a:b] *= gain

    buf = np.tanh(buf * 0.85)
    peak = float(np.max(np.abs(buf)))
    if peak > 0:
        buf *= 0.89 / peak
    return buf[: int(total * sr) + sr // 2]


def render_to_file(doc: Document, path: str, sr: int | None = None) -> tuple[int, float]:
    import soundfile as sf

    sr = sr or doc.sample_rate
    audio = render(doc, sr)
    sf.write(path, audio, sr, subtype="PCM_16")
    return len(expand(doc)), audio.shape[0] / sr
