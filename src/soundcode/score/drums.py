"""Drums, per voice: onsets grouped by GM drum voice, onset F1, hit decay.

Spec 2026-09-28-benchmark-scorer, Task 3. `voice_onsets` runs tsumugi
`drums_v1_5` on a wav (its own external checkout; see `..tsumugi`) and groups
the resulting MIDI note onsets by `gm.drum_voice` into kick / snare / hat /
clap / other, caching the grouped onsets by the wav's sha1 under `work` so a
second call on the same file skips transcription entirely.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .. import gm, tsumugi

VOICES = ("kick", "snare", "hat", "clap", "other")


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _voice_of(note: int) -> str:
    """Map a GM drum note to kick/snare/hat/clap/other via `gm.drum_voice`'s name."""
    name = gm.drum_voice(note)
    for fam in ("kick", "snare", "hat", "clap"):
        if name == fam or name.startswith(fam + "."):
            return fam
    return "other"


def _midi_onsets(midi_path: Path) -> list[tuple[float, int]]:
    """(onset time, GM pitch) for every note in `midi_path`, any track."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    onsets = [(n.start, n.pitch) for inst in pm.instruments for n in inst.notes]
    onsets.sort()
    return onsets


def voice_onsets(wav: Path, work: Path) -> dict[str, np.ndarray]:
    """Per-voice onset times (seconds) for `wav`'s drums, cached by sha1 in `work`."""
    wav = Path(wav)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    sha1 = _sha1_file(wav)
    cache = work / f"{sha1}.drum_voices.json"

    if cache.exists():
        grouped: dict[str, list[float]] = json.loads(cache.read_text())
    else:
        midi_path = work / "drums.mid"
        tsumugi.transcribe(wav, midi_path, "drums_v1_5")
        grouped = {v: [] for v in VOICES}
        for t, note in _midi_onsets(midi_path):
            grouped[_voice_of(note)].append(t)
        cache.write_text(json.dumps(grouped))

    return {v: np.asarray(sorted(grouped.get(v, [])), dtype=float) for v in VOICES}


def _in_range(onsets: np.ndarray, a: float, b: float) -> np.ndarray:
    return onsets[(onsets >= a) & (onsets < b)]


def voice_f1(ref: dict, est: dict, a: float, b: float, tol: float = 0.05) -> dict[str, float | None]:
    """Onset F1 per voice within [a, b), tolerance `tol` (default ±50 ms).

    A voice absent on both sides is None; present on only one side is 0.0.
    """
    import mir_eval

    out: dict[str, float | None] = {}
    for v in VOICES:
        r = _in_range(np.asarray(ref.get(v, [])), a, b)
        e = _in_range(np.asarray(est.get(v, [])), a, b)
        if r.size == 0 and e.size == 0:
            out[v] = None
        elif r.size == 0 or e.size == 0:
            out[v] = 0.0
        else:
            out[v] = float(mir_eval.onset.f_measure(r, e, window=tol)[0])
    return out


def _envelope(seg: np.ndarray, sr: int, frame_s: float = 0.005) -> np.ndarray:
    """Short-time RMS envelope, non-overlapping `frame_s`-second frames."""
    frame = max(int(round(frame_s * sr)), 1)
    n = seg.shape[0] // frame
    if n == 0:
        return np.zeros(0)
    trimmed = seg[: n * frame].reshape(n, frame).astype(np.float64)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def decay_s(y: np.ndarray, sr: int, onsets: np.ndarray, window_s: float = 1.0,
            frame_s: float = 0.005) -> float | None:
    """Median time from each isolated hit's peak to -20 dB.

    Each onset gets its own window: from the onset to the next onset (or
    `window_s`, whichever is sooner), so overlapping hits never bleed into
    each other's decay. A hit that never reaches -20 dB within its window is
    skipped (not counted as 0 or as an error). None if no hit reaches -20 dB.
    """
    y = np.asarray(y, dtype=np.float64)
    times = sorted(float(t) for t in onsets)
    if not times:
        return None

    decays = []
    for i, t in enumerate(times):
        next_t = times[i + 1] if i + 1 < len(times) else t + window_s
        end_t = min(next_t, t + window_s, y.shape[0] / sr)
        i0 = max(int(round(t * sr)), 0)
        i1 = max(int(round(end_t * sr)), i0)
        seg = y[i0:i1]
        env = _envelope(seg, sr, frame_s)
        if env.size < 2:
            continue
        peak_idx = int(np.argmax(env))
        peak = env[peak_idx]
        if peak <= 1e-12:
            continue
        target = peak * 10.0 ** (-20.0 / 20.0)
        tail = env[peak_idx:]
        below = np.where(tail <= target)[0]
        if below.size == 0:
            continue
        decays.append(below[0] * frame_s)

    return float(np.median(decays)) if decays else None
