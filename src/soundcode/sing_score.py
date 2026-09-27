"""The sung part of a .sc as DiffSinger input: phonemes, frame durations, f0.

Ported from the 2026-09-26 singing spike (ds_render.py). f0 comes from
:contour.vox where present, else from the notes with ~30 ms portamento —
never from the bank's own pitch predictor, which drifts ~80 cents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

SR, HOP = 44100, 512
CONS_S = 0.07
VOWELS = {"aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy",
          "ow", "oy", "uh", "uw"}
_LETTER = {"a": "ae", "e": "eh", "i": "ih", "o": "aa", "u": "ah", "y": "iy"}
_cmu = None


class SingError(RuntimeError):
    """The .sc cannot be sung; the message says why."""


@dataclass
class Score:
    phonemes: list[str]
    frames: list[int]
    f0_hz: np.ndarray
    n_frames: int
    warnings: list[str] = field(default_factory=list)


def vocal_stream(doc) -> str:
    for s in doc.streams:
        if s.kind != "notes":
            continue
        if s.fields.get("inst", "").startswith("voice.lead") or \
                s.meta.get("stem") in ("lead_vocals", "vocals") or s.name == "notes.vox":
            return s.name
    raise SingError("no lead vocal stream (inst=voice.lead, stem=lead_vocals, or :notes.vox)")


def vocal_notes(doc, stream: str) -> list[tuple[float, float, int]]:
    from .expand import expand

    ns = sorted((n.start, n.start + n.dur, int(round(n.cents / 100)))
                for n in expand(doc) if n.stream == stream and n.cents is not None)
    if not ns:
        return []
    median = float(np.median([p for _, _, p in ns]))
    out: list[tuple[float, float, int]] = []
    for a, b, p in ns:
        if out and a < out[-1][1] - 1e-6:                 # overlap: keep the nearer-register note
            if abs(p - median) < abs(out[-1][2] - median):
                out[-1] = (out[-1][0], a, out[-1][2])
                out.append((a, b, p))
            continue
        out.append((a, b, p))
    return [(round(a, 6), round(b, 6), p) for a, b, p in out if b > a]


def words(doc) -> list[tuple[float, float, str]]:
    from .expand import _duration, _seconds, build_grid

    s = doc.stream("text.vox")
    if s is None:
        return []
    grid = build_grid(doc)
    out = []
    for e in s.events:
        if not e.text:
            continue
        a = _seconds(e, grid)
        b = a + (_duration(e, grid) if e.dur else 0.3)
        w = re.sub(r"[^a-z']", "", e.text.lower()).strip("'")
        if w:
            out.append((a, b, w))
    return sorted(out)


def g2p(word: str) -> tuple[list[str], bool]:
    global _cmu
    if _cmu is None:
        import cmudict
        _cmu = cmudict.dict()
    prons = _cmu.get(word) or _cmu.get(word.replace("'", "")) or \
        (_cmu.get(word + "g") if word.endswith("in") else None)
    if prons:
        out = []
        for p in prons[0]:
            base = re.sub(r"\d", "", p).lower()
            if base == "ah" and p.endswith("0"):
                base = "ax"
            out.append("en/" + base)
        return out, True
    guess = [("en/" + _LETTER[c]) if c in _LETTER else ("en/" + c) for c in word if c.isalpha()]
    return guess or ["en/ah"], False


def _syllables(phs: list[str]) -> list[list[str]]:
    vpos = [i for i, p in enumerate(phs) if p[3:] in VOWELS]
    if not vpos:
        return [phs]
    syls, start = [], 0
    for k, vi in enumerate(vpos):
        if k + 1 < len(vpos):
            nxt = vpos[k + 1]
            cons = list(range(vi + 1, nxt))
            cut = nxt if not cons else cons[-1]
            syls.append(phs[start:cut])
            start = cut
        else:
            syls.append(phs[start:])
    return syls


def _f0(doc, notes, n: int) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    from .contour import STEP_S, read_contour

    t = (np.arange(n) + 0.5) * HOP / SR
    midi = np.zeros(n)
    for a, b, p in notes:
        midi[(t >= a) & (t < b)] = p
    idx = np.where(midi > 0)[0]
    if len(idx):
        midi = np.interp(t, t[idx], midi[idx])
    else:
        midi[:] = 60
    cents = gaussian_filter1d(midi, sigma=2.5) * 100.0
    for start, vals in read_contour(doc):
        seg_t = start + np.arange(len(vals)) * STEP_S
        inside = (t >= seg_t[0]) & (t <= seg_t[-1] + STEP_S / 2)
        cents[inside] = np.interp(t[inside], seg_t, vals)
    return (440.0 * 2 ** ((cents - 6900) / 1200)).astype(np.float32)


def build(doc, duration: float | None = None) -> Score:
    stream = vocal_stream(doc)
    notes = vocal_notes(doc, stream)
    dur = duration or doc.duration or (max((b for _, b, _ in notes), default=1.0) + 1.0)
    n = int(round(dur * SR / HOP))
    warn: list[str] = []
    ws = words(doc)
    if not notes:
        raise SingError(f"{stream} has no notes")
    if not ws:
        warn.append("no lyrics: singing on 'ah'")
        ws = [(a, b, "ah") for a, b, _ in notes]

    seq: list[tuple[str, float, float]] = []
    cur = 0.0

    def gap(a: float, b: float) -> None:
        if b - a < 0.02:
            return
        if b - a > 0.45:
            seq.append(("SP", a, b - 0.3))
            seq.append(("AP", b - 0.3, b))
        else:
            seq.append(("SP", a, b))

    for i, (t, end, w) in enumerate(ws):
        nxt = ws[i + 1][0] if i + 1 < len(ws) else dur
        end = min(max(end, t + 0.12), nxt)
        if t < cur:                                        # overlapping words: squeeze
            t = cur
            if end - t < 0.06:
                warn.append(f"dropped '{w}' (no room)")
                continue
        if t > cur:
            gap(cur, t)
        phs, hit = g2p(w)
        if not hit:
            warn.append(f"no dictionary entry for '{w}': letter guess")
        syls = _syllables(phs)
        inside = [x for x in notes if t - 0.03 <= x[0] < end]
        if len(inside) >= len(syls) > 1:
            bounds = [max(t, inside[k][0]) for k in range(len(syls))] + [end]
        else:
            bounds = list(np.linspace(t, end, len(syls) + 1))
        for k, syl in enumerate(syls):
            a, b = bounds[k], bounds[k + 1]
            vi = next((j for j, p in enumerate(syl) if p[3:] in VOWELS), 0)
            n_coda = len(syl) - vi - 1
            cd = min(CONS_S, (b - a) / (len(syl) + 1))
            s = a
            for j, p in enumerate(syl):
                d = cd if j != vi else max(0.03, b - n_coda * cd - s)
                seq.append((p, s, s + d))
                s += d
            cur = s
    if cur < dur:
        gap(cur, dur)

    seq = [(p, a, b) for p, a, b in seq if b > a]
    frames, acc = [], 0
    for i, (_, a, b) in enumerate(seq):
        end_f = n if i == len(seq) - 1 else int(round(b * SR / HOP))
        frames.append(max(1, end_f - acc))
        acc += frames[-1]
    frames[-1] += n - sum(frames)
    if frames[-1] < 1:
        raise SingError("phoneme timing overflowed the song length")
    return Score([p for p, _, _ in seq], frames, _f0(doc, notes, n), n, warn)
