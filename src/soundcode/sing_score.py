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
# letter-to-ARPAbet guess for words CMUdict lacks; every value is a bank phoneme
_LETTER = {"a": ["ae"], "b": ["b"], "c": ["k"], "d": ["d"], "e": ["eh"], "f": ["f"],
           "g": ["g"], "h": ["hh"], "i": ["ih"], "j": ["jh"], "k": ["k"], "l": ["l"],
           "m": ["m"], "n": ["n"], "o": ["aa"], "p": ["p"], "q": ["k"], "r": ["r"],
           "s": ["s"], "t": ["t"], "u": ["ah"], "v": ["v"], "w": ["w"], "x": ["k", "s"],
           "y": ["iy"], "z": ["z"]}
_cmu = None


class SingError(RuntimeError):
    """The .sc cannot be sung; the message says why."""


class NoVocalError(SingError):
    """The .sc has nothing to sing (no vocal stream, or no notes in it)."""


@dataclass
class Score:
    phonemes: list[str]
    frames: list[int]
    f0_hz: np.ndarray
    n_frames: int
    warnings: list[str] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)   # phoneme index range per word


def vocal_stream(doc) -> str:
    for s in doc.streams:
        if s.kind != "notes":
            continue
        if s.fields.get("inst", "").startswith("voice.lead") or \
                s.meta.get("stem") in ("lead_vocals", "vocals") or s.name == "notes.vox":
            return s.name
    raise NoVocalError("no lead vocal stream (inst=voice.lead, stem=lead_vocals, or :notes.vox)")


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
    guess = ["en/" + p for c in word if c in _LETTER for p in _LETTER[c]]
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
        inside = (t >= seg_t[0]) & (t < seg_t[-1] + STEP_S)     # half-open: no gap to the next line
        cents[inside] = np.interp(t[inside], seg_t, vals)
    return (440.0 * 2 ** ((cents - 6900) / 1200)).astype(np.float32)


def build(doc, duration: float | None = None, durations: str = "heuristic") -> Score:
    """durations: "heuristic" (70 ms consonants, vowel fills the note) or "model"
    (the bank's own duration model, rescaled inside each word's span)."""
    stream = vocal_stream(doc)
    notes = vocal_notes(doc, stream)
    dur = duration or doc.duration or (max((b for _, b, _ in notes), default=1.0) + 1.0)
    n = int(round(dur * SR / HOP))
    warn: list[str] = []
    ws = words(doc)
    if not notes:
        raise NoVocalError(f"{stream} has no notes")
    if not ws:
        warn.append("no lyrics: singing on 'ah'")
        ws = [(a, b, "ah") for a, b, _ in notes]
    else:
        # every note is sung: a note no word covers gets an 'ah'
        uncovered = [(a, b) for a, b, _ in notes
                     if not any(w0 - 0.03 <= a < w1 for w0, w1, _ in ws)]
        if uncovered:
            warn.append(f"{len(uncovered)} note(s) without words sung on 'ah'")
            ws = sorted(ws + [(a, b, "ah") for a, b in uncovered])

    seq: list[tuple[str, float, float, int]] = []          # phoneme, start, end, word id
    cur = 0.0
    wid = [0]

    def next_id() -> int:
        wid[0] += 1
        return wid[0]

    def gap(a: float, b: float) -> None:
        if b - a < 0.02:
            return
        if b - a > 0.45:
            seq.append(("SP", a, b - 0.3, next_id()))
            seq.append(("AP", b - 0.3, b, next_id()))
        else:
            seq.append(("SP", a, b, next_id()))

    late = [w for t, _, w in ws if t >= dur - 0.05]
    if late:
        warn.append(f"dropped {len(late)} word(s) past the end of the song: {' '.join(late)}")
        ws = [x for x in ws if x[0] < dur - 0.05]
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
        word_id = next_id()
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
                seq.append((p, s, s + d, word_id))
                s += d
            cur = s
    if cur < dur:
        gap(cur, dur)

    seq = [x for x in seq if x[2] > x[1]]
    frames, acc = [], 0
    for i, (_, a, b, _w) in enumerate(seq):
        end_f = n if i == len(seq) - 1 else int(round(b * SR / HOP))
        frames.append(max(1, end_f - acc))
        acc += frames[-1]
    frames[-1] += n - sum(frames)
    if frames[-1] < 1:
        raise SingError("phoneme timing overflowed the song length")
    spans, start = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i][3] != seq[start][3]:
            spans.append((start, i))
            start = i
    phonemes = [x[0] for x in seq]
    f0 = _f0(doc, notes, n)
    if durations == "model":
        from . import diffsinger
        starts = np.concatenate([[0], np.cumsum(frames)[:-1]])
        ph_midi = [int(round(69 + 12 * np.log2(max(float(f0[min(int(k), n - 1)]), 1.0) / 440.0)))
                   for k in starts]
        frames = diffsinger.predict_durations(phonemes, [i1 - i0 for i0, i1 in spans],
                                              [int(sum(frames[i0:i1])) for i0, i1 in spans], ph_midi)
    return Score(phonemes, frames, f0, n, warn, spans)
