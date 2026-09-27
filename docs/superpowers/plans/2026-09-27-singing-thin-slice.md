# Singing Thin Slice — Implementation Plan (Milestone 2, step 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- The encoder writes the lead vocal's f0 curve (`:contour.vox`) and lyrics with performed timing.
- `soundcode render --with-vocals` sings the lead vocal in the original singer's voice (DiffSinger → Seed-VC) and mixes it with the instruments.
- `compare --with-vocals` scores the sung vocal on pitch error in cents and on voice similarity.

**Architecture:**
- `contour.py` extracts, writes and reads f0.
- `sing_score.py` (pure) turns the `.sc` vocal note stream, `:text.vox` and `:contour.vox` into phonemes, frame durations and an f0 curve.
- `diffsinger.py` drives an OpenUtau ONNX bank with onnxruntime.
- `seedvc.py` runs Seed-VC in its own venv as a subprocess.
- `sing.py` orchestrates score → DiffSinger → Seed-VC, with a cache.
- `render_sf` and `compare` gain `with_vocals`.

**Tech Stack:** Python 3.11, torchcrepe (installed), onnxruntime (installed), cmudict (new), resemblyzer (new); Seed-VC (GPL-3.0, own venv, commit `51383ef`); DiffSinger bank Azure Cobalt v0.4.28 (CC BY-SA 4.0) and vocoder `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` (CC BY-NC-SA).

**Spec:** `docs/superpowers/specs/2026-09-27-singing-thin-slice-design.md`. Evidence: `docs/research/2026-09-26-singing-spike.md`.

## Global Constraints

- Python `>=3.11,<3.12`. Install packages with `uv pip install --python .venv/bin/python <pkg>`.
- **Big files go on the external drive `/Volumes/ExFAT 2/infinity-engine/`.** It is exFAT: no venvs there, and verify copies with `diff -rq -x '._*'`. Keep venvs and code on the internal disk, which has about 11 GB free; stop if it would drop below 3 GB.
  - DiffSinger bank and vocoder: `/Volumes/ExFAT 2/infinity-engine/models/diffsinger/` (`$SOUNDCODE_DIFFSINGER` overrides).
  - Seed-VC checkpoints: `/Volumes/ExFAT 2/infinity-engine/models/seed-vc-checkpoints/`, symlinked as `external/seed-vc/checkpoints`.
  - The spike already downloaded all of these: bank and vocoder at `/Volumes/ExFAT 2/infinity-engine/scratch/svs/banks/`, Seed-VC checkout and checkpoints at `/Volumes/ExFAT 2/infinity-engine/scratch/svs/seed-vc/`. Move, don't re-download.
- DiffSinger frames are 44.1 kHz with hop 512 (11.6 ms). The contour is written at 50 Hz (20 ms).
- The bank's pitch predictor (`dspitch`) is never used: it drifts 80 cents. f0 always comes from the contour or the notes.
- Renders without `--with-vocals` must be byte-identical to today's behaviour.
- Open, free tools only. `out/`, `external/` and `models/` stay gitignored. No `Co-Authored-By` in commits.

## Review Focus

1. **A song with no vocal** (vocal stem gated, no `:notes` voice stream, or an empty `:text.vox`). `render --with-vocals` renders instruments and says one line about the missing vocal; `encode` writes no `:contour.vox`. (Task 1 and Task 5 tests.)
2. **Words the dictionary doesn't know** (names, contractions like "meetin'", misheard tokens). G2P falls back to a letter-based guess and warns, never crashes. (Task 3 test.)
3. **More words than notes, or notes with no words** (melisma, humming). Every note is sung; extra words are squeezed or dropped with a warning. (Task 3 test.)
4. **The cache being stale** after the `.sc` or the reference changes. The cache key must include everything that shapes the output. (Task 5 test.)
5. **The external drive not mounted.** One-line error naming the path, exit 2. (Task 4 and Task 5 tests.)

---

### Task 1: `:contour.vox`: extract, write, read

**Files:**
- Create: `src/soundcode/contour.py`
- Modify: `src/soundcode/encode.py`, adding `stage_contour` and wiring it after the tsumugi stage
- Test: `tests/test_sing.py`

**Interfaces:**
- Produces:
  - `RATE_HZ = 50`, `GAP_S = 0.06`
  - `phrases(times: np.ndarray, cents: np.ndarray, voiced: np.ndarray) -> list[tuple[float, list[int]]]`: input at any hop, output resampled to 20 ms. Gaps under `GAP_S` are interpolated; longer gaps split phrases.
  - `contour_lines(phrases, per_line: int = 50) -> list[str]`, giving lines like `"f0  @12.340  6912 6915 …"`. Long phrases are split into consecutive lines of at most 50 values.
  - `read_contour(doc, stream: str = "contour.vox") -> list[tuple[float, np.ndarray]]`, the (start seconds, cents per 20 ms) segments
  - `extract(stem: Path, sr: int = 16000) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]`, returning (times, cents, voiced, mean periodicity) via torchcrepe
  - `encode.stage_contour(stem: Path | None, stem_name: str, mix: np.ndarray, sr: int) -> Stage`, named `contour.vox`, with `header_fields {"rate": "50"}`, `meta stem=…`

- [ ] **Step 1: Write the failing tests** in `tests/test_sing.py`:

```python
"""Singing thin slice (spec 2026-09-27)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import contour as ct  # noqa: E402
from soundcode.parser import parse  # noqa: E402


def test_phrases_resample_to_20ms_and_split_on_long_gaps():
    t = np.arange(0, 2.0, 0.01)                       # 10 ms frames
    cents = np.full_like(t, 6900.0)
    voiced = np.ones_like(t, bool)
    voiced[(t >= 0.50) & (t < 0.53)] = False          # 30 ms: filled
    voiced[(t >= 1.00) & (t < 1.20)] = False          # 200 ms: split
    ph = ct.phrases(t, cents, voiced)
    assert [round(s, 2) for s, _ in ph] == [0.0, 1.2]
    assert len(ph[0][1]) == 50 and all(v == 6900 for v in ph[0][1])


def test_vibrato_survives_at_50hz():
    t = np.arange(0, 1.0, 0.01)
    cents = 6900 + 50 * np.sin(2 * np.pi * 5.5 * t)   # 5.5 Hz, +/-50 c
    (start, vals), = ct.phrases(t, cents, np.ones_like(t, bool))
    assert max(vals) - min(vals) >= 90


def test_contour_lines_round_trip_through_the_parser():
    ph = [(12.34, [6912, 6915, 6920]), (14.02, list(range(6700, 6760)))]
    text = "%sc 0.3\n\n:contour.vox rate=50\n" + "\n".join(ct.contour_lines(ph)) + "\n"
    segs = ct.read_contour(parse(text))
    assert [round(s, 2) for s, _ in segs] == [12.34, 14.02, 15.02]       # 60 values -> 50 + 10
    assert list(segs[0][1]) == [6912, 6915, 6920]
    assert len(segs[1][1]) == 50 and len(segs[2][1]) == 10


def test_no_voiced_frames_gives_no_phrases():
    t = np.arange(0, 1.0, 0.01)
    assert ct.phrases(t, np.zeros_like(t), np.zeros_like(t, bool)) == []


def test_stage_contour_on_a_gated_stem_writes_nothing(tmp_path):
    import soundfile as sf
    from soundcode import encode as enc
    sr = 16000
    p = tmp_path / "lead_vocals.wav"
    sf.write(str(p), np.zeros(sr * 4, np.float32), sr)
    st = enc.stage_contour(p, "lead_vocals", np.random.default_rng(0).standard_normal(sr * 4) * 0.1, sr)
    assert not st.ok
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q`
Expected: collection error `cannot import name 'contour'`.

- [ ] **Step 3: Implement** `src/soundcode/contour.py`:

```python
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
                                 device="cpu")
    per = torchcrepe.filter.median(per, 3)
    hz = torchcrepe.filter.median(hz, 3)
    hz, per = hz[0].numpy(), per[0].numpy()
    voiced = per >= 0.5
    cents = np.where(voiced, 1200 * np.log2(np.maximum(hz, 1e-6) / 440.0) + 6900, 0.0)
    times = np.arange(len(hz)) * 0.01
    return times, cents, voiced, float(per[voiced].mean()) if voiced.any() else 0.0
```

Add `stage_contour` to `encode.py` (next to `stage_lyrics`):

```python
def stage_contour(stem: Path | None, stem_name: str, mix: np.ndarray, sr: int) -> Stage:
    """The lead vocal's f0 curve (:contour.vox), gated like every stem."""
    import librosa

    from . import contour as ct

    st = Stage("contour.vox", src="torchcrepe:full", stem=stem_name)
    st.header_fields = {"rate": str(ct.RATE_HZ)}
    if stem is None:
        st.warns.append("no vocal stem")
        return st
    y, _ = librosa.load(str(stem), sr=sr, mono=True)
    mask = active_blocks(y, mix, sr)
    if not mask.any():
        st.warns.append("vocal stem silent (below the loudness gate)")
        return st
    try:
        times, cents, voiced, conf = ct.extract(stem)
    except Exception as exc:                             # noqa: BLE001
        st.warns.append(f"f0 extraction failed: {exc}")
        return st
    # blank frames in gated blocks, so bleed never becomes a contour
    blk = np.minimum((times / GATE_BLOCK_S).astype(int), len(mask) - 1)
    voiced = voiced & mask[blk]
    st.lines = ct.contour_lines(ct.phrases(times, cents, voiced))
    st.conf, st.ok = conf, bool(st.lines)
    return st
```

In `encode()`, after the `ts_stages` line, add:

```python
    _log("vocal f0 contour")
    contour_st = stage_contour(stems.get("vocals"), vox_name or "lead_vocals", mono_mix, sr)
```

Then add `contour_st` to the output tuple right after `*ts_stages`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q`
Expected: `5 passed`. If the parser doesn't keep `f0 @…` lines as statements in a `contour` stream, fix the parser's statement rule for that stream kind, add a parser test, and record a Ruling.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/contour.py src/soundcode/encode.py tests/test_sing.py
git commit -m "contour: lead-vocal f0 curve (:contour.vox, 20 ms, cents) extracted, written, read"
```

---

### Task 2: Lyrics as performed

**Files:**
- Modify: `src/soundcode/parser.py` (text events keep `dur`)
- Modify: `src/soundcode/encode.py` (`stage_lyrics` positions and durations)
- Test: `tests/test_sing.py` (append)

**Interfaces:**
- Produces:
  - `Event.dur` is set for text events when a duration token follows the quoted text: `3:2.125 "river" 0.500b ?0.71`
  - `encode.lyric_cells(words: list[tuple[float, float, str, float]], grid: dict) -> list[str]`, where each word is (start, end, text, prob). It is pure, and collisions are nudged 20 ms apart.

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- lyrics as performed ------------------------------------------------------------------

from soundcode import encode as enc  # noqa: E402

GRID = {"downbeat": 1.0, "bar_dur": 2.0}


def test_text_events_keep_their_duration():
    doc = parse('%sc 0.3\n\n:text.vox\n1:2.125 "river" 0.500b ?0.71\n@0.400 "I" 0.200s\n')
    evs = doc.stream("text.vox").events
    assert [(e.text, e.dur) for e in evs] == [("river", "0.500b"), ("I", "0.200s")]


def test_lyric_cells_use_performed_timing_and_durations():
    words = [(1.5625, 1.9375, "river", 0.9), (0.4, 0.6, "I", 0.5)]
    cells = enc.lyric_cells(words, GRID)
    assert cells == ['1:2.125 "river" 0.750b', '@0.400 "I" 0.200s ?0.50']


def test_colliding_words_are_nudged_20ms():
    words = [(1.5, 1.8, "a", 0.9), (1.5, 1.9, "b", 0.9)]
    cells = enc.lyric_cells(words, GRID)
    assert cells[1].startswith("1:2.040")                         # +20 ms at 0.5 s per beat
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q -k "duration or performed or nudged"`
Expected: 3 failures: text `dur` is `None`, and `lyric_cells` is missing.

- [ ] **Step 3: Implement.** In `parser.parse_event`, after `ev.text = rest[0].strip('"'); rest = rest[1:]`, add:

```python
        if rest and _DUR_RE.match(rest[0]):
            ev.dur = rest[0]
            rest = rest[1:]
```

In `encode.py`, add:

```python
def lyric_cells(words: list[tuple[float, float, str, float]], grid: dict) -> list[str]:
    """Words at performed timing (3-decimal beats), with durations."""
    from . import tsumugi_sc as tsc

    beat_s = grid["bar_dur"] / 4
    cells, last = [], -1.0
    for start, end, word, prob in sorted(words, key=lambda w: w[0]):
        if not word:
            continue
        start = max(start, last + 0.02) if last >= 0 else start
        last = start
        pos = tsc.position(start, grid)
        dur = max(end - start, 0.05)
        d = f"{dur:.3f}s" if pos.startswith("@") else f"{dur / beat_s:.3f}b"
        mark = "" if prob >= 0.80 else f" ?{prob:.2f}"
        cells.append(f'{pos} "{word}" {d}{mark}')
    return cells
```

In `stage_lyrics`, collect `(w.start, w.end, w.word.strip(), w.probability)` (and the same four fields in the whisper branch: `w["start"], w["end"], …`). Replace the positioning loop with `cells = lyric_cells(words, grid)`, keeping the 5-per-line grouping, and compute `st.conf` from `w[3]`.

- [ ] **Step 4: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (132 + 8 = 140).

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/parser.py src/soundcode/encode.py tests/test_sing.py
git commit -m "lyrics: performed timing (3-decimal beats) with word durations; parser keeps text dur"
```

---

### Task 3: Vocal score: phonemes, durations, f0 (pure)

**Files:**
- Create: `src/soundcode/sing_score.py`
- Test: `tests/test_sing.py` (append)

**Interfaces:**
- Consumes: `expand.expand`, `expand.build_grid`, `expand._seconds`, `expand._duration`, `contour.read_contour`.
- Produces:
  - `SR = 44100`, `HOP = 512`
  - `class SingError(RuntimeError)`
  - `vocal_stream(doc) -> str`: the first `notes.*` stream with `inst` starting `voice.lead`, or with `meta stem` in (`lead_vocals`, `vocals`), or named `notes.vox`. Raises `SingError("no lead vocal stream")`.
  - `vocal_notes(doc, stream) -> list[tuple[float, float, int]]`: monophonic (start, end, midi). At overlaps it keeps the note nearest the median register.
  - `words(doc) -> list[tuple[float, float, str]]`: (start, end, lowercased word stripped of punctuation), from `:text.vox`
  - `g2p(word) -> tuple[list[str], bool]`: phonemes like `["en/r", "en/ih", "en/v", "en/er"]`, and whether it was a dictionary hit
  - `@dataclass Score(phonemes: list[str], frames: list[int], f0_hz: np.ndarray, n_frames: int, warnings: list[str])`
  - `build(doc, duration: float | None = None) -> Score`

- [ ] **Step 1: Install cmudict**

Run: `uv pip install --python .venv/bin/python cmudict`
Expected: `+ cmudict==…`.

- [ ] **Step 2: Write the failing tests** (append):

```python
# --- vocal score --------------------------------------------------------------------------

from soundcode import sing_score as ss  # noqa: E402

SONG = """%sc 0.3
@duration 4.0

:grid
meter @0.000 4/4
anchor bar 1 @0.000
tempo @0.000 120

:notes.lead inst=voice.lead
meta stem=lead_vocals
1:1.000  C4  1.000b 90
1:2.000  D4  1.000b 90
1:3.000  E4  2.000b 90

:text.vox
1:1.000 "hello" 1.000b | 1:3.000 "river" 2.000b
"""


def test_vocal_stream_and_notes():
    doc = parse(SONG)
    assert ss.vocal_stream(doc) == "notes.lead"
    assert ss.vocal_notes(doc, "notes.lead") == [(0.0, 0.5, 60), (0.5, 1.0, 62), (1.0, 2.0, 64)]


def test_words_and_g2p():
    doc = parse(SONG)
    assert [(round(a, 2), round(b, 2), w) for a, b, w in ss.words(doc)] == [(0.0, 0.5, "hello"), (1.0, 2.0, "river")]
    ph, hit = ss.g2p("river")
    assert hit and ph[0] == "en/r" and "en/er" in ph


def test_unknown_word_falls_back_and_warns():
    ph, hit = ss.g2p("zzxqv")
    assert not hit and ph                                   # letter-based guess, never empty


def test_score_frames_cover_the_song_and_f0_follows_notes():
    sc = ss.build(parse(SONG))
    assert sum(sc.frames) == sc.n_frames == int(round(4.0 * ss.SR / ss.HOP))
    t = (np.arange(sc.n_frames) + 0.5) * ss.HOP / ss.SR
    mid_c4 = sc.f0_hz[(t > 0.1) & (t < 0.4)]
    assert np.all(np.abs(1200 * np.log2(mid_c4 / 261.63)) < 5)
    assert "SP" in sc.phonemes and sc.phonemes.count("en/ow") == 1


def test_contour_overrides_notes_where_present():
    doc = parse(SONG + "\n:contour.vox rate=50\nf0  @0.100  " + " ".join(["6030"] * 10) + "\n")
    sc = ss.build(doc)
    t = (np.arange(sc.n_frames) + 0.5) * ss.HOP / ss.SR
    seg = sc.f0_hz[(t > 0.12) & (t < 0.28)]
    assert np.all(np.abs(1200 * np.log2(seg / 261.63) - 30) < 3)        # C4 + 30 c from contour


def test_melisma_and_extra_words():
    doc = parse(SONG.replace('| 1:3.000 "river" 2.000b', '| 1:2.500 "river" 0.500b | 1:2.750 "run" 0.250b'))
    sc = ss.build(doc)
    assert sum(sc.frames) == sc.n_frames
    assert not any("crash" in w for w in sc.warnings)
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q -k "vocal_stream or g2p or unknown_word or score_frames or contour_overrides or melisma"`
Expected: collection error `cannot import name 'sing_score'`.

- [ ] **Step 4: Implement** `src/soundcode/sing_score.py`. This ports the spike's `ds_render.py` phoneme logic, reading the `.sc` model instead of regexes:

```python
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
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q`
Expected: `14 passed`. If `g2p("hello")` has no `en/ow`, the CMUdict entry differs (`HH AH0 L OW1` → `en/hh en/ax en/l en/ow`). Check that before changing the test.

- [ ] **Step 6: Commit**

```bash
git add src/soundcode/sing_score.py tests/test_sing.py
git commit -m "sing_score: .sc vocal to phonemes, frame durations and f0 (contour first)"
```

---

### Task 4: DiffSinger ONNX engine

**Files:**
- Create: `src/soundcode/diffsinger.py`
- Create: `scripts/install_diffsinger.sh`
- Test: `tests/test_sing.py` (append)

**Interfaces:**
- Consumes: `sing_score.Score`, `SR`, `HOP`.
- Produces:
  - `bank_dir() -> Path` and `vocoder_path() -> Path`, defaulting to `/Volumes/ExFAT 2/infinity-engine/models/diffsinger/{AC0.4.28, pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx}` (`$SOUNDCODE_DIFFSINGER` replaces the parent directory). Both raise `SingError` naming the path when missing.
  - `render(score: Score, mode: str = "01CORE", steps: int = 20, depth: float = 0.6) -> np.ndarray`: mono float32 at 44.1 kHz, peak 0.8

- [ ] **Step 1: Move the spike's bank and vocoder into place.** `scripts/install_diffsinger.sh`:

```bash
#!/bin/zsh
# DiffSinger bank (Azure Cobalt v0.4.28, CC BY-SA 4.0) + NSF-HiFiGAN vocoder
# (CC BY-NC-SA) onto the external drive. Reuses the 2026-09-26 spike download.
set -e
DEST=${SOUNDCODE_DIFFSINGER:-"/Volumes/ExFAT 2/infinity-engine/models/diffsinger"}
SPIKE="/Volumes/ExFAT 2/infinity-engine/scratch/svs/banks"
[ -d "${DEST:h}" ] || { echo "external drive not mounted: ${DEST:h}"; exit 2; }
mkdir -p "$DEST"
[ -d "$DEST/AC0.4.28" ] || cp -R "$SPIKE/ac/AC0.4.28" "$DEST/"
[ -f "$DEST/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx" ] || \
  cp "$SPIKE/voc/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx" "$DEST/"
diff -rq -x '._*' "$SPIKE/ac/AC0.4.28" "$DEST/AC0.4.28" && echo "diffsinger bank ready at $DEST"
```

Run: `chmod +x scripts/install_diffsinger.sh && scripts/install_diffsinger.sh`
Expected: `diffsinger bank ready at …`.

- [ ] **Step 2: Write the failing tests** (append):

```python
# --- DiffSinger engine -------------------------------------------------------------------

from soundcode import diffsinger as ds  # noqa: E402


def test_missing_bank_is_one_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCODE_DIFFSINGER", str(tmp_path / "nope"))
    with pytest.raises(ss.SingError, match="nope"):
        ds.bank_dir()


def _bank_present():
    try:
        ds.bank_dir(), ds.vocoder_path()
        return True
    except ss.SingError:
        return False


@pytest.mark.skipif(not _bank_present(), reason="DiffSinger bank not installed")
def test_diffsinger_sings_two_bars_at_the_requested_pitch():
    import librosa
    sc = ss.build(parse(SONG))
    y = ds.render(sc)
    assert abs(len(y) / ss.SR - 4.0) < 0.1 and np.abs(y).max() > 0.1
    f0, v, _ = librosa.pyin(y, fmin=100, fmax=600, sr=ss.SR, frame_length=2048)
    t = librosa.times_like(f0, sr=ss.SR)
    seg = f0[(t > 1.3) & (t < 1.8) & v]                    # the E4 note
    assert len(seg) and abs(np.median(1200 * np.log2(seg / 329.63))) < 50
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q -k "bank or diffsinger"`
Expected: collection error `cannot import name 'diffsinger'`.

- [ ] **Step 4: Implement** `src/soundcode/diffsinger.py`:

```python
"""Headless DiffSinger: an OpenUtau ONNX voicebank driven with onnxruntime.

linguistic -> variance (breathiness/voicing/tension) -> acoustic -> NSF-HiFiGAN.
Pitch is always the explicit f0 from sing_score (never the bank's dspitch).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .sing_score import SingError, Score

DEFAULT = Path("/Volumes/ExFAT 2/infinity-engine/models/diffsinger")
BANK = "AC0.4.28"
VOCODER = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"


def _root() -> Path:
    return Path(os.environ.get("SOUNDCODE_DIFFSINGER", DEFAULT))


def bank_dir() -> Path:
    p = _root() / BANK
    if not (p / "dsmain" / "acoustic.onnx").exists():
        raise SingError(f"DiffSinger bank missing at {p}; run scripts/install_diffsinger.sh")
    return p


def vocoder_path() -> Path:
    p = _root() / VOCODER
    if not p.exists():
        raise SingError(f"DiffSinger vocoder missing at {p}; run scripts/install_diffsinger.sh")
    return p


def render(score: Score, mode: str = "01CORE", steps: int = 20, depth: float = 0.6) -> np.ndarray:
    import onnxruntime as ort

    b = bank_dir()
    ids = json.loads((b / "dsmain" / "phonemes.json").read_text())
    missing = sorted({p for p in score.phonemes if p not in ids})
    if missing:
        raise SingError(f"bank has no phonemes {missing}")
    nf = score.n_frames
    tokens = np.array([[ids[p] for p in score.phonemes]], dtype=np.int64)
    langs = np.array([[1 if p.startswith("en/") else 0 for p in score.phonemes]], dtype=np.int64)
    ph_dur = np.array([score.frames], dtype=np.int64)
    spk = np.fromfile(b / "embeds" / f"{mode}.emb", dtype=np.float32).reshape(1, 1, 384)
    spk_f = np.repeat(spk, nf, axis=1)
    pitch_midi = (69 + 12 * np.log2(score.f0_hz / 440.0)).astype(np.float32)[None]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    prov = ["CPUExecutionProvider"]
    z = np.zeros((1, nf), np.float32)
    st = np.array(steps, np.int64)

    ling = ort.InferenceSession(str(b / "dsvariance" / "linguistic.onnx"), opts, providers=prov)
    enc, _ = ling.run(None, {"tokens": tokens, "languages": langs, "ph_dur": ph_dur})
    var = ort.InferenceSession(str(b / "dsvariance" / "variance.onnx"), opts, providers=prov)
    br, vo, te = var.run(None, {"encoder_out": enc, "ph_dur": ph_dur, "pitch": pitch_midi,
                                "breathiness": z, "voicing": z, "tension": z,
                                "retake": np.ones((1, nf, 3), bool), "spk_embed": spk_f,
                                "steps": st})
    ac = ort.InferenceSession(str(b / "dsmain" / "acoustic.onnx"), opts, providers=prov)
    f0 = score.f0_hz[None].astype(np.float32)
    mel = ac.run(None, {"tokens": tokens, "languages": langs, "durations": ph_dur, "f0": f0,
                        "breathiness": br.astype(np.float32), "voicing": vo.astype(np.float32),
                        "tension": te.astype(np.float32), "gender": z,
                        "velocity": np.ones((1, nf), np.float32), "spk_embed": spk_f,
                        "depth": np.array(depth, np.float32), "steps": st})[0]
    voc = ort.InferenceSession(str(vocoder_path()), opts, providers=prov)
    wav = voc.run(None, {"mel": mel.astype(np.float32), "f0": f0})[0][0]
    return (wav / max(1e-6, float(np.abs(wav).max())) * 0.8).astype(np.float32)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q`
Expected: `16 passed` (the DiffSinger test runs, since the bank is installed). If the phoneme ids use a different key format (check `dsmain/phonemes.json` for `en/r` vs `r`), adapt `g2p`'s prefix and record a Ruling.

- [ ] **Step 6: Commit**

```bash
git add src/soundcode/diffsinger.py scripts/install_diffsinger.sh tests/test_sing.py
git commit -m "diffsinger: headless ONNX bank render with explicit f0"
```

---

### Task 5: Seed-VC, the `sing` orchestrator, `render --with-vocals`

**Files:**
- Create: `src/soundcode/seedvc.py`, `src/soundcode/sing.py`, `scripts/install_seedvc.sh`, `scripts/seedvc.patch`
- Modify: `src/soundcode/render_sf.py`, `src/soundcode/cli.py`
- Test: `tests/test_sing.py` (append)

**Interfaces:**
- Produces:
  - `seedvc.home() -> Path` (`$SOUNDCODE_SEEDVC` or `external/seed-vc`, needing `.venv/bin/python` there; otherwise `SingError`)
  - `seedvc.convert(src: Path, ref: Path, out: Path, steps: int = 30) -> Path`
  - `sing.voice_ref(doc, explicit: Path | None) -> Path`: explicit, else `out/stems/<Path(@source).stem>/lead_vocals.wav`, else `vocals.wav`; otherwise `SingError` naming `--voice-ref`
  - `sing.cache_key(doc, ref: Path, settings: dict) -> str`: the sha1 of the vocal stream's lines, `:text.vox`, `:contour.vox`, the ref path + mtime + size, and the settings
  - `sing.sing(doc, ref: Path | None = None, cache: Path = Path("out/sing"), steps: int = 30) -> tuple[Path, list[str]]`: the cached wav path and warnings
  - `render_sf.render(doc, sr=None, sf2=None, with_vocals=False, voice_ref=None)` and `render_to_file(doc, path, sr=None, with_vocals=False, voice_ref=None)`. With vocals, the sung wav replaces the vocal stream's SoundFont render, level-matched to its `meta level`.
  - CLI: `render … --with-vocals [--voice-ref WAV]`. A `SingError` gives exit 2 with a one-line message.

- [ ] **Step 1: Install Seed-VC from the spike checkout.** Write `scripts/seedvc.patch` as the diff below. It is the spike's working changes to `inference.py` at `51383ef`:

```diff
--- a/inference.py
+++ b/inference.py
@@ -326,8 +326,8 @@ def main(args):
         F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
         F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)
 
-        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
-        F0_alt = torch.from_numpy(F0_alt).to(device)[None]
+        F0_ori = torch.from_numpy(F0_ori.astype(np.float32)).to(device)[None]
+        F0_alt = torch.from_numpy(F0_alt.astype(np.float32)).to(device)[None]
 
         voiced_F0_ori = F0_ori[F0_ori > 1]
         voiced_F0_alt = F0_alt[F0_alt > 1]
@@ -404,7 +404,7 @@ def main(args):
     source_name = os.path.basename(source).split(".")[0]
     target_name = os.path.basename(target_name).split(".")[0]
     os.makedirs(args.output, exist_ok=True)
-    torchaudio.save(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.cpu(), sr)
+    import soundfile as _sf; _sf.write(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.cpu().numpy().T, sr)
```

Then write `scripts/install_seedvc.sh`:

```bash
#!/bin/zsh
# Seed-VC (GPL-3.0) singing voice conversion, own venv; checkpoints on the external drive.
set -e
ROOT=${0:A:h:h}
DEST=${SOUNDCODE_SEEDVC:-$ROOT/external/seed-vc}
CK="/Volumes/ExFAT 2/infinity-engine/models/seed-vc-checkpoints"
SPIKE="/Volumes/ExFAT 2/infinity-engine/scratch/svs/seed-vc/checkpoints"
[ -d "/Volumes/ExFAT 2" ] || { echo "external drive not mounted"; exit 2; }
[ -d "$DEST/.git" ] || git clone https://github.com/Plachtaa/seed-vc.git "$DEST"
git -C "$DEST" checkout --quiet 51383ef
git -C "$DEST" apply --check "$ROOT/scripts/seedvc.patch" 2>/dev/null && git -C "$DEST" apply "$ROOT/scripts/seedvc.patch"
if [ ! -e "$DEST/checkpoints" ]; then
  mkdir -p "$CK"; [ -d "$SPIKE" ] && cp -R "$SPIKE/." "$CK/"
  ln -s "$CK" "$DEST/checkpoints"
fi
cd "$DEST" && uv venv --quiet --python 3.11 .venv
uv pip install --quiet --python .venv/bin/python torch==2.13.0 torchaudio numpy==1.26.4 scipy==1.13.1 \
  librosa==0.10.2 munch einops "huggingface-hub>=0.28.1" transformers==4.46.3 soundfile pyyaml \
  descript-audio-codec==1.0.0 "setuptools<80"
echo "seed-vc 51383ef ready at $DEST"
```

Run: `chmod +x scripts/install_seedvc.sh && scripts/install_seedvc.sh && df -h /System/Volumes/Data | tail -1`
Expected: `seed-vc 51383ef ready …`, with free space still above 3 GB.

- [ ] **Step 2: Write the failing tests** (append):

```python
# --- sing orchestrator + render --with-vocals ---------------------------------------------

from soundcode import render_sf, sing  # noqa: E402


def test_voice_ref_defaults_to_the_songs_lead_stem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "out" / "stems" / "river-30s"
    d.mkdir(parents=True)
    (d / "lead_vocals.wav").write_bytes(b"RIFF")
    doc = parse("%sc 0.3\n@source river-30s.wav\n")
    assert sing.voice_ref(doc, None) == Path("out/stems/river-30s/lead_vocals.wav")
    with pytest.raises(ss.SingError, match="--voice-ref"):
        sing.voice_ref(parse("%sc 0.3\n@source other.wav\n"), None)


def test_cache_key_changes_with_every_input(tmp_path):
    ref = tmp_path / "r.wav"
    ref.write_bytes(b"x")
    k = sing.cache_key(parse(SONG), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG.replace("E4", "F4")), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG.replace('"river"', '"rover"')), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG + "\n:contour.vox rate=50\nf0  @0.1  6000\n"), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG), ref, {"steps": 50})


def test_sing_uses_the_cache_and_chains_the_engines(tmp_path, monkeypatch):
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(4410, np.float32), 44100)
    calls = []
    monkeypatch.setattr(sing.diffsinger, "render", lambda score, **k: calls.append("ds") or np.zeros(44100, np.float32))
    def fake_convert(src, r, out, steps=30):
        calls.append("vc"); sf.write(str(out), np.zeros(44100, np.float32), 44100); return out
    monkeypatch.setattr(sing.seedvc, "convert", fake_convert)
    p1, _ = sing.sing(parse(SONG), ref, cache=tmp_path / "c")
    p2, _ = sing.sing(parse(SONG), ref, cache=tmp_path / "c")
    assert p1 == p2 and p1.exists() and calls == ["ds", "vc"]


def test_render_with_vocals_mixes_the_sung_stream_level_matched(tmp_path, monkeypatch):
    import soundfile as sf
    sung = tmp_path / "sung.wav"
    t = np.arange(44100 * 4) / 44100
    sf.write(str(sung), (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 44100)
    monkeypatch.setattr(sing, "sing", lambda doc, ref=None, **k: (sung, []))
    doc = parse(SONG.replace("meta stem=lead_vocals", "meta stem=lead_vocals level=-30.0"))
    keys = np.zeros((44100 * 5, 2), np.float32)
    monkeypatch.setattr(render_sf, "render_streams", lambda d, sr=None, sf2=None: {"notes.lead": keys})
    without = render_sf.render(doc, 44100)
    with_v = render_sf.render(doc, 44100, with_vocals=True)
    assert np.abs(without).max() == 0 and np.abs(with_v).max() > 0.1
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q -k "voice_ref or cache_key or chains or with_vocals"`
Expected: collection error `cannot import name 'sing'`.

- [ ] **Step 4: Implement** `src/soundcode/seedvc.py`:

```python
"""Seed-VC (Plachtaa/seed-vc @51383ef, GPL-3.0): singing voice conversion.

Runs in its own checkout and venv (external/seed-vc) as a subprocess.
Reference = the original singer (separated lead stem); f0 is preserved.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .sing_score import SingError


def home() -> Path:
    root = Path(os.environ.get("SOUNDCODE_SEEDVC",
                               Path(__file__).resolve().parents[2] / "external" / "seed-vc"))
    if not (root / ".venv" / "bin" / "python").exists():
        raise SingError(f"Seed-VC not installed at {root}; run scripts/install_seedvc.sh")
    return root


def convert(src: Path, ref: Path, out: Path, steps: int = 30) -> Path:
    root = home()
    src, ref, out = Path(src).resolve(), Path(ref).resolve(), Path(out).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [str(root / ".venv" / "bin" / "python"), "inference.py", "--source", str(src),
               "--target", str(ref), "--output", tmp, "--diffusion-steps", str(steps),
               "--length-adjust", "1.0", "--inference-cfg-rate", "0.7", "--f0-condition", "True",
               "--auto-f0-adjust", "False", "--semi-tone-shift", "0", "--fp16", "False"]
        env = {**os.environ, "HF_HUB_CACHE": str(root / "checkpoints" / "hf_cache")}
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, env=env,
                              timeout=1800)
        wavs = sorted(Path(tmp).glob("vc_*.wav"))
        if proc.returncode != 0 or not wavs:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            raise SingError(f"Seed-VC failed: {tail[-1] if tail else proc.returncode}")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(wavs[0]), out)
    return out
```

`src/soundcode/sing.py`:

```python
"""Sing the lead vocal of a .sc in the original singer's voice.

score (sing_score) -> DiffSinger (explicit f0) -> Seed-VC (reference = the
original lead stem). Cached: Seed-VC takes minutes per 30 s.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import diffsinger, seedvc
from . import sing_score as ss
from .sing_score import SingError


def voice_ref(doc, explicit: Path | None) -> Path:
    if explicit is not None:
        if not Path(explicit).exists():
            raise SingError(f"--voice-ref {explicit} does not exist")
        return Path(explicit)
    stem = Path(doc.header.get("source", "")).stem
    for name in ("lead_vocals.wav", "vocals.wav"):
        p = Path("out") / "stems" / stem / name
        if stem and p.exists():
            return p
    raise SingError("no voice reference found; pass --voice-ref <wav> (the original singer)")


def _stream_text(doc, name: str) -> str:
    s = doc.stream(name)
    if s is None:
        return ""
    return json.dumps([s.fields, s.meta, s.statements, [e.raw for e in s.events]],
                      sort_keys=True, default=str)


def cache_key(doc, ref: Path, settings: dict) -> str:
    h = hashlib.sha1()
    for name in (ss.vocal_stream(doc), "text.vox", "contour.vox", "grid"):
        h.update(_stream_text(doc, name).encode())
    st = Path(ref).stat()
    h.update(f"{Path(ref).resolve()}|{st.st_mtime_ns}|{st.st_size}".encode())
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()[:16]


def sing(doc, ref: Path | None = None, cache: Path = Path("out/sing"),
         steps: int = 30) -> tuple[Path, list[str]]:
    import soundfile as sf

    ref = voice_ref(doc, ref)
    settings = {"steps": steps, "bank": diffsinger.BANK, "mode": "01CORE"}
    out = Path(cache) / f"{cache_key(doc, ref, settings)}.wav"
    score = ss.build(doc)
    if out.exists():
        return out, score.warnings
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".diffsinger.wav")
    sf.write(str(raw), diffsinger.render(score, mode=settings["mode"]), ss.SR)
    seedvc.convert(raw, ref, out, steps=steps)
    return out, score.warnings
```

In `render_sf.py`, change `render` and `render_to_file`:

```python
def render(doc: Document, sr: int | None = None, sf2: Path | None = None,
           with_vocals: bool = False, voice_ref: Path | None = None) -> np.ndarray:
    sr = sr or doc.sample_rate
    streams = render_streams(doc, sr, sf2)
    if with_vocals:
        from . import sing
        from .sing_score import vocal_stream
        name = vocal_stream(doc)
        wav, _ = sing.sing(doc, voice_ref)
        streams[name] = _load_stream(wav, sr, max((y.shape[0] for y in streams.values()), default=0),
                                     doc.stream(name))
    return mix(doc, streams, sr, with_vocals)


def _load_stream(path: Path, sr: int, n: int, stream) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    n = max(n, len(y))
    out = np.zeros((n, 2), np.float32)
    out[: len(y), 0] = out[: len(y), 1] = y
    level = stream.meta.get("level") if stream is not None else None
    have = _rms_db(out, sr)
    if level is not None and np.isfinite(have):
        out *= 10 ** ((float(level.rstrip("dB")) - have) / 20)
    return out


def render_to_file(doc: Document, path: str, sr: int | None = None,
                   with_vocals: bool = False, voice_ref: Path | None = None) -> tuple[int, float]:
    import soundfile as sf

    sr = sr or doc.sample_rate
    audio = render(doc, sr, with_vocals=with_vocals, voice_ref=voice_ref)
    sf.write(path, audio, sr, subtype="PCM_16")
    return len(expand(doc)), audio.shape[0] / sr
```

In `cli.py`, add `p_render.add_argument("--voice-ref", default=None)`. The sf2 wrapper calls `render_sf.render_to_file(doc, out, sr, with_vocals=args.with_vocals, voice_ref=Path(args.voice_ref) if args.voice_ref else None)`. The `except` in the render handler also maps `SingError` to `render failed: …` / exit 2.

- [ ] **Step 5: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. The Task 3/4 tests plus these 4 give about 150.

- [ ] **Step 6: Real smoke test (3 s).** Render the Task 3 test song through DiffSinger → Seed-VC with the River lead stem as the reference:

```bash
.venv/bin/python -c "
from pathlib import Path; from soundcode import sing; from soundcode.parser import parse
import tests.test_sing as T
p, w = sing.sing(parse(T.SONG), Path('out/stems/river-30s/lead_vocals.wav'), cache=Path('out/sing-smoke'))
print(p, w)"
```

Expected: a wav path, after a few minutes the first time (Seed-VC loads its models).

- [ ] **Step 7: Commit**

```bash
git add src/soundcode/seedvc.py src/soundcode/sing.py src/soundcode/render_sf.py src/soundcode/cli.py \
        scripts/install_seedvc.sh scripts/seedvc.patch tests/test_sing.py
git commit -m "sing: DiffSinger -> Seed-VC in the original singer's voice; render --with-vocals"
```

---

### Task 6: `compare --with-vocals`: pitch and voice metrics

**Files:**
- Modify: `src/soundcode/compare.py`, `src/soundcode/cli.py`
- Test: `tests/test_compare.py` (append)

**Interfaces:**
- Produces:
  - `compare.pitch_error_cents(y_ref, y_est, sr) -> float | None`: torchcrepe on both, median absolute cents over frames voiced in both (periodicity ≥ 0.5)
  - `compare.voice_similarity(y_ref, y_est, sr) -> float | None`: resemblyzer embedding cosine; `None` if either side is silent or resemblyzer is absent
  - `run(..., with_vocals: bool = False, voice_ref: Path | None = None)`: with vocals, the lead stem's render is the sung wav (via `sing.sing`), and the `lead_vocals` row gains `pitch_cents` and `voice_sim`
  - CLI `compare … --with-vocals [--voice-ref WAV]`, which prints `pitch_cents` and `voice_sim` under the table

- [ ] **Step 1: Install resemblyzer**

Run: `uv pip install --python .venv/bin/python resemblyzer`
Expected: `+ resemblyzer…` (about 20 MB).

- [ ] **Step 2: Write the failing tests** (append to `tests/test_compare.py`):

```python
# --- vocal metrics --------------------------------------------------------------------------

def _sweep(freq, cents_off=0.0, secs=2.0, sr=16000):
    t = np.arange(int(secs * sr)) / sr
    f = freq * 2 ** (cents_off / 1200) * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    return (0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)).astype(np.float32)


def test_pitch_error_is_zero_for_the_same_line_and_measures_detune():
    a = _sweep(220.0)
    assert cmp.pitch_error_cents(a, a, 16000) < 5
    assert 40 < cmp.pitch_error_cents(a, _sweep(220.0, 50.0), 16000) < 60


def test_pitch_error_of_silence_is_none():
    assert cmp.pitch_error_cents(_sweep(220.0), np.zeros(32000, np.float32), 16000) is None


def test_voice_similarity_is_none_on_silence():
    assert cmp.voice_similarity(_sweep(220.0), np.zeros(32000, np.float32), 16000) is None
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q -k "pitch_error or voice_similarity"`
Expected: 3 failures, `AttributeError`.

- [ ] **Step 4: Implement** in `compare.py`:

```python
def pitch_error_cents(y_ref, y_est, sr) -> float | None:
    import torch
    import torchcrepe

    if _silent(y_ref) or _silent(y_est):
        return None

    def f0(y):
        hz, per = torchcrepe.predict(torch.tensor(y, dtype=torch.float32)[None], sr,
                                     hop_length=sr // 100, fmin=50.0, fmax=1100.0, model="full",
                                     return_periodicity=True, batch_size=512, device="cpu")
        return hz[0].numpy(), torchcrepe.filter.median(per, 3)[0].numpy()

    n = min(len(y_ref), len(y_est))
    (hr, pr), (he, pe) = f0(y_ref[:n]), f0(y_est[:n])
    m = min(len(hr), len(he))
    both = (pr[:m] >= 0.5) & (pe[:m] >= 0.5)
    if both.sum() < 10:
        return None
    return float(np.median(np.abs(1200 * np.log2(he[:m][both] / hr[:m][both]))))


def voice_similarity(y_ref, y_est, sr) -> float | None:
    if _silent(y_ref) or _silent(y_est):
        return None
    try:
        import librosa
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return None
    enc = VoiceEncoder("cpu", verbose=False)
    to16 = lambda y: preprocess_wav(librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=16000))  # noqa: E731
    a, b = enc.embed_utterance(to16(y_ref)), enc.embed_utterance(to16(y_est))
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
```

In `run`, add the `with_vocals` and `voice_ref` parameters. After the `.sc` render loop, when `with_vocals` is set:

```python
        from . import sing
        wav, _ = sing.sing(doc, voice_ref)
        y, _sr = sf_read_mono(wav)
        rend["lead_vocals"] = np.zeros(n, np.float32)
        m = _mono(y, _sr)[:n]
        rend["lead_vocals"][:len(m)] = m
```

`sf_read_mono` is `librosa.load(str(p), sr=None, mono=True)`. For the `lead_vocals` row, when `with_vocals` is set, add `row["pitch_cents"] = pitch_error_cents(yo, yr, SR)` and `row["voice_sim"] = voice_similarity(yo, yr, SR)`. CLI: `--with-vocals` and `--voice-ref` flags; after the table, print `lead_vocals  pitch {pitch_cents} c  voice_sim {voice_sim}`.

- [ ] **Step 5: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/soundcode/compare.py src/soundcode/cli.py tests/test_compare.py
git commit -m "compare --with-vocals: pitch error (cents) and voice similarity for the sung lead"
```

---

### Task 7: Measure and listen

**Files:**
- Create: `docs/results/2026-09-27-singing-thin-slice.md`
- Modify: `README.md`

- [ ] **Step 1: Encode River with the new stages**

Run: `.venv/bin/soundcode encode audio/test/river-30s.wav -o out/sc/m2/river-30s.sc --workdir out/work`
Expected: the `.sc` has `:contour.vox rate=50` with `f0 @…` lines, and `:text.vox` words at 3-decimal beats with durations.

- [ ] **Step 2: Render and compare with vocals**

```bash
.venv/bin/soundcode render out/sc/m2/river-30s.sc --with-vocals -o out/sc/m2/river-30s.sung.render.wav
.venv/bin/soundcode compare audio/test/river-30s.wav out/sc/m2/river-30s.sc --with-vocals -o out/compare/m2/river-30s
```

Expected: `pitch_cents` ≤ 30 and `voice_sim` ≥ 0.88 on `lead_vocals`.

- [ ] **Step 3: Check the pitch lever.** Copy the `.sc` without its `:contour.vox` stream (`python3 - <<'EOF'` that drops the block from `:contour.vox` to the next blank line) to `out/sc/m2/river-30s.nocontour.sc`. Run `compare --with-vocals` on it into `out/compare/m2/river-30s-nocontour`.
Expected: a higher `pitch_cents` than step 2.

- [ ] **Step 4: The other four clips.** Encode each, then `render --with-vocals`. The loop should report per-song success or a `SingError` line: no crash, and numbers recorded without a bar. discipline and lights have lead vocals. 999999 and corona may produce `SingError` or a "no lyrics" warning, and that's acceptable.

- [ ] **Step 5: Listening checkpoint.** `afplay -t 30 out/sc/m2/river-30s.sung.render.wav`. Record the user's verdict.

- [ ] **Step 6: Write the results file** with the River numbers (with and without the contour), the other songs, and render timings. Add a README Status line for `render --with-vocals`. Run all tests, then commit:

```bash
.venv/bin/python -m pytest -q
git add docs/results/2026-09-27-singing-thin-slice.md README.md
git commit -m "docs: singing thin-slice results"
```
