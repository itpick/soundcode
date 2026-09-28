# Correct Words + Clearer Singing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- The `.sc` carries the correct, performance-timed lyrics: large ASR, reconciled with LRCLIB's published lyrics.
- The sung vocal becomes intelligible and better sounding, by picking the better of an improved DiffSinger→Seed-VC chain and SoulX-Singer.
- Everything is measured with `lyric_wer` and `sung_wer`.

**Architecture:**
- `lyrics.py`: LRCLIB lookup + cache, reference-word extraction for the clip window, and reconcile (difflib alignment).
- `encode.stage_lyrics` uses `large-v3-turbo` and reconciles when a reference exists.
- `compare` adds `lyric_wer` and `sung_wer`.
- `sing_score` gains DiffSinger `dsdur` durations.
- `soulx.py` adapts `.sc` → SoulX-Singer metadata and runs it over ssh on `framepick`.
- `sing.sing(…, singer=)` selects the engine.

**Tech Stack:** faster-whisper (`large-v3-turbo`), requests/urllib (LRCLIB), difflib, onnxruntime (`dsdur`), SoulX-Singer (Apache-2.0, conda py3.10, CUDA on framepick).

**Spec:** `docs/superpowers/specs/2026-09-27-lyrics-and-voice-design.md`.

## Global Constraints

- Open, free tools only. LRCLIB is free with no key: send `User-Agent: infinity-engine/0.1 (private research)`, one request per lookup, and cache responses under `out/lyrics/`.
- Copyrighted lyrics stay private: the `out/` cache and `.sc` files are gitignored, and test fixtures use made-up lyrics.
- Large models go on the external drive (`/Volumes/ExFAT 2/infinity-engine/models/…`). Internal free space must stay above 3 GB.
- SoulX-Singer runs on `framepick` in its own env under `~/infinity-engine/soulx-singer`. Use `ssh -o BatchMode=yes` and scp with BatchMode, as `seedvc.py` does, including `LD_LIBRARY_PATH=/run/opengl-driver/lib`.
- Renders and encodes without a reference lyric, or with `--singer diffsinger`, behave as today apart from the better ASR.
- No `Co-Authored-By` in commits.

## Review Focus

1. **No LRCLIB match, offline, or HTTP 429/5xx.** The encode proceeds with ASR only and `meta warn`; nothing crashes; 429 is not retried in a loop. (Task 1 test.)
2. **The clip starts mid-song** (`@offset` > 0), **or the song is a different recording** (live, remix). Only lines inside the clip window are used, and a low alignment ratio (< 0.5) falls back to ASR-only with a warning instead of forcing wrong words. (Task 2 test.)
3. **Repeated choruses:** the same line appears several times. The alignment must stay monotonic in time and never map a word to the wrong repetition. (Task 2 test.)
4. **Ad-libs and words the reference lacks:** kept with `?conf`, never deleted. (Task 2 test.)
5. **SoulX-Singer failing on a segment** (too long, unknown phoneme). Fall back per render to the DiffSinger chain with a one-line note. (Task 5 test.)

---

### Task 1: LRCLIB lookup

**Files:** Create `src/soundcode/lyrics.py`. Test `tests/test_lyrics.py`.

**Interfaces:**
- Produces:
  - `@dataclass Line(t: float | None, text: str)`
  - `guess_title_artist(path: Path, title: str | None = None, artist: str | None = None) -> tuple[str, str | None]`
  - `lookup(title: str, artist: str | None, duration: float | None, cache: Path = Path("out/lyrics")) -> list[Line] | None`: synced lines when present, else plain lines with `t=None`; `None` when there is no match or on any error
  - `ref_words(lines: list[Line], start: float, end: float) -> list[tuple[float | None, str]]`: normalised words (lowercase, no punctuation) from lines inside `[start − 1, end]`. When no line has a time, every word is used with `t=None`.
  - `normalise(word: str) -> str`

- [ ] **Step 1: Write the failing tests** in `tests/test_lyrics.py` (fictional lyrics only):

```python
"""Lyrics: LRCLIB lookup and reconciliation (spec 2026-09-27-lyrics-and-voice)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import lyrics as ly  # noqa: E402

LRC = "[00:01.00] Walk me down to the harbor\n[00:05.50] Lights are low and the tide is high\n[00:40.00] Walk me down to the harbor\n"


class FakeResp:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def read(self):
        return json.dumps(self.body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_guess_title_artist_from_file_names():
    assert ly.guess_title_artist(Path("The River-JordanFelix.mp3")) == ("The River", "Jordan Felix")
    assert ly.guess_title_artist(Path("Some Artist - A Song.wav")) == ("A Song", "Some Artist")
    assert ly.guess_title_artist(Path("x.wav"), "T", "A") == ("T", "A")


def test_lookup_parses_synced_lines_and_caches(tmp_path, monkeypatch):
    calls = []
    def fake_open(req, timeout=20):
        calls.append(req.full_url)
        return FakeResp({"trackName": "Harbor", "artistName": "Nobody", "duration": 60.0,
                         "syncedLyrics": LRC, "plainLyrics": "x"})
    monkeypatch.setattr(ly.urllib.request, "urlopen", fake_open)
    lines = ly.lookup("Harbor", "Nobody", 60.0, cache=tmp_path)
    assert lines[0] == ly.Line(1.0, "Walk me down to the harbor")
    assert ly.lookup("Harbor", "Nobody", 60.0, cache=tmp_path) == lines
    assert len(calls) == 1                                   # second call from the cache


def test_lookup_failure_is_none_not_a_crash(tmp_path, monkeypatch):
    import urllib.error
    def boom(req, timeout=20):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
    monkeypatch.setattr(ly.urllib.request, "urlopen", boom)
    assert ly.lookup("Harbor", "Nobody", 60.0, cache=tmp_path) is None


def test_ref_words_keep_only_the_clip_window():
    lines = [ly.Line(1.0, "Walk me down to the harbor"), ly.Line(5.5, "Lights are low, and the tide is high!"),
             ly.Line(40.0, "Walk me down to the harbor")]
    words = [w for _, w in ly.ref_words(lines, 0.0, 30.0)]
    assert words[:3] == ["walk", "me", "down"] and "high" in words and words.count("harbor") == 1
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_lyrics.py -q`
Expected: collection error `cannot import name 'lyrics'`.

- [ ] **Step 3: Implement** `src/soundcode/lyrics.py`:

```python
"""Published lyrics (LRCLIB, free, no key) reconciled with what ASR heard.

The reference fixes mishearings ("wider" -> "whiter"); the ASR keeps the
performed timing and the ad-libs the reference leaves out. Responses are
cached under out/lyrics/ (gitignored: lyrics are copyrighted, private use).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = "https://lrclib.net/api"
UA = "infinity-engine/0.1 (private research)"
_LRC = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")


@dataclass(frozen=True)
class Line:
    t: float | None
    text: str


def normalise(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower()).strip("'")


def guess_title_artist(path: Path, title: str | None = None,
                       artist: str | None = None) -> tuple[str, str | None]:
    if title:
        return title, artist
    stem = Path(path).stem
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return t.strip(), a.strip()
    if "-" in stem:
        t, a = stem.rsplit("-", 1)
        a = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", a).strip()          # JordanFelix -> Jordan Felix
        return t.strip(), a or None
    return stem, artist


def _get(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _lines(rec: dict) -> list[Line]:
    if rec.get("syncedLyrics"):
        out = []
        for raw in rec["syncedLyrics"].splitlines():
            m = _LRC.match(raw.strip())
            if m and m.group(3).strip():
                out.append(Line(int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
        return out
    return [Line(None, ln.strip()) for ln in (rec.get("plainLyrics") or "").splitlines() if ln.strip()]


def lookup(title: str, artist: str | None, duration: float | None,
           cache: Path = Path("out/lyrics")) -> list[Line] | None:
    key = re.sub(r"[^a-z0-9]+", "-", f"{artist or ''}-{title}".lower()).strip("-")
    path = Path(cache) / f"{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        return _lines(rec) if rec else None
    try:
        q = {"track_name": title} | ({"artist_name": artist} if artist else {})
        rec = None
        if duration:
            try:
                rec = _get(f"{API}/get?" + urllib.parse.urlencode(q | {"duration": round(duration)}))
            except Exception:                                       # noqa: BLE001 - try search
                rec = None
        if not rec:
            found = _get(f"{API}/search?" + urllib.parse.urlencode(q)) or []
            found = [r for r in found if r.get("syncedLyrics") or r.get("plainLyrics")]
            if duration:
                found.sort(key=lambda r: abs((r.get("duration") or 0) - duration))
            rec = found[0] if found else None
    except Exception:                                               # noqa: BLE001 - offline, 429, 5xx
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec or {}))
    return _lines(rec) if rec else None


def ref_words(lines: list[Line], start: float, end: float) -> list[tuple[float | None, str]]:
    timed = any(ln.t is not None for ln in lines)
    out = []
    for ln in lines:
        if timed and (ln.t is None or not (start - 1.0 <= ln.t <= end)):
            continue
        out += [(ln.t, w) for w in (normalise(x) for x in ln.text.split()) if w]
    return out
```

**Duration note:** when the clip is shorter than the song (a 30 s excerpt), `lookup` receives the *song* duration if known. Otherwise pass `None`, and search picks the first result. The River clip's full song is 196 s. `encode` passes `None` for clips, and the plan accepts search order.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_lyrics.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/lyrics.py tests/test_lyrics.py
git commit -m "lyrics: LRCLIB lookup (cached, failure-safe), clip-window reference words"
```

---

### Task 2: Reconcile ASR with the reference; better ASR in the encoder

**Files:** Modify `src/soundcode/lyrics.py`, `src/soundcode/encode.py` (`stage_lyrics`, `encode()` signature and the `--artist` flag in `cli.py`). Test `tests/test_lyrics.py` (append).

**Interfaces:**
- Produces:
  - `reconcile(asr: list[tuple[float, float, str, float]], ref: list[tuple[float | None, str]]) -> tuple[list[tuple[float, float, str, float, str | None]], float]`: the words (start, end, text, conf, alt) plus the alignment ratio
  - `ASR_MODEL = os.environ.get("SOUNDCODE_ASR_MODEL", "large-v3-turbo")`
  - `stage_lyrics(stem, sr, grid, reference: list[tuple[float | None, str]] | None = None) -> Stage`
  - `encode(..., artist: str | None = None)`; the CLI `encode --artist`

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- reconcile -------------------------------------------------------------------------------

ASR = [(1.0, 1.3, "walk", 0.9), (1.3, 1.5, "me", 0.95), (1.5, 1.8, "down", 0.9), (1.8, 2.0, "to", 0.9),
       (2.0, 2.2, "the", 0.9), (2.2, 2.9, "harder", 0.6),                  # misheard 'harbor'
       (3.0, 3.3, "yeah", 0.7)]                                           # ad-lib
REF = [(1.0, w) for w in "walk me down to the harbor".split()]


def test_reconcile_fixes_mishearings_and_keeps_adlibs():
    words, ratio = ly.reconcile(ASR, REF)
    texts = [w[2] for w in words]
    assert texts == ["walk", "me", "down", "to", "the", "harbor", "yeah"]
    harbor = words[5]
    assert harbor[0] == 2.2 and harbor[4] == "harder"                     # ASR timing, alt kept
    assert words[6][3] == 0.7 and ratio > 0.7


def test_missed_reference_words_are_interpolated_and_marked():
    asr = [w for w in ASR if w[2] != "to"]
    words, _ = ly.reconcile(asr, REF)
    to = next(w for w in words if w[2] == "to")
    assert 1.8 <= to[0] <= 2.0 and to[3] == 0.5


def test_wrong_recording_falls_back_to_asr():
    ref = [(1.0, w) for w in "completely different words about nothing here".split()]
    words, ratio = ly.reconcile(ASR, ref)
    assert ratio < 0.5 and [w[2] for w in words] == [w[2] for w in ASR]


def test_repeated_chorus_stays_monotonic():
    asr = [(float(i), float(i) + 0.4, w, 0.9) for i, w in enumerate("la la land la la land".split())]
    ref = [(None, w) for w in "la la land la la land".split()]
    words, _ = ly.reconcile(asr, ref)
    assert [w[0] for w in words] == sorted(w[0] for w in words) and len(words) == 6
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_lyrics.py -q`
Expected: 4 failures (`reconcile` is missing).

- [ ] **Step 3: Implement** in `lyrics.py`:

```python
def reconcile(asr: list[tuple[float, float, str, float]],
              ref: list[tuple[float | None, str]]):
    """Reference words with ASR timing. Returns (words, ratio); below 0.5 the
    reference is judged not to match this recording and ASR is kept as heard."""
    import difflib

    a = [normalise(w[2]) for w in asr]
    b = [w for _, w in ref]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ratio = sm.ratio()
    if ratio < 0.5 or not ref:
        return [(s, e, w, p, None) for s, e, w, p in asr], ratio
    out: list[tuple[float, float, str, float, str | None]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out += [(asr[i][0], asr[i][1], b[j1 + (i - i1)], max(asr[i][3], 0.9), None)
                    for i in range(i1, i2)]
        elif tag == "replace":
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                if k < i2 - i1 and k < j2 - j1:
                    s, e, heard, p = asr[i1 + k]
                    out.append((s, e, b[j1 + k], 0.7, heard))
                elif k < j2 - j1:                                      # extra reference word
                    out.append((None, None, b[j1 + k], 0.5, None))
                else:                                                  # extra ASR word
                    s, e, heard, p = asr[i1 + k]
                    out.append((s, e, heard, p, None))
        elif tag == "delete":                                          # ASR only: ad-lib
            out += [(asr[i][0], asr[i][1], asr[i][2], asr[i][3], None) for i in range(i1, i2)]
        else:                                                          # insert: reference only
            out += [(None, None, b[j], 0.5, None) for j in range(j1, j2)]
    # time the reference-only words between their neighbours
    for k, w in enumerate(out):
        if w[0] is None:
            prev = next((x[1] for x in reversed(out[:k]) if x[1] is not None), 0.0)
            nxt = next((x[0] for x in out[k + 1:] if x[0] is not None), prev + 0.4)
            span = max(nxt - prev, 0.1)
            out[k] = (prev, prev + span, w[2], 0.5, None)
    return out, ratio
```

This is monotonic by construction, because opcodes walk both sequences in order.

In `encode.py`, change `stage_lyrics`:
- use `WhisperModel(ASR_MODEL, device="cpu", compute_type="int8", download_root=<external>/models/whisper if the drive is mounted else None)`;
- collect `(start, end, text, prob)`;
- if `reference` is given, `words, ratio = lyrics.reconcile(words, reference)`, and set `st.src = f"lrclib+{ASR_MODEL}"` when `ratio >= 0.5`, otherwise add a warn `reference lyrics did not match this recording (ratio …)`;
- pass `alt` through to `lyric_cells`: extend the cell with ` alt="<heard>"` when not None, and the parser keeps `alt=`.

In `encode()`, add the `artist` parameter:

```python
    from . import lyrics as ly
    t, a = ly.guess_title_artist(src, title, artist)
    lines = ly.lookup(t, a, None)
    reference = ly.ref_words(lines, 0.0, duration) if lines else None
    if lines is None:
        STEM_NOTES.append(f"no published lyrics found for '{t}' / '{a}'; ASR only")
```

Then pass `reference` to `stage_lyrics`. In the CLI, add `p_encode.add_argument("--artist", default=None)` and thread it through.

- [ ] **Step 4: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/lyrics.py src/soundcode/encode.py src/soundcode/cli.py tests/test_lyrics.py
git commit -m "lyrics: reconcile ASR with published lyrics; large-v3-turbo ASR; encode --artist"
```

---

### Task 3: `lyric_wer` and `sung_wer` in compare

**Files:** Modify `src/soundcode/compare.py`, `src/soundcode/cli.py`. Test `tests/test_compare.py` (append).

**Interfaces:**
- Produces:
  - `compare.wer(ref: list[str], hyp: list[str]) -> float` (Levenshtein over words)
  - `report["lyrics"] = {"lyric_wer": float | None, "sung_wer": float | None}`
    - `lyric_wer` needs the cached reference (via `lyrics.lookup` on the doc's `@source`/`@title`).
    - `sung_wer` needs `--with-vocals`: whisper-small on the sung vocal vs the `.sc` words.
  - The CLI prints `lyrics: wer vs published X, sung intelligibility wer Y`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_compare.py`):

```python
def test_wer():
    assert cmp.wer("a b c d".split(), "a b c d".split()) == 0.0
    assert cmp.wer("a b c d".split(), "a x c".split()) == pytest.approx(0.5)
    assert cmp.wer([], ["a"]) == 1.0
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q -k wer`
Expected: `AttributeError`.

- [ ] **Step 3: Implement.** Add `wer` to `compare.py`:

```python
def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)
```

In `run`:
- **`lyric_wer`:** get the reference via `lyrics.lookup(*guess_title_artist(Path(doc.header.get("source","")), doc.header.get("title")), None)` (cache only: the cached file exists after encode; do not hit the network from compare, i.e. check the cache path first and skip otherwise). Then `wer(ref_words_in_window, [w for _,_,w in sing_score.words(doc)])`.
- **`sung_wer`:** when `with_vocals`, `faster_whisper.WhisperModel("small", device="cpu", compute_type="int8")` on the sung wav, normalised, `wer(sc_words, heard)`.

Store both under `report["lyrics"]` and print them in the CLI.

- [ ] **Step 4: Run all tests, then commit**

```bash
.venv/bin/python -m pytest -q
git add src/soundcode/compare.py src/soundcode/cli.py tests/test_compare.py
git commit -m "compare: lyric_wer (vs published lyrics) and sung_wer (intelligibility of the sung vocal)"
```

---

### Task 4: DiffSinger phoneme durations from the bank's `dsdur` model; Seed-VC 50 steps

**Files:** Modify `src/soundcode/sing_score.py`, `src/soundcode/diffsinger.py`, `src/soundcode/sing.py`. Test `tests/test_sing.py` (append).

**Interfaces:**
- Produces:
  - `diffsinger.predict_durations(phonemes: list[str], word_div: list[int], word_frames: list[int], ph_midi: list[int]) -> list[int]`. It runs `dsdur/linguistic.onnx` (tokens, languages, word_div, word_dur) → `dsdur/dur.onnx` (encoder_out, x_masks, ph_midi, spk_embed) → `ph_dur_pred`. The predictions are rescaled per word so they sum exactly to that word's frames.
  - `sing_score.build(doc, duration=None, durations="heuristic"|"model")`: `"model"` builds words and syllables as today, then replaces each word's phoneme frames with `predict_durations` for that word, keeping word boundaries.
  - `sing.sing(…, steps=50, durations="model")` becomes the default settings, and both enter the cache key.

- [ ] **Step 1: Write the failing tests** (append; the bank-dependent one is skipped when the bank is absent):

```python
def test_model_durations_keep_word_boundaries_and_total_length():
    if not _bank_present():
        pytest.skip("no bank")
    heur = ss.build(parse(SONG))
    mod = ss.build(parse(SONG), durations="model")
    assert sum(mod.frames) == mod.n_frames == heur.n_frames
    assert mod.phonemes == heur.phonemes
    assert mod.frames != heur.frames                        # the model actually changed timing
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_sing.py -q -k model_durations`
Expected: `TypeError: build() got an unexpected keyword argument 'durations'`.

- [ ] **Step 3: Implement.** In `sing_score.build`, record each word's phoneme index range while building `seq` (a list of `(i0, i1)` per word, with `SP`/`AP` as single-phoneme "words"). After computing `frames`, if `durations == "model"`:

```python
        from . import diffsinger
        word_div = [i1 - i0 for i0, i1 in spans]
        word_frames = [sum(frames[i0:i1]) for i0, i1 in spans]
        ph_midi = [int(round(69 + 12 * np.log2(max(f0[min(sum(frames[:k]), len(f0) - 1)], 1) / 440)))
                   for k in range(len(frames))]
        frames = diffsinger.predict_durations([p for p, _, _ in seq], word_div, word_frames, ph_midi)
```

`diffsinger.predict_durations` runs the two dsdur graphs with the bank's `embeds/01CORE.emb` repeated per token, then rescales per word: `np.round(pred / pred.sum() * word_frames)`, fixing rounding on the longest phoneme and keeping each phoneme ≥ 1.

In `sing.sing`, the defaults become `steps=50` and `durations="model"`, and both go into `settings`.

- [ ] **Step 4: Run all tests and confirm they pass; commit**

```bash
.venv/bin/python -m pytest -q
git add src/soundcode/sing_score.py src/soundcode/diffsinger.py src/soundcode/sing.py tests/test_sing.py
git commit -m "sing: phoneme durations from the bank's dsdur model; Seed-VC 50 steps"
```

---

### Task 5: SoulX-Singer on framepick

**Files:** Create `src/soundcode/soulx.py`, `scripts/install_soulx_remote.sh`. Modify `src/soundcode/sing.py`, `src/soundcode/cli.py` (`--singer`). Test `tests/test_soulx.py`.

**Interfaces:**
- Produces:
  - `soulx.metadata(doc, t0: float, t1: float) -> list[dict]`: SoulX segments (`index`, `language`, `time` "[ms0, ms1]", `duration`, `text`, `phoneme`, `note_pitch`, `note_type`, `f0`) for the vocal between `t0` and `t1`, split at rests into segments of at most 15 s.
    - Notes come from `sing_score.vocal_notes`, words from `sing_score.words`, and phonemes are CMUdict ARPAbet with stress, in the `en_R-IH1-V-ER0` format.
    - **The `note_type` and `f0` hop semantics are read from SoulX-Singer's `preprocess/` code in Step 1 and pinned in a test.**
  - `soulx.render(doc, ref_wav: Path, prompt: tuple[float, float]) -> np.ndarray`. It builds target metadata for the whole song and prompt metadata for the reference span (default: the first 8 s of lead-vocal notes), uploads them with the prompt clip, runs `python -m cli.inference --control melody --auto_shift …` remotely, and downloads the wav atomically.
  - `sing.sing(…, singer="diffsinger"|"soulx")`. `soulx` falls back to the DiffSinger chain on `SingError`, with a warning.

- [ ] **Step 1: Install on framepick and pin the format.** Write `scripts/install_soulx_remote.sh`:
  - clone `https://github.com/Soul-AILab/SoulX-Singer` into `~/infinity-engine/soulx-singer`;
  - create a uv venv with Python 3.10 (instead of conda) and `pip install -r requirements.txt`;
  - `hf download Soul-AILab/SoulX-Singer` and `Soul-AILab/SoulX-Singer-Preprocess`;
  - print the commit hash.

Run it. Then read `preprocess/` on the remote (`ssh framepick 'grep -rn note_type ~/infinity-engine/soulx-singer/preprocess | head'`) and the example `example/audio/en_target.json` to pin down:
  - what `note_type` 1/2/3 mean (rest, word start, continuation);
  - the `f0` hop size and sample rate;
  - the phoneme format for word continuations (melisma).

Record them in a comment at the top of `soulx.py` and as constants.

- [ ] **Step 2: Write the failing tests** in `tests/test_soulx.py`:
  - **`test_metadata_matches_the_pinned_format`:** from `tests/test_sing.SONG`, `soulx.metadata(doc, 0, 4)` gives one segment. Its `text` token count equals its `duration` count, equals its `note_pitch` count, equals its `note_type` count. The rests are `<SP>` with pitch 0, the phoneme of "river" is `en_R-IH1-V-ER0`, and the f0 length equals the segment duration divided by the pinned hop.
  - **`test_long_songs_split_at_rests_under_15s`:** a synthetic 40 s vocal with rests every 8 s splits into segments of ≤ 15 s that cover every note.
  - **`test_soulx_failure_falls_back_to_diffsinger`:** monkeypatch `soulx.render` to raise `SingError`, and check `sing.sing(…, singer="soulx")` returns the DiffSinger-chain wav with a warning.

- [ ] **Step 3: Run the tests and confirm they fail, then implement** `soulx.py` (a metadata builder, plus the remote runner modelled on `seedvc._remote`: the same ssh/scp options, `.part` rename and cleanup) and the `sing.sing` / CLI wiring. The `singer` value goes into the cache key.

- [ ] **Step 4: Real run.** River 30 s: `render --with-vocals --singer soulx -o out/sc/lv/river-30s.soulx.wav`. It should produce audio; record the time taken.

- [ ] **Step 5: Run all tests, then commit**

```bash
.venv/bin/python -m pytest -q
git add src/soundcode/soulx.py src/soundcode/sing.py src/soundcode/cli.py scripts/install_soulx_remote.sh tests/test_soulx.py
git commit -m "sing: SoulX-Singer (zero-shot, melody-conditioned) on the GPU box as an alternative singer"
```

---

### Task 6: Measure, choose, listen

**Files:** Create `docs/results/2026-09-27-lyrics-and-voice.md`. Modify `README.md`, and `sing.py` (the default singer = the winner).

- [ ] **Step 1: Baseline.** Using the current `out/sc/fx/river-30s.sc` (whisper-base lyrics, old chain): run `compare --with-vocals` and record `lyric_wer`, `sung_wer`, `voice_sim` and `pitch_cents`.
- [ ] **Step 2: Re-encode River** with the new lyrics stage into `out/sc/lv/river-30s.sc`, with `--title "The River" --artist "Jordan Feliz"`. Check the three lines whisper-base misheard, and that `lyric_wer` ≤ 0.10.
- [ ] **Step 3: Head-to-head** on the new `.sc`: `compare --with-vocals` with `--singer diffsinger` (the improved chain), then with `--singer soulx`. Record the four numbers for each.
- [ ] **Step 4: Choose the default** by `sung_wer`, then `voice_sim`, then `pitch_cents`. Set `sing.DEFAULT_SINGER`. The winner's `sung_wer` must be at least 30% below the Step 1 baseline.
- [ ] **Step 5: Listening checkpoint.** Play the original (25 s), then the winner's full rebuild (25 s), then the other singer's (25 s), with copies in `~/Downloads/`. Record the user's verdict.
- [ ] **Step 6: Results file + README.** Run the tests and commit:

```bash
.venv/bin/python -m pytest -q
git add docs/results/2026-09-27-lyrics-and-voice.md README.md src/soundcode/sing.py
git commit -m "docs: correct words + clearer singing results; default singer"
```
