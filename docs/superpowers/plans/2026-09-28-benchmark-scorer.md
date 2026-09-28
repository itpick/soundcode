# Benchmark Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score every part of a rebuild (vocals, keys, bass, drums, guitar, other, mix) from 0 to 100, per section and per 20 s window, over a benchmark set of 30 s to full-length songs, with drift flags, HTML reports and a committed run history.

**Architecture:**
- A new package, `src/soundcode/score/`. It holds pure metric functions on numpy slices, plus an orchestrator that runs the existing pipeline (`separate` → `encode` → `render --with-vocals --parts`) with caching.
- Rebuilt parts (`render --parts`) are compared with the original's separated stems. MERT embeddings drive the Sound axis.
- Scores map to 0–100 through the floor and ceiling anchors in `anchors.json`.

**Tech Stack:**
- Python 3.11, numpy, librosa, mir_eval, torchcrepe and resemblyzer (all already installed);
- pyloudnorm (new, MIT);
- transformers + MERT-v1-95M (new model, CC BY-NC 4.0; `transformers` is already a dependency of the ASR/tsumugi envs, so check the venv);
- tsumugi `drums_v1_5` via `soundcode.tsumugi`.

**Spec:** `docs/superpowers/specs/2026-09-28-benchmark-scorer-design.md`

## Global Constraints

- **Part keys:** exactly `render_sf.PART_KEYS` (`lead_vocals`, `backing_vocals`, `piano`, `guitar`, `bass`, `drums`, `other`, `residual`), plus `mix`. `residual` is never scored.
- **Gate:** an original part or slice whose loudest 100 ms is below −50 dBFS is *silent*: not scored and not a failure.
- **Tolerances:**
  - note F1 cover: onset ±100 ms, pitch ±100 cents, no offset;
  - onset F1: ±50 ms;
  - drift flag: |lag| > 30 ms, with lag searched in ±1.5 s.
- **Windows:** 20 s long with a 10 s hop. Sections come from `:struct` when it has ≥ 2 distinct labels, otherwise from 8-bar grid windows.
- **Scores:** `score = 100 × clamp((m − floor)/(ceiling − floor), 0, 1)`, inverted where lower is better.
  - Part score: `0.4·What + 0.4·Sound + 0.2·Dyn`.
  - Self-optimised metrics (`spectral_db`, level difference) weigh 0.1 inside their axis.
  - Song score: the part scores weighted by energy share, blended 50/50 with the mix.
- **MERT:** model `m-a-p/MERT-v1-95M`, 24 kHz mono input, layer-averaged hidden states, pooled per slice, scored by cosine. It lives in `"/Volumes/ExFAT 2/infinity-engine/models/mert"` when mounted, otherwise the HF cache. Embeddings are cached by file sha1 in `out/bench/cache/`.
- **History:** `docs/results/benchmark/history.jsonl` holds scores only: no audio, no lyrics.
- **Commits:** no Co-Authored-By or AI attribution. Tests run with `.venv/bin/python -m pytest -q`, never loading real models (MERT, Whisper and tsumugi are faked); the real-data tests skip when their inputs are absent.

## Review Focus

1. **Parts of unequal length.** The rebuild tail runs 31.5 s against 30.0 s, and SoulX output lengths differ from the original. Every metric trims to the common length, and a window past either end is *missing*, never an index error.
2. **Parts that are entirely silent, or NaN from a flat signal.** `std() == 0` in correlations, empty onset lists, no voiced frames. The metric returns `None`, the axis averages only the non-None metrics, and an axis with none is `None`, left out of the part score, not scored as 0.
3. **A missing rebuild part for an active original part.** For example backing vocals, which are never rebuilt. The part scores 0 with `missing: true`, and the report says "not rebuilt".
4. **Loop cost on full songs.** Corona Radiata runs 453 s: torchcrepe, MERT and chroma are computed **once per file** and sliced, never recomputed per window.
5. **An interrupted bench run.** A crash leaves no half-written `history.jsonl` line: the line is appended only after the whole run succeeds, and the README is written atomically.

---

## File Structure

| File | Contents |
|---|---|
| `src/soundcode/score/__init__.py` | exports `score_song`, `run_bench` |
| `src/soundcode/score/slices.py` | `Slice` dataclass, `sections(doc, duration)`, `windows(duration)`, `active(y, sr, a, b)` |
| `src/soundcode/score/align.py` | `lag_ms(y_ref, y_est, sr, a, b)`, `drift(y_ref, y_est, sr, windows)` |
| `src/soundcode/score/metrics.py` | `Features` (per-file precomputed arrays) and per-slice metric functions |
| `src/soundcode/score/drums.py` | per-voice onsets via tsumugi drums, per-voice F1 and decay |
| `src/soundcode/score/embed.py` | MERT loading, frame embeddings, cache, per-slice cosine |
| `src/soundcode/score/anchors.json` + `anchors.py` | floor/ceiling/weight/direction per (part type, metric); `to_score` |
| `src/soundcode/score/scorer.py` | `score_part`, `score_song` → a nested dict |
| `src/soundcode/score/report.py` | terminal table, HTML report, excerpts |
| `src/soundcode/score/bench.py` | tiers, cached pipeline, run dir, history, README, diff |
| `src/soundcode/cli.py` | `score` and `bench` subcommands |
| `src/soundcode/render_sf.py` | no misleading SoulX warnings |
| `tests/test_score_*.py` | one test file per module |

---

### Task 1: Slices and alignment

**Files:**
- Create: `src/soundcode/score/__init__.py`, `src/soundcode/score/slices.py`, `src/soundcode/score/align.py`
- Test: `tests/test_score_slices.py`

**Interfaces:**
- Produces:
  - `Slice(kind: str, label: str, a: float, b: float)`, where kind is `"section"`, `"window"` or `"song"`;
  - `sections(doc, duration: float) -> list[Slice]`;
  - `windows(duration: float, length=20.0, hop=10.0) -> list[Slice]`;
  - `song(duration) -> Slice`;
  - `active(y, sr, a, b, gate_db=-50.0) -> bool`, true if the loudest 100 ms RMS in [a, b) is ≥ gate;
  - `lag_ms(y_ref, y_est, sr, a, b, search_s=1.5) -> float | None`: positive means the rebuild is late, computed on the onset-strength envelopes (hop 10 ms), with the best Pearson correlation over the lags; None if either is silent;
  - `drift(y_ref, y_est, sr, wins) -> list[dict(label, a, b, lag_ms, drift: bool)]`, with drift when |lag| > 30.

- [ ] **Step 1: Write failing tests**

```python
"""Slices and alignment (spec 2026-09-28-benchmark-scorer)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundcode.parser import parse  # noqa: E402
from soundcode.score import align, slices  # noqa: E402

GRID = "%sc 0.3\n@duration 40.0\n\n:grid\nmeter @0.000 4/4\nanchor bar 1 @0.000\ntempo @0.000 120\n"


def clicks(sr, times, n):
    y = np.zeros(n, np.float32)
    for t in times:
        i = int(t * sr)
        y[i:i + 200] = np.hanning(200)
    return y


def test_windows_cover_the_song_with_hop():
    w = slices.windows(45.0)
    assert [(s.a, s.b) for s in w][:3] == [(0.0, 20.0), (10.0, 30.0), (20.0, 40.0)]
    assert w[-1].b == 45.0 and all(s.kind == "window" for s in w)


def test_sections_fall_back_to_8_bar_windows_when_struct_is_weak():
    doc = parse(GRID)                                   # no :struct → 8 bars at 120 bpm 4/4 = 16 s
    s = slices.sections(doc, 40.0)
    assert [(x.a, x.b) for x in s] == [(0.0, 16.0), (16.0, 32.0), (32.0, 40.0)]
    assert s[0].label == "bars 1-8"


def test_sections_use_struct_when_it_has_two_labels():
    doc = parse(GRID + "\n:struct\nintro 1-4 inst energy=0.2\nverse 5-12 vocal energy=0.5\n")
    s = slices.sections(doc, 40.0)
    assert [x.label for x in s][:2] == ["intro", "verse"]
    assert s[0].a == 0.0 and s[0].b == pytest.approx(8.0)


def test_active_gate():
    sr = 8000
    y = np.zeros(sr * 4, np.float32)
    y[sr:sr + 800] = 0.1                                # -20 dBFS burst in second 1
    assert slices.active(y, sr, 1.0, 2.0) and not slices.active(y, sr, 2.0, 4.0)


def test_lag_and_drift():
    sr = 16000
    t = np.arange(0.5, 19.5, 0.37)
    ref = clicks(sr, t, sr * 20)
    late = clicks(sr, t + 0.2, sr * 20)
    assert align.lag_ms(ref, ref, sr, 0, 20) == pytest.approx(0, abs=10)
    assert align.lag_ms(ref, late, sr, 0, 20) == pytest.approx(200, abs=10)
    d = align.drift(ref, late, sr, [slices.Slice("window", "0:00", 0.0, 20.0)])
    assert d[0]["drift"] and d[0]["lag_ms"] == pytest.approx(200, abs=10)
    assert align.lag_ms(np.zeros(sr * 20, np.float32), ref, sr, 0, 20) is None
```

- [ ] **Step 2: Run:** `.venv/bin/python -m pytest -q tests/test_score_slices.py`. Expected: FAIL (`ModuleNotFoundError: soundcode.score`).

- [ ] **Step 3: Implement.**
  - `slices.sections`:
    - read `:struct` with the grid from `expand.build_grid(doc)`;
    - struct lines look like `intro 1-4 inst energy=0.22` (bars inclusive), with times from `grid.time_of(bar, 1.0)`;
    - use `:struct` only when there are ≥ 2 distinct labels;
    - otherwise cut 8-bar windows (`bars 1-8`, `bars 9-16`, …) up to `duration`, clamping the last one;
    - with no grid, use 16 s windows.
  - `active`: 100 ms frames, max RMS in dBFS ≥ `gate_db`.
  - `align.lag_ms`: `librosa.onset.onset_strength(y, sr, hop_length=sr//100)` on both slices (`y[int(a*sr):int(b*sr)]`), trimmed to equal length. For lags of −150…+150 frames, take the Pearson correlation of the overlapping parts, pick the maximum, and return `lag*10.0`. Return None when either signal is silent (std = 0 or RMS < 1e-5).

- [ ] **Step 4: Run** the same command. Expected: 5 passed.
- [ ] **Step 5: Commit:** `git add src/soundcode/score tests/test_score_slices.py && git commit -m "score: slices (sections, 20 s windows, gate) and lag/drift"`

### Task 2: Metrics on precomputed features

**Files:**
- Create: `src/soundcode/score/metrics.py`
- Test: `tests/test_score_metrics.py`

**Interfaces:**
- Consumes: `Slice` from Task 1, and `compare.SR`, `compare.note_f1`, `compare.transcribe`.
- Produces:
  - `Features.of(y: np.ndarray, sr: int, pitched: bool, f0: bool)`: computed **once per file**. It stores:
    - `y` (mono, resampled to 22050);
    - `onsets` (librosa onset times);
    - `chroma` (chroma_cqt, hop 512);
    - `rms` (hop 512);
    - `logspec`: a 1/12-octave log-spectrum per frame, from a 4096-point STFT (hop 2048) pooled into 12-per-octave bands from 40 Hz to 16 kHz;
    - `notes` (basic-pitch `(iv, hz)`, only when `pitched`);
    - `f0` / `voiced` (torchcrepe at 100 Hz, weighted_argmax, only when `f0`).
  - `slice_metrics(ref: Features, est: Features, s: Slice, part: str) -> dict[str, float | None]`. Keys by part type:
    - `note_f1`: pitched parts, cover tolerance (onset 0.1 s, pitch 100 c): filter both note lists to onsets in [a, b), then call a new `compare.note_f1(..., onset_tolerance=0.1, pitch_tolerance=100.0)`. Add those two keyword arguments to `compare.note_f1`, defaulting to today's 0.05/50.0 so existing callers are unchanged.
    - `chroma`: cosine of the mean chroma over the slice.
    - `onset_f1`: `mir_eval.onset.f_measure` on onsets inside [a, b), window 0.05.
    - `f0_cents`: median |cents| over frames voiced in both; bass and vocal parts.
    - `env_corr`: Pearson of the RMS over the slice.
    - `level_diff_db`: RMS dB of est − ref over the slice.
    - `logspec_db`: mean |dB difference| of the slice-mean 1/12-octave spectra, each normalised to its own mean, over bands within 50 dB of the reference's maximum.
  - Every function returns None on silence, flat signals, or a slice past the end (Review Focus 1–2).

- [ ] **Step 1: Write failing tests** (synthetic signals; basic-pitch and torchcrepe faked by monkeypatching `metrics._notes` and `metrics._f0` with deterministic stubs):

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundcode.score import metrics  # noqa: E402
from soundcode.score.slices import Slice  # noqa: E402

SR = 22050


def tone(freq, secs, sr=SR, amp=0.3):
    t = np.arange(int(secs * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr(metrics, "_notes", lambda y, sr: (np.array([[0.5, 1.5], [2.0, 3.0]]), np.array([440.0, 494.0])))
    monkeypatch.setattr(metrics, "_f0", lambda y, sr: (np.full(400, 440.0), np.ones(400, bool)))


def test_identical_parts_are_perfect():
    y = tone(440, 4.0)
    f = metrics.Features.of(y, SR, pitched=True, f0=True)
    m = metrics.slice_metrics(f, f, Slice("song", "song", 0.0, 4.0), "piano")
    assert m["chroma"] == pytest.approx(1.0) and m["note_f1"] == pytest.approx(1.0)
    assert m["level_diff_db"] == pytest.approx(0.0, abs=0.01) and m["logspec_db"] == pytest.approx(0.0, abs=0.01)


def test_quieter_and_silent_parts():
    y = tone(440, 4.0)
    ref = metrics.Features.of(y, SR, pitched=False, f0=False)
    est = metrics.Features.of(y * 0.5, SR, pitched=False, f0=False)
    m = metrics.slice_metrics(ref, est, Slice("song", "song", 0.0, 4.0), "other")
    assert m["level_diff_db"] == pytest.approx(-6.02, abs=0.1)
    z = metrics.Features.of(np.zeros_like(y), SR, pitched=False, f0=False)
    m0 = metrics.slice_metrics(ref, z, Slice("song", "song", 0.0, 4.0), "other")
    assert m0["env_corr"] is None and m0["chroma"] is None


def test_slice_past_the_end_is_none_not_an_error():
    f = metrics.Features.of(tone(440, 4.0), SR, pitched=False, f0=False)
    short = metrics.Features.of(tone(440, 2.0), SR, pitched=False, f0=False)
    m = metrics.slice_metrics(f, short, Slice("window", "0:03", 3.0, 4.0), "other")
    assert all(v is None for v in m.values())


def test_cover_tolerance_note_f1_forgives_60ms_and_60_cents():
    from soundcode import compare
    ref_iv, ref_hz = np.array([[1.0, 2.0]]), np.array([440.0])
    est_iv, est_hz = np.array([[1.06, 2.0]]), np.array([440.0 * 2 ** (60 / 1200)])
    assert compare.note_f1(ref_iv, ref_hz, est_iv, est_hz) == 0.0            # default ±50 ms/±50 c unchanged
    assert compare.note_f1(ref_iv, ref_hz, est_iv, est_hz, onset_tolerance=0.1, pitch_tolerance=100.0) == 1.0
```

- [ ] **Step 2: Run:** `.venv/bin/python -m pytest -q tests/test_score_metrics.py`. Expected: FAIL (no module).
- [ ] **Step 3: Implement**, following the interface above.
  - Module-level `_notes(y, sr)` wraps `compare.transcribe` on a temp WAV.
  - Module-level `_f0(y, sr)` wraps torchcrepe `full` with weighted_argmax at 16 kHz, hop 10 ms, returning `(hz, voiced = periodicity ≥ 0.5)`.
- [ ] **Step 4: Run.** Expected: 4 passed, plus the existing compare tests still passing: `.venv/bin/python -m pytest -q tests/test_compare.py`.
- [ ] **Step 5: Commit:** `git commit -m "score: per-slice metrics on per-file features; note_f1 tolerances are parameters"`

### Task 3: Drums per voice, and the MERT embedding

**Files:**
- Create: `src/soundcode/score/drums.py`, `src/soundcode/score/embed.py`, `scripts/install_mert.sh`
- Test: `tests/test_score_drums_embed.py`

**Interfaces:**
- Produces:
  - `drums.voice_onsets(wav: Path, work: Path) -> dict[str, np.ndarray]`: calls `tsumugi.transcribe(wav, work/"drums.mid", "drums_v1_5")`, then groups the MIDI note onsets by `gm.drum_voice(note)` into `kick`, `snare`, `hat`, `clap` and `other`. It is cached by the wav's sha1 in `work`.
  - `drums.voice_f1(ref: dict, est: dict, a, b) -> dict[str, float | None]`: onset F1 per voice within [a, b), ±50 ms. A voice absent on both sides is None; one present on only one side is 0.0.
  - `drums.decay_s(y, sr, onsets) -> float | None`: the median time from each isolated hit's peak to −20 dB.
  - `embed.frames(wav: Path) -> np.ndarray` of shape (T, 768) at MERT's 75 Hz frame rate. It loads the model once (`AutoModel.from_pretrained(path, trust_remote_code=True)` + `Wav2Vec2FeatureExtractor`), runs 24 kHz mono in 30 s chunks, averages the hidden states over all layers, and caches the result to `out/bench/cache/<sha1>.mert.npy`.
  - `embed.cosine(fr_ref, fr_est, a, b) -> float | None`: the mean-pool of each over [a, b), then their cosine.
  - `embed.MODEL_DIR`: the drive path if mounted, else None (the HF cache).
  - `scripts/install_mert.sh`: downloads `m-a-p/MERT-v1-95M` into the drive's `models/mert` with `huggingface-cli download`, and installs `nnAudio` if MERT's remote code needs it.

- [ ] **Step 1: Write failing tests.**
  - Monkeypatch `drums._midi_onsets` to return `[(0.5, 36), (1.0, 38), (1.5, 42), (2.0, 36)]` for the reference, and the same with the snare missing for the estimate. Assert `voice_f1` gives kick 1.0, snare 0.0, hat 1.0, and clap None.
  - `decay_s` on a synthetic exponential decay with τ chosen so that −20 dB is reached at 0.3 s gives ≈ 0.3 (±0.03).
  - `embed.cosine` with monkeypatched `embed.frames` (random fixed arrays): the same array gives 1.0; an orthogonal constructed pair gives ≈ 0; a slice past the end gives None.
  - A real MERT smoke test, `@pytest.mark.skipif(not model available)`: 3 s of a tone against itself has cosine > 0.99.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_score_drums_embed.py`. Expected: FAIL.
- [ ] **Step 3: Implement** following the interfaces. Keep `_midi_onsets(midi_path) -> list[(t, note)]` a separate, monkeypatchable function using `pretty_midi` or `mido` (whichever `tsumugi_sc.read_tracks` already uses).
- [ ] **Step 4: Run the install script once** (`bash scripts/install_mert.sh`) so the smoke test runs. Then run the tests. Expected: all pass, with the smoke test executed, not skipped.
- [ ] **Step 5: Commit:** `git commit -m "score: per-voice drum onsets and decay; MERT frame embeddings with cache"`

### Task 4: Anchors and the scorer

**Files:**
- Create: `src/soundcode/score/anchors.json`, `src/soundcode/score/anchors.py`, `src/soundcode/score/scorer.py`
- Test: `tests/test_score_scorer.py`

**Interfaces:**
- **`anchors.json`:** `{ "<part type>": { "<metric>": {"floor": x, "ceiling": y, "weight": w, "axis": "what"|"sound"|"dyn"} } }`.
  - Part types: `pitched` (piano, guitar, other), `bass`, `drums`, `vocal` (lead_vocals, backing_vocals), `mix`.
  - Initial values, until calibration in Task 7:

    | Metric | Floor | Ceiling | Weight | Axis |
    |---|---|---|---|---|
    | `note_f1` | 0.0 | 0.9 | 1 | what |
    | `chroma` | 0.5 | 0.98 | 1 | what |
    | `onset_f1` | 0.0 | 0.9 | 1 | what |
    | `f0_cents` | 300 | 10 | 1 | what |
    | `lag_ms_abs` | 150 | 5 | 1 | what |
    | `sung_wer_excess` | 0.6 | 0.0 | 1 | what (vocal only) |
    | `mert` | 0.6 | 0.97 | 1 | sound |
    | `logspec_db` | 12 | 1.5 | 1 | sound |
    | `voice_sim` | 0.6 | 0.95 | 1 | sound (vocal only) |
    | `decay_ratio_err` | 1.0 | 0.05 | 1 | sound (drums only) |
    | `env_corr` | 0.0 | 0.95 | 1 | dyn |
    | `level_diff_db_abs` | 12 | 0.5 | 0.1 | dyn |
    | `spectral_db` | 12 | 1.0 | 0.1 | sound |
    | `drum_voice_f1` | 0.0 | 0.9 | 1 | what (drums only) |
    | `lufs_diff_abs` | 10 | 0.5 | 1 | dyn (mix only) |
    | `word_mae_s` | 0.5 | 0.05 | 1 | what (vocal only) |
    | `width_diff` | 0.5 | 0.02 | 1 | sound (mix only) |

  - Lower-is-better is implied by `floor > ceiling`.
- **`anchors.to_score(value, a) -> float | None`**: `100·clamp((v−floor)/(ceiling−floor), 0, 1)`, None if v is None.
- **`scorer.score_slice(metrics: dict, part_type) -> dict`**: returns `{axes: {what, sound, dyn}, score, metrics: {name: {raw, score}}}`.
  - An axis is the weighted mean of its non-None metric scores, or None.
  - The score is `0.4 what + 0.4 sound + 0.2 dyn`, renormalised over the axes that aren't None; None if all are None.
- **`scorer.score_part(...)`**: combines the song slice, sections and windows into `{song, sections:[…], windows:[…], worst: {label, score}, missing, silent}`.
- **`scorer.song_score(parts: dict, energy: dict, mix) -> float`**: the energy-share-weighted mean of the non-silent parts, 50/50 with the mix. A missing active part counts as 0.
- **Tests:**
  - `to_score` both directions and clamping;
  - an axis with all-None metrics is excluded, not zero;
  - weights (`spectral_db` at 0.1 barely moves the axis);
  - the song score with a missing active part vs a silent part;
  - the worst section is picked correctly.
- **Steps:** tests → FAIL → implement → PASS → commit `"score: anchors (floor/ceiling/weight per metric) and the part/section/song scorer"`.

### Task 5: `score_song` and the report

**Files:**
- Create: `src/soundcode/score/report.py`
- Modify: `src/soundcode/score/scorer.py` (add `score_song`), `src/soundcode/score/__init__.py`
- Test: `tests/test_score_song.py`

**Interfaces:**
- **`score_song(original: Path, stems_dir: Path, sc: Path, parts_dir: Path, rebuild_mix: Path, out_dir: Path) -> dict`**. For each part key with an original stem or a rebuilt part:
  - build `Features` once per file;
  - skip `residual`;
  - mark the part silent if the original is below the gate for the whole song, or missing if the rebuild lacks an active part;
  - compute slice metrics for the song, sections and windows, plus drift from `align`, MERT cosine from `embed`, and drum voices from `drums` (drums only);
  - vocals only:
    - `voice_sim` = `compare.voice_similarity` over the song;
    - `sung_wer_excess` = `compare.sung_wer(doc, rebuilt part) − compare.sung_wer(doc, original stem)` (floor at 0), for the song only;
    - `word_mae_s`: faster-whisper `small` word timestamps on both the original stem and the rebuilt part. Align the words with `difflib.SequenceMatcher` on normalised words, then take the median |start difference| over matched words. Song level, plus per section over the words inside it. Share one transcription per file with `sung_wer`, which means refactoring `compare.sung_wer` to accept precomputed words, keeping its signature for existing callers;
  - the mix: chroma, onset_f1, lag, mert, `lufs_diff_abs` (`pyloudnorm`, add it to `pyproject.toml`), `width_diff` (|width_ref − width_est|, with width = side RMS / (mid RMS + side RMS) on the stereo files), and `spectral_db`.

  It returns `{song: <name>, duration, parts: {...}, mix: {...}, score, worst: {part, label, score}, drift: {part: [...]}}` and writes `scores.json` to `out_dir`.
- **`report.table(result) -> str`**: fixed-width rows `part | what | sound | dyn | score | worst (label @ m:ss)`, sorted by score ascending, with `—` for silent parts and `not rebuilt` for missing ones.
- **`report.html(result, out_dir)`**: builds `report.html` with inline CSS/JS and no external resources:
  - a heatmap table of parts × sections, one cell per score, colored red→green with the number shown;
  - SVG polylines for lag per window per part (a drift threshold line at ±30 ms), vocal `f0_cents` per window, and level diff per window;
  - a "Worst slices" table: the three lowest (part, section) slices, each with `<audio controls>` for `excerpts/<n>-original.wav` and `excerpts/<n>-rebuild.wav`. Each excerpt is 15 s centred in the slice, cut from the original stem and the rebuilt part.
  - DOM text uses escaped strings; build it with `html.escape` in Python.
- **Tests:** synthetic `score_song` on a tiny fixture.
  - Build a 12 s song with two parts (piano tone and drum clicks) as original stems. The rebuild parts are the same audio, with the piano 200 ms late and quieter. Use a small `.sc` with a grid.
  - Monkeypatch `embed.frames`, `drums.voice_onsets`, `metrics._notes` and `metrics._f0`.
  - Assert:
    - the drum part scores > 90;
    - the piano part is flagged drift in every window;
    - the piano What axis is < drums What by ≥ 30;
    - `table()` lists piano first;
    - `report.html` exists and contains `Worst slices`, and 3 excerpt pairs exist.
- **Steps:** tests → FAIL → implement → PASS → commit `"score: score_song over parts/sections/windows with drift; terminal table and HTML report"`.

### Task 6: `bench`, history, CLI, and the SoulX warnings

**Files:**
- Create: `src/soundcode/score/bench.py`
- Modify: `src/soundcode/cli.py`, `src/soundcode/render_sf.py` (`_streams`: when the singer is soulx, don't print `sing_score`'s "dropped … (no room)" warnings; keep "past the end" and every other warning), `.gitignore` (`out/` is already ignored; `audio/` too)
- Test: `tests/test_score_bench.py`, `tests/test_render_sf.py` (warnings)

**Interfaces:**
- **`TIERS: dict[str, list[dict]]`**, where an entry is `{name, source, start=None, end=None, title, artist}`.
  - A: `river-30s` (audio/test/river-30s.wav, "The River", "Jordan Feliz"), `discipline-30s`, `lights_in_the_sky-30s`, `999999-30s`, `corona_radiata-30s` (audio/test/*.mp3, NIN titles).
  - B: `discipline-100s` (audio/nin/discipline-100s.wav), `lights_in_the_sky-60s` (audio/nin/lights_in_the_sky.mp3, 30–90 s), `corona_radiata-60s` (audio/nin/corona_radiata.mp3, 0–60 s), `999999-full` (audio/nin/999999.mp3).
  - C: `discipline-full`, `lights_in_the_sky-full`, `corona_radiata-full` (audio/nin/*.mp3), `river-full` (`audio/uploads/The River-JordanFelix.mp3`).
- **`prepare(entry, root, force, run=subprocess.run) -> dict(paths)`**, cached by mtime per stage:
  - cut B excerpts with ffmpeg to `audio/bench/<name>.wav`;
  - `separate` → `out/stems/<name>/`;
  - `encode` → `out/bench/<name>/<name>.sc` (`--title/--artist`);
  - `render --with-vocals --parts out/bench/<name>/parts -o out/bench/<name>/rebuild.wav`;
  - each stage's wall time is recorded in the returned dict;
  - a failing stage raises `RuntimeError("<name>: <stage> failed (<last stderr line>)")`.
- **`run_bench(tier, label, root, force=False, run=subprocess.run) -> dict`**:
  - prepares and scores every entry, writing to `out/bench/runs/<YYYY-MM-DD-HHMM>-<slug(label)>/<name>/`;
  - then, **only after all succeed**:
    - appends one JSON line to `docs/results/benchmark/history.jsonl`: `{date, label, commit: git rev-parse --short HEAD, tier, songs: {name: {score, worst, parts: {part: {score, what, sound, dyn, sections: {label: score}}}}}, timing: {name: {stage: s}}}`;
    - rewrites `docs/results/benchmark/README.md` atomically (write `.part`, then replace). It holds the latest table per tier, the change from the previous run of the same tier per song and part (`+13`), and lists of "Improved (≥ +5)" and "Regressed (≤ −5)".
- **`diff(prev_line, cur_line) -> list[(song, part, before, after)]`**.
- **CLI:**
  - `soundcode score ORIGINAL SC [--stems DIR] [--out DIR]`: renders the parts if needed, calls `score_song`, prints `report.table`, and gives the report path.
  - `soundcode bench [--tier A|B|C|all] [--label TEXT] [--force] [--calibrate]` (calibrate lands in Task 7; for now it prints "not yet").
- **Tests:**
  - a fake runner that writes the stage outputs (like `tests/test_site.py`'s `_fake_run`), with `score_song` monkeypatched to return fixed dicts;
  - a two-run sequence: the history has 2 lines, and the README shows `+` and `−` changes;
  - a failing stage names the song and stage, and history is unchanged;
  - an interrupted run (score_song raising on song 2) leaves the history and README untouched;
  - the SoulX warnings test in `test_render_sf.py`.
- **Steps:** tests → FAIL → implement → PASS → full suite → commit `"score: bench tiers with cached pipeline, run history and README; score/bench CLI; SoulX renders stop printing heuristic drop warnings"`.

### Task 7: Calibration, validation and the first real runs

**Files:**
- Modify: `src/soundcode/score/bench.py` (`calibrate()`), `src/soundcode/score/anchors.json` (measured values)
- Create: `tests/test_score_real.py` (skips when inputs are absent), `docs/results/benchmark/history.jsonl`, `docs/results/benchmark/README.md`

**Interfaces:**
- **`calibrate(root)`**, per part type and metric:
  - **ceiling** = the median over tier-A songs of the metric between the original stem and a **second separation** of the same clip. Run `separate` again into `out/bench/calib/<name>/`, since demucs/roformer vary by about ±1 dB.
  - **floor** = the median over tier-A song pairs (a ≠ b) of the metric between song a's stem and song b's same stem, trimmed to equal length.
  - Write these into `anchors.json`, keeping the weights and axes. Degenerate anchors are left as they are, with a warning: those where |ceiling − floor| is tiny, or where the direction flips against the expected one.

**Steps:**
- [ ] **Step 1:** Implement `calibrate`, with a unit test on fake metric values (median logic, a flip guard). Run `soundcode bench --calibrate` for real; it takes several minutes. Commit the updated `anchors.json` with a summary of floors and ceilings in the commit message.
- [ ] **Step 2: Real validation tests** in `tests/test_score_real.py` (skipif the inputs are missing):
  - original stem against itself: the part score is ≥ 95 for every non-silent part of `discipline-30s`;
  - `discipline-30s` lead vocal stem against `lights_in_the_sky-30s` lead vocal stem: ≤ 10;
  - the **pre-fix full Discipline vocal** (`out/bench/fixtures/discipline-full-vocal-autoshift.wav`) against the original lead-vocal stem (`out/stems/discipline/lead_vocals.wav`) scores lead_vocals at least 30 points below the fixed one (`…-fixed.wav`);
  - a 200 ms-delayed copy of the `discipline-30s` drum stem is flagged drift in every window, and drops What by ≥ 30.

  Run them. If an anchor assertion fails, fix the anchors or metrics, not the test thresholds.
- [ ] **Step 3:** Run `soundcode bench --tier A --label baseline`, and commit `docs/results/benchmark/`.
- [ ] **Step 4:** Run `soundcode bench --tier C --label baseline` (Discipline full, about 1 hour cold). Commit the history and README.
- [ ] **Step 5 (user checkpoint):** open the full-Discipline `report.html` for the user, and play the three worst-slice listen pairs.
- [ ] **Step 6: Commit:** `git commit -m "score: calibrated anchors, real validation tests, tier A and C baselines"`
