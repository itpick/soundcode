# Milestone 1: Clean Separation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `soundcode separate song.mp3` splits any song into seven clean stems:
- lead vocals and backing vocals
- drums, bass, guitar, piano and other

It also writes a vocals-only mix, an instrumental mix, and a residual, and reports how well the stems rebuild the original. The encoder and the listening server use these stems.

**Architecture:**
- A new `src/soundcode/separate.py` runs three model passes through a small `Backend` protocol:
  1. BS-RoFormer: mix → vocals + instrumental
  2. Karaoke Mel-RoFormer: vocals → lead + backing
  3. HTDemucs 6-stem: instrumental → drums / bass / guitar / piano / other
- The real backend wraps the `audio-separator` package. Tests use a fake numpy backend, so the pipeline logic is tested without downloading models.
- The residual (original − sum of stems) is always written. That way stems + residual rebuild the mix exactly, and the residual's level measures what separation lost.

**Tech Stack:** Python 3.11, numpy, soundfile, librosa (<1.0), audio-separator 0.47 (PyTorch on MPS/CPU, ONNX Runtime), pytest.

**Spec:** `docs/superpowers/plans/2026-09-26-infinity-engine-roadmap.md` (Milestone 1) and `docs/superpowers/specs/2026-07-31-soundcode-design.md` §5 (encoder stages, fail-soft rule).

## Global Constraints

- Python `>=3.11,<3.12` (from `pyproject.toml`). librosa must stay `<1.0`, because 1.0 requires 3.12.
- Stem names are exactly: `lead_vocals`, `backing_vocals`, `drums`, `bass`, `guitar`, `piano`, `other`. The derived mixes are exactly: `vocals`, `instrumental`, `residual`.
- The sample rate is 44100 Hz and all audio is stereo `(2, n)` float32. Stems are written as 32-bit float WAV so nothing clips.
- Sum-check pass: `|level_diff_db| <= 1.0` and `residual_db <= -15.0`.
- Model weights download into `models/` (already gitignored), or into `$SOUNDCODE_MODELS` if set. Output defaults to `out/stems/<audio stem>/` (gitignored). Private use only: never commit audio, stems or lyrics.
- `audio_separator` is imported lazily, only inside the real backend. The core package and the test suite must import without it.
- Encoder stages fail soft. A separation failure is recorded in `STEM_FAILURE` and the encode continues on the full mix.
- Commits have no `Co-Authored-By` or other Claude attribution lines.

## Review Focus

1. **Filenames with spaces or parentheses** (`The River-JordanFelix.mp3`, `Song (Live).mp3`). Stems must still be labelled correctly. The mix is copied to `_work/mix.wav` before pass 1, and labels come from the last `_(Label)` group (Task 2 and Task 3 tests).
2. **Mono or short input.** A mono file must produce stereo stems of the same length as the input. Model outputs that are a few samples longer or shorter than the input must be padded or trimmed (Task 2 and Task 3 tests).
3. **A silent file** must not crash the sum check or write `NaN`/`Infinity` into `manifest.json` (Task 2 and Task 3 tests).
4. **Re-running into an existing output folder** must not leave stale stems from an earlier run. Only files the tool owns are replaced (Task 3 test).
5. **Model missing or download failure** (offline, or `audio-separator` not installed). `soundcode separate` should print a one-line error and exit 2. `soundcode encode` should continue with a `STEM_FAILURE` note (Task 4 and Task 5 tests).

---

### Task 1: Dependencies and test config

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Pin librosa, add audio-separator, and scope pytest.** In `pyproject.toml`, replace the `encode` list and add a pytest section:

```toml
encode = [
    "demucs>=4.1.0",
    "audio-separator[cpu]>=0.47",
    "basic-pitch>=0.4",
    "torchcrepe>=0.0.23",
    "whisperx>=3.1",
    "beat-this",
    "librosa>=0.10,<1.0",
]
```

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Install into the project venv.** The venv has no pip, so use uv:

Run: `uv pip install --python .venv/bin/python "audio-separator[cpu]>=0.47"`
Expected: installs audio-separator 0.47.x. torch and numpy stay unchanged (check with `.venv/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"` before and after: 2.13.x and 2.4.x).

- [ ] **Step 3: Confirm a bare `pytest` now works.**

Run: `.venv/bin/python -m pytest -q`
Expected: `17 passed`. There should be no collection errors from `external/` or `scripts/`.

- [ ] **Step 4: Commit.**

```bash
git add pyproject.toml
git commit -m "build: add audio-separator, pin librosa<1.0, scope pytest to tests/"
```

---

### Task 2: Audio helpers, output labelling and sum check

**Files:**
- Create: `src/soundcode/separate.py`
- Test: `tests/test_separate.py`

**Interfaces:**
- Produces:
  - `SR: int = 44100`
  - `STEMS: tuple[str, ...]`
  - `read_stereo(path: Path | str, sr: int = SR, length: int | None = None) -> np.ndarray`, which returns shape `(2, n)` float32
  - `fit_length(y: np.ndarray, n: int) -> np.ndarray`
  - `write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None`
  - `rms_db(y: np.ndarray) -> float`, which returns `-inf` for silence
  - `label_outputs(paths: Iterable[Path]) -> dict[str, Path]`
  - `@dataclass(frozen=True) SumReport(mix_db, sum_db, level_diff_db, residual_db, ok)`
  - `sum_check(mix: np.ndarray, stems: Mapping[str, np.ndarray], max_level_diff_db: float = 1.0, max_residual_db: float = -15.0) -> SumReport`

- [ ] **Step 1: Write the failing tests** in `tests/test_separate.py`:

```python
"""Stem separation (Milestone 1). Models are never loaded here: a fake
backend stands in for audio-separator, so these tests run in milliseconds."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import separate as sep  # noqa: E402

SR = sep.SR


def tone(freq: float, secs: float = 1.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * secs)) / SR
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, x])


# --- audio helpers ------------------------------------------------------------

def test_read_stereo_turns_mono_into_two_channels(tmp_path):
    mono = tone(440)[0]
    sf.write(tmp_path / "m.wav", mono, SR)
    y = sep.read_stereo(tmp_path / "m.wav")
    assert y.shape == (2, SR)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y[0], y[1])


def test_fit_length_pads_and_trims():
    y = tone(440, 0.5)
    assert sep.fit_length(y, SR).shape == (2, SR)
    assert sep.fit_length(y, 100).shape == (2, 100)
    assert np.all(sep.fit_length(y, SR)[:, len(y[0]):] == 0)


def test_rms_db_of_silence_is_minus_inf():
    assert sep.rms_db(np.zeros((2, 10), np.float32)) == -math.inf
    # a full-scale square wave is 0 dBFS
    assert sep.rms_db(np.ones((2, 10), np.float32)) == pytest.approx(0.0)


# --- labelling audio-separator outputs --------------------------------------

def test_label_outputs_uses_the_last_parenthesised_group():
    paths = [
        Path("/w/mix_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755.wav"),
        Path("/w/Song (Live)_(Instrumental)_model_bs_roformer.wav"),
    ]
    labels = sep.label_outputs(paths)
    assert labels == {"vocals": paths[0], "instrumental": paths[1]}


def test_label_outputs_ignores_unlabelled_files():
    assert sep.label_outputs([Path("/w/readme.wav")]) == {}


# --- sum check ----------------------------------------------------------------

def test_sum_check_passes_when_stems_rebuild_the_mix():
    a, b = tone(220), tone(330)
    report = sep.sum_check(a + b, {"a": a, "b": b})
    assert report.ok
    assert report.level_diff_db == pytest.approx(0.0, abs=1e-6)
    assert report.residual_db == -math.inf


def test_sum_check_fails_when_a_stem_is_missing():
    a, b = tone(220), tone(330)
    report = sep.sum_check(a + b, {"a": a})
    assert not report.ok
    assert report.residual_db > -15.0


def test_sum_check_on_silence_does_not_crash():
    z = np.zeros((2, SR), np.float32)
    report = sep.sum_check(z, {"a": z})
    assert report.ok
    assert report.level_diff_db == 0.0
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/bin/python -m pytest tests/test_separate.py -q`
Expected: collection error `ImportError: cannot import name 'separate'`.

- [ ] **Step 3: Implement** `src/soundcode/separate.py`:

```python
"""Stem separation: split a mix into voice and instrument stems.

Three passes, each a pretrained model run through `audio-separator`:

    1. BS-RoFormer        mix          -> vocals, instrumental
    2. Karaoke RoFormer   vocals       -> lead_vocals, backing_vocals
    3. HTDemucs 6-stem    instrumental -> drums, bass, guitar, piano, other

Whatever the models leave behind (original minus the sum of the stems) is
written as `residual.wav`, so the stems plus the residual always rebuild the
mix exactly, and the level of the residual says how much separation lost.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import soundfile as sf

SR = 44100
STEMS = ("lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other")

_LABEL_RE = re.compile(r"_\(([^()]+)\)")


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------

def read_stereo(path: Path | str, sr: int = SR, length: int | None = None) -> np.ndarray:
    """Any audio file -> (2, n) float32 at `sr`; mono is duplicated."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    y = y[:2].astype(np.float32)
    return fit_length(y, length) if length is not None else y


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad or trim to exactly n samples; models are often off by a few."""
    if y.shape[1] >= n:
        return y[:, :n]
    return np.pad(y, ((0, 0), (0, n - y.shape[1])))


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    # float WAV: separated stems can exceed full scale and must not clip
    sf.write(str(path), y.T, sr, subtype="FLOAT")


def rms_db(y: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20 * math.log10(r) if r > 0 else -math.inf


def label_outputs(paths: Iterable[Path]) -> dict[str, Path]:
    """audio-separator names outputs `<input>_(<Stem>)_<model>.wav`; the stem
    label is the last parenthesised group, lowercased."""
    out: dict[str, Path] = {}
    for p in paths:
        groups = _LABEL_RE.findall(Path(p).name)
        if groups:
            out[groups[-1].strip().lower()] = Path(p)
    return out


# --------------------------------------------------------------------------
# sum check
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SumReport:
    mix_db: float          # RMS of the original, dBFS
    sum_db: float          # RMS of all stems summed, dBFS
    level_diff_db: float   # sum_db - mix_db
    residual_db: float     # RMS of (mix - sum) relative to the mix, dB
    ok: bool


def sum_check(mix: np.ndarray, stems: Mapping[str, np.ndarray],
              max_level_diff_db: float = 1.0,
              max_residual_db: float = -15.0) -> SumReport:
    total = np.zeros_like(mix)
    for y in stems.values():
        total += y
    mix_db, sum_db = rms_db(mix), rms_db(total)
    res_db = rms_db(mix - total)

    if mix_db == -math.inf:
        # a silent mix: fine only if the stems are silent too
        diff = 0.0 if sum_db == -math.inf else math.inf
        rel = -math.inf if res_db == -math.inf else math.inf
    else:
        diff = sum_db - mix_db
        rel = res_db - mix_db
    ok = abs(diff) <= max_level_diff_db and rel <= max_residual_db
    return SumReport(mix_db, sum_db, diff, rel, ok)
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/bin/python -m pytest tests/test_separate.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit.**

```bash
git add src/soundcode/separate.py tests/test_separate.py
git commit -m "separate: audio helpers, output labelling, stem sum check"
```

---

### Task 3: Three-pass pipeline with a pluggable backend

**Files:**
- Modify: `src/soundcode/separate.py` (append)
- Test: `tests/test_separate.py` (append)

**Interfaces:**
- Consumes: everything from Task 2.
- Produces:
  - `class SeparationError(RuntimeError)`
  - `class Backend(Protocol)`, with `run(self, model: str, audio: Path, out_dir: Path) -> dict[str, Path]` returning `{lowercase label: wav}`
  - `@dataclass(frozen=True) Pass(model: str, source: str, outputs: dict[str, str])`
  - `PASSES: tuple[Pass, ...]`
  - `@dataclass SeparationResult(source: Path, out_dir: Path, stems: dict[str, Path], mixes: dict[str, Path], levels: dict[str, float], report: SumReport, warnings: list[str])`
  - `separate(audio: Path | str, out_dir: Path | str, backend: Backend | None = None) -> SeparationResult`
  - Files written to `out_dir`:
    - `<stem>.wav` for each of `STEMS`
    - `vocals.wav`, `instrumental.wav`, `residual.wav`
    - `manifest.json`
    - the `_work/` folder

- [ ] **Step 1: Write the failing tests** by appending to `tests/test_separate.py`:

```python
# --- pipeline with a fake backend --------------------------------------------

import json  # noqa: E402


class FakeBackend:
    """Splits by fixed gains so the expected stems are known exactly.

    Each output is the input times a gain, and the gains of one pass sum to
    1.0, so the stems rebuild the mix. `drift` makes outputs 7 samples
    longer, as real models sometimes are."""

    GAINS = {
        "vocals": {"vocals": 0.4, "instrumental": 0.6},
        "karaoke": {"vocals": 0.75, "instrumental": 0.25},
        "demucs": {"drums": 0.3, "bass": 0.2, "guitar": 0.2, "piano": 0.1,
                   "other": 0.15, "vocals": 0.05},
    }

    def __init__(self, drift: int = 0, fail_on: str | None = None):
        self.drift, self.fail_on, self.calls = drift, fail_on, []

    def run(self, model, audio, out_dir):
        self.calls.append((model, Path(audio).name))
        if self.fail_on and self.fail_on in model:
            raise RuntimeError("model download failed")
        kind = ("karaoke" if "karaoke" in model else
                "demucs" if "demucs" in model else "vocals")
        y, _ = sf.read(str(audio), always_2d=True)
        y = y.T.astype(np.float32)
        if self.drift:
            y = np.pad(y, ((0, 0), (0, self.drift)))
        out = {}
        for label, g in self.GAINS[kind].items():
            p = Path(out_dir) / f"{Path(audio).stem}_({label.title()})_{model}.wav"
            sf.write(str(p), (y * g).T, SR, subtype="FLOAT")
            out[label] = p
        return out


def test_separate_writes_all_stems_mixes_and_manifest(tmp_path):
    src = tmp_path / "Song (Live).wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    res = sep.separate(src, tmp_path / "out", backend=FakeBackend())

    assert set(res.stems) == set(sep.STEMS)
    assert set(res.mixes) == {"vocals", "instrumental", "residual"}
    for p in [*res.stems.values(), *res.mixes.values()]:
        assert p.exists() and p.parent == tmp_path / "out"
    assert res.report.ok

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["stems"]["lead_vocals"] == "lead_vocals.wav"
    assert [m["model"] for m in manifest["passes"]] == [p.model for p in sep.PASSES]


def test_separate_routes_each_pass_to_the_right_input(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    fake = FakeBackend()
    sep.separate(src, tmp_path / "out", backend=fake)
    # pass 1 sees the copied mix, pass 2 the vocals, pass 3 the instrumental
    assert fake.calls[0][1] == "mix.wav"
    assert fake.calls[1][1] == "vocals.wav"
    assert fake.calls[2][1] == "instrumental.wav"


def test_separate_gains_are_correct_and_demucs_vocal_bleed_goes_to_other(tmp_path):
    src = tmp_path / "a.wav"
    mix = tone(220, 0.5)
    sf.write(str(src), mix.T, SR)
    res = sep.separate(src, tmp_path / "out", backend=FakeBackend())
    lead = sep.read_stereo(res.stems["lead_vocals"])
    other = sep.read_stereo(res.stems["other"])
    np.testing.assert_allclose(lead, mix * 0.4 * 0.75, atol=1e-5)
    np.testing.assert_allclose(other, mix * 0.6 * (0.15 + 0.05), atol=1e-5)


def test_separate_fixes_length_drift_and_mono_input(tmp_path):
    src = tmp_path / "mono.wav"
    sf.write(str(src), tone(220, 0.5)[0], SR)
    res = sep.separate(src, tmp_path / "out", backend=FakeBackend(drift=7))
    for p in res.stems.values():
        assert sep.read_stereo(p).shape == (2, SR // 2)


def test_separate_silence_writes_valid_json(tmp_path):
    src = tmp_path / "silent.wav"
    sf.write(str(src), np.zeros((SR // 4, 2), np.float32), SR)
    sep.separate(src, tmp_path / "out", backend=FakeBackend())
    text = (tmp_path / "out" / "manifest.json").read_text()
    assert "Infinity" not in text and "NaN" not in text
    json.loads(text)


def test_rerun_replaces_owned_files_and_keeps_others(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    (out / "drums.wav").write_bytes(b"stale")
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    sep.separate(src, out, backend=FakeBackend())
    assert (out / "notes.txt").read_text() == "mine"
    assert sep.read_stereo(out / "drums.wav").shape == (2, SR // 2)


def test_backend_failure_raises_separation_error_naming_the_model(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    with pytest.raises(sep.SeparationError, match="karaoke"):
        sep.separate(src, tmp_path / "out", backend=FakeBackend(fail_on="karaoke"))
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/bin/python -m pytest tests/test_separate.py -q`
Expected: 7 new failures with `AttributeError: module 'soundcode.separate' has no attribute 'separate'`.

- [ ] **Step 3: Implement** by appending to `src/soundcode/separate.py`. Also add `import json`, `import shutil` and `from typing import Protocol` next to the existing imports at the top.

```python
# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

class SeparationError(RuntimeError):
    """A separation pass failed; the message names the model."""


class Backend(Protocol):
    def run(self, model: str, audio: Path, out_dir: Path) -> dict[str, Path]:
        """Run one model on `audio`; return {lowercase stem label: wav path}."""


@dataclass(frozen=True)
class Pass:
    model: str
    source: str                 # "mix", or an output name of an earlier pass
    outputs: dict[str, str]     # model label -> our name; repeats are summed


PASSES = (
    Pass("model_bs_roformer_ep_317_sdr_12.9755.ckpt", "mix",
         {"vocals": "vocals", "instrumental": "instrumental"}),
    Pass("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", "vocals",
         {"vocals": "lead_vocals", "instrumental": "backing_vocals"}),
    # Run on the instrumental, so its "vocals" output is only bleed from
    # pass 1; folding it into `other` keeps the stems summing to the mix.
    Pass("htdemucs_6s.yaml", "instrumental",
         {"drums": "drums", "bass": "bass", "guitar": "guitar",
          "piano": "piano", "other": "other", "vocals": "other"}),
)

_VOCAL_STEMS = ("lead_vocals", "backing_vocals")
_OWNED = (*STEMS, "vocals", "instrumental", "residual")


@dataclass
class SeparationResult:
    source: Path
    out_dir: Path
    stems: dict[str, Path]
    mixes: dict[str, Path]
    levels: dict[str, float]    # RMS dBFS per stem and mix
    report: SumReport
    warnings: list[str]


def separate(audio: Path | str, out_dir: Path | str,
             backend: Backend | None = None) -> SeparationResult:
    """Split `audio` into STEMS under `out_dir`. See the module docstring."""
    src, out = Path(audio), Path(out_dir)
    backend = backend or AudioSeparatorBackend()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    for name in _OWNED:
        (out / f"{name}.wav").unlink(missing_ok=True)

    mix = read_stereo(src)
    n = mix.shape[1]
    # models see a plain-named copy of exactly the audio we sum-check against
    inputs: dict[str, Path] = {"mix": work / "mix.wav"}
    write_wav(inputs["mix"], mix)

    audio_by_name: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    for i, p in enumerate(PASSES, 1):
        pass_dir = work / f"pass{i}"
        pass_dir.mkdir()
        try:
            produced = backend.run(p.model, inputs[p.source], pass_dir)
        except Exception as exc:  # any model/backend failure
            raise SeparationError(f"{p.model}: {exc}") from exc
        for label, name in p.outputs.items():
            if label not in produced:
                warnings.append(f"{p.model} produced no '{label}' output")
                continue
            y = read_stereo(produced[label], length=n)
            audio_by_name[name] = audio_by_name.get(name, 0) + y
            # intermediate results become inputs for later passes
            if name not in STEMS:
                inputs[name] = work / f"{name}.wav"
                write_wav(inputs[name], audio_by_name[name])

    stems_audio = {s: audio_by_name.get(s, np.zeros_like(mix)) for s in STEMS}
    for s in STEMS:
        if s not in audio_by_name:
            warnings.append(f"no audio for stem '{s}'; wrote silence")

    vocals = stems_audio["lead_vocals"] + stems_audio["backing_vocals"]
    instrumental = sum(y for s, y in stems_audio.items() if s not in _VOCAL_STEMS)
    mixes_audio = {"vocals": vocals, "instrumental": instrumental,
                   "residual": mix - vocals - instrumental}

    stems, mixes = {}, {}
    for s, y in stems_audio.items():
        stems[s] = out / f"{s}.wav"
        write_wav(stems[s], y)
    for m, y in mixes_audio.items():
        mixes[m] = out / f"{m}.wav"
        write_wav(mixes[m], y)

    levels = {k: rms_db(y) for k, y in {**stems_audio, **mixes_audio}.items()}
    report = sum_check(mix, stems_audio)
    result = SeparationResult(src, out, stems, mixes, levels, report, warnings)
    _write_manifest(result, n)
    return result


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None      # JSON has no Infinity


def _write_manifest(r: SeparationResult, n: int) -> None:
    manifest = {
        "source": str(r.source),
        "sr": SR,
        "duration": n / SR,
        "passes": [{"model": p.model, "source": p.source} for p in PASSES],
        "stems": {s: p.name for s, p in r.stems.items()},
        "mixes": {m: p.name for m, p in r.mixes.items()},
        "levels_db": {k: _finite(v) for k, v in r.levels.items()},
        "sum_check": {k: (_finite(v) if isinstance(v, float) else v)
                      for k, v in r.report.__dict__.items()},
        "warnings": r.warnings,
    }
    (r.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class AudioSeparatorBackend:
    """Placeholder until Task 4; importing audio_separator happens there."""

    def run(self, model: str, audio: Path, out_dir: Path) -> dict[str, Path]:
        raise SeparationError("audio-separator backend not implemented yet")
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `32 passed` (17 core + 15 separate).

- [ ] **Step 5: Commit.**

```bash
git add src/soundcode/separate.py tests/test_separate.py
git commit -m "separate: three-pass pipeline, derived mixes, residual, manifest"
```

---

### Task 4: Real audio-separator backend and the `soundcode separate` command

**Files:**
- Modify: `src/soundcode/separate.py` (replace the `AudioSeparatorBackend` placeholder)
- Modify: `src/soundcode/cli.py`
- Test: `tests/test_separate.py` (append)

**Interfaces:**
- Consumes: `separate`, `SeparationError`, `SeparationResult`, `label_outputs` from Tasks 2 and 3.
- Produces:
  - `AudioSeparatorBackend(model_dir: Path | None = None)`
  - `default_out_dir(audio: Path | str) -> Path`, which returns `out/stems/<audio stem>`
  - the `soundcode separate <file> [-o DIR]` CLI command, exiting 0 if the sum check passes, 1 if it fails, and 2 on error

- [ ] **Step 1: Write the failing tests** by appending to `tests/test_separate.py`:

```python
# --- CLI -----------------------------------------------------------------------

from soundcode import cli  # noqa: E402


def test_default_out_dir_is_under_out_stems():
    assert sep.default_out_dir("audio/test/Song (Live).mp3") == Path("out/stems/Song (Live)")


def test_cli_separate_prints_levels_and_exits_zero(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    monkeypatch.setattr(sep, "AudioSeparatorBackend", FakeBackend)
    code = cli.main(["separate", str(src), "-o", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0
    assert "lead_vocals" in out and "sum check: OK" in out


def test_cli_separate_reports_backend_failure_and_exits_two(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    monkeypatch.setattr(sep, "AudioSeparatorBackend",
                        lambda: FakeBackend(fail_on="roformer_ep_317"))
    code = cli.main(["separate", str(src), "-o", str(tmp_path / "out")])
    assert code == 2
    assert "separation failed" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/bin/python -m pytest tests/test_separate.py -q`
Expected: 3 failures: `default_out_dir` missing, and `invalid choice: 'separate'`.

- [ ] **Step 3: Replace the placeholder backend** in `separate.py` with the real one. Add `import os` at the top.

```python
def default_out_dir(audio: Path | str) -> Path:
    return Path("out") / "stems" / Path(audio).stem


class AudioSeparatorBackend:
    """Runs one model through the `audio-separator` package.

    Weights download on first use into `models/` (or $SOUNDCODE_MODELS).
    PyTorch models use MPS on Apple Silicon; ops MPS lacks fall back to CPU.
    """

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir or os.environ.get("SOUNDCODE_MODELS", "models"))

    def run(self, model: str, audio: Path, out_dir: Path) -> dict[str, Path]:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            from audio_separator.separator import Separator
        except ImportError as exc:
            raise SeparationError(
                "audio-separator is not installed "
                "(uv pip install --python .venv/bin/python 'audio-separator[cpu]')"
            ) from exc
        self.model_dir.mkdir(parents=True, exist_ok=True)
        separator = Separator(output_dir=str(out_dir),
                              model_file_dir=str(self.model_dir),
                              output_format="WAV")
        separator.load_model(model_filename=model)
        files = separator.separate(str(audio))
        paths = [Path(f) if Path(f).is_absolute() else Path(out_dir) / Path(f).name
                 for f in files]
        return label_outputs(paths)
```

`separate()` keeps `backend = backend or AudioSeparatorBackend()`. The name is looked up in module globals at call time, so tests that monkeypatch `sep.AudioSeparatorBackend` reach it.

- [ ] **Step 4: Add the CLI command** in `cli.py`. Put the parser next to `p_encode`:

```python
    p_sep = sub.add_parser("separate", help="split audio into vocal and instrument stems")
    p_sep.add_argument("file")
    p_sep.add_argument("-o", "--out", default=None,
                       help="output folder (default out/stems/<name>)")
```

Put the handler before `if args.cmd == "encode":`:

```python
        if args.cmd == "separate":
            from . import separate as sep

            out = args.out or str(sep.default_out_dir(args.file))
            try:
                res = sep.separate(args.file, out, backend=sep.AudioSeparatorBackend())
            except sep.SeparationError as exc:
                print(f"separation failed: {exc}", file=sys.stderr)
                return 2
            for name in (*sep.STEMS, "vocals", "instrumental", "residual"):
                db = res.levels[name]
                print(f"  {name:<15} {db:7.1f} dBFS" if db > -200 else f"  {name:<15}  silent")
            r = res.report
            verdict = "OK" if r.ok else "FAILED"
            print(f"sum check: {verdict}  (level diff {r.level_diff_db:+.2f} dB, "
                  f"residual {r.residual_db:.1f} dB)  -> {out}")
            for w in res.warnings:
                print(f"  warn: {w}")
            return 0 if r.ok else 1
```

Update the module docstring's command list to include `soundcode separate <audio> [-o dir]`.

- [ ] **Step 5: Run the tests and confirm they pass.**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `35 passed`.

- [ ] **Step 6: Real run on a 30-second clip** (downloads about 1 GB of weights on the first run):

Run: `.venv/bin/soundcode separate audio/test/999999-30s.mp3`
Expected:
- ten level lines, then `sum check: OK`, with a level diff within ±1 dB and a residual ≤ −15 dB
- files in `out/stems/999999-30s/`
- the terminal log shows `mps` in use for the RoFormer passes

If the karaoke pass warns that it produced no `vocals`/`instrumental`, run `ls out/stems/999999-30s/_work/pass2` and fix the labels in `PASSES[1].outputs` to match the real filenames. Rerun the tests after any fix.

- [ ] **Step 7: Commit.**

```bash
git add src/soundcode/separate.py src/soundcode/cli.py tests/test_separate.py
git commit -m "separate: audio-separator backend and 'soundcode separate' command"
```

---

### Task 5: Encoder uses the new stems (and transcribes guitar and piano)

**Files:**
- Modify: `src/soundcode/encode.py:67-96` (`separate_stems`) and `:563-576`, `:610-611` (stage list)
- Test: `tests/test_separate.py` (append)

**Interfaces:**
- Consumes: `separate`, `SeparationError`, `SeparationResult`.
- Produces:
  - `encoder_stems(result: SeparationResult) -> dict[str, Path]`, which maps the old keys `vocals` → lead vocals and keeps `drums`, `bass`, `guitar`, `piano`, `other`
  - `separate_stems(path, workdir)`, with the same signature and contract as before

- [ ] **Step 1: Write the failing tests** by appending:

```python
# --- encoder integration -------------------------------------------------------

from soundcode import encode as enc  # noqa: E402


def test_encoder_stems_maps_lead_vocals_to_vocals(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    res = sep.separate(src, tmp_path / "out", backend=FakeBackend())
    stems = enc.encoder_stems(res)
    assert stems["vocals"] == res.stems["lead_vocals"]
    assert {"drums", "bass", "guitar", "piano", "other"} <= set(stems)


def test_separate_stems_records_failure_and_returns_empty(tmp_path, monkeypatch):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    monkeypatch.setattr(sep, "AudioSeparatorBackend", lambda: FakeBackend(fail_on="htdemucs"))
    enc.STEM_FAILURE.clear()
    assert enc.separate_stems(str(src), tmp_path / "wd") == {}
    assert "htdemucs" in enc.STEM_FAILURE[0]
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/bin/python -m pytest tests/test_separate.py -q -k "encoder_stems or separate_stems"`
Expected: `AttributeError: module 'soundcode.encode' has no attribute 'encoder_stems'`.

- [ ] **Step 3: Replace `separate_stems`** in `encode.py` (lines 67–91). Keep `_module_available`, which other code may use.

```python
def encoder_stems(result) -> dict[str, Path]:
    """Separation stems under the names the stages below expect. The lead
    vocal drives melody and lyrics; backing vocals are left for Milestone 2."""
    stems = {k: v for k, v in result.stems.items()
             if k in ("drums", "bass", "guitar", "piano", "other")}
    stems["vocals"] = result.stems["lead_vocals"]
    return stems


def separate_stems(path: str, workdir: Path) -> dict[str, Path]:
    """Seven-stem separation (see separate.py); fails soft into STEM_FAILURE."""
    from . import separate as sep

    _log("separating stems (RoFormer vocals, karaoke lead/backing, demucs 6-stem)")
    try:
        result = sep.separate(path, workdir / "stems")
    except sep.SeparationError as exc:
        _log(f"separation FAILED: {exc}")
        STEM_FAILURE.append(str(exc))
        return {}
    if not result.report.ok:
        _log(f"stem sum check failed: residual {result.report.residual_db:.1f} dB")
    return encoder_stems(result)
```

- [ ] **Step 4: Transcribe guitar and piano.** After the `other_st = ...` line (around `encode.py:574`), add:

```python
    _log("guitar/piano notes")
    guitar_st = stage_notes_poly(stems.get("guitar"), "guitar", grid)
    piano_st = stage_notes_poly(stems.get("piano"), "piano", grid)
```

Then add both to the output loop (around `encode.py:610`):

```python
    for st in (grid_st, struct_st, harm_st, perc_st, bass_st, vox_st,
               guitar_st, piano_st, other_st, text_st, mix_st):
```

- [ ] **Step 5: Run all tests and confirm they pass.**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `37 passed`.

- [ ] **Step 6: Real encode.**

Run: `.venv/bin/soundcode encode audio/test/999999-30s.mp3 -o out/sc/999999-30s.sc --workdir out/work`
Expected:
- the log shows the three passes and then the stages
- `.venv/bin/soundcode check out/sc/999999-30s.sc` lists `:notes.guitar` and/or `:notes.piano`, or has `# omitted` lines for them
- there is no stem-failure WARNING

- [ ] **Step 7: Commit.**

```bash
git add src/soundcode/encode.py tests/test_separate.py
git commit -m "encode: use seven-stem separation; transcribe guitar and piano stems"
```

---

### Task 6: Stems in the listening server

**Files:**
- Modify: `src/soundcode/server.py:35-68` (`_discover_tracks`)
- Test: `tests/test_separate.py` (append)

**Interfaces:**
- Consumes: the `out/stems/<song>/*.wav` layout from Task 4.
- Produces: `/api/tracks` entries with `kind == "stem"` for stem files. Files under `_work/` are hidden.

- [ ] **Step 1: Write the failing test** by appending:

```python
# --- listening server ------------------------------------------------------------

from soundcode import server  # noqa: E402


def test_server_lists_stems_and_hides_work_files(tmp_path, monkeypatch):
    d = tmp_path / "out" / "stems" / "song"
    (d / "_work" / "pass1").mkdir(parents=True)
    for name in ("lead_vocals.wav", "instrumental.wav", "_work/mix.wav",
                 "_work/pass1/x_(Vocals)_m.wav"):
        (d / name).write_bytes(b"RIFF")
    monkeypatch.setenv("SOUNDCODE_ROOT", str(tmp_path))
    tracks = server._discover_tracks()
    labels = {t["label"]: t["kind"] for t in tracks}
    assert labels == {"stems/song/lead_vocals": "stem",
                      "stems/song/instrumental": "stem"}
```

- [ ] **Step 2: Run the test and confirm it fails.**

Run: `.venv/bin/python -m pytest tests/test_separate.py -q -k server`
Expected: FAIL. The kinds come out as `other`, and the `_work` files are listed.

- [ ] **Step 3: Implement** in `_discover_tracks`. Right after the suffix check, skip work files:

```python
        if "_work" in path.relative_to(out).parts:
            continue              # separation intermediates, not for listening
```

Then add a `stem` branch before `elif "mock" in label:`:

```python
        elif label.startswith("stems/"):
            kind = "stem"
```

Add it to the sort order as well:

```python
    order = {"ref": 0, "mock": 1, "cover": 2, "stem": 3, "other": 4}
```

- [ ] **Step 4: Run all tests and confirm they pass.**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `38 passed`.

- [ ] **Step 5: Listen.**

Run: `.venv/bin/soundcode serve`, open http://127.0.0.1:8720, and put `ref/999999-30s` in A and `stems/999999-30s/instrumental` in B.
Expected:
- the stems appear with the tag `stem`
- the A/B switch between the original and the instrumental stays in sync
- the vocals-only and instrumental mixes sound clean

- [ ] **Step 6: Commit.**

```bash
git add src/soundcode/server.py tests/test_separate.py
git commit -m "serve: list separated stems, hide separation work files"
```

---

### Task 7: Milestone 1 acceptance run

**Files:**
- Modify: `README.md` (Status and Usage sections)

- [ ] **Step 1: Separate all four 30-second test clips** and record the results:

```bash
for f in audio/test/*.mp3; do .venv/bin/soundcode separate "$f"; done
```

Expected: all four print `sum check: OK`. Note the residual dB for each.

- [ ] **Step 2: Separate one full song with an awkward filename.**

```bash
.venv/bin/soundcode separate "audio/uploads/The River-JordanFelix.mp3"
```

Expected: `sum check: OK`, all stems 196 s long, and the time taken noted.

- [ ] **Step 3: Listen** in the server to the lead vocals, backing vocals, instrumental and drums of two songs.

- [ ] **Step 4: Update the README.**
  - Add the `separate` command to Usage with a one-line description.
  - Change the Status line to say Milestone 1 (separation) is done.
  - Include a small table with the four clips' residual dB values and the full-song run time.

- [ ] **Step 5: Run the full test suite, then commit.**

```bash
.venv/bin/python -m pytest -q
git add README.md
git commit -m "docs: Milestone 1 separation results"
```
