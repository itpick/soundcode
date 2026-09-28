# Production Matching + Song's Own Drum Kit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- The encoder measures each stem's production (EQ curve, reverb, width, pan, crest) and writes it as an `fx` line on every note/perc stream.
- It also cuts a drum kit from the song's own drum stem.
- The renderer applies both.
- `compare` gains `spectral_db`, which says whether a part *sounds* like the original.

**Architecture:**
- `src/soundcode/fx.py` holds pure measurement and application functions: band spectra, rt60/wet, width/pan, crest, EQ matching, plus pedalboard reverb/compressor.
- `src/soundcode/kit.py` selects isolated hits, cuts samples and plays them round-robin.
- In the encoder, a post-pass `attach_fx` adds `fx` lines using each stage's stem, and the drums stage builds the kit. `encode()` copies the kit next to the `.sc` as `<name>.kit/` and writes `meta kit=<name>.kit`.
- `render_sf.render_streams` applies `fx` (unless `no_fx`) and plays kit drums.
- `parse_file` records the document's path, so relative kit paths resolve.

**Tech Stack:** Python 3.11, numpy, librosa (STFT), pedalboard 0.9.25 (new; Reverb, Compressor), soundfile.

**Spec:** `docs/superpowers/specs/2026-09-27-production-match-design.md`.

## Global Constraints

- Python `>=3.11,<3.12`. Install with `uv pip install --python .venv/bin/python pedalboard` and add it to the `encode` extra in `pyproject.toml`.
- **A render of a `.sc` with no `fx` lines and no `meta kit` must be sample-identical to today's render.** Old files must not change.
- fx and the kit must not change note timing: onset F1 and note F1 of re-rendered stems stay within ±0.02 of the `--no-fx`, no-kit render.
- EQ gains are clamped to ±15 dB. Reverb `wet` is 0–1 and `rt60` 0.1–4 s. Compression ratio is 1.5–6:1 and only engages when render crest exceeds target by more than 3 dB.
- Kit samples are 44.1 kHz float WAV. The kit folder sits next to the `.sc` (`<stem>.kit/`) under `out/` (gitignored).
- Open, free tools only. No `Co-Authored-By` in commits.

## Review Focus

1. **Silent or very short stems** (one note, a 0.5 s part). The fx measurement returns defaults, not NaN, and the fx line is still parseable. (Task 1 test.)
2. **A drum voice with no isolated hit, or a kit folder deleted after encoding.** Those hits fall back to the GM kit, with `meta warn` at render time, and never crash. (Task 4 test.)
3. **Hand-written `fx` lines with missing fields, or garbage values.** `parse_fx` ignores unknown or invalid fields and applies only what parses. (Task 1 test.)
4. **Stereo vs mono stems and sample-rate mismatches** (a kit cut at 44.1 kHz, a render at 22.05 kHz). Resample on load. (Task 4 test.)
5. **An EQ that would boost noise:** bands the rendered part has no energy in must not get +15 dB. The gain is limited where the render is more than 40 dB below its own peak band. (Task 2 test.)

---

### Task 1: fx measurement and the `fx` line

**Files:**
- Create: `src/soundcode/fx.py`
- Test: `tests/test_fx.py`

**Interfaces:**
- Produces:
  - `BANDS = 20.0 * 2 ** (np.arange(31) / 3)`, the 31 third-octave centres from 20 Hz to about 20 kHz
  - `band_db(y: np.ndarray, sr: int) -> np.ndarray`, 31 values in dB relative to the mean of bands within 60 dB of the peak band. Empty bands are `-60`, and silence gives all zeros.
  - `@dataclass Fx(eq: list[int], rt60: float = 0.3, wet: float = 0.1, width: float = 0.0, pan: float = 0.0, crest: float = 12.0)`
  - `measure(stereo: np.ndarray, sr: int, offsets: list[float]) -> Fx`. `stereo` has shape `(2, n)`; `offsets` are note end times in seconds, used for rt60 and wet.
  - `fx_line(f: Fx) -> str`, giving `"fx      eq=… rt60=0.62s wet=0.18 width=0.35 pan=-0.10 crest=14.2dB"`
  - `parse_fx(stream) -> Fx | None`, which reads the stream's `fx` statement and tolerates missing or bad fields

- [ ] **Step 1: Install pedalboard**

Run: `uv pip install --python .venv/bin/python pedalboard`
Expected: `+ pedalboard==0.9.25`. Also add `"pedalboard>=0.9"` to the `encode` list in `pyproject.toml`.

- [ ] **Step 2: Write the failing tests** in `tests/test_fx.py`:

```python
"""Production matching (spec 2026-09-27-production-match)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import fx  # noqa: E402
from soundcode.parser import parse  # noqa: E402

SR = 44100


def noise(secs=3.0, seed=0):
    return np.random.default_rng(seed).standard_normal(int(secs * SR)).astype(np.float32) * 0.1


def tilt(y, db_per_oct):
    """Apply a spectral tilt of db_per_oct around 1 kHz."""
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    g = 10 ** (db_per_oct * np.log2(np.maximum(f, 20) / 1000) / 20)
    return np.fft.irfft(Y * g, len(y)).astype(np.float32)


def test_band_db_sees_a_tilt():
    flat, dark = fx.band_db(noise(), SR), fx.band_db(tilt(noise(), -6), SR)
    assert len(flat) == 31
    hi = fx.BANDS > 4000
    assert dark[hi].mean() < flat[hi].mean() - 10


def test_band_db_of_silence_is_zeros():
    assert np.allclose(fx.band_db(np.zeros(SR, np.float32), SR), 0)


def test_rt60_of_an_exponential_decay():
    t = np.arange(int(2.0 * SR)) / SR
    y = np.zeros_like(t, dtype=np.float32)
    burst = noise(0.3, 1)
    y[:burst.size] = burst
    tail = noise(1.7, 2) * np.exp(-6.91 * (t[burst.size:] - 0.3) / 0.8)[: int(1.7 * SR)]  # rt60 0.8 s
    y[burst.size:burst.size + tail.size] = tail * (np.abs(burst[-2000:]).mean() / 0.08)
    f = fx.measure(np.stack([y, y]), SR, offsets=[0.3])
    assert 0.5 < f.rt60 < 1.2
    assert f.wet > 0.2


def test_width_and_pan():
    l, r = noise(seed=1), noise(seed=2)
    wide = fx.measure(np.stack([l, r]), SR, [])
    mono_left = fx.measure(np.stack([l, l * 0.25]), SR, [])
    assert wide.width > 0.8 and mono_left.width < 0.2
    assert mono_left.pan < -0.5


def test_measure_on_silence_or_a_blip_gives_defaults_not_nan():
    f = fx.measure(np.zeros((2, SR), np.float32), SR, [0.5])
    assert all(np.isfinite([f.rt60, f.wet, f.width, f.pan, f.crest]))
    assert fx.parse_fx(parse("%sc 0.3\n\n:notes.x\n" + fx.fx_line(f) + "\n").stream("notes.x"))


def test_fx_line_round_trips_and_tolerates_garbage():
    f = fx.Fx(eq=list(range(-15, 16)), rt60=0.62, wet=0.18, width=0.35, pan=-0.1, crest=14.2)
    doc = parse("%sc 0.3\n\n:notes.piano\n" + fx.fx_line(f) + "\n")
    g = fx.parse_fx(doc.stream("notes.piano"))
    assert g.eq == f.eq and g.rt60 == pytest.approx(0.62) and g.pan == pytest.approx(-0.1)
    bad = parse("%sc 0.3\n\n:notes.p\nfx  rt60=banana  wet=0.3  eq=1,2\n").stream("notes.p")
    h = fx.parse_fx(bad)
    assert h.wet == pytest.approx(0.3) and h.rt60 == fx.Fx(eq=[]).rt60 and h.eq == []
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_fx.py -q`
Expected: collection error `cannot import name 'fx'`.

- [ ] **Step 4: Implement** `src/soundcode/fx.py`, measurement half:

```python
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
    mid, side = (l + r) / 2, (l - r) / 2
    em, es = np.mean(mid ** 2), np.mean(side ** 2)
    f.width = float(np.clip(es / (em + es + 1e-20) * 2, 0, 1))
    el, er = np.mean(l ** 2), np.mean(r ** 2)
    f.pan = float(np.clip((er - el) / (er + el + 1e-20), -1, 1))
    f.crest = float(20 * np.log10(np.abs(mono).max() / (np.sqrt(np.mean(mono.astype(np.float64) ** 2)) + 1e-20)))
    # room: energy decay after isolated note ends
    hop = int(0.01 * sr)
    env = np.array([_rms_db(mono[i:i + hop]) for i in range(0, len(mono) - hop, hop)])
    slopes, wets = [], []
    for t in offsets:
        k = int(t / 0.01)
        body, tail = env[max(0, k - 10):k], env[k + 5:k + 40]
        if len(body) < 5 or len(tail) < 20:
            continue
        x = np.arange(len(tail)) * 0.01
        slope = np.polyfit(x, tail, 1)[0]                      # dB per second
        if slope < -1:
            slopes.append(slope)
        wets.append(np.mean(tail[:15]) - np.mean(body))
    if slopes:
        f.rt60 = float(np.clip(-60.0 / np.median(slopes), 0.1, 4.0))
    if wets:
        f.wet = float(np.clip((np.median(wets) + 30) / 30, 0, 1))
    return f


def fx_line(f: Fx) -> str:
    eq = ",".join(str(int(v)) for v in f.eq)
    return (f"fx      eq={eq}  rt60={f.rt60:.2f}s  wet={f.wet:.2f}  width={f.width:.2f}  "
            f"pan={f.pan:.2f}  crest={f.crest:.1f}dB")


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
                    vals = [int(x) for x in v.split(",") if x.strip()]
                    f.eq = vals if len(vals) == len(BANDS) else []
                elif k in ("rt60", "crest"):
                    setattr(f, k, float(v.rstrip("sdB")))
                elif k in ("wet", "width", "pan"):
                    setattr(f, k, float(v))
            except ValueError:
                continue
        return f
    return None
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_fx.py -q`
Expected: `6 passed`. If `test_rt60…` is off, check the tail fit window (50–400 ms after the offset) before loosening the bounds, and record a Ruling if it changes.

- [ ] **Step 6: Commit**

```bash
git add src/soundcode/fx.py tests/test_fx.py pyproject.toml
git commit -m "fx: measure a stem's production (EQ curve, rt60, wet, width, pan, crest); fx line"
```

---

### Task 2: Apply fx in the renderer

**Files:**
- Modify: `src/soundcode/fx.py` (apply half), `src/soundcode/render_sf.py`, `src/soundcode/cli.py`
- Test: `tests/test_fx.py` (append)

**Interfaces:**
- Produces:
  - `fx.apply(stereo: np.ndarray, sr: int, f: Fx) -> np.ndarray`, taking and returning `(n, 2)` float32
  - `fx.eq_match(stereo, sr, target_eq) -> stereo`
  - `render_sf.render_streams(doc, sr=None, sf2=None, no_fx=False)`, which applies `parse_fx(stream)` after synthesis and before level matching
  - `render(…, no_fx=False)`, `render_to_file(…, no_fx=False)`, and the CLI `render --no-fx`. In `mix`, a stream with fx skips the family `_PAN`, because fx already placed it.

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- applying fx -------------------------------------------------------------------------------

def test_eq_match_moves_the_spectrum_toward_the_target():
    src = np.stack([noise(seed=3)] * 2, 1)
    target = [int(round(v)) for v in fx.band_db(tilt(noise(seed=4), -6), SR)]
    before = np.abs(fx.band_db(src.T, SR) - target)[fx.BANDS < 10000].mean()
    out = fx.eq_match(src, SR, target)
    after = np.abs(fx.band_db(out.T, SR) - target)[fx.BANDS < 10000].mean()
    assert after < before * 0.5


def test_eq_never_boosts_empty_bands():
    t = np.arange(SR * 2) / SR
    sine = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32)
    out = fx.eq_match(sine, SR, [0] * 31)                   # flat target vs a single sine
    assert np.abs(out).max() < 4 * np.abs(sine).max()


def test_reverb_and_width_change_the_signal_but_not_its_start():
    t = np.arange(SR) / SR
    y = np.zeros((SR * 2, 2), np.float32)
    y[:SR, 0] = y[:SR, 1] = 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 0.2)
    out = fx.apply(y, SR, fx.Fx(eq=[], rt60=1.5, wet=0.6, width=0.8, pan=0.0))
    assert np.abs(out[int(0.4 * SR):int(0.8 * SR)]).mean() > np.abs(y[int(0.4 * SR):int(0.8 * SR)]).mean() + 1e-3
    onset = lambda a: int(np.argmax(np.abs(a[:, 0]) > 1e-3))  # noqa: E731
    assert abs(onset(out) - onset(y)) < int(0.002 * SR)


def test_render_without_fx_lines_is_unchanged():
    from soundcode import render_sf
    sf2 = Path(__file__).resolve().parents[1] / "models" / "soundfonts" / "GeneralUser-GS.sf2"
    if not sf2.exists():
        pytest.skip("no SoundFont")
    doc = parse("%sc 0.3\n@duration 2.0\n\n:notes.keys inst=keys.piano\n@0.0 C4 0.5s 100\n")
    a = render_sf.render_streams(doc, sf2=sf2)
    b = render_sf.render_streams(doc, sf2=sf2, no_fx=True)
    np.testing.assert_array_equal(a["notes.keys"], b["notes.keys"])
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_fx.py -q`
Expected: 4 failures (`eq_match`, `apply` and `no_fx` are missing).

- [ ] **Step 3: Implement.** Append to `fx.py`:

```python
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
    boards = []
    have_crest = 20 * np.log10(np.abs(y).max() / (np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-20) + 1e-20)
    if have_crest - f.crest > 3:
        ratio = float(np.clip(1 + (have_crest - f.crest) / 6, 1.5, 6.0))
        boards.append(pb.Compressor(threshold_db=-24, ratio=ratio, attack_ms=10, release_ms=120))
    if f.wet > 0.02:
        boards.append(pb.Reverb(room_size=_room_size(f.rt60), wet_level=f.wet,
                                dry_level=1 - f.wet / 2, width=1.0))
    if boards:
        y = pb.Pedalboard(boards)(y.T.astype(np.float32), sr).T
    mid, side = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
    side_now = np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12)
    want = f.width / max(2 - f.width, 1e-3)                        # inverse of measure()'s map
    side = side * (np.clip(want / side_now, 0, 4) if side_now > 1e-4 else 0.0)
    if side_now <= 1e-4 and f.width > 0.05:                        # mono render: decorrelate
        side = np.roll(mid, int(0.011 * sr)) * np.sqrt(want)
    l, r = mid + side, mid - side
    theta = (f.pan + 1) * np.pi / 4
    out = np.stack([l * np.cos(theta) * np.sqrt(2), r * np.sin(theta) * np.sqrt(2)], 1)
    return out.astype(np.float32)
```

`have` is relative to its own mean, so it is directly comparable to the target curve.

In `render_sf.render_streams`, add the `no_fx: bool = False` parameter. After `y = _synth_stream(...)` and before the level block, add:

```python
        if not no_fx:
            from .fx import apply as apply_fx, parse_fx
            f = parse_fx(s)
            if f is not None:
                y = apply_fx(y, sr, f)
```

(`s = doc.stream(name)` must be fetched before this block; move that line up.) In `mix`, when `parse_fx(s)` is not None, set `pan = 0.0`. Thread `no_fx` through `render`, `render_to_file` and the CLI (`p_render.add_argument("--no-fx", action="store_true")`).

- [ ] **Step 4: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (160 + 10 = 170).

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/fx.py src/soundcode/render_sf.py src/soundcode/cli.py tests/test_fx.py
git commit -m "render: apply each stream's fx (EQ match, reverb, compression, width/pan); --no-fx"
```

---

### Task 3: `spectral_db` in compare

**Files:**
- Modify: `src/soundcode/compare.py`, `src/soundcode/cli.py`
- Test: `tests/test_compare.py` (append)

**Interfaces:**
- Produces:
  - `compare.spectral_db(y_ref, y_est, sr) -> float | None`: the mean |Δ| over bands where the reference is within 50 dB of its peak band; `None` if either side is silent
  - `report["stems"][s]["spectral_db"]`, a new `spec dB` column in the CLI table and the HTML table

- [ ] **Step 1: Write the failing tests** (append to `tests/test_compare.py`):

```python
# --- spectral match ----------------------------------------------------------------------------

def test_spectral_db_is_zero_for_the_same_sound_and_grows_with_a_tilt():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(SR * 3) * 0.1).astype(np.float32)
    assert cmp.spectral_db(a, a, SR) < 0.5
    Y = np.fft.rfft(a); f = np.fft.rfftfreq(len(a), 1 / SR)
    dark = np.fft.irfft(Y * 10 ** (-6 * np.log2(np.maximum(f, 20) / 1000) / 20), len(a)).astype(np.float32)
    assert cmp.spectral_db(a, dark, SR) > 6
    assert cmp.spectral_db(a, np.zeros_like(a), SR) is None
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q -k spectral`
Expected: `AttributeError: … 'spectral_db'`.

- [ ] **Step 3: Implement** in `compare.py`:

```python
def spectral_db(y_ref, y_est, sr) -> float | None:
    from .fx import band_db

    if _silent(y_ref) or _silent(y_est):
        return None
    a, b = band_db(y_ref, sr), band_db(y_est, sr)
    live = a > -50
    return float(np.mean(np.abs(a[live] - b[live]))) if live.any() else None
```

In `run`, add `row["spectral_db"] = spectral_db(yo, yr, SR)`. In the CLI table, add a `spec dB` column after `energy` (`f(r.get('spectral_db'), '.1f')`). In `_html`, add a `<th>spec dB</th>` column.

- [ ] **Step 4: Run all tests, fixing the CLI table test fixture** (`test_cli_compare_prints_a_row_per_stem` must include `"spectral_db": 3.2` in its fake row)

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/compare.py src/soundcode/cli.py tests/test_compare.py
git commit -m "compare: spectral_db — does the part sound like the original (1/3-octave spectrum)"
```

---

### Task 4: The song's own drum kit

**Files:**
- Create: `src/soundcode/kit.py`
- Modify: `src/soundcode/parser.py` (`parse_file` sets `doc.path`), `src/soundcode/render_sf.py` (kit playback)
- Test: `tests/test_kit.py`

**Interfaces:**
- Produces:
  - `kit.isolated(hits: list[tuple[float, str, int]], before: float = 0.08, after: float = 0.15) -> dict[str, list[float]]`: per voice, onsets with no other hit within the window, ordered by closeness to the voice's median velocity, at most 4
  - `kit.build(drum_stem: Path, hits: list[tuple[float, str, int]], out_dir: Path) -> dict[str, list[Path]]`: writes `<voice>_<k>.wav`
  - `kit.load(kit_dir: Path, sr: int) -> dict[str, list[np.ndarray]]`, returning `(n, 2)` float32 arrays resampled to `sr`
  - `kit.play(hits: list[tuple[float, str, int]], samples: dict[str, list[np.ndarray]], n: int) -> tuple[np.ndarray, list[tuple[float, str, int]]]`: the mixed `(n, 2)` audio, plus the hits whose voice had no sample (for GM fallback). Round-robin per voice, scaled by `(vel/127)^1.5`.
  - `Document.path: Path | None`, set by `parse_file`
  - `render_sf.render_streams` plays a `perc` stream whose `meta kit` resolves (relative to `doc.path.parent`), with GM fallback for the remaining hits. A missing folder gives a warn and a full GM fallback.

- [ ] **Step 1: Write the failing tests** in `tests/test_kit.py`:

```python
"""The song's own drum kit (spec 2026-09-27-production-match §2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import kit  # noqa: E402
from soundcode.parser import parse, parse_file  # noqa: E402

SR = 44100


def test_isolated_hits_skip_crowded_ones_and_prefer_typical_velocity():
    hits = [(0.0, "kick", 100), (0.05, "hat", 60), (1.0, "kick", 90), (2.0, "kick", 40),
            (3.0, "kick", 95), (3.1, "snare", 80)]
    got = kit.isolated(hits)
    assert got["kick"][0] in (1.0,)                       # 0.0 crowded by the hat; 3.0 by the snare
    assert 2.0 in got["kick"] and 0.0 not in got["kick"]
    assert "hat" not in got                               # its only hit is crowded


def test_build_cuts_with_fades_and_play_round_robins(tmp_path):
    y = np.zeros((SR * 4, 2), np.float32)
    for t in (0.5, 1.5, 2.5):
        i = int(t * SR)
        y[i:i + 2000] = np.hanning(4000)[2000:, None] * 0.8
    stem = tmp_path / "drums.wav"
    sf.write(str(stem), y, SR)
    made = kit.build(stem, [(0.5, "clap", 100), (1.5, "clap", 100), (2.5, "clap", 100)], tmp_path / "kit")
    assert len(made["clap"]) == 3
    s0, _ = sf.read(str(made["clap"][0]))
    assert abs(s0[0]).max() < 0.05                         # fade-in, starts 5 ms early
    samples = kit.load(tmp_path / "kit", SR)
    out, missing = kit.play([(0.1, "clap", 127), (0.6, "clap", 127), (1.1, "kick", 100)], samples, SR * 2)
    assert missing == [(1.1, "kick", 100)]
    assert np.abs(out[int(0.1 * SR):int(0.2 * SR)]).max() > 0.3


def test_kit_resamples_on_load(tmp_path):
    (tmp_path / "k").mkdir()
    sf.write(str(tmp_path / "k" / "snare_0.wav"), np.zeros((4410, 2), np.float32), SR)
    got = kit.load(tmp_path / "k", 22050)
    assert got["snare"][0].shape == (2205, 2)


def test_parse_file_records_the_path(tmp_path):
    p = tmp_path / "s.sc"
    p.write_text("%sc 0.3\n")
    assert parse_file(str(p)).path == p


def test_render_uses_the_kit_and_falls_back_when_it_is_gone(tmp_path):
    import pytest
    from soundcode import render_sf
    sf2 = Path(__file__).resolve().parents[1] / "models" / "soundfonts" / "GeneralUser-GS.sf2"
    if not sf2.exists():
        pytest.skip("no SoundFont")
    (tmp_path / "s.kit").mkdir()
    t = np.arange(4410) / SR
    sf.write(str(tmp_path / "s.kit" / "clap_0.wav"),
             np.stack([np.sin(2 * np.pi * 3000 * t)] * 2, 1).astype(np.float32) * 0.5, SR)
    sc = tmp_path / "s.sc"
    sc.write_text("%sc 0.3\n@duration 2.0\n\n:perc.drums inst=drums.kit\nmeta kit=s.kit\n"
                  "@0.5 clap 127\n@1.0 kick 100\n")
    doc = parse_file(str(sc))
    y = render_sf.render_streams(doc, sf2=sf2)["perc.drums"]
    assert np.abs(y[int(0.5 * SR):int(0.55 * SR)]).max() > 0.2        # the kit clap
    assert np.abs(y[int(1.0 * SR):int(1.2 * SR)]).max() > 0.01        # GM kick fallback
    import shutil
    shutil.rmtree(tmp_path / "s.kit")
    y2 = render_sf.render_streams(parse_file(str(sc)), sf2=sf2)["perc.drums"]
    assert np.abs(y2).max() > 0.01                                     # full GM fallback, no crash
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_kit.py -q`
Expected: collection error `cannot import name 'kit'`.

- [ ] **Step 3: Implement** `src/soundcode/kit.py`:

```python
"""The song's own drum kit: one-shots cut from the separated drum stem at the
transcribed hit times, played back round-robin at the rebuilt pattern.

These samples are audio from the original recording (the faithful-rebuild
path); the .sc still carries every hit as code, and without `meta kit` the
render falls back to the General MIDI kit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_PER_VOICE = 4
PRE_S, MAX_S, FADE_IN_S, FADE_OUT_S = 0.005, 0.6, 0.005, 0.02


def isolated(hits: list[tuple[float, str, int]], before: float = 0.08,
             after: float = 0.15) -> dict[str, list[float]]:
    times = sorted(t for t, _, _ in hits)
    by_voice: dict[str, list[tuple[float, int]]] = {}
    for t, v, vel in hits:
        crowded = any(0 < t - u <= before or 0 < u - t <= after for u in times if u != t)
        if not crowded:
            by_voice.setdefault(v, []).append((t, vel))
    out = {}
    for v, lst in by_voice.items():
        med = float(np.median([vel for _, vel in lst]))
        out[v] = [t for t, _ in sorted(lst, key=lambda x: abs(x[1] - med))][:MAX_PER_VOICE]
    return out


def build(drum_stem: Path, hits: list[tuple[float, str, int]], out_dir: Path) -> dict[str, list[Path]]:
    import soundfile as sf

    y, sr = sf.read(str(drum_stem), always_2d=True)
    y = y.astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_t = sorted(t for t, _, _ in hits)
    made: dict[str, list[Path]] = {}
    for v, ts in isolated(hits).items():
        for k, t in enumerate(ts):
            nxt = next((u for u in all_t if u > t + 1e-6), t + MAX_S)
            a, b = int(max(0, t - PRE_S) * sr), int(min(nxt - PRE_S, t + MAX_S) * sr)
            seg = y[a:b].copy()
            if len(seg) < int(0.02 * sr):
                continue
            fi, fo = int(FADE_IN_S * sr), int(FADE_OUT_S * sr)
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
    for p in sorted(Path(kit_dir).glob("*.wav")):
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
```

`play` needs the render's sample rate. `render_sf` passes `sr=sr`. The kit test calls it at 44.1 kHz, the default.

In `parser.parse_file`, set `doc.path = Path(path)` (add `path: Path | None = None` to the `Document` dataclass).

In `render_sf.render_streams`, for a stream with `kind == "perc"` and `s.meta.get("kit")`:

```python
            kit_dir = Path(s.meta["kit"])
            if not kit_dir.is_absolute() and getattr(doc, "path", None):
                kit_dir = doc.path.parent / kit_dir
            if kit_dir.is_dir():
                from . import kit as kitmod
                hits = [(x.start, x.voice or "", x.vel) for x in stream_notes]
                y_kit, missing = kitmod.play(hits, kitmod.load(kit_dir, sr), n, sr)
                gm_notes = [x for x in stream_notes if (x.start, x.voice or "", x.vel) in set(missing)]
                y = y_kit + (_synth_stream(stream_events(gm_notes, target), target, sfpath, sr, n)
                             if gm_notes else 0)
            else:
                s.warns.append(f"drum kit {kit_dir} not found; General MIDI kit used")
                y = _synth_stream(stream_events(stream_notes, target), target, sfpath, sr, n)
```

The existing `_synth_stream` call is otherwise unchanged for non-kit streams.

- [ ] **Step 4: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/kit.py src/soundcode/parser.py src/soundcode/model.py src/soundcode/render_sf.py tests/test_kit.py
git commit -m "kit: the song's own drum one-shots (isolated hits), round-robin playback, GM fallback"
```

---

### Task 5: Encoder writes fx lines and the kit

**Files:**
- Modify: `src/soundcode/encode.py`
- Test: `tests/test_fx.py` (append)

**Interfaces:**
- Consumes: `fx.measure`, `fx.fx_line`, `kit.build`; `Stage.stem`; the stem paths from `encode()` (`ts_stems` + `stems`).
- Produces:
  - `Stage.extra_meta: dict[str, str]`, written by `_stage_lines` as `meta k=v` lines
  - `encode.attach_fx(stages: list[Stage], stem_paths: dict[str, Path], sr: int) -> None`: for every `ok` stage with a `stem` in `stem_paths`, measure fx on the stereo stem (offsets = the stage's note ends, parsed back from its lines via the grid) and insert `fx_line` as the stage's first line. Offsets can be approximated from the stem's onsets when parsing is inconvenient: use `librosa.onset.onset_detect` on the stem, and each offset = the next onset − 10 ms.
  - The drums stage builds the kit into `wd/"kit"` from the stem and the tsumugi hit list, and sets `st.kit_src = <dir>`.
  - `encode()` copies `kit_src` to `dest.with_suffix(".kit")` after computing `dest`, and sets `st.extra_meta["kit"] = dest.with_suffix(".kit").name`, all before building the text.

- [ ] **Step 1: Write the failing test** (append to `tests/test_fx.py`):

```python
# --- encoder attaches fx ---------------------------------------------------------------------

def test_attach_fx_adds_a_line_to_stem_backed_stages(tmp_path):
    import soundfile as sf
    from soundcode import encode as enc
    p = tmp_path / "piano.wav"
    sf.write(str(p), np.stack([noise(seed=5), noise(seed=6)], 1), SR)
    st = enc.Stage("notes.piano", src="x", ok=True, stem="piano")
    st.lines = ["1:1.000  C4  1.000b 90"]
    other = enc.Stage("notes.x", src="x", ok=True)
    other.lines = ["1:1.000  C4  1.000b 90"]
    enc.attach_fx([st, other], {"piano": p}, SR)
    assert st.lines[0].startswith("fx      eq=")
    assert other.lines == ["1:1.000  C4  1.000b 90"]
    text = "\n".join(enc._stage_lines(st))
    assert fx.parse_fx(parse("%sc 0.3\n\n" + text).stream("notes.piano")) is not None
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_fx.py -q -k attach`
Expected: `AttributeError: … 'attach_fx'`.

- [ ] **Step 3: Implement.**
  - Add `extra_meta: dict[str, str] = field(default_factory=dict)` and `kit_src: Path | None = None` to `Stage`. In `_stage_lines`, after the `stem=`/`level=` line, write `meta    {k}={v}` for each `extra_meta` item.
  - Add `attach_fx` as described, using `librosa.load(path, sr=sr, mono=False)` for the stereo stem and `librosa.onset.onset_detect(..., units="time")` for the offsets.
  - In `stage_tsumugi`'s drums branch, after the stage is built:

    ```python
                if st.ok:
                    from . import kit as kitmod
                    hits = [(s0, gm.drum_voice(p), v) for t in tracks for s0, _, p, v in t.notes]
                    made = kitmod.build(path, hits, work / "kit")
                    if made:
                        st.kit_src = work / "kit"
    ```

  - In `encode()`: after all stages exist, call `attach_fx([...all stages...], {**ts_stems, **{"piano": stems.get("piano"), …}}, sr)` over the stem paths that exist. Then, after `dest` is computed and before `text` is assembled:

    ```python
    for st in all_stages:
        if st.kit_src and Path(st.kit_src).is_dir():
            kit_dir = dest.with_suffix(".kit")
            shutil.rmtree(kit_dir, ignore_errors=True)
            shutil.copytree(st.kit_src, kit_dir)
            st.extra_meta["kit"] = kit_dir.name
    ```

    This means the line-assembly block moves below the `dest` computation. Keep the output identical otherwise, and re-add `import shutil`.

- [ ] **Step 4: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/encode.py tests/test_fx.py
git commit -m "encode: fx line per stem-backed stream; drum kit cut from the song's drum stem next to the .sc"
```

---

### Task 6: Measure and listen

**Files:**
- Create: `docs/results/2026-09-27-production-match.md`
- Modify: `README.md`

- [ ] **Step 1: Re-encode River and discipline** into `out/sc/fx/`, with `SOUNDCODE_SEEDVC_HOST=framepick` set for the vocal later. Check that each `.sc` has `fx` lines on its note/perc streams, and that `river-30s.kit/` contains `clap_*.wav`.

- [ ] **Step 2: Compare three renders per song**, all with the same `.sc`:
  - **A:** `--no-fx`, with the kit disabled (a copy of the `.sc` without its `meta kit` line). This is the baseline.
  - **B:** fx on, kit disabled.
  - **C:** fx on, kit on.

  Do this with `compare` (add `--no-fx` to `compare`, threaded to `render_streams`, if it is not there yet; record a Ruling). Record `spectral_db`, note F1 and onset F1 per stem.
  Expected: `spectral_db` drops from A to B on every active stem, and drums drop further from B to C. Note F1 and onset F1 stay within ±0.02 across A, B and C.

- [ ] **Step 3: Listening checkpoint.** Play the River original (25 s), then C with vocals (`render --with-vocals`). Then the discipline original, then C. Copy both renders to `~/Downloads/`.

- [ ] **Step 4: Write** the results file (the A/B/C tables and the user's verdict), add a README Status line, run the tests, and commit:

```bash
.venv/bin/python -m pytest -q
git add docs/results/2026-09-27-production-match.md README.md
git commit -m "docs: production matching + drum kit results"
```
