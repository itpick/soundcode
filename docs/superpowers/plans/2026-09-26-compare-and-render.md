# Compare + Sampled Renderer + Encoder Fixes — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- `soundcode render` plays a `.sc` through real sampled instruments.
- `soundcode compare` scores a render against the original, stem by stem, and writes a visual report.
- The encoder stops inventing notes, records each stream's source stem and level, and fixes the +33 cent pitch bias.
- Measured before and after on the test songs.

**Architecture:**
- `gm.py` maps `.sc` streams to General MIDI targets.
- `render_sf.py` turns expanded `Note`s into timed MIDI events and renders them offline through `tinysoundfont` per stream, then mixes.
- `compare.py` pairs each rendered stream with its source stem, computes metrics with `mir_eval`/`librosa`, and writes `report.json`, `report.html` and PNGs.
- Encoder fixes live in `encode.py` (`stage_notes_poly` plus a new pure `active_blocks` helper).

Plan 2 (instrument inventory, full taxonomy) is written after this plan lands, using `compare` numbers.

**Tech Stack:** Python 3.11, numpy, librosa 0.11, soundfile, tinysoundfont 0.3.7, mir_eval, matplotlib, basic-pitch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-instruments-design.md` (Parts A1, A2, B; §0 findings; Evaluation item 0).

## Global Constraints

- Python `>=3.11,<3.12`, and librosa `<1.0`. Install packages with `uv pip install --python .venv/bin/python <pkg>`; the venv has no pip.
- Open, free tools only. No paid APIs.
- SoundFont:
  - GeneralUser GS at `models/soundfonts/GeneralUser-GS.sf2`, sha256 `9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe`.
  - URL: `https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2`.
  - `$SOUNDCODE_SOUNDFONT` overrides it.
- Stem names come from `separate.STEMS`: `lead_vocals backing_vocals drums bass guitar piano other`.
- Renders are stereo float32 at the document sample rate (44100 unless `@sr` says otherwise).
- JSON outputs never contain NaN or Infinity (`allow_nan=False`); non-finite numbers become `null`.
- Private use only: `out/`, `models/`, `data/` and `audio/` stay gitignored. Never commit audio.
- The encoder is fail-soft (spec §5): a stage that cannot produce evidence is omitted with a reason, never guessed.
- No `Co-Authored-By` or other Claude attribution lines in commits.

## Spec deviations decided in this plan

- **Spec B2 "`minimum_note_length` 58 → 80 ms"** is wrong about the default. basic-pitch's default is already 127.7 ms. It stays at the default; B2 changes only `onset_threshold`, `frame_threshold`, the amplitude floor and same-pitch merging.
- **Spec A2 "`:contour` vibrato as pitch bend"**: the encoder emits no `:contour` streams yet (review finding). Plan 1 applies the static cents offset as pitch bend on monophonic streams. Vibrato bend is deferred until a `:contour` stage exists (Milestone 2).
- **Spec A1 "CLAP sound similarity"**: msclap is not installed yet. `compare` reports `sound: null` with a note when it is absent. Plan 2 installs msclap for the tagger, which enables it.

## Review Focus

1. **A `.sc` with no pitched streams, or a stem with no notes on one side** (e.g. an original stem silent where the render has notes). The metrics must return `null` or 0, not crash. The report must show the stem, not drop it. (Task 5 tests.)
2. **Stream names outside the known set** (`notes.lead2`, `notes.pad.other`, hand-written files). `compare` pairs them with `other`; the renderer falls back to piano. Neither crashes. (Task 1 and Task 6 tests.)
3. **Documents with `@seconds` events, no `:grid`, or a `@sr` other than 44100.** The renderer honours `@sr`. `compare` falls back to 2 s blocks without a grid. (Task 3 and Task 5 tests.)
4. **The SoundFont missing, offline, or with a bad checksum.** One-line error, exit 2, and a half-downloaded file is not left in place. (Task 3 and Task 4 tests.)
5. **The pitch fix must not erase real detuning**, and the loudness gate must not silence quiet-but-real parts: a sparse soft piano between loud sections. (Task 8 and Task 9 tests.)

---

### Task 1: General MIDI targets for streams

**Files:**
- Create: `src/soundcode/gm.py`
- Test: `tests/test_render_sf.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Target(bank: int, preset: int, drums: bool, family: str)`
  - `target_for(stream_name: str, inst: str) -> Target`
  - `drum_note(voice: str) -> int`
  - `FAMILY_PROGRAM: dict[str, int]`
  - `MONO_FAMILIES: frozenset[str]`

- [ ] **Step 1: Write the failing tests**

```python
"""Sampled renderer (spec Part A2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import gm  # noqa: E402


def test_inst_family_picks_the_program():
    assert gm.target_for("notes.x", "bass.electric") == gm.Target(0, 33, False, "bass")
    assert gm.target_for("notes.x", "keys.piano").preset == 0
    assert gm.target_for("notes.x", "voice.lead").preset == 53


def test_stream_name_is_the_fallback_when_inst_is_missing_or_unknown():
    assert gm.target_for("notes.bass", "unknown").family == "bass"
    assert gm.target_for("notes.vox", "unknown").family == "voice"
    assert gm.target_for("notes.guitar", "unknown").preset == 27
    assert gm.target_for("notes.lead2", "unknown") == gm.Target(0, 0, False, "unknown")


def test_percussion_streams_use_the_drum_kit():
    t = gm.target_for("perc.drums", "unknown")
    assert t.drums and t.bank == 128 and t.family == "drums"


def test_drum_voice_map():
    assert [gm.drum_note(v) for v in ("kick", "snare", "hat", "crash")] == [36, 38, 42, 49]
    assert gm.drum_note("cowbell-ish") == 39
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_render_sf.py -q`
Expected: collection error `cannot import name 'gm'`.

- [ ] **Step 3: Implement `src/soundcode/gm.py`**

```python
"""General MIDI targets for .sc streams.

Plan 1 maps at family level (`bass.electric` -> the bass program). Plan 2
replaces FAMILY_PROGRAM with the full instrument taxonomy's render targets.
"""

from __future__ import annotations

from dataclasses import dataclass

# GM programs, 0-indexed, as GeneralUser GS names them
FAMILY_PROGRAM = {
    "keys": 0,        # Acoustic Grand Piano
    "mallet": 11,     # Vibraphone
    "organ": 16,      # Drawbar Organ
    "gtr": 27,        # Clean Guitar
    "bass": 33,       # Finger Bass
    "strings": 48,    # String Ensemble
    "voice": 53,      # Voice Oohs
    "brass": 61,      # Brass Section
    "winds": 65,      # Alto Sax
    "synth": 89,      # Warm Pad
    "unknown": 0,
}
# stream track name -> family, for files whose streams carry no inst=
STREAM_FAMILY = {"bass": "bass", "vox": "voice", "bvox": "voice",
                 "guitar": "gtr", "piano": "keys"}
DRUM_NOTES = {"kick": 36, "snare": 38, "hat": 42, "hat.open": 46, "crash": 49,
              "ride": 51, "tom.hi": 50, "tom.mid": 47, "tom.lo": 43}
DRUM_DEFAULT = 39   # hand clap: audible, and obviously "unmapped"
MONO_FAMILIES = frozenset({"voice"})


@dataclass(frozen=True)
class Target:
    bank: int
    preset: int
    drums: bool
    family: str


def target_for(stream_name: str, inst: str) -> Target:
    if stream_name.startswith("perc.") or inst.startswith(("drums", "perc")):
        return Target(128, 0, True, "drums")
    family = inst.split(".")[0] if inst and inst != "unknown" else \
        STREAM_FAMILY.get(stream_name.split(".")[-1], "unknown")
    if family not in FAMILY_PROGRAM:
        family = "unknown"
    return Target(0, FAMILY_PROGRAM[family], False, family)


def drum_note(voice: str) -> int:
    return DRUM_NOTES.get(voice, DRUM_DEFAULT)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_render_sf.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/gm.py tests/test_render_sf.py
git commit -m "gm: General MIDI targets for .sc streams"
```

---

### Task 2: Notes → MIDI events

**Files:**
- Create: `src/soundcode/render_sf.py`
- Test: `tests/test_render_sf.py` (append)

**Interfaces:**
- Consumes: `gm.target_for`, `gm.drum_note`, `gm.MONO_FAMILIES`; `expand.Note` (fields `start dur cents voice vel stream inst`).
- Produces:
  - `@dataclass(frozen=True, order=True) MidiEvent(time: float, order: int, kind: str, key: int = 0, value: int = 0)`. `kind` is one of `"on"`, `"off"`, `"bend"`. `order` sorts offs before bends before ons at equal times.
  - `stream_events(notes: list[Note], target: gm.Target) -> list[MidiEvent]`, for one stream's notes.
  - `BEND_RANGE_SEMITONES = 2`

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- notes -> MIDI events ------------------------------------------------------

from soundcode import render_sf  # noqa: E402
from soundcode.expand import Note  # noqa: E402


def note(start, cents, dur=0.5, stream="notes.keys", inst="keys.piano", voice=None, vel=90):
    return Note(start=start, dur=dur, cents=cents, voice=voice, vel=vel,
                stream=stream, inst=inst)


def test_polyphonic_streams_round_to_the_nearest_semitone_without_bend():
    t = gm.target_for("notes.keys", "keys.piano")
    ev = render_sf.stream_events([note(0.0, 6040.0), note(0.0, 6360.0)], t)
    ons = [e for e in ev if e.kind == "on"]
    assert sorted(e.key for e in ons) == [60, 64]
    assert not [e for e in ev if e.kind == "bend"]


def test_monophonic_streams_send_the_cents_offset_as_pitch_bend():
    t = gm.target_for("notes.vox", "voice.lead")
    ev = render_sf.stream_events([note(1.0, 6930.0, stream="notes.vox", inst="voice.lead")], t)
    bend = next(e for e in ev if e.kind == "bend")
    on = next(e for e in ev if e.kind == "on")
    assert on.key == 69 and bend.time == on.time and bend < on
    # +30 cents of a +/-2 semitone range: 8192 + 0.30/2 * 8192
    assert bend.value == 8192 + round(0.15 * 8192)


def test_offs_follow_ons_by_duration_and_sort_before_ons_at_the_same_time():
    t = gm.target_for("notes.keys", "keys.piano")
    ev = sorted(render_sf.stream_events([note(0.0, 6000.0, dur=0.5),
                                        note(0.5, 6000.0, dur=0.5)], t))
    kinds = [(round(e.time, 3), e.kind) for e in ev]
    assert kinds == [(0.0, "on"), (0.5, "off"), (0.5, "on"), (1.0, "off")]


def test_drum_notes_use_the_voice_map():
    t = gm.target_for("perc.drums", "unknown")
    ev = render_sf.stream_events([note(0.0, None, stream="perc.drums", voice="snare")], t)
    assert next(e for e in ev if e.kind == "on").key == 38
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_render_sf.py -q`
Expected: 4 new failures, `cannot import name 'render_sf'`.

- [ ] **Step 3: Implement** `src/soundcode/render_sf.py`:

```python
"""Sampled renderer: .sc -> audio through a General MIDI SoundFont.

Each stream is rendered on its own (so `compare` can score it against its
source stem), then mixed. Pitch: chords round to the nearest semitone; a
monophonic line keeps its cents offset as pitch bend, so tuning survives.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import gm
from .expand import Note

BEND_RANGE_SEMITONES = 2
_ORDER = {"off": 0, "bend": 1, "on": 2}


@dataclass(frozen=True, order=True)
class MidiEvent:
    time: float
    order: int
    kind: str
    key: int = 0
    value: int = 0


def _event(time: float, kind: str, key: int = 0, value: int = 0) -> MidiEvent:
    return MidiEvent(time, _ORDER[kind], kind, key, value)


def stream_events(notes: list[Note], target: gm.Target) -> list[MidiEvent]:
    mono = target.family in gm.MONO_FAMILIES
    out: list[MidiEvent] = []
    for n in notes:
        if target.drums:
            key = gm.drum_note(n.voice or "")
        elif n.cents is None:
            continue
        else:
            key = int(round(n.cents / 100.0))
            if mono:
                offset = (n.cents - key * 100) / 100.0          # semitones
                value = 8192 + round(offset / BEND_RANGE_SEMITONES * 8192)
                out.append(_event(n.start, "bend", value=max(0, min(16383, value))))
        key = max(0, min(127, key))
        out.append(_event(n.start, "on", key, max(1, min(127, n.vel))))
        out.append(_event(n.start + max(n.dur, 0.02), "off", key))
    return out
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_render_sf.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/render_sf.py tests/test_render_sf.py
git commit -m "render_sf: notes to MIDI events, cents as pitch bend on mono lines"
```

---

### Task 3: SoundFont rendering per stream, level matching, mix

**Files:**
- Modify: `src/soundcode/render_sf.py` (append)
- Test: `tests/test_render_sf.py` (append)

**Interfaces:**
- Consumes: `stream_events`, `MidiEvent`, `gm.target_for`; `render._GAIN`, `render._PAN`, `render.section_gains`; `expand.expand`; `Document.streams`, `Stream.meta`.
- Produces:
  - `class SoundFontError(RuntimeError)`
  - `soundfont_path() -> Path`, which resolves `$SOUNDCODE_SOUNDFONT`, else the default path, downloading and verifying the checksum if missing.
  - `render_streams(doc: Document, sr: int | None = None, sf2: Path | None = None) -> dict[str, np.ndarray]`, giving each stream name a `(n, 2)` float32 array. All arrays have the same length. A stream with `meta level=<dB>` is scaled to that RMS; other streams keep their raw level.
  - `mix(doc: Document, streams: dict[str, np.ndarray], sr: int) -> np.ndarray`
  - `render(doc, sr=None, sf2=None) -> np.ndarray`
  - `render_to_file(doc, path, sr=None) -> tuple[int, float]`, with the same return shape as `render.render_to_file`.

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- SoundFont rendering -------------------------------------------------------------

from soundcode.parser import parse  # noqa: E402

SF2 = Path(__file__).resolve().parents[1] / "models" / "soundfonts" / "GeneralUser-GS.sf2"
needs_sf2 = pytest.mark.skipif(not SF2.exists(), reason="SoundFont not downloaded")

SCALE_SC = """%sc 0.3
@duration 5.0
@sr 44100

:notes.keys inst=keys.piano
@0.0 C4 0.4s 100
@0.5 D4 0.4s 100
@1.0 E4 0.4s 100
@1.5 F4 0.4s 100
@2.0 G4 0.4s 100
"""


def _f0(x, sr):
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo = int(sr / 1000)
    return sr / (np.argmax(ac[lo:]) + lo)


@needs_sf2
def test_scale_renders_in_tune_and_to_length():
    doc = parse(SCALE_SC)
    streams = render_sf.render_streams(doc, sf2=SF2)
    y = streams["notes.keys"]
    assert y.shape[1] == 2 and y.dtype == np.float32
    assert abs(y.shape[0] / 44100 - 6.5) < 0.01          # @duration + 1.5 s release tail
    for i, hz in enumerate([261.63, 293.66, 329.63, 349.23, 392.0]):
        seg = y[int((i * 0.5 + 0.05) * 44100): int((i * 0.5 + 0.30) * 44100), 0]
        cents = 1200 * np.log2(_f0(seg, 44100) / hz)
        assert abs(cents) < 20, (i, cents)


@needs_sf2
def test_meta_level_sets_the_stream_rms():
    doc = parse(SCALE_SC.replace(":notes.keys inst=keys.piano",
                                 ":notes.keys inst=keys.piano\nmeta level=-30.0"))
    y = render_sf.render_streams(doc, sf2=SF2)["notes.keys"]
    active = y[np.abs(y).max(1) > 1e-4]
    rms_db = 20 * np.log10(np.sqrt(np.mean(active ** 2)))
    assert abs(rms_db - (-30.0)) < 0.5


@needs_sf2
def test_mix_is_normalised_and_honours_sample_rate():
    doc = parse(SCALE_SC.replace("@sr 44100", "@sr 22050"))
    y = render_sf.render(doc, sf2=SF2)
    assert 0.8 < float(np.abs(y).max()) <= 0.9
    assert abs(y.shape[0] / 22050 - 6.5) < 0.01


def test_missing_soundfont_offline_is_one_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCODE_SOUNDFONT", str(tmp_path / "nope.sf2"))
    with pytest.raises(render_sf.SoundFontError, match="SOUNDCODE_SOUNDFONT"):
        render_sf.soundfont_path()


def test_bad_download_is_not_left_behind(tmp_path, monkeypatch):
    monkeypatch.delenv("SOUNDCODE_SOUNDFONT", raising=False)
    monkeypatch.setattr(render_sf, "DEFAULT_SF2", tmp_path / "sf" / "GeneralUser-GS.sf2")
    monkeypatch.setattr(render_sf, "_download", lambda url, dest: dest.write_bytes(b"junk"))
    with pytest.raises(render_sf.SoundFontError, match="checksum"):
        render_sf.soundfont_path()
    assert not (tmp_path / "sf" / "GeneralUser-GS.sf2").exists()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_render_sf.py -q`
Expected: 5 new failures (`AttributeError: ... 'render_streams'` / `'SoundFontError'`).

- [ ] **Step 3: Implement** by appending to `render_sf.py`. Add these imports at the top: `import hashlib`, `import os`, `import urllib.request`, `from pathlib import Path`, `import numpy as np`, `from .model import Document`, and `from .expand import expand` (next to `Note`).

```python
SF2_URL = "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2"
SF2_SHA256 = "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe"
DEFAULT_SF2 = Path("models") / "soundfonts" / "GeneralUser-GS.sf2"
_TAIL_S = 1.5          # release tails after the last note
_CHUNK = 512           # samples per generate() call between events


class SoundFontError(RuntimeError):
    """No usable SoundFont; the message says how to get one."""


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as fh:
        fh.write(r.read())


def soundfont_path() -> Path:
    override = os.environ.get("SOUNDCODE_SOUNDFONT")
    if override:
        p = Path(override)
        if not p.is_file():
            raise SoundFontError(f"SOUNDCODE_SOUNDFONT={p} does not exist")
        return p
    p = DEFAULT_SF2
    if p.is_file():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    part = p.with_suffix(".part")
    try:
        _download(SF2_URL, part)
    except Exception as exc:  # offline, HTTP error
        part.unlink(missing_ok=True)
        raise SoundFontError(f"could not download the SoundFont ({exc}); set "
                             "SOUNDCODE_SOUNDFONT to any .sf2, or use --engine mock") from exc
    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    if digest != SF2_SHA256:
        part.unlink(missing_ok=True)
        raise SoundFontError(f"SoundFont download failed its checksum ({digest[:12]}…); "
                             "set SOUNDCODE_SOUNDFONT to any .sf2, or use --engine mock")
    part.rename(p)
    return p


def _synth_stream(events: list[MidiEvent], target: gm.Target, sfpath: Path,
                  sr: int, n: int) -> np.ndarray:
    import tinysoundfont

    synth = tinysoundfont.Synth(samplerate=sr)
    sfid = synth.sfload(str(sfpath))
    ch = 9 if target.drums else 0
    synth.program_select(ch, sfid, target.bank, target.preset, target.drums)
    synth.pitchbend_range(ch, BEND_RANGE_SEMITONES)
    out = np.zeros((n, 2), dtype=np.float32)
    pos = 0
    for ev in sorted(events):
        stop = min(int(ev.time * sr), n)
        while pos < stop:
            k = min(_CHUNK, stop - pos)
            out[pos:pos + k] = np.frombuffer(synth.generate(k), dtype=np.float32).reshape(-1, 2)
            pos += k
        if ev.kind == "on":
            synth.noteon(ch, ev.key, ev.value)
        elif ev.kind == "off":
            synth.noteoff(ch, ev.key)
        else:
            synth.pitchbend(ch, ev.value)
    while pos < n:
        k = min(_CHUNK, n - pos)
        out[pos:pos + k] = np.frombuffer(synth.generate(k), dtype=np.float32).reshape(-1, 2)
        pos += k
    return out


def _rms_db(y: np.ndarray) -> float:
    active = y[np.abs(y).max(1) > 1e-4] if y.size else y
    if not active.size:
        return float("-inf")
    return 20 * float(np.log10(np.sqrt(np.mean(active.astype(np.float64) ** 2))))


def render_streams(doc: Document, sr: int | None = None,
                   sf2: Path | None = None) -> dict[str, np.ndarray]:
    sr = sr or doc.sample_rate
    sfpath = sf2 or soundfont_path()
    notes = expand(doc)
    total = doc.duration or (max((x.start + x.dur for x in notes), default=1.0))
    n = int((total + _TAIL_S) * sr)
    by_stream: dict[str, list[Note]] = {}
    for x in notes:
        by_stream.setdefault(x.stream, []).append(x)

    out: dict[str, np.ndarray] = {}
    for name, stream_notes in by_stream.items():
        target = gm.target_for(name, stream_notes[0].inst)
        y = _synth_stream(stream_events(stream_notes, target), target, sfpath, sr, n)
        s = doc.stream(name)
        level = s.meta.get("level") if s is not None else None
        if level is not None:
            have = _rms_db(y)
            if np.isfinite(have):
                y *= 10 ** ((float(level.rstrip("dB")) - have) / 20)
        out[name] = y
    return out


def mix(doc: Document, streams: dict[str, np.ndarray], sr: int) -> np.ndarray:
    from .render import _GAIN, _PAN, section_gains

    n = max((y.shape[0] for y in streams.values()), default=sr)
    buf = np.zeros((n, 2), dtype=np.float32)
    for name, y in streams.items():
        s = doc.stream(name)
        target = gm.target_for(name, s.fields.get("inst", "unknown") if s else "unknown")
        has_level = s is not None and "level" in s.meta
        gain = 1.0 if has_level else _GAIN.get(target.family, _GAIN["unknown"])
        pan = _PAN.get(target.family, 0.0)
        buf[:, 0] += y[:, 0] * gain * (1.0 - max(pan, 0.0))
        buf[:, 1] += y[:, 1] * gain * (1.0 + min(pan, 0.0))
    for start_s, end_s, g in section_gains(doc):
        a, b = int(start_s * sr), min(int(end_s * sr), n)
        if b > a:
            buf[a:b] *= g
    peak = float(np.abs(buf).max())
    if peak > 0:
        buf *= 0.89 / peak
    return buf


def render(doc: Document, sr: int | None = None, sf2: Path | None = None) -> np.ndarray:
    sr = sr or doc.sample_rate
    return mix(doc, render_streams(doc, sr, sf2), sr)


def render_to_file(doc: Document, path: str, sr: int | None = None) -> tuple[int, float]:
    import soundfile as sf

    sr = sr or doc.sample_rate
    audio = render(doc, sr)
    sf.write(path, audio, sr, subtype="PCM_16")
    return len(expand(doc)), audio.shape[0] / sr
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_render_sf.py -q`
Expected: `13 passed`. The SoundFont exists locally (downloaded by the spike). If the in-tune test fails on octave or pitch, check `_f0`'s search range before touching the renderer.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/render_sf.py tests/test_render_sf.py
git commit -m "render_sf: SoundFont rendering per stream, level matching, mix"
```

---

### Task 4: `render --engine`, and `render` kind in the server

**Files:**
- Modify: `src/soundcode/cli.py` (render parser and handler)
- Modify: `src/soundcode/server.py` (`_discover_tracks`)
- Test: `tests/test_render_sf.py` (append)

**Interfaces:**
- Consumes: `render_sf.render_to_file`, `render_sf.SoundFontError`.
- Produces:
  - `soundcode render FILE [-o OUT] [--sr N] [--engine sf2|mock]`. The default is `sf2`, and the default output is `<file stem>.render.wav`. `mock` keeps `<stem>.mock.wav`.
  - Server kind `"render"` for labels containing `.render` or `render-`.

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- CLI + server ----------------------------------------------------------------

from soundcode import cli, server  # noqa: E402


def test_render_defaults_to_sf2_and_names_the_output(tmp_path, monkeypatch):
    sc = tmp_path / "song.sc"
    sc.write_text(SCALE_SC)
    calls = {}
    monkeypatch.setattr(render_sf, "render_to_file",
                        lambda doc, path, sr=None: calls.setdefault("path", path) and (5, 5.0))
    assert cli.main(["render", str(sc)]) == 0
    assert calls["path"].endswith("song.render.wav")


def test_render_reports_missing_soundfont_and_exits_two(tmp_path, monkeypatch, capsys):
    sc = tmp_path / "song.sc"
    sc.write_text(SCALE_SC)

    def boom(*a, **k):
        raise render_sf.SoundFontError("could not download the SoundFont; set SOUNDCODE_SOUNDFONT")
    monkeypatch.setattr(render_sf, "render_to_file", boom)
    assert cli.main(["render", str(sc)]) == 2
    assert "SOUNDCODE_SOUNDFONT" in capsys.readouterr().err


def test_server_tags_render_files(tmp_path, monkeypatch):
    (tmp_path / "out" / "sc").mkdir(parents=True)
    (tmp_path / "out" / "sc" / "song.render.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("SOUNDCODE_ROOT", str(tmp_path))
    assert server._discover_tracks()[0]["kind"] == "render"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_render_sf.py -q`
Expected: 3 new failures: the default path is `.mock.wav`, the SoundFontError is uncaught, and the kind is `other`.

- [ ] **Step 3: Implement.** In `cli.py`, add to `p_render`:

```python
    p_render.add_argument("--engine", choices=("sf2", "mock"), default="sf2",
                          help="sf2: sampled instruments (default); mock: crude synth")
```

Replace the `render` handler with:

```python
        if args.cmd == "render":
            doc = parse_file(args.file)
            if args.engine == "mock":
                from .render import render_to_file
                suffix = ".mock.wav"
            else:
                from . import render_sf
                render_to_file = render_sf.render_to_file
                suffix = ".render.wav"
            out = args.out or str(Path(args.file).with_suffix(suffix))
            try:
                count, secs = render_to_file(doc, out, args.sr)
            except Exception as exc:
                from .render_sf import SoundFontError
                if isinstance(exc, SoundFontError):
                    print(f"render failed: {exc}", file=sys.stderr)
                    return 2
                raise
            print(f"rendered {count} events -> {out}  ({secs:.2f}s @ "
                  f"{args.sr or doc.sample_rate} Hz, engine {args.engine})")
            return 0
```

In `server._discover_tracks`, add before `elif "mock" in label:`:

```python
        elif ".render" in label or "render-" in label:
            kind = "render"
```

Then update `order` to `{"ref": 0, "render": 1, "mock": 2, "cover": 3, "stem": 4, "other": 5}`.

- [ ] **Step 4: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: `62 passed` (46 existing + 16 in `test_render_sf.py`).

- [ ] **Step 5: Real render**

Run: `.venv/bin/soundcode render examples/signal-lost.v3.sc -o out/signal-lost.render.wav && afplay -t 15 out/signal-lost.render.wav`
Expected: `rendered 261 events -> out/signal-lost.render.wav (... engine sf2)`, and audible sampled instruments.

- [ ] **Step 6: Commit**

```bash
git add src/soundcode/cli.py src/soundcode/server.py tests/test_render_sf.py
git commit -m "render: --engine sf2 (default) | mock; server tags sampled renders"
```

---

### Task 5: Comparison metrics

**Files:**
- Create: `src/soundcode/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Produces (pure functions; audio is mono float32):
  - `SR = 22050`
  - `level_db(y) -> float | None`, which is `None` for silence
  - `note_f1(ref_iv, ref_hz, est_iv, est_hz, octave_agnostic=False) -> float | None`. Intervals are `(n, 2)` seconds and pitches are Hz. Returns `None` when both sides are empty, and 0.0 when only one side is.
  - `chroma_blocks(y_ref, y_est, blocks: list[tuple[float, float]]) -> list[float | None]`, a cosine per block, `None` where either side is silent
  - `onset_f1(y_ref, y_est) -> float | None`
  - `energy_corr(y_ref, y_est) -> float | None`
  - `blocks_from_grid(doc, duration) -> list[tuple[float, float]]`, which gives bars from the `:grid` or 2 s blocks
  - `json_safe(obj)`, which recursively turns non-finite floats into `None`

- [ ] **Step 1: Write the failing tests** in `tests/test_compare.py`:

```python
"""soundcode compare (spec Part A1)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import compare as cmp  # noqa: E402

SR = cmp.SR


def tone(freq, secs=2.0, amp=0.3, start=0.0, total=None):
    total = total or secs + start
    y = np.zeros(int(total * SR), np.float32)
    t = np.arange(int(secs * SR)) / SR
    y[int(start * SR):int(start * SR) + t.size] = amp * np.sin(2 * np.pi * freq * t)
    return y


def clicks(times, total=4.0):
    y = np.zeros(int(total * SR), np.float32)
    for t in times:
        i = int(t * SR)
        y[i:i + 200] = np.hanning(200)
    return y


def test_level_of_silence_is_none():
    assert cmp.level_db(np.zeros(100, np.float32)) is None
    assert cmp.level_db(np.ones(100, np.float32)) == pytest.approx(0.0)


def test_note_f1_identical_is_one_and_octave_agnostic_forgives_octaves():
    iv = np.array([[0.0, 0.5], [0.5, 1.0]])
    hz = np.array([440.0, 523.25])
    assert cmp.note_f1(iv, hz, iv, hz) == pytest.approx(1.0)
    assert cmp.note_f1(iv, hz, iv, hz * 2) == pytest.approx(0.0)
    assert cmp.note_f1(iv, hz, iv, hz * 2, octave_agnostic=True) == pytest.approx(1.0)


def test_note_f1_empty_sides():
    e = np.zeros((0, 2))
    assert cmp.note_f1(e, np.zeros(0), e, np.zeros(0)) is None
    assert cmp.note_f1(np.array([[0.0, 0.5]]), np.array([440.0]), e, np.zeros(0)) == 0.0


def test_chroma_blocks_same_pitch_high_different_low_silent_none():
    a, b = tone(440, 4.0), tone(440, 4.0)
    c = tone(311.13, 4.0)
    blocks = [(0.0, 2.0), (2.0, 4.0)]
    assert min(cmp.chroma_blocks(a, b, blocks)) > 0.95
    assert max(cmp.chroma_blocks(a, c, blocks)) < 0.6
    z = np.zeros_like(a)
    assert cmp.chroma_blocks(a, z, blocks) == [None, None]


def test_onset_f1_matches_same_clicks():
    ref = clicks([0.5, 1.0, 1.5, 2.0, 2.5])
    assert cmp.onset_f1(ref, ref) == pytest.approx(1.0)
    assert cmp.onset_f1(ref, clicks([3.5])) < 0.3


def test_energy_corr_identical_is_one_and_silence_is_none():
    y = tone(440, 1.0, start=1.0, total=3.0)
    assert cmp.energy_corr(y, y) == pytest.approx(1.0)
    assert cmp.energy_corr(y, np.zeros_like(y)) is None


def test_blocks_fall_back_to_two_seconds_without_grid():
    from soundcode.parser import parse
    doc = parse("%sc 0.3\n@duration 5.0\n")
    assert cmp.blocks_from_grid(doc, 5.0) == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]


def test_json_safe_strips_non_finite():
    out = cmp.json_safe({"a": math.inf, "b": [math.nan, 1.0], "c": {"d": -math.inf}})
    assert json.dumps(out, allow_nan=False) == '{"a": null, "b": [null, 1.0], "c": {"d": null}}'
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: collection error `cannot import name 'compare'`.

- [ ] **Step 3: Implement** `src/soundcode/compare.py`:

```python
"""Compare a render against the original, stem by stem.

Every metric returns None when it has nothing to measure (a silent side),
never NaN: a report must say "no evidence", not print a number that lies.
"""

from __future__ import annotations

import math

import numpy as np

SR = 22050
_HOP = 512
_SILENT_DB = -70.0


def json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def level_db(y: np.ndarray) -> float | None:
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    return 20 * math.log10(r) if r > 0 else None


def _silent(y: np.ndarray) -> bool:
    lv = level_db(y)
    return lv is None or lv < _SILENT_DB


def note_f1(ref_iv, ref_hz, est_iv, est_hz, octave_agnostic: bool = False) -> float | None:
    import mir_eval

    if len(ref_iv) == 0 and len(est_iv) == 0:
        return None
    if len(ref_iv) == 0 or len(est_iv) == 0:
        return 0.0
    ref_hz, est_hz = np.asarray(ref_hz, float), np.asarray(est_hz, float)
    if octave_agnostic:
        fold = lambda h: 440.0 * 2 ** (((12 * np.log2(h / 440.0)) % 12) / 12)  # noqa: E731
        ref_hz, est_hz = fold(ref_hz), fold(est_hz)
    return float(mir_eval.transcription.precision_recall_f1_overlap(
        np.asarray(ref_iv, float), ref_hz, np.asarray(est_iv, float), est_hz,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)[2])


def chroma_blocks(y_ref, y_est, blocks) -> list[float | None]:
    import librosa

    n = min(len(y_ref), len(y_est))
    y_ref, y_est = y_ref[:n], y_est[:n]
    cr = librosa.feature.chroma_cqt(y=y_ref, sr=SR, hop_length=_HOP)
    ce = librosa.feature.chroma_cqt(y=y_est, sr=SR, hop_length=_HOP)
    out: list[float | None] = []
    for a, b in blocks:
        i, j = int(a * SR), int(b * SR)
        if _silent(y_ref[i:j]) or _silent(y_est[i:j]):
            out.append(None)
            continue
        fa, fb = int(a * SR / _HOP), max(int(b * SR / _HOP), int(a * SR / _HOP) + 1)
        u, v = cr[:, fa:fb].mean(1), ce[:, fa:fb].mean(1)
        out.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
    return out


def onset_f1(y_ref, y_est) -> float | None:
    import librosa
    import mir_eval

    if _silent(y_ref) and _silent(y_est):
        return None
    on_r = librosa.onset.onset_detect(y=y_ref, sr=SR, units="time")
    on_e = librosa.onset.onset_detect(y=y_est, sr=SR, units="time")
    if len(on_r) == 0 or len(on_e) == 0:
        return 0.0
    return float(mir_eval.onset.f_measure(on_r, on_e, window=0.07)[0])


def energy_corr(y_ref, y_est) -> float | None:
    import librosa

    n = min(len(y_ref), len(y_est))
    if _silent(y_ref[:n]) or _silent(y_est[:n]):
        return None
    a = librosa.feature.rms(y=y_ref[:n], hop_length=_HOP)[0]
    b = librosa.feature.rms(y=y_est[:n], hop_length=_HOP)[0]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def blocks_from_grid(doc, duration: float) -> list[tuple[float, float]]:
    from .expand import build_grid

    if doc.stream("grid") is not None:
        grid = build_grid(doc)
        edges, bar = [], 1
        while True:
            t = grid.time_of(bar, 1.0)
            if t >= duration or bar > 10000:
                break
            edges.append(t)
            bar += 1
        edges.append(duration)
        blocks = [(a, b) for a, b in zip(edges, edges[1:]) if b > a]
        if blocks:
            return blocks
    edges = list(np.arange(0.0, duration, 2.0)) + [duration]
    return [(float(a), float(b)) for a, b in zip(edges, edges[1:]) if b > a]
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/compare.py tests/test_compare.py
git commit -m "compare: per-stem metrics (level, notes, chroma, onsets, energy)"
```

---

### Task 6: Compare pipeline, report, CLI and server

**Files:**
- Modify: `src/soundcode/compare.py` (append)
- Modify: `src/soundcode/cli.py`, `src/soundcode/server.py`
- Test: `tests/test_compare.py` (append)

**Interfaces:**
- Consumes: Task 5 metrics; `render_sf.render_streams`; `separate.separate`, `separate.STEMS`, `separate.default_out_dir`; `parser.parse_file`.
- Produces:
  - `stem_for_stream(doc, name: str) -> str`, which uses `meta stem=`, else the name mapping, else `"other"`
  - `transcribe(path) -> tuple[np.ndarray, np.ndarray]` (basic-pitch intervals and Hz; empty arrays when there are no notes)
  - `run(original: Path, sc_or_wav: Path, out_dir: Path, engine: str = "sf2", render_streams=None, stems_dir: Path | None = None) -> dict`, which writes `report.json`, `report.html`, `<stem>.png`, `bars.png`, and `stem-<name>.wav` for both sides, and returns the report dict
  - the CLI `soundcode compare ORIGINAL SC_OR_WAV [-o DIR] [--engine sf2|mock]`
  - server `GET /compare/<path>` serving files under `out/compare/`

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- pipeline ------------------------------------------------------------------------

from soundcode.parser import parse  # noqa: E402


def test_stream_to_stem_pairing():
    doc = parse("%sc 0.3\n\n:notes.pad inst=synth.pad\nmeta stem=other\n@0.0 C4 1s 90\n"
                "\n:notes.vox\n@0.0 C4 1s 90\n\n:notes.lead2\n@0.0 C4 1s 90\n")
    assert cmp.stem_for_stream(doc, "notes.pad") == "other"
    assert cmp.stem_for_stream(doc, "notes.vox") == "lead_vocals"
    assert cmp.stem_for_stream(doc, "perc.drums") == "drums"
    assert cmp.stem_for_stream(doc, "notes.piano") == "piano"
    assert cmp.stem_for_stream(doc, "notes.lead2") == "other"


def test_run_scores_a_perfect_render_highly_and_writes_the_report(tmp_path, monkeypatch):
    import soundfile as sf
    from soundcode import separate as sep

    # original stems: piano plays A4 then C5; everything else silent
    sr = 44100
    t = np.arange(sr) / sr
    a4 = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    c5 = (0.3 * np.sin(2 * np.pi * 523.25 * t)).astype(np.float32)
    piano = np.concatenate([a4, c5, np.zeros(sr, np.float32)])
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    for s in sep.STEMS:
        y = piano if s == "piano" else np.zeros_like(piano)
        sf.write(str(stems_dir / f"{s}.wav"), np.stack([y, y]).T, sr, subtype="FLOAT")
    orig = tmp_path / "orig.wav"
    sf.write(str(orig), np.stack([piano, piano]).T, sr)
    sc = tmp_path / "song.sc"
    sc.write_text("%sc 0.3\n@duration 3.0\n\n:notes.piano inst=keys.piano\n"
                  "@0.0 A4 1s 90\n@1.0 C5 1s 90\n")

    fake = lambda doc, sr=None, sf2=None: {"notes.piano": np.stack([piano, piano]).T}  # noqa: E731
    fake_notes = {  # transcriptions: identical on both sides
        "piano": (np.array([[0.0, 1.0], [1.0, 2.0]]), np.array([440.0, 523.25])),
    }
    monkeypatch.setattr(cmp, "transcribe",
                        lambda p: fake_notes["piano"] if "piano" in Path(p).name
                        else (np.zeros((0, 2)), np.zeros(0)))
    report = cmp.run(orig, sc, tmp_path / "cmp", render_streams=fake, stems_dir=stems_dir)

    piano_row = report["stems"]["piano"]
    assert piano_row["notes_f1"] == pytest.approx(1.0)
    assert piano_row["level_diff_db"] == pytest.approx(0.0, abs=0.1)
    assert report["stems"]["bass"]["notes_f1"] is None           # silent both sides
    for f in ("report.json", "report.html", "piano.png", "bars.png"):
        assert (tmp_path / "cmp" / f).exists()
    json.loads((tmp_path / "cmp" / "report.json").read_text())


def test_cli_compare_prints_a_row_per_stem(tmp_path, monkeypatch, capsys):
    from soundcode import cli
    monkeypatch.setattr(cmp, "run", lambda *a, **k: {"stems": {
        "piano": {"level_orig_db": -20.0, "level_render_db": -21.0, "level_diff_db": -1.0,
                  "notes_f1": 0.5, "notes_f1_octave": 0.6, "chroma": 0.9,
                  "onset_f1": 0.7, "energy_corr": 0.8}}, "report": "x/report.html"})
    assert cli.main(["compare", "a.wav", "b.sc", "-o", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "piano" in out and "0.50" in out and "report.html" in out


def test_server_serves_compare_reports(tmp_path, monkeypatch):
    from soundcode import server
    d = tmp_path / "out" / "compare" / "song"
    d.mkdir(parents=True)
    (d / "report.html").write_text("<html>ok</html>")
    monkeypatch.setenv("SOUNDCODE_ROOT", str(tmp_path))
    assert server._compare_file("song/report.html") == d / "report.html"
    assert server._compare_file("../../etc/passwd") is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: 4 new failures (`stem_for_stream`, `run`, `invalid choice: 'compare'` and `_compare_file` are missing).

- [ ] **Step 3: Implement** by appending to `compare.py`. Add `import json`, `from pathlib import Path` at the top.

```python
_NAME_STEM = {"vox": "lead_vocals", "bvox": "backing_vocals", "drums": "drums",
              "bass": "bass", "guitar": "guitar", "piano": "piano", "other": "other"}


def stem_for_stream(doc, name: str) -> str:
    from .separate import STEMS

    s = doc.stream(name)
    if s is not None and s.meta.get("stem") in STEMS:
        return s.meta["stem"]
    track = name.split(".", 1)[1] if "." in name else name
    return _NAME_STEM.get(track, "other")


def transcribe(path) -> tuple[np.ndarray, np.ndarray]:
    from basic_pitch.inference import predict

    _, _, events = predict(str(path))
    if not events:
        return np.zeros((0, 2)), np.zeros(0)
    events = sorted(events, key=lambda e: e[0])
    iv = np.array([[e[0], max(e[1], e[0] + 0.01)] for e in events])
    hz = 440.0 * 2 ** ((np.array([e[2] for e in events]) - 69) / 12)
    return iv, hz


def _mono(y: np.ndarray, sr: int) -> np.ndarray:
    """(n, 2) render output -> mono at SR. (separate.read_stereo is (2, n): use .mean(0).)"""
    import librosa

    m = y.mean(1) if y.ndim == 2 else y
    return librosa.resample(m.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else m


def _write_wav(path: Path, y_mono: np.ndarray) -> None:
    import soundfile as sf
    sf.write(str(path), y_mono, SR, subtype="FLOAT")


def run(original, sc_or_wav, out_dir, engine: str = "sf2",
        render_streams=None, stems_dir=None) -> dict:
    from .parser import parse_file
    from .separate import STEMS, default_out_dir, read_stereo, separate

    original, sc_or_wav, out = Path(original), Path(sc_or_wav), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes_side: dict[str, str] = {}

    # original stems
    sd = Path(stems_dir) if stems_dir else default_out_dir(original)
    if not (sd / "manifest.json").exists() and stems_dir is None:
        separate(original, sd)
    orig = {s: read_stereo(sd / f"{s}.wav", sr=SR).mean(0) for s in STEMS}
    n = max(len(y) for y in orig.values())

    # rendered side, per stem
    rend = {s: np.zeros(n, np.float32) for s in STEMS}
    doc = None
    if sc_or_wav.suffix == ".sc":
        doc = parse_file(str(sc_or_wav))
        if render_streams is None:
            if engine == "mock":
                raise ValueError("compare needs per-stream renders; use --engine sf2")
            from .render_sf import render_streams
        streams = render_streams(doc, sr=doc.sample_rate)
        for name, y in streams.items():
            m = _mono(y, doc.sample_rate)[:n]
            rend[stem_for_stream(doc, name)][:len(m)] += m
    else:
        rs = default_out_dir(sc_or_wav)
        separate(sc_or_wav, rs)
        notes_side["render"] = "stems from re-separation (less reliable pairing)"
        for s in STEMS:
            m = read_stereo(rs / f"{s}.wav", sr=SR).mean(0)[:n]
            rend[s][:len(m)] = m

    duration = n / SR
    blocks = blocks_from_grid(doc, duration) if doc is not None else blocks_from_grid(
        type("D", (), {"stream": lambda self, k: None})(), duration)
    report: dict = {"original": str(original), "render": str(sc_or_wav), "engine": engine,
                    "notes": notes_side, "blocks": blocks, "stems": {}}
    for s in STEMS:
        yo, yr = orig[s][:n], rend[s][:n]
        po, pr = out / f"stem-{s}.orig.wav", out / f"stem-{s}.render.wav"
        _write_wav(po, yo)
        _write_wav(pr, yr)
        lo, lr = level_db(yo), level_db(yr)
        row = {"level_orig_db": lo, "level_render_db": lr,
               "level_diff_db": (lr - lo) if lo is not None and lr is not None else None,
               "onset_f1": onset_f1(yo, yr), "energy_corr": energy_corr(yo, yr),
               "chroma_blocks": chroma_blocks(yo, yr, blocks)}
        vals = [v for v in row["chroma_blocks"] if v is not None]
        row["chroma"] = float(np.mean(vals)) if vals else None
        if s == "drums":
            row["notes_f1"] = row["notes_f1_octave"] = None
        else:
            (io, ho), (ir, hr) = transcribe(po), transcribe(pr)
            row["notes_f1"] = note_f1(io, ho, ir, hr)
            row["notes_f1_octave"] = note_f1(io, ho, ir, hr, octave_agnostic=True)
        row["sound"] = None                      # CLAP: enabled in Plan 2
        report["stems"][s] = row

    _plots(out, orig, rend, report)
    (out / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    _html(out, report)
    report["report"] = str(out / "report.html")
    return report


def _plots(out: Path, orig: dict, rend: dict, report: dict) -> None:
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for s in orig:
        fig, ax = plt.subplots(1, 4, figsize=(22, 3.2))
        for j, (y, tag) in enumerate([(orig[s], "original"), (rend[s], "render")]):
            S = librosa.amplitude_to_db(np.abs(librosa.stft(y + 1e-9, hop_length=_HOP)), ref=np.max)
            librosa.display.specshow(S, sr=SR, hop_length=_HOP, x_axis="time", y_axis="log",
                                     ax=ax[j], vmin=-80)
            ax[j].set_title(f"{s} — {tag} spectrogram")
            C = librosa.feature.chroma_cqt(y=y + 1e-9, sr=SR, hop_length=_HOP)
            librosa.display.specshow(C, sr=SR, hop_length=_HOP, x_axis="time", y_axis="chroma",
                                     ax=ax[2 + j])
            ax[2 + j].set_title(f"{s} — {tag} chroma")
        fig.tight_layout()
        fig.savefig(out / f"{s}.png", dpi=60)
        plt.close(fig)

    names = list(report["stems"])
    grid = np.array([[np.nan if v is None else v for v in report["stems"][s]["chroma_blocks"]]
                     for s in names], dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, grid.shape[1] * 0.3), 3))
    im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("bar (or 2 s block)")
    ax.set_title("chroma similarity: where the render diverges")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "bars.png", dpi=70)
    plt.close(fig)


def _fmt(v, spec=".2f"):
    return "—" if v is None else format(v, spec)


def _html(out: Path, report: dict) -> None:
    rows = []
    for s, r in report["stems"].items():
        rows.append(
            f"<tr><td>{s}</td><td>{_fmt(r['level_orig_db'], '.1f')}</td>"
            f"<td>{_fmt(r['level_render_db'], '.1f')}</td><td>{_fmt(r['level_diff_db'], '+.1f')}</td>"
            f"<td>{_fmt(r['notes_f1'])}</td><td>{_fmt(r['notes_f1_octave'])}</td>"
            f"<td>{_fmt(r['chroma'])}</td><td>{_fmt(r['onset_f1'])}</td><td>{_fmt(r['energy_corr'])}</td>"
            f"<td><audio controls preload=none src='stem-{s}.orig.wav'></audio></td>"
            f"<td><audio controls preload=none src='stem-{s}.render.wav'></audio></td></tr>")
    imgs = "".join(f"<h3>{s}</h3><img src='{s}.png' style='max-width:100%'>" for s in report["stems"])
    notes = "".join(f"<p><b>note:</b> {v}</p>" for v in report["notes"].values())
    (out / "report.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>compare — {Path(report['original']).name}</title>
<style>body{{font:14px system-ui;margin:16px}}td,th{{padding:4px 8px;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;overflow-x:auto;display:block}}</style>
<h1>{Path(report['original']).name} vs {Path(report['render']).name}</h1>{notes}
<table><tr><th>stem</th><th>orig dB</th><th>render dB</th><th>Δ dB</th><th>note F1</th>
<th>F1 any-oct</th><th>chroma</th><th>onset F1</th><th>energy r</th><th>original</th><th>render</th></tr>
{''.join(rows)}</table><h2>Where it diverges</h2><img src='bars.png' style='max-width:100%'>{imgs}
""", encoding="utf-8")
```

In `cli.py`, put the parser next to `p_sep`:

```python
    p_cmp = sub.add_parser("compare", help="score a render against the original, per stem")
    p_cmp.add_argument("original")
    p_cmp.add_argument("render", help=".sc file (rendered per stream) or a rendered .wav")
    p_cmp.add_argument("-o", "--out", default=None, help="default out/compare/<name>")
    p_cmp.add_argument("--engine", choices=("sf2", "mock"), default="sf2")
```

Put the handler before `if args.cmd == "encode":`:

```python
        if args.cmd == "compare":
            from . import compare as cmp

            out = args.out or str(Path("out") / "compare" / Path(args.original).stem)
            rep = cmp.run(args.original, args.render, out, engine=args.engine)
            f = cmp._fmt
            print(f"{'stem':<15}{'orig dB':>8}{'rend dB':>8}{'Δ dB':>7}{'noteF1':>8}"
                  f"{'anyOct':>8}{'chroma':>8}{'onsetF1':>8}{'energy':>8}")
            for s, r in rep["stems"].items():
                print(f"{s:<15}{f(r['level_orig_db'], '.1f'):>8}{f(r['level_render_db'], '.1f'):>8}"
                      f"{f(r['level_diff_db'], '+.1f'):>7}{f(r['notes_f1']):>8}"
                      f"{f(r['notes_f1_octave']):>8}{f(r['chroma']):>8}"
                      f"{f(r['onset_f1']):>8}{f(r['energy_corr']):>8}")
            print(f"report: {rep['report']}")
            return 0
```

In `server.py`, add a helper next to `_sc_sources`:

```python
def _compare_file(rel: str) -> Path | None:
    base = (_project_root() / "out" / "compare").resolve()
    target = (base / rel).resolve()
    if base not in target.parents or not target.is_file():
        return None
    return target
```

Add a route in `do_GET` before the `/audio/` branch:

```python
        elif route.startswith("/compare/"):
            target = _compare_file(route[len("/compare/"):])
            if target is None:
                self.send_error(404)
                return
            ctype = {".html": "text/html; charset=utf-8", ".png": "image/png",
                     ".json": "application/json", ".wav": "audio/wav"}.get(target.suffix,
                                                                            "application/octet-stream")
            self._send_file(target, ctype)
```

In `_discover_tracks`, skip compare stem files so they don't flood the track list: add `if "compare" in path.relative_to(out).parts: continue` next to the `_work` skip.

- [ ] **Step 4: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: `74 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/compare.py src/soundcode/cli.py src/soundcode/server.py tests/test_compare.py
git commit -m "compare: per-stem report (json, html, plots), CLI and server route"
```

---

### Task 7: Baseline

**Files:**
- Create: `docs/results/2026-09-26-compare-baseline.md`

- [ ] **Step 1: Make the six clips.** Produce 30 s clips for the four test clips (already 30 s in `audio/test/`) and *The River* (0–30 s). Write the River clip to `audio/test/river-30s.wav`:

```bash
.venv/bin/python -c "
import librosa, soundfile as sf
y, sr = librosa.load('audio/uploads/The River-JordanFelix.mp3', sr=44100, mono=False, duration=30.0)
sf.write('audio/test/river-30s.wav', y.T, sr)"
```

- [ ] **Step 2: Encode with the current encoder** (before Part B) into `out/sc/baseline/`:

```bash
for f in audio/test/*.mp3 audio/test/river-30s.wav; do n=$(basename "${f%.*}"); \
  .venv/bin/soundcode encode "$f" -o "out/sc/baseline/$n.sc" --workdir out/work >/dev/null 2>&1; done; ls out/sc/baseline
```

Expected: 5 `.sc` files.

- [ ] **Step 3: Compare each** into `out/compare/baseline/<name>/` and keep the CLI tables:

```bash
for f in audio/test/*.mp3 audio/test/river-30s.wav; do n=$(basename "${f%.*}"); \
  echo "## $n"; .venv/bin/soundcode compare "$f" "out/sc/baseline/$n.sc" -o "out/compare/baseline/$n" 2>/dev/null | grep -v INFO; done \
  | tee /tmp/baseline.txt
```

Expected: a table per song. The River's bass shows a large positive Δ dB, as in spec §0.

- [ ] **Step 4: Write the results file.** Create `docs/results/2026-09-26-compare-baseline.md` with the six tables from `/tmp/baseline.txt` under per-song headings, plus one line per song naming its worst stem. Read `out/compare/baseline/river-30s/bars.png` and `piano.png` and record in one sentence each what they show.

- [ ] **Step 5: Commit**

```bash
git add docs/results/2026-09-26-compare-baseline.md
git commit -m "docs: compare baseline before encoder fixes"
```

---

### Task 8: Pitch fix (B4)

**Files:**
- Modify: `src/soundcode/encode.py:~438-442` (the bend→cents line in `stage_notes_poly`)
- Create: `tests/test_transcription_calibration.py`

**Interfaces:**
- Produces: `encode.bend_to_cents(bend) -> float`, a pure function, and `encode.BASIC_PITCH_BEND_ZERO = 1.0`.

- [ ] **Step 1: Write the failing tests**:

```python
"""basic-pitch reports +1 bend bin on in-tune notes; the encoder must not."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import encode as enc  # noqa: E402


def test_bend_zero_point_is_one_bin():
    assert enc.bend_to_cents([1, 1, 1]) == pytest.approx(0.0)
    assert enc.bend_to_cents([2, 2]) == pytest.approx(100 / 3)
    assert enc.bend_to_cents(None) == 0.0


basic_pitch = pytest.importorskip("basic_pitch")


def _tone_cents(tmp_path, hz):
    from basic_pitch.inference import predict
    sr = 22050
    t = np.arange(int(sr * 1.5)) / sr
    y = 0.4 * np.sin(2 * np.pi * hz * t) * np.minimum(1, (1.5 - t) * 10)
    p = tmp_path / f"{hz:.2f}.wav"
    sf.write(str(p), y, sr)
    _, _, ev = predict(str(p))
    start, end, midi, amp, bend = max(ev, key=lambda e: e[1] - e[0])
    true = 1200 * np.log2(hz / 440.0) + 6900
    return midi * 100 + enc.bend_to_cents(bend) - true


@pytest.mark.parametrize("hz", [110.0, 261.63, 440.0])
def test_in_tune_tones_transcribe_in_tune(tmp_path, hz):
    assert abs(_tone_cents(tmp_path, hz)) <= 10


def test_a_sharp_tone_still_reads_sharp(tmp_path):
    hz = 440.0 * 2 ** (30 / 1200)
    assert 15 <= _tone_cents(tmp_path, hz) + 30 <= 45
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_transcription_calibration.py -q`
Expected: failures with `AttributeError: ... 'bend_to_cents'`.

- [ ] **Step 3: Implement.** In `encode.py`, add near `_DRUM_VOICES`:

```python
# basic-pitch reports pitch bend in 1/3-semitone bins, and an in-tune note
# reads +1 bin (measured on pure tones), so that is the zero point.
BASIC_PITCH_BEND_ZERO = 1.0


def bend_to_cents(bend) -> float:
    if bend is None or not np.size(bend):
        return 0.0
    return (float(np.mean(bend)) - BASIC_PITCH_BEND_ZERO) * 100.0 / 3.0
```

In `stage_notes_poly`, replace the cents computation:

```python
        cents = midi * 100 + bend_to_cents(bend)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (74 + 5 = 79). If `test_a_sharp_tone_still_reads_sharp` fails because basic-pitch quantises bends coarsely, that means real detuning below one bin is unrepresentable. Record it in the ledger as a Ruling, relax that one test to "reads ≥ 0 c", and keep the in-tune tests strict.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/encode.py tests/test_transcription_calibration.py
git commit -m "encode: correct basic-pitch bend zero point (+33 c bias)"
```

---

### Task 9: Loudness gate, stricter notes, stem/level metadata (B1–B3)

**Files:**
- Modify: `src/soundcode/encode.py` (`Stage`, `stage_notes_poly`, `stage_percussion`, `encode()`, the header writer)
- Test: `tests/test_encode_gate.py`

**Interfaces:**
- Produces:
  - `GATE_ABS_DB = -50.0`, `GATE_REL_DB = 35.0`, `GATE_BLOCK_S = 2.0`, `AMP_FLOOR = 0.40`, `ONSET_THRESHOLD = 0.6`, `FRAME_THRESHOLD = 0.4`
  - `active_blocks(stem: np.ndarray, mix: np.ndarray, sr: int) -> np.ndarray[bool]`, one entry per 2 s block
  - `merge_same_pitch(events: list[tuple]) -> list[tuple]`, which merges basic-pitch events with the same `midi` that overlap or touch (gap < 30 ms)
  - `active_level_db(stem: np.ndarray, mask: np.ndarray, sr: int) -> float | None`
  - `Stage` gains `stem: str = ""` and `level_db: float | None = None`. The header writer emits `meta stem=<stem> level=<x.x>dB` when they are set.
  - `stage_notes_poly(stem, name, grid, min_amp=AMP_FLOOR, mix=None, sr=44100, stem_name="")`, whose new keyword parameters default to the old behaviour when `mix is None`.

- [ ] **Step 1: Write the failing tests** in `tests/test_encode_gate.py`:

```python
"""Loudness gate and note filtering (spec Part B1-B3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import encode as enc  # noqa: E402

SR = 22050


def noise(secs, db):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(secs * SR)) * 10 ** (db / 20)).astype(np.float32)


def test_gate_closes_on_silence_and_bleed_opens_on_real_parts():
    mix = noise(8, -12)
    stem = np.concatenate([noise(4, -70), noise(4, -20)])     # silent-ish, then playing
    mask = enc.active_blocks(stem, mix, SR)
    assert mask.tolist() == [False, False, True, True]


def test_gate_keeps_a_quiet_but_real_part():
    mix = noise(4, -30)                                       # a soft passage
    stem = noise(4, -45)                                      # 15 dB under a soft mix
    assert enc.active_blocks(stem, mix, SR).all()


def test_gate_closes_on_bleed_far_below_a_loud_mix():
    mix = noise(4, -10)
    stem = noise(4, -48)                                      # 38 dB under the mix
    assert not enc.active_blocks(stem, mix, SR).any()


def test_same_pitch_overlaps_merge():
    ev = [(0.0, 0.5, 60, 0.9, [1]), (0.48, 1.0, 60, 0.8, [1]), (0.5, 1.0, 64, 0.7, [1])]
    merged = enc.merge_same_pitch(ev)
    assert [(e[0], e[1], e[2]) for e in merged] == [(0.0, 1.0, 60), (0.5, 1.0, 64)]


def test_active_level_uses_only_active_blocks():
    stem = np.concatenate([np.zeros(int(2 * SR), np.float32), noise(2, -20)])
    mask = np.array([False, True])
    assert abs(enc.active_level_db(stem, mask, SR) - (-20)) < 0.5
    assert enc.active_level_db(stem, np.array([False, False]), SR) is None


def test_header_writes_stem_and_level_meta():
    st = enc.Stage("notes.bass", src="basic-pitch:onnx", ok=True, stem="bass", level_db=-31.24)
    st.lines = ["1:1.0  E1  1.0b 90"]
    text = "\n".join(enc._stage_lines(st))
    assert "meta    stem=bass  level=-31.2dB" in text
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_encode_gate.py -q`
Expected: failures (`active_blocks`, `merge_same_pitch`, `active_level_db`, `_stage_lines` and the `Stage` fields are missing).

- [ ] **Step 3: Implement** in `encode.py`.

(a) Add fields to `Stage`: `stem: str = ""` and `level_db: float | None = None`.

(b) Add the helpers after `bend_to_cents`:

```python
# Loudness gate: a 2 s block of a stem is transcribed only if it is above an
# absolute floor AND within GATE_REL_DB of the full mix. Separation leaves
# faint bleed in every stem; basic-pitch's floor is relative to the stem's own
# peak, so a near-silent stem otherwise yields confident invented notes.
GATE_ABS_DB = -50.0
GATE_REL_DB = 35.0
GATE_BLOCK_S = 2.0
AMP_FLOOR = 0.40
ONSET_THRESHOLD = 0.6
FRAME_THRESHOLD = 0.4


def _block_db(y: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(np.ceil(len(y) / (GATE_BLOCK_S * sr))))
    out = np.full(n, -np.inf)
    for i in range(n):
        seg = y[int(i * GATE_BLOCK_S * sr): int((i + 1) * GATE_BLOCK_S * sr)]
        r = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))) if seg.size else 0.0
        out[i] = 20 * math.log10(r) if r > 0 else -np.inf
    return out


def active_blocks(stem: np.ndarray, mix: np.ndarray, sr: int) -> np.ndarray:
    s, m = _block_db(stem, sr), _block_db(mix, sr)
    m = np.pad(m, (0, max(0, len(s) - len(m))), constant_values=-np.inf)[: len(s)]
    return (s >= GATE_ABS_DB) & (s >= m - GATE_REL_DB)


def active_level_db(stem: np.ndarray, mask: np.ndarray, sr: int) -> float | None:
    parts = [stem[int(i * GATE_BLOCK_S * sr): int((i + 1) * GATE_BLOCK_S * sr)]
             for i, on in enumerate(mask) if on]
    if not parts:
        return None
    y = np.concatenate(parts)
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20 * math.log10(r) if r > 0 else None


def merge_same_pitch(events: list[tuple]) -> list[tuple]:
    out: list[list] = []
    last: dict[int, int] = {}
    for e in sorted(events, key=lambda e: e[0]):
        start, end, midi, amp, bend = e
        j = last.get(midi)
        if j is not None and start <= out[j][1] + 0.03:
            out[j][1] = max(out[j][1], end)
            out[j][3] = max(out[j][3], amp)
            continue
        last[midi] = len(out)
        out.append([start, end, midi, amp, bend])
    return [tuple(e) for e in out]
```

(c) Factor the header writer out of `encode()` into `_stage_lines(st) -> list[str]`. Its body is the current loop body, plus the stem/level line after the `src/conf` meta line:

```python
def _stage_lines(st: Stage) -> list[str]:
    out = [f":{st.name}", f"meta    src={st.src}  conf={st.conf:.2f}"]
    if st.stem:
        level = f"  level={st.level_db:.1f}dB" if st.level_db is not None else ""
        out.append(f"meta    stem={st.stem}{level}")
    out += [f'meta    warn="{w}"' for w in st.warns]
    return out + st.lines + [""]
```

In `encode()`, the loop becomes: `if not st.ok: ...omitted...; continue` and then `lines += _stage_lines(st)`.

(d) Change the signature of `stage_notes_poly` to `(stem, name, grid, min_amp=AMP_FLOOR, mix=None, sr=44100, stem_name="")`. Change the `predict` call to `predict(str(stem), onset_threshold=ONSET_THRESHOLD, frame_threshold=FRAME_THRESHOLD)`. Then, after `events` is obtained:

```python
    st.stem = stem_name
    mask = None
    if mix is not None:
        import librosa
        y_stem, _ = librosa.load(str(stem), sr=sr, mono=True)
        mask = active_blocks(y_stem, mix, sr)
        st.level_db = active_level_db(y_stem, mask, sr)
        if not mask.any():
            st.warns = ["stem silent (below the loudness gate throughout)"]
            return st
        events = [e for e in events
                  if mask[min(int(e[0] / GATE_BLOCK_S), len(mask) - 1)]]
    events = merge_same_pitch(events)
    if not events:
        st.warns.append("no notes above the loudness gate")
        return st
```

(e) In `encode()`, compute `mono_mix = y.mean(0)` once. Pass `mix=mono_mix, sr=sr, stem_name=<stem>` to each `stage_notes_poly` call, using `bass`/`lead_vocals`/`other`/`guitar`/`piano` as `stem_name`. The `vox` call uses `stem_name="lead_vocals"`, even when `encoder_stems` substituted the full vocals mix.

In `stage_percussion`, set `st.stem = "drums"`. When the stem exists, set `st.level_db = active_level_db(d, active_blocks(d, y.mean(0), sr), sr)`.

- [ ] **Step 4: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (79 + 6 = 85).

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/encode.py tests/test_encode_gate.py
git commit -m "encode: loudness gate, stricter note filtering, stem/level metadata"
```

---

### Task 10: Re-measure, tune, acceptance

**Files:**
- Modify: `src/soundcode/encode.py` (constants only, if tuning changes them)
- Create: `docs/results/2026-09-26-compare-after-fixes.md`
- Modify: `README.md` (Status: a short Plan 1 results section)

- [ ] **Step 1: Re-encode and compare** the six clips into `out/sc/after/` and `out/compare/after/`, using the Task 7 commands with `baseline` → `after`. Tee to `/tmp/after.txt`.

- [ ] **Step 2: Check the acceptance bars** from spec Evaluation item 0, per song:
  - every stem's |Δ dB| ≤ 3 dB, among stems where both sides have a level;
  - the River bass has no notes before 17 s. Check with `grep -n "^[1-9]:" out/sc/after/river-30s.sc` on the `:notes.bass` block; the earliest bar must start at or after 17 s by the grid. The River `other` stem is omitted (`# :notes.other omitted — stem silent`);
  - mean `notes_f1` over the pitched stems is above the baseline for every song.

- [ ] **Step 3: Tune only if a bar fails.** Adjust one constant at a time: `GATE_REL_DB` first (±5 dB), then `AMP_FLOOR` (±0.05), then `ONSET_THRESHOLD` (±0.05). Re-run steps 1–2 after each change, and record every tried value and its effect in the results file. Stop when all bars pass, or after 6 tries. In the latter case, record which bar still fails as a Ruling in the ledger and in the results file.

- [ ] **Step 4: Read the report images.** Read `out/compare/after/river-30s/bars.png` and the worst stem's PNG. Write one paragraph in the results file on what still diverges and why; this feeds Plan 2.

- [ ] **Step 5: Write the results file.** Put the before/after tables in `docs/results/2026-09-26-compare-after-fixes.md`. Add a README Status paragraph: Plan 1 done, render and compare commands, and a link to both results files.

- [ ] **Step 6: Run all tests, then commit**

```bash
.venv/bin/python -m pytest -q
git add src/soundcode/encode.py docs/results/2026-09-26-compare-after-fixes.md README.md
git commit -m "docs: compare results after encoder fixes; tuned gate constants"
```
