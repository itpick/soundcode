# tsumugi Transcription Backbone — Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The encoder transcribes every stem with tsumugi and writes one note stream per identified instrument (drums with claps included). Confidence comes from agreement between the per-stem refinement and a mix-level vote, so the rebuilt song uses the right instruments with performed timing.

**Architecture:**
- tsumugi lives in `external/tsumugi`, in its own venv, pinned to commit `020edc1`.
- `src/soundcode/tsumugi.py` runs its CLIs as subprocesses and returns MIDI and JSON paths.
- Pure modules turn that output into `.sc`:
  - `tsumugi_sc.py` converts MIDI to stream lines;
  - `inventory.py` decides the instrument per stem by voting.
- `encode.py` gets one new stage that uses these and falls back to the existing basic-pitch path on any tsumugi failure.

**Tech Stack:** Python 3.11, pretty_midi (already installed via basic-pitch), numpy, subprocess; tsumugi (MIT; torch 2.13 in its own venv, MPS).

**Spec:** `docs/superpowers/specs/2026-09-26-instruments-design.md`, Part C (revised) and Evaluation item 1. Research: `docs/research/2026-09-26-open-song-analyzers.md`.

## Global Constraints

- Python `>=3.11,<3.12` for soundcode. tsumugi runs in its own venv at `external/tsumugi/.venv` (Python 3.11 via `uv sync --locked --python 3.11`).
- tsumugi is pinned to commit `020edc1`, URL `https://github.com/anime-song/tsumugi.git`, license MIT. `$SOUNDCODE_TSUMUGI` overrides the checkout path.
- `external/`, `out/`, `models/` and `audio/` stay gitignored. Never commit audio, MIDI or lyrics.
- The disk is nearly full (about 7 GB free). The tsumugi venv is about 700 MB and its checkpoints are 55 MB each. Check `df -h /System/Volumes/Data` before installing, and stop if free space would drop below 3 GB.
- The encoder is fail-soft (spec §5): a tsumugi failure means the basic-pitch path runs, with `meta warn`.
- Note positions are written as `bar:beat` with 3 decimal beats. A note before the first downbeat is written as `@<seconds>`.
- Vocals are transcribed but still left out of renders unless `--with-vocals`.
- Open, free tools only. No `Co-Authored-By` or other Claude attribution in commits.

## Review Focus

1. **tsumugi emits zero tracks for a stem** (silence, very short input) **or a track with 1–2 notes.** No stream is written, with an `# omitted — …` line, and nothing crashes. (Task 3 tests.)
2. **Notes before the first downbeat, or past the end of the grid.** The first are written as `@seconds`; nothing is dropped silently and nothing gets a negative beat. (Task 3 test.)
3. **Two stems resolve to the same instrument** (e.g. the piano and "guitar" stems both become `keys.piano`). Stream names must not clash (`:notes.piano`, `:notes.piano.guitar`). (Task 3 and Task 5 tests.)
4. **The mix-level vote has too few matched notes to decide** (fewer than 5). The stem keeps its refinement result, and the confidence says so. (Task 4 test.)
5. **tsumugi missing, a subprocess non-zero exit, or unparseable JSON.** A `TsumugiError` naming the step, and the encoder falls back per stem rather than aborting the whole encode. (Task 1 and Task 5 tests.)

---

### Task 1: Installer and subprocess wrapper

**Files:**
- Create: `scripts/install_tsumugi.sh`
- Create: `src/soundcode/tsumugi.py`
- Test: `tests/test_tsumugi.py`

**Interfaces:**
- Produces:
  - `class TsumugiError(RuntimeError)`
  - `PINNED_COMMIT = "020edc1"`
  - `STEM_MODEL = {"piano": "default", "guitar": "guitar_v1_5", "bass": "bass_v2", "other": "other_v1_5", "drums": "drums_v1_5", "lead_vocals": "vocal_harmony_v1_5", "backing_vocals": "vocal_harmony_v1_5", "vocals": "vocal_harmony_v1_5"}`
  - `REFINE_STEM = {"piano": "piano", "guitar": "guitar", "bass": "bass", "other": "other", "lead_vocals": "vocals", "backing_vocals": "vocals", "vocals": "vocals"}`
  - `home() -> Path`, which is `$SOUNDCODE_TSUMUGI` or `<project root>/external/tsumugi`, raising `TsumugiError` if `.venv/bin/python` is missing
  - `_run(module: str, args: list[str]) -> subprocess.CompletedProcess`, which raises `TsumugiError(f"{module}: {last stderr line}")` on a non-zero exit
  - `transcribe(audio: Path, out_midi: Path, model: str) -> Path`
  - `refine(audio: Path, midi: Path, stem_name: str, out_json: Path) -> list[tuple[str, float]]`, the top candidates from `global_candidates`
  - `add_velocity(midi: Path, stem_wav: Path, out_midi: Path) -> Path`

- [ ] **Step 1: Write the failing tests** in `tests/test_tsumugi.py`:

```python
"""tsumugi subprocess wrapper (spec Part C)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import tsumugi as ts  # noqa: E402


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    py = tmp_path / "tsumugi" / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setenv("SOUNDCODE_TSUMUGI", str(tmp_path / "tsumugi"))
    return tmp_path / "tsumugi"


def test_missing_install_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCODE_TSUMUGI", str(tmp_path / "nope"))
    with pytest.raises(ts.TsumugiError, match="install_tsumugi.sh"):
        ts.home()


def test_transcribe_builds_the_cli_call(fake_home, monkeypatch, tmp_path):
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"], seen["cwd"] = cmd, kw.get("cwd")
        Path(cmd[cmd.index("--output-midi") + 1]).write_bytes(b"MThd")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    out = ts.transcribe(tmp_path / "piano.wav", tmp_path / "piano.mid", "default")
    assert out.exists()
    cmd = seen["cmd"]
    assert cmd[0].endswith(".venv/bin/python") and cmd[1:3] == ["-m", "instrument_agnostic_amt.amt.cli.infer"]
    assert cmd[cmd.index("--type") + 1] == "default"
    assert "--device" in cmd and "--disable-tqdm" in cmd
    assert seen["cwd"] == str(fake_home)


def test_nonzero_exit_names_the_step(fake_home, monkeypatch, tmp_path):
    monkeypatch.setattr(ts.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom\nRuntimeError: bad audio"))
    with pytest.raises(ts.TsumugiError, match="amt.cli.infer: RuntimeError: bad audio"):
        ts.transcribe(tmp_path / "x.wav", tmp_path / "x.mid", "default")


def test_refine_reads_global_candidates(fake_home, monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        report = {"global_candidates": [{"class_name": "piano", "probability": 0.73},
                                        {"class_name": "electric_piano", "probability": 0.27}]}
        Path(cmd[cmd.index("--output-json") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    got = ts.refine(tmp_path / "g.wav", tmp_path / "g.mid", "piano", tmp_path / "g.json")
    assert got == [("piano", 0.73), ("electric_piano", 0.27)]


def test_refine_bad_json_is_a_tsumugi_error(fake_home, monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        Path(cmd[cmd.index("--output-json") + 1]).write_text("{not json")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    with pytest.raises(ts.TsumugiError, match="refine"):
        ts.refine(tmp_path / "g.wav", tmp_path / "g.mid", "piano", tmp_path / "g.json")


def test_stem_model_table_covers_every_separation_stem():
    from soundcode.separate import STEMS
    assert set(STEMS) <= set(ts.STEM_MODEL)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_tsumugi.py -q`
Expected: collection error `cannot import name 'tsumugi'`.

- [ ] **Step 3: Implement** `src/soundcode/tsumugi.py`:

```python
"""tsumugi (anime-song/tsumugi, MIT): instrument-labelled transcription.

Runs in its own checkout and venv (external/tsumugi, pinned) as a subprocess,
because it pins its own PyTorch. Every failure is a TsumugiError naming the
step; the encoder falls back per stem.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PINNED_COMMIT = "020edc1"
STEM_MODEL = {"piano": "default", "guitar": "guitar_v1_5", "bass": "bass_v2",
              "other": "other_v1_5", "drums": "drums_v1_5",
              "lead_vocals": "vocal_harmony_v1_5", "backing_vocals": "vocal_harmony_v1_5",
              "vocals": "vocal_harmony_v1_5"}
REFINE_STEM = {"piano": "piano", "guitar": "guitar", "bass": "bass", "other": "other",
               "lead_vocals": "vocals", "backing_vocals": "vocals", "vocals": "vocals"}
DEVICE = os.environ.get("SOUNDCODE_TSUMUGI_DEVICE", "mps")


class TsumugiError(RuntimeError):
    """A tsumugi step failed; the message names the step."""


def home() -> Path:
    root = Path(os.environ.get("SOUNDCODE_TSUMUGI",
                               Path(__file__).resolve().parents[2] / "external" / "tsumugi"))
    if not (root / ".venv" / "bin" / "python").exists():
        raise TsumugiError(f"tsumugi not installed at {root}; run scripts/install_tsumugi.sh")
    return root


def _run(module: str, args: list[str]) -> subprocess.CompletedProcess:
    root = home()
    cmd = [str(root / ".venv" / "bin" / "python"), "-m", module, *args]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        short = module.replace("instrument_agnostic_amt.", "")
        raise TsumugiError(f"{short}: {tail[-1] if tail else 'exit ' + str(proc.returncode)}")
    return proc


def transcribe(audio: Path, out_midi: Path, model: str) -> Path:
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    _run("instrument_agnostic_amt.amt.cli.infer",
         ["--audio", str(audio), "--output-midi", str(out_midi), "--type", model,
          "--device", DEVICE, "--disable-tqdm"])
    if not out_midi.exists():
        raise TsumugiError(f"amt.cli.infer: no MIDI written for {audio.name}")
    return out_midi


def refine(audio: Path, midi: Path, stem_name: str, out_json: Path) -> list[tuple[str, float]]:
    _run("instrument_agnostic_amt.instrument_refinement.cli.infer",
         ["--audio", str(audio), "--midi", str(midi), "--stem-name", stem_name,
          "--mode", "single", "--output-json", str(out_json), "--device", DEVICE,
          "--disable-tqdm"])
    try:
        report = json.loads(out_json.read_text())
        return [(c["class_name"], float(c["probability"])) for c in report["global_candidates"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TsumugiError(f"refine: unreadable report {out_json.name} ({exc})") from exc


def add_velocity(midi: Path, stem_wav: Path, out_midi: Path) -> Path:
    _run("instrument_agnostic_amt.velocity.cli.infer_velocity",
         ["--midi", str(midi), "--stem-files", str(stem_wav), "--output-midi", str(out_midi),
          "--device", DEVICE])
    if not out_midi.exists():
        raise TsumugiError(f"velocity: no MIDI written for {midi.name}")
    return out_midi
```

Create `scripts/install_tsumugi.sh`:

```bash
#!/bin/zsh
# Install tsumugi (MIT) into external/tsumugi at the tested commit, with its own venv.
set -e
ROOT=${0:A:h:h}
DEST=${SOUNDCODE_TSUMUGI:-$ROOT/external/tsumugi}
COMMIT=020edc1
free_gb=$(df -g /System/Volumes/Data | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt 3 ]; then echo "only ${free_gb} GB free; need ~1 GB (stop at 3 GB)"; exit 1; fi
[ -d "$DEST/.git" ] || git clone https://github.com/anime-song/tsumugi.git "$DEST"
git -C "$DEST" fetch --quiet origin && git -C "$DEST" checkout --quiet "$COMMIT"
(cd "$DEST" && uv sync --locked --python 3.11)
echo "tsumugi $COMMIT ready at $DEST"
```

Then `chmod +x scripts/install_tsumugi.sh`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_tsumugi.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Install for real and smoke-test**

Run: `scripts/install_tsumugi.sh && .venv/bin/python -c "from pathlib import Path; from soundcode import tsumugi as t; print(t.transcribe(Path('out/stems/river-30s/piano.wav'), Path('/tmp/sm/piano.mid'), 'default'))"`

Expected: `tsumugi 020edc1 ready …`, then the MIDI path is printed. The first run downloads a 55 MB checkpoint. If the scratchpad already has a clone (`…/scratchpad/tsumugi`), copying it saves the download: `cp -R` it to `external/tsumugi`, then run the script, which re-checks out the commit and re-syncs.

- [ ] **Step 6: Commit**

```bash
git add scripts/install_tsumugi.sh src/soundcode/tsumugi.py tests/test_tsumugi.py
git commit -m "tsumugi: pinned installer and subprocess wrapper"
```

---

### Task 2: Class map, fine-grained GM targets, full drum kit

**Files:**
- Create: `src/soundcode/data/tsumugi_classes.json`
- Modify: `src/soundcode/gm.py`
- Test: `tests/test_render_sf.py` (append)

**Interfaces:**
- Produces:
  - `gm.TSUMUGI: dict[str, dict]`, each entry with keys `inst` (our vocab), `program` (int), `family` (our family) and `programs` (list[int])
  - `gm.tsumugi_class_for_program(program: int) -> str`
  - `gm.target_for` resolves an `inst=` found among `TSUMUGI` entries' `inst` to that entry's program before falling back to the family
  - `DRUM_NOTES` covers the GM kit (keys below), and `gm.drum_voice(pitch: int) -> str` is its inverse (unknown pitch → `f"gm{pitch}"`)

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- tsumugi classes + full drum kit -------------------------------------------------

def test_every_tsumugi_class_has_an_inst_program_and_family():
    assert len(gm.TSUMUGI) == 36
    for name, e in gm.TSUMUGI.items():
        assert set(e) >= {"inst", "program", "family", "programs"}, name
        assert 0 <= e["program"] <= 127


def test_fine_grained_inst_picks_its_own_program():
    assert gm.target_for("notes.x", "keys.ep").preset == 4
    assert gm.target_for("notes.x", "gtr.electric.distortion").preset == 30
    assert gm.target_for("notes.x", "bass.electric").preset == 33      # family table still works


def test_programs_map_back_to_tsumugi_classes():
    assert gm.tsumugi_class_for_program(5) == "electric_piano"
    assert gm.tsumugi_class_for_program(30) == "distorted_guitar"


def test_drum_voices_round_trip_including_clap():
    for pitch in (35, 36, 37, 38, 39, 42, 44, 46, 49, 51, 56):
        assert gm.drum_note(gm.drum_voice(pitch)) == (36 if pitch == 35 else pitch)
    assert gm.drum_voice(39) == "clap"
    assert gm.drum_voice(81) == "gm81" and gm.drum_note("gm81") == 81
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_render_sf.py -q -k "tsumugi or fine_grained or round_trip"`
Expected: 4 failures (`TSUMUGI` and `drum_voice` are missing).

- [ ] **Step 3: Create `src/soundcode/data/tsumugi_classes.json`.** It holds the 36 classes of tsumugi@020edc1 (`instrument_merge.json` labels plus `melody`, `vocal_harmony` and `wind_chimes`). `program` is the representative GM program we render with; `programs` are the members.

```json
{
  "accordion_family":      {"inst": "keys.accordion",          "family": "keys",    "program": 21,  "programs": [21, 23]},
  "acoustic_bass":         {"inst": "bass.upright",            "family": "bass",    "program": 32,  "programs": [32]},
  "acoustic_guitar":       {"inst": "gtr.acoustic",            "family": "gtr",     "program": 25,  "programs": [24, 25]},
  "brass":                 {"inst": "brass.section",           "family": "brass",   "program": 61,  "programs": [56, 57, 58, 59, 60, 61, 62, 63]},
  "choir":                 {"inst": "voice.choir",             "family": "voice",   "program": 52,  "programs": [52, 53, 54]},
  "chromatic_percussion":  {"inst": "mallet.vibraphone",       "family": "mallet",  "program": 11,  "programs": [8, 9, 10, 11, 12, 13, 14, 15]},
  "distorted_guitar":      {"inst": "gtr.electric.distortion", "family": "gtr",     "program": 30,  "programs": [29, 30]},
  "drums":                 {"inst": "drums.kit.standard",      "family": "drums",   "program": 0,   "programs": []},
  "electric_bass":         {"inst": "bass.electric",           "family": "bass",    "program": 33,  "programs": [33, 34, 35]},
  "electric_guitar_clean": {"inst": "gtr.electric.clean",      "family": "gtr",     "program": 27,  "programs": [26, 27]},
  "electric_guitar_muted": {"inst": "gtr.electric.muted",      "family": "gtr",     "program": 28,  "programs": [28]},
  "electric_piano":        {"inst": "keys.ep",                 "family": "keys",    "program": 4,   "programs": [4, 5, 2]},
  "ethnic":                {"inst": "ethnic.plucked",          "family": "other",   "program": 104, "programs": [104, 105, 106, 107, 108, 109, 110, 111]},
  "flute_pipe":            {"inst": "winds.flute",             "family": "winds",   "program": 73,  "programs": [72, 73, 74, 75, 76, 77, 78, 79]},
  "guitar_harmonics":      {"inst": "gtr.harmonics",           "family": "gtr",     "program": 31,  "programs": [31]},
  "harmonica":             {"inst": "winds.harmonica",         "family": "winds",   "program": 22,  "programs": [22]},
  "melody":                {"inst": "voice.lead",              "family": "voice",   "program": 53,  "programs": []},
  "orchestra_hit":         {"inst": "synth.stab.orchestra",    "family": "synth",   "program": 55,  "programs": [55]},
  "orchestral_harp":       {"inst": "strings.harp",            "family": "strings", "program": 46,  "programs": [46]},
  "orchestral_woodwind":   {"inst": "winds.clarinet",          "family": "winds",   "program": 71,  "programs": [68, 69, 70, 71]},
  "organ":                 {"inst": "organ.drawbar",           "family": "organ",   "program": 16,  "programs": [16, 17, 18, 19, 20]},
  "percussive_fx":         {"inst": "fx.percussive",           "family": "other",   "program": 115, "programs": [112, 113, 114, 115, 116, 117, 118, 119]},
  "piano":                 {"inst": "keys.piano",              "family": "keys",    "program": 0,   "programs": [0, 1, 3]},
  "pizzicato_strings":     {"inst": "strings.pizzicato",       "family": "strings", "program": 45,  "programs": [45]},
  "plucked_keyboard":      {"inst": "keys.clav",               "family": "keys",    "program": 7,   "programs": [6, 7]},
  "sax":                   {"inst": "winds.sax",               "family": "winds",   "program": 65,  "programs": [64, 65, 66, 67]},
  "slap_bass":             {"inst": "bass.slap",               "family": "bass",    "program": 36,  "programs": [36, 37]},
  "sound_fx":              {"inst": "fx.sound",                "family": "other",   "program": 122, "programs": [120, 121, 122, 123, 124, 125, 126, 127]},
  "strings":               {"inst": "strings.ensemble",        "family": "strings", "program": 48,  "programs": [40, 41, 42, 43, 44, 48, 49, 50, 51]},
  "synth_bass":            {"inst": "synth.bass",              "family": "bass",    "program": 38,  "programs": [38, 39]},
  "synth_fx":              {"inst": "fx.synth",                "family": "synth",   "program": 96,  "programs": [96, 97, 98, 99, 100, 101, 102, 103]},
  "synth_lead":            {"inst": "synth.lead",              "family": "synth",   "program": 81,  "programs": [80, 81, 82, 83, 84, 85, 86, 87]},
  "synth_pad":             {"inst": "synth.pad",               "family": "synth",   "program": 89,  "programs": [88, 89, 90, 91, 92, 93, 94, 95]},
  "timpani":               {"inst": "perc.timpani",            "family": "other",   "program": 47,  "programs": [47]},
  "vocal_harmony":         {"inst": "voice.backing",           "family": "voice",   "program": 53,  "programs": []},
  "wind_chimes":           {"inst": "fx.chimes",               "family": "other",   "program": 124, "programs": []}
}
```

Add to `pyproject.toml` under `[tool.hatch.build.targets.wheel]`:

```toml
artifacts = ["src/soundcode/data/*.json"]
```

- [ ] **Step 4: Extend `gm.py`.** Add `import json` and `from pathlib import Path`. After `DRUM_DEFAULT`:

```python
TSUMUGI: dict[str, dict] = json.loads(
    (Path(__file__).parent / "data" / "tsumugi_classes.json").read_text())
_INST_PROGRAM = {e["inst"]: e["program"] for e in TSUMUGI.values() if e["family"] != "drums"}
_PROGRAM_CLASS = {p: name for name, e in TSUMUGI.items() for p in e["programs"]}

DRUM_NOTES.update({"kick.acoustic": 35, "stick": 37, "clap": 39, "snare.electric": 40,
                   "tom.floor.lo": 41, "hat.pedal": 44, "tom.lo.mid": 45, "tom.hi.mid": 48,
                   "crash.2": 57, "ride.bell": 53, "tamb": 54, "splash": 55, "cowbell": 56,
                   "ride.2": 59, "china": 52, "shaker": 70})
_DRUM_VOICE = {v: k for k, v in DRUM_NOTES.items()}
_DRUM_VOICE[35] = "kick"                       # both GM kicks read as a kick


def drum_voice(pitch: int) -> str:
    return _DRUM_VOICE.get(pitch, f"gm{pitch}")


def tsumugi_class_for_program(program: int) -> str:
    return _PROGRAM_CLASS.get(program, "piano")
```

Change `drum_note`:

```python
def drum_note(voice: str) -> int:
    if voice.startswith("gm") and voice[2:].isdigit():
        return int(voice[2:])
    return DRUM_NOTES.get(voice, DRUM_DEFAULT)
```

In `target_for`, before the family resolution, add:

```python
    if inst in _INST_PROGRAM:
        return Target(0, _INST_PROGRAM[inst], False, inst.split(".")[0] if
                      inst.split(".")[0] in FAMILY_PROGRAM else "unknown")
```

Also add `"synth"` stays and `"other": 0` to `FAMILY_PROGRAM`, so `ethnic`, `fx` and `perc` resolve.

- [ ] **Step 5: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (102 + 4 in `test_render_sf.py` = 106, counting Task 1's 6).

- [ ] **Step 6: Commit**

```bash
git add src/soundcode/data/tsumugi_classes.json src/soundcode/gm.py pyproject.toml tests/test_render_sf.py
git commit -m "gm: tsumugi's 36 classes as fine-grained render targets; full GM drum kit"
```

---

### Task 3: tsumugi MIDI → `.sc` stream lines

**Files:**
- Create: `src/soundcode/tsumugi_sc.py`
- Test: `tests/test_tsumugi.py` (append)

**Interfaces:**
- Consumes: `gm.TSUMUGI`, `gm.tsumugi_class_for_program`, `gm.drum_voice`; the encoder's grid dict (`downbeat`, `bar_dur`).
- Produces:
  - `@dataclass Track(klass: str, inst: str, notes: list[tuple[float, float, int, int]])`, where each note is `(start_s, end_s, pitch, velocity)`
  - `read_tracks(midi: Path) -> list[Track]`, with drum tracks as `klass == "drums"`
  - `position(t: float, grid: dict) -> str`, which returns `"b:beat.xxx"`, or `"@t.xxx"` before the downbeat
  - `note_lines(track: Track, grid: dict) -> list[str]`, e.g. `"3:2.125  C4  0.750b 88"`. Drums give `"3:2.125 clap 96"`.
  - `drop_bleed(tracks: list[Track], min_notes: int = 3, min_share: float = 0.02) -> tuple[list[Track], list[str]]`, returning the kept tracks and reasons for the dropped ones
  - `stream_name(inst: str, stem: str, taken: set[str]) -> str`: `keys.piano` gives `notes.piano`, and a clash gives `notes.piano.<stem>`

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- MIDI -> .sc lines --------------------------------------------------------------------

import pretty_midi  # noqa: E402
from soundcode import tsumugi_sc as tsc  # noqa: E402

GRID = {"downbeat": 1.0, "bar_dur": 2.0}          # 120 bpm, 4/4, bar 1 at 1.0 s


def _midi(tmp_path, tracks):
    pm = pretty_midi.PrettyMIDI()
    for program, is_drum, notes in tracks:
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
        for s, e, p, v in notes:
            inst.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=e))
        pm.instruments.append(inst)
    path = tmp_path / "t.mid"
    pm.write(str(path))
    return path


def test_read_tracks_labels_by_program_and_drums(tmp_path):
    p = _midi(tmp_path, [(5, False, [(1.0, 1.5, 60, 90)]), (0, True, [(1.0, 1.1, 39, 100)])])
    tracks = tsc.read_tracks(p)
    assert [(t.klass, t.inst) for t in tracks] == [("electric_piano", "keys.ep"),
                                                   ("drums", "drums.kit.standard")]


def test_positions_keep_performed_timing_and_pickups():
    assert tsc.position(1.0, GRID) == "1:1.000"
    assert tsc.position(1.0 + 0.5 + 0.0625, GRID) == "1:2.125"     # a 32nd after beat 2
    assert tsc.position(3.0, GRID) == "2:1.000"
    assert tsc.position(0.4, GRID) == "@0.400"                     # before the first downbeat


def test_note_lines_pitched_and_drums():
    t = tsc.Track("piano", "keys.piano", [(1.5625, 1.9375, 60, 88)])
    assert tsc.note_lines(t, GRID) == ["1:2.125  C4  0.750b 88"]
    d = tsc.Track("drums", "drums.kit.standard", [(3.0, 3.1, 39, 96)])
    assert tsc.note_lines(d, GRID) == ["2:1.000 clap 96"]


def test_bleed_tracks_are_dropped_with_a_reason():
    big = tsc.Track("piano", "keys.piano", [(i, i + 0.5, 60, 90) for i in range(200)])
    tiny = tsc.Track("strings", "strings.ensemble", [(0, 1, 60, 90), (1, 2, 62, 90)])
    rare = tsc.Track("organ", "organ.drawbar", [(i, i + 0.5, 60, 90) for i in range(3)])
    kept, why = tsc.drop_bleed([big, tiny, rare])
    assert [t.klass for t in kept] == ["piano"]
    assert any("strings" in w and "2 notes" in w for w in why)
    assert any("organ" in w for w in why)                          # 3 of 205 < 2 %


def test_empty_midi_gives_no_tracks(tmp_path):
    assert tsc.read_tracks(_midi(tmp_path, [])) == []


def test_stream_names_do_not_clash():
    taken: set[str] = set()
    assert tsc.stream_name("keys.piano", "piano", taken) == "notes.piano"
    assert tsc.stream_name("keys.piano", "guitar", taken) == "notes.piano.guitar"
    assert tsc.stream_name("keys.ep", "guitar", taken) == "notes.ep"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_tsumugi.py -q`
Expected: collection error `cannot import name 'tsumugi_sc'`.

- [ ] **Step 3: Implement** `src/soundcode/tsumugi_sc.py`:

```python
"""tsumugi MIDI -> .sc stream lines.

Positions keep the performed timing (3-decimal beats), not a quantised grid:
the point of the file is a near-original rebuild. Notes before the first
downbeat are written in absolute seconds rather than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import gm
from .pitch import cents_to_name

BEATS_PER_BAR = 4


@dataclass
class Track:
    klass: str
    inst: str
    notes: list[tuple[float, float, int, int]] = field(default_factory=list)


def read_tracks(midi: Path) -> list[Track]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi))
    out = []
    for inst in pm.instruments:
        if not inst.notes:
            continue
        klass = "drums" if inst.is_drum else gm.tsumugi_class_for_program(inst.program)
        notes = sorted((n.start, n.end, n.pitch, n.velocity) for n in inst.notes)
        out.append(Track(klass, gm.TSUMUGI[klass]["inst"], notes))
    return out


def position(t: float, grid: dict) -> str:
    down, bar_dur = grid["downbeat"], grid["bar_dur"]
    if t < down - 1e-6:
        return f"@{t:.3f}"
    beats = (t - down) / (bar_dur / BEATS_PER_BAR)
    beats = round(beats, 3)
    bar, beat = divmod(beats, BEATS_PER_BAR)
    return f"{int(bar) + 1}:{beat + 1:.3f}"


def note_lines(track: Track, grid: dict) -> list[str]:
    beat_s = grid["bar_dur"] / BEATS_PER_BAR
    lines = []
    for start, end, pitch, vel in track.notes:
        pos = position(start, grid)
        if track.klass == "drums":
            lines.append(f"{pos} {gm.drum_voice(pitch)} {vel}")
        else:
            dur = max(end - start, 0.01) / beat_s
            lines.append(f"{pos}  {cents_to_name(pitch * 100)}  {dur:.3f}b {vel}")
    return lines


def drop_bleed(tracks: list[Track], min_notes: int = 3,
               min_share: float = 0.02) -> tuple[list[Track], list[str]]:
    total = sum(len(t.notes) for t in tracks) or 1
    kept, why = [], []
    for t in tracks:
        n = len(t.notes)
        if n < min_notes or n / total < min_share:
            why.append(f"{t.klass}: bleed ({n} notes, {n / total:.0%} of stem)")
        else:
            kept.append(t)
    return kept, why


def stream_name(inst: str, stem: str, taken: set[str]) -> str:
    short = inst.split(".")[1] if "." in inst else inst
    name = f"notes.{short}"
    if name in taken:
        name = f"notes.{short}.{stem}"
    taken.add(name)
    return name
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_tsumugi.py -q`
Expected: `12 passed`. If `cents_to_name(6000)` prints `C4` with a trailing cents suffix, check `pitch.cents_to_name` for exact semitones and adjust the test expectation only if the golden-file convention differs. Record a Ruling either way.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/tsumugi_sc.py tests/test_tsumugi.py
git commit -m "tsumugi_sc: MIDI tracks to .sc lines with performed timing, bleed dropping"
```

---

### Task 4: Instrument decision by voting

**Files:**
- Create: `src/soundcode/inventory.py`
- Test: `tests/test_tsumugi.py` (append)

**Interfaces:**
- Consumes: `tsumugi_sc.Track`, `gm.TSUMUGI`.
- Produces:
  - `REASSIGN_SHARE = 0.60`, `MIN_VOTES = 5`, `LOW_CONF = 0.5`
  - `mix_votes(stem_tracks: list[Track], mix_tracks: list[Track]) -> dict[str, int]`, giving family → count of stem notes matched by a mix note (onset ±50 ms, pitch ±1)
  - `@dataclass Decision(family: str, reassigned: bool, conf: float, votes: dict[str, int], warn: str | None)`
  - `decide(stem: str, prior_family: str, refine_top: list[tuple[str, float]], votes: dict[str, int]) -> Decision`
  - `STEM_FAMILY = {"piano": "keys", "guitar": "gtr", "bass": "bass", "other": "other", "drums": "drums", "lead_vocals": "voice", "backing_vocals": "voice", "vocals": "voice"}`
  - `FAMILY_REFINE_STEM = {"keys": "piano", "gtr": "guitar", "bass": "bass", "voice": "vocals"}`, the refinement prior to re-run under after a reassignment (other families → `"other"`)

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- inventory voting ------------------------------------------------------------------------

from soundcode import inventory as inv  # noqa: E402


def _t(klass, onsets, pitch=60):
    return tsc.Track(klass, gm.TSUMUGI[klass]["inst"], [(o, o + 0.3, pitch, 90) for o in onsets])


def test_mix_votes_count_matching_notes_by_family():
    stem = [_t("distorted_guitar", [1.0, 2.0, 3.0, 4.0])]
    mix = [_t("electric_piano", [1.02, 2.01, 3.03]), _t("strings", [4.0])]
    assert inv.mix_votes(stem, mix) == {"keys": 3, "strings": 1}


def test_guitar_stem_that_the_mix_calls_keys_is_reassigned():
    d = inv.decide("guitar", "gtr", [("distorted_guitar", 0.97)], {"keys": 8, "gtr": 1, "strings": 1})
    assert d.family == "keys" and d.reassigned
    assert "reassigned" in d.warn and "0.80" in d.warn


def test_agreement_keeps_the_stem_family_with_combined_confidence():
    d = inv.decide("piano", "keys", [("piano", 0.97)], {"keys": 9, "gtr": 1})
    assert d.family == "keys" and not d.reassigned
    assert abs(d.conf - 0.97 * 0.9) < 1e-9


def test_too_few_votes_leaves_refinement_in_charge():
    d = inv.decide("bass", "bass", [("electric_bass", 0.88)], {"keys": 3})
    assert d.family == "bass" and not d.reassigned and d.conf == 0.88
    assert "few mix votes" in d.warn
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_tsumugi.py -q -k "votes or reassigned or agreement or few"`
Expected: 4 failures, `cannot import name 'inventory'`.

- [ ] **Step 3: Implement** `src/soundcode/inventory.py`:

```python
"""Which instrument is a stem really? Two sources vote.

The separator's stem name is only a prior. tsumugi's refinement model says
what the stem sounds like *within* that prior; the mix-level transcription
(no stem prior at all) says which family each of the stem's notes belongs to.
Agreement is confidence; strong disagreement reassigns the stem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import gm
from .tsumugi_sc import Track

REASSIGN_SHARE = 0.60
MIN_VOTES = 5
LOW_CONF = 0.5
STEM_FAMILY = {"piano": "keys", "guitar": "gtr", "bass": "bass", "other": "other",
               "drums": "drums", "lead_vocals": "voice", "backing_vocals": "voice",
               "vocals": "voice"}
FAMILY_REFINE_STEM = {"keys": "piano", "gtr": "guitar", "bass": "bass", "voice": "vocals"}


@dataclass
class Decision:
    family: str
    reassigned: bool
    conf: float
    votes: dict[str, int] = field(default_factory=dict)
    warn: str | None = None


def _family(klass: str) -> str:
    return gm.TSUMUGI[klass]["family"]


def mix_votes(stem_tracks: list[Track], mix_tracks: list[Track]) -> dict[str, int]:
    mix = [(s, p, _family(t.klass)) for t in mix_tracks for s, _, p, _ in t.notes]
    votes: dict[str, int] = {}
    for t in stem_tracks:
        for s, _, p, _ in t.notes:
            hit = next((f for ms, mp, f in mix if abs(ms - s) <= 0.05 and abs(mp - p) <= 1), None)
            if hit is not None:
                votes[hit] = votes.get(hit, 0) + 1
    return votes


def decide(stem: str, prior_family: str, refine_top: list[tuple[str, float]],
           votes: dict[str, int]) -> Decision:
    p_refine = refine_top[0][1] if refine_top else 0.0
    total = sum(votes.values())
    if total < MIN_VOTES:
        return Decision(prior_family, False, p_refine, votes,
                        f"few mix votes ({total}); refinement only")
    top_family, top_n = max(votes.items(), key=lambda kv: kv[1])
    share = top_n / total
    if top_family != prior_family and share >= REASSIGN_SHARE:
        return Decision(top_family, True, share, votes,
                        f"{stem} stem reassigned to {top_family} by mix-level vote {share:.2f}")
    agree = votes.get(prior_family, 0) / total
    return Decision(prior_family, False, p_refine * agree, votes, None)
```

After a reassignment, the encoder (Task 5) re-runs refinement under `FAMILY_REFINE_STEM[family]` and multiplies that probability into `conf`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_tsumugi.py -q`
Expected: `16 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/soundcode/inventory.py tests/test_tsumugi.py
git commit -m "inventory: stem instrument decided by refinement + mix-level vote"
```

---

### Task 5: Encoder stage

**Files:**
- Modify: `src/soundcode/encode.py`: a new `stage_tsumugi`, wired into `encode()`
- Test: `tests/test_tsumugi.py` (append)

**Interfaces:**
- Consumes: Tasks 1–4; `active_blocks`, `active_level_db`, `Stage`, `_stage_lines`, `STEM_NOTES`.
- Produces:
  - `stage_tsumugi(stems: dict[str, Path], mix_path: Path, mix: np.ndarray, sr: int, grid: dict, work: Path) -> tuple[list[Stage], set[str]]`. It returns the stages (`:instruments` first, then note streams and `:perc.drums`) and the set of stem names it handled. Stems it could not handle (a `TsumugiError` for that stem, or tsumugi absent) are left out of the set, so `encode()` runs the basic-pitch path for exactly those.
  - In `encode()`: `stems` is keyed by encoder names (`vocals`, `bass`, …). `stage_tsumugi` gets `{"lead_vocals" or "vocals": stems["vocals"], ...}` using `vocal_stem_name`.

- [ ] **Step 1: Write the failing tests** (append):

```python
# --- encoder stage (tsumugi faked) ---------------------------------------------------------

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from soundcode import encode as enc  # noqa: E402


def _stem(tmp_path, name, db=-20.0, secs=8):
    sr = 22050
    y = (np.random.default_rng(1).standard_normal(sr * secs) * 10 ** (db / 20)).astype(np.float32)
    p = tmp_path / f"{name}.wav"
    sf.write(str(p), y, sr)
    return p, y, sr


def _fake_tsumugi(monkeypatch, tmp_path, per_model, mix_tracks, refine_top):
    def _write(out_midi, tracks):
        out_midi.parent.mkdir(parents=True, exist_ok=True)
        _midi(tmp_path, tracks).replace(out_midi)
        return out_midi

    def transcribe(audio, out_midi, model):
        tracks = mix_tracks if Path(audio).name == "mix.wav" else per_model[model]
        return _write(out_midi, tracks)
    monkeypatch.setattr(ts, "transcribe", transcribe)
    monkeypatch.setattr(ts, "refine", lambda audio, midi, stem, js: refine_top[stem])
    monkeypatch.setattr(ts, "add_velocity", lambda midi, wav, out: midi)


def test_stage_tsumugi_reassigns_the_river_guitar_and_keeps_claps(tmp_path, monkeypatch):
    piano, y, sr = _stem(tmp_path, "piano")
    guitar, _, _ = _stem(tmp_path, "guitar", -25)
    drums, _, _ = _stem(tmp_path, "drums", -22)
    mix = np.concatenate([y])
    onsets = [1.0 + 0.5 * i for i in range(12)]
    per_model = {
        "default": [(0, False, [(o, o + 0.4, 60, 90) for o in onsets])],
        "guitar_v1_5": [(30, False, [(o, o + 0.4, 64, 80) for o in onsets])],
        "drums_v1_5": [(0, True, [(o, o + 0.1, 39, 100) for o in onsets])],
    }
    mix_tracks = [(4, False, [(o + 0.01, o + 0.4, 64, 80) for o in onsets] +
                            [(o + 0.01, o + 0.4, 60, 90) for o in onsets])]
    refine_top = {"piano": [("piano", 0.97)], "guitar": [("distorted_guitar", 0.97)]}
    _fake_tsumugi(monkeypatch, tmp_path, per_model, mix_tracks, refine_top)
    grid = {"downbeat": 1.0, "bar_dur": 2.0}
    stages, handled = enc.stage_tsumugi({"piano": piano, "guitar": guitar, "drums": drums},
                                        tmp_path / "mix.wav", mix, sr, grid, tmp_path / "work")
    text = "\n".join(line for st in stages if st.ok for line in enc._stage_lines(st))
    assert handled == {"piano", "guitar", "drums"}
    assert ":instruments" in text and "reassigned to keys" in text
    headers = [ln for ln in text.splitlines() if ln.startswith(":notes.")]
    assert any("inst=keys." in h for h in headers)
    assert not any("inst=gtr." in h for h in headers)
    assert " clap " in text


def test_stage_tsumugi_falls_back_per_stem_on_error(tmp_path, monkeypatch):
    piano, y, sr = _stem(tmp_path, "piano")
    bass, _, _ = _stem(tmp_path, "bass")
    def transcribe(audio, out_midi, model):
        if model == "bass_v2":
            raise ts.TsumugiError("amt.cli.infer: boom")
        out_midi.parent.mkdir(parents=True, exist_ok=True)
        _midi(tmp_path, [(0, False, [(1.0 + i * 0.5, 1.4 + i * 0.5, 60, 90) for i in range(8)])]).replace(out_midi)
        return out_midi
    monkeypatch.setattr(ts, "transcribe", transcribe)
    monkeypatch.setattr(ts, "refine", lambda *a: [("piano", 0.9)])
    monkeypatch.setattr(ts, "add_velocity", lambda midi, wav, out: midi)
    stages, handled = enc.stage_tsumugi({"piano": piano, "bass": bass}, tmp_path / "mix.wav",
                                        y, sr, {"downbeat": 1.0, "bar_dur": 2.0}, tmp_path / "w")
    assert handled == {"piano"}
    assert any("bass" in n and "boom" in n for n in enc.STEM_NOTES)


def test_silent_stem_is_gated_before_tsumugi(tmp_path, monkeypatch):
    piano, y, sr = _stem(tmp_path, "piano")
    other, _, _ = _stem(tmp_path, "other", -90)
    calls = []
    monkeypatch.setattr(ts, "transcribe", lambda a, o, m: calls.append(m) or (_ for _ in ()).throw(ts.TsumugiError("x")))
    enc.stage_tsumugi({"other": other}, tmp_path / "mix.wav", y, sr,
                      {"downbeat": 1.0, "bar_dur": 2.0}, tmp_path / "w")
    assert "other_v1_5" not in calls
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_tsumugi.py -q -k "stage_tsumugi or silent_stem"`
Expected: `AttributeError: module 'soundcode.encode' has no attribute 'stage_tsumugi'`.

- [ ] **Step 3: Implement `stage_tsumugi`** in `encode.py`, after `stage_notes_poly`:

```python
def stage_tsumugi(stems: dict[str, Path], mix_path: Path, mix: np.ndarray, sr: int,
                  grid: dict, work: Path) -> tuple[list[Stage], set[str]]:
    """tsumugi per stem + a mix-level vote -> :instruments, note streams, drums.

    Stems that fail (TsumugiError) are left out of `handled`; encode() runs
    the basic-pitch path for those, so one bad stem never loses the rest.
    """
    import librosa
    import soundfile as sf

    from . import inventory as inv
    from . import tsumugi as ts
    from . import tsumugi_sc as tsc

    work.mkdir(parents=True, exist_ok=True)
    if not grid:
        return [], set()
    handled: set[str] = set()
    inventory = Stage("instruments", src=f"tsumugi@{ts.PINNED_COMMIT}+mix-vote")
    stages: list[Stage] = []
    taken: set[str] = set()
    confs: list[float] = []

    if not mix_path.exists():
        sf.write(str(mix_path), mix, sr)
    try:
        mix_tracks = tsc.read_tracks(ts.transcribe(mix_path, work / "mix.mid", "default"))
    except ts.TsumugiError as exc:
        STEM_NOTES.append(f"tsumugi mix-level vote unavailable ({exc})")
        mix_tracks = []

    for stem, path in stems.items():
        y_stem, _ = librosa.load(str(path), sr=sr, mono=True)
        mask = active_blocks(y_stem, mix, sr)
        level = active_level_db(y_stem, mask, sr)
        if not mask.any():
            inventory.lines.append(f"# {stem}: silent (below the loudness gate)")
            handled.add(stem)
            continue
        try:
            midi = ts.transcribe(path, work / f"{stem}.mid", ts.STEM_MODEL[stem])
            try:
                midi = ts.add_velocity(midi, path, work / f"{stem}.vel.mid")
            except ts.TsumugiError as exc:
                STEM_NOTES.append(f"{stem}: velocity model failed ({exc}); velocities are flat")
            tracks, why = tsc.drop_bleed(tsc.read_tracks(midi))
        except ts.TsumugiError as exc:
            STEM_NOTES.append(f"tsumugi failed on {stem} ({exc}); basic-pitch used")
            continue
        handled.add(stem)

        if stem == "drums":
            st = Stage("perc.drums", src=f"tsumugi:drums_v1_5@{ts.PINNED_COMMIT}", conf=0.69,
                       stem="drums", level_db=level)
            st.warns += why
            for t in tracks:
                st.lines += tsc.note_lines(t, grid)
            st.ok = bool(st.lines)
            stages.append(st)
            inventory.lines.append(f"drums   {'drums.kit' if st.ok else 'silent'}")
            continue

        prior = inv.STEM_FAMILY.get(stem, "other")
        refine_as = ts.REFINE_STEM.get(stem, "other")
        try:
            top = ts.refine(path, midi, refine_as, work / f"{stem}.refine.json")
        except ts.TsumugiError as exc:
            top = []
            STEM_NOTES.append(f"{stem}: refinement failed ({exc})")
        decision = inv.decide(stem, prior, top, inv.mix_votes(tracks, mix_tracks))
        if decision.reassigned:
            again = inv.FAMILY_REFINE_STEM.get(decision.family, "other")
            try:
                top = ts.refine(path, midi, again, work / f"{stem}.refine.{again}.json")
                decision.conf *= top[0][1] if top else 1.0
            except ts.TsumugiError:
                pass
        klass = top[0][0] if top and gm_family(top[0][0]) == decision.family else None
        for t in tracks:
            if klass is not None:
                t.klass, t.inst = klass, gm_tsumugi_inst(klass)
            name = tsc.stream_name(t.inst, stem, taken)
            st = Stage(name, src=f"tsumugi:{ts.STEM_MODEL[stem]}@{ts.PINNED_COMMIT}",
                       conf=decision.conf, stem=vocal_or(stem), level_db=level)
            st.header_fields = {"inst": t.inst}
            if decision.warn:
                st.warns.append(decision.warn)
            st.warns += why
            st.lines = tsc.note_lines(t, grid)
            st.ok = bool(st.lines)
            stages.append(st)
        runner = f"   | {top[1][0]} {top[1][1]:.2f}" if len(top) > 1 else ""
        mark = "" if decision.conf >= 0.80 else f" ?{decision.conf:.2f}"
        label = tracks[0].inst if tracks else "none"
        inventory.lines.append(f"{stem:<8} {label}{mark}{runner}")
        if decision.warn:
            inventory.warns.append(decision.warn)
        confs.append(decision.conf)

    inventory.conf = float(np.mean(confs)) if confs else 0.0
    inventory.ok = bool(inventory.lines)
    return ([inventory] + stages if inventory.ok else stages), handled
```

Add two small helpers next to it, which keep `encode.py` free of `gm` internals:

```python
def gm_family(klass: str) -> str:
    from . import gm
    return gm.TSUMUGI[klass]["family"]


def gm_tsumugi_inst(klass: str) -> str:
    from . import gm
    return gm.TSUMUGI[klass]["inst"]


def vocal_or(stem: str) -> str:
    return stem
```

Add `header_fields: dict[str, str] = field(default_factory=dict)` to `Stage`. In `_stage_lines`, render the stream header with those fields:

```python
    fields = "".join(f" {k}={v}" for k, v in st.header_fields.items())
    out = [f":{st.name}{fields}", f"meta    src={st.src}  conf={st.conf:.2f}"]
```

- [ ] **Step 4: Wire it into `encode()`.** After the grid stage and before the `percussion`/`notes` stages:

```python
    ts_stems = {("lead_vocals" if k == "vocals" and vocal_stem_name(v) == "lead_vocals"
                 else "vocals" if k == "vocals" else k): v for k, v in stems.items()}
    ts_stages, handled = stage_tsumugi(ts_stems, wd / "mix.wav", y.mean(0), sr, grid,
                                       wd / "tsumugi") if stems else ([], set())
```

Guard each existing stage with its stem: skip `stage_percussion` if `"drums" in handled`; skip each `stage_notes_poly` whose `stem_name` is in `handled` (the `vox` one checks either `lead_vocals` or `vocals`). Output order: grid, struct, harmony, then `ts_stages`, then whichever fallback stages ran, then text and mix.

- [ ] **Step 5: Run all tests and confirm they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. If the first test's `inst=keys.piano` assertion is awkward because of stream order, simplify it to "a `:notes.` stream with `inst=keys.piano` exists and no stream header starts with `:notes.` plus a gtr instrument". Keep the reassignment and clap assertions strict.

- [ ] **Step 6: Commit**

```bash
git add src/soundcode/encode.py tests/test_tsumugi.py
git commit -m "encode: tsumugi stage (per-stem transcription, voting inventory, drums with claps), per-stem fallback"
```

---

### Task 6: Measure, listen, record

**Files:**
- Create: `docs/results/2026-09-26-compare-tsumugi.md`
- Modify: `README.md` (Status)

- [ ] **Step 1: Encode and compare** the five 30 s clips with the tsumugi encoder into `out/sc/tsumugi/` and `out/compare/tsumugi/`. Use the measure loop from Plan 1 Task 7 with the label `tsumugi`, and write per-encode logs to `out/work/`.

- [ ] **Step 2: Check acceptance** (spec Evaluation item 1):
  - mean note F1 over active stems (original stem > −60 dBFS) is above `after2` for every song (`docs/results/2026-09-26-compare-after-fixes.md`);
  - `out/sc/tsumugi/river-30s.sc` has an `:instruments` stream that reassigns the guitar stem to keys; no `:notes.*` stream has a `gtr.` instrument; `:perc.drums` contains `clap`.

- [ ] **Step 3: Listening checkpoint.** Render `out/sc/tsumugi/river-30s.sc` (instruments only), then play it: `afplay -t 30 out/sc/tsumugi/river-30s.render.wav`. Record the user's verdict in the results file.

- [ ] **Step 4: Write** the results file: the tables, the acceptance checks, and the River `:instruments` block quoted. Add a README Status line: the encoder uses tsumugi for notes, drums and the instrument inventory, with a link to the results.

- [ ] **Step 5: Run all tests, then commit**

```bash
.venv/bin/python -m pytest -q
git add docs/results/2026-09-26-compare-tsumugi.md README.md
git commit -m "docs: compare results with the tsumugi backbone"
```
