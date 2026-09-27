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
    # tsumugi runs with its own checkout as cwd: every path must be absolute
    audio, out_midi = Path(audio).resolve(), Path(out_midi).resolve()
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    _run("instrument_agnostic_amt.amt.cli.infer",
         ["--audio", str(audio), "--output-midi", str(out_midi), "--type", model,
          "--device", DEVICE, "--disable-tqdm"])
    if not out_midi.exists():
        raise TsumugiError(f"amt.cli.infer: no MIDI written for {audio.name}")
    return out_midi


def refine(audio: Path, midi: Path, stem_name: str, out_json: Path) -> list[tuple[str, float]]:
    audio, midi, out_json = Path(audio).resolve(), Path(midi).resolve(), Path(out_json).resolve()
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
    midi, stem_wav, out_midi = Path(midi).resolve(), Path(stem_wav).resolve(), Path(out_midi).resolve()
    _run("instrument_agnostic_amt.velocity.cli.infer_velocity",
         ["--midi", str(midi), "--stem-files", str(stem_wav), "--output-midi", str(out_midi),
          "--device", DEVICE])
    if not out_midi.exists():
        raise TsumugiError(f"velocity: no MIDI written for {midi.name}")
    return out_midi
