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

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol

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

# Loud masters decode above full scale, and audio-separator rescales any input
# or output peaking over 0.9 without restoring the level. Every pass input is
# written with its peak at or below HEADROOM (5 dB under that ceiling, since a
# stem can peak above its input) and the gain is undone on the outputs; an
# output that still reached the ceiling is reported in the warnings.
HEADROOM = 0.5
_MODEL_CEILING = 0.9

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
    owned = [out / f"{name}.wav" for name in _OWNED] + [out / "manifest.json"]
    if src.resolve() in {p.resolve() for p in owned}:
        raise SeparationError(f"refusing to overwrite the input {src} with a stem; "
                              "choose another output folder")

    # read before touching anything, so a bad input leaves the last run intact
    mix = read_stereo(src)
    n = mix.shape[1]

    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    for p in owned:
        p.unlink(missing_ok=True)
    # models see a plain-named copy of exactly the audio we sum-check against
    inputs: dict[str, tuple[Path, float]] = {"mix": _write_input(work / "mix.wav", mix)}

    audio_by_name: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    for i, p in enumerate(PASSES, 1):
        pass_dir = work / f"pass{i}"
        pass_dir.mkdir()
        in_path, gain = inputs[p.source]
        try:
            produced = backend.run(p.model, in_path, pass_dir)
        except Exception as exc:  # any model/backend failure
            raise SeparationError(f"{p.model}: {exc}") from exc
        for label, name in p.outputs.items():
            if label not in produced:
                warnings.append(f"{p.model} produced no '{label}' output")
                continue
            y = read_stereo(produced[label], length=n)
            if float(np.abs(y).max()) >= _MODEL_CEILING - 1e-3:
                warnings.append(f"{p.model} '{label}' output hit the {_MODEL_CEILING} "
                                "ceiling and was probably rescaled by the model")
            y = y / gain
            audio_by_name[name] = audio_by_name.get(name, 0) + y
            # intermediate results become inputs for later passes
            if name not in STEMS:
                inputs[name] = _write_input(work / f"{name}.wav", audio_by_name[name])

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


def _write_input(path: Path, y: np.ndarray) -> tuple[Path, float]:
    """Write a model input with its peak at or below HEADROOM; return the gain."""
    peak = float(np.abs(y).max())
    gain = HEADROOM / peak if peak > HEADROOM else 1.0
    write_wav(path, y * gain)
    return path, gain


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
