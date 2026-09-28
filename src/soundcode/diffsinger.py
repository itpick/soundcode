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
    if missing:                    # never let one odd word cost the whole vocal
        score.warnings.append(f"bank has no phonemes {missing}; sung as 'ah'")
        score.phonemes = [p if p in ids else "en/ah" for p in score.phonemes]
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


def predict_durations(phonemes: list[str], word_div: list[int], word_frames: list[int],
                      ph_midi: list[int], mode: str = "01CORE") -> list[int]:
    """Phoneme durations from the bank's duration model, rescaled so each word
    keeps exactly its frames (word boundaries, and so timing, never move)."""
    import onnxruntime as ort

    b = bank_dir()
    ids = json.loads((b / "dsdur" / "phonemes.json").read_text())
    tokens = np.array([[ids.get(p, ids["SP"]) for p in phonemes]], dtype=np.int64)
    langs = np.array([[1 if p.startswith("en/") else 0 for p in phonemes]], dtype=np.int64)
    opts = ort.SessionOptions()
    prov = ["CPUExecutionProvider"]
    ling = ort.InferenceSession(str(b / "dsdur" / "linguistic.onnx"), opts, providers=prov)
    enc, masks = ling.run(None, {"tokens": tokens, "languages": langs,
                                 "word_div": np.array([word_div], dtype=np.int64),
                                 "word_dur": np.array([word_frames], dtype=np.int64)})
    spk = np.fromfile(b / "embeds" / f"{mode}.emb", dtype=np.float32).reshape(1, 1, 384)
    dur = ort.InferenceSession(str(b / "dsdur" / "dur.onnx"), opts, providers=prov)
    pred = dur.run(None, {"encoder_out": enc, "x_masks": masks,
                          "ph_midi": np.array([ph_midi], dtype=np.int64),
                          "spk_embed": np.repeat(spk, len(phonemes), axis=1)})[0][0]
    out: list[int] = []
    k = 0
    for n_ph, wf in zip(word_div, word_frames):
        p = np.maximum(pred[k:k + n_ph].astype(np.float64), 1e-3)
        f = np.maximum(np.round(p / p.sum() * wf).astype(int), 1)
        f[int(np.argmax(f))] += wf - int(f.sum())          # exact word total
        if f.min() < 1:                                     # tiny words: fall back to even split
            f = np.full(n_ph, wf // n_ph)
            f[-1] += wf - int(f.sum())
        out += f.tolist()
        k += n_ph
    return out
