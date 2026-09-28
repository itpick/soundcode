"""MERT frame embeddings, cached per file hash.

Spec 2026-09-28-benchmark-scorer, Task 3. `m-a-p/MERT-v1-95M` (CC BY-NC 4.0,
allowed for private research) is loaded once per process — from the external
drive when mounted (`scripts/install_mert.sh`), else the HF cache — and run
on 24 kHz mono audio in `CHUNK_S`-second chunks so memory stays bounded on
full songs; hidden states are averaged over every transformer layer to get
one 768-dim vector per frame at MERT's ~75 Hz frame rate. `frames` caches its
result to `out/bench/cache/<sha1>.mert.npy`, keyed by the source file's
content.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

MODEL_ID = "m-a-p/MERT-v1-95M"
_DRIVE_MODELS = Path("/Volumes/ExFAT 2/infinity-engine/models")
MODEL_DIR = (_DRIVE_MODELS / "mert") if _DRIVE_MODELS.exists() else None

TARGET_SR = 24000
CHUNK_S = 30.0
FRAME_RATE = 75.0
CACHE_DIR = Path("out/bench/cache")

_MODEL = None
_EXTRACTOR = None
_DEVICE = None


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_path() -> str:
    if MODEL_DIR is not None and (MODEL_DIR / "config.json").exists():
        return str(MODEL_DIR)
    return os.environ.get("SOUNDCODE_MERT_MODEL", MODEL_ID)


def _load():
    """Load the model + feature extractor once per process; cache in globals."""
    global _MODEL, _EXTRACTOR, _DEVICE
    if _MODEL is not None:
        return _MODEL, _EXTRACTOR, _DEVICE

    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    path = _model_path()
    model = AutoModel.from_pretrained(path, trust_remote_code=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, trust_remote_code=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()

    _MODEL, _EXTRACTOR, _DEVICE = model, extractor, device
    return _MODEL, _EXTRACTOR, _DEVICE


def _load_audio(wav: Path) -> np.ndarray:
    import librosa

    y, _sr = librosa.load(str(wav), sr=TARGET_SR, mono=True)
    return np.asarray(y, dtype=np.float32)


def _run_chunk(chunk: np.ndarray) -> np.ndarray:
    """One chunk -> (T, 768): hidden states averaged over every layer."""
    import torch

    model, extractor, device = _load()
    inputs = extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        out = model(input_values, output_hidden_states=True)
    layers = torch.stack(out.hidden_states, dim=0).squeeze(1)   # (n_layers+1, T, 768)
    avg = layers.mean(dim=0)                                     # (T, 768)
    return avg.to("cpu").float().numpy()


def frames(wav: Path) -> np.ndarray:
    """MERT frame embeddings for `wav`: (T, 768) at ~75 Hz, cached by sha1."""
    wav = Path(wav)
    sha1 = _sha1_file(wav)
    cache = CACHE_DIR / f"{sha1}.mert.npy"
    if cache.exists():
        return np.load(cache)

    y = _load_audio(wav)
    chunk_len = int(round(CHUNK_S * TARGET_SR))
    parts = []
    for start in range(0, y.size, chunk_len):
        chunk = y[start:start + chunk_len]
        if chunk.size == 0:
            continue
        parts.append(_run_chunk(chunk))
    out = np.concatenate(parts, axis=0) if parts else np.zeros((0, 768), dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def _pool(fr: np.ndarray, a: float, b: float) -> np.ndarray | None:
    i0 = max(int(round(a * FRAME_RATE)), 0)
    i1 = min(int(round(b * FRAME_RATE)), fr.shape[0])
    if i1 <= i0:
        return None
    return fr[i0:i1].mean(axis=0)


def cosine(fr_ref: np.ndarray, fr_est: np.ndarray, a: float, b: float) -> float | None:
    """Cosine of the mean-pooled embeddings of each array over [a, b)."""
    u = _pool(np.asarray(fr_ref), a, b)
    v = _pool(np.asarray(fr_est), a, b)
    if u is None or v is None:
        return None
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 1e-12:
        return None
    return float(np.dot(u, v) / denom)
