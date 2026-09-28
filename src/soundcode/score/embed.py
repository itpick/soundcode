"""MERT frame embeddings, cached per file hash.

Spec 2026-09-28-benchmark-scorer, Task 3. `m-a-p/MERT-v1-95M` (CC BY-NC 4.0,
allowed for private research) is loaded once per process — from the external
drive when mounted (`scripts/install_mert.sh`), else the HF cache — and run
on 24 kHz mono audio in `CHUNK_S`-second chunks so memory stays bounded on
full songs; hidden states are averaged over every transformer layer to get
one 768-dim vector per frame at MERT's ~75 Hz frame rate.

MERT's feature encoder is a stack of *valid* (unpadded) convolutions, so a
chunk run on its own loses a little over ~1 frame of context at its own
edges compared to what a continuous pass over the same span would produce.
Naively concatenating independently-run chunks therefore drifts by about a
frame per chunk boundary (measured: +2 frames by 65 s / 2 boundaries on a
real run) — on a full song that reaches a fifth of a second, enough to pool
the wrong frames in `cosine`. `_frames_for_audio` fixes this with
overlap-and-discard: each chunk (bar the song's own start/end) is run with
`CONTEXT_S` of audio on both sides, and only the frames whose *nominal* time
(`slice_start + frame_idx / FRAME_RATE`) falls inside the chunk's own
`[start, end)` are kept — so frame `i` of the stitched array is always
`i / FRAME_RATE` seconds in, independent of chunk boundaries, to within the
one frame that is unavoidably lost at the true start and end of the file
(no context available there, same as a single continuous pass would lose).

`frames` caches its result to
`out/bench/cache/<sha1>.<model-id-slug>-tf<transformers version>.mert.npy`,
keyed by the source file's content AND the model identity, so a model or
`transformers` upgrade can't silently reuse embeddings from the old one.

Caveat: MERT's transformer layers use full self-attention over everything in
one forward call, so a chunk's embeddings are never bit-identical to a
continuous pass's — only frame *index* alignment is exact, not the values.
See `_frames_for_audio` and `tests/test_score_drums_embed.py`'s
`test_mert_chunking_matches_a_continuous_pass` for the measured gap.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

MODEL_ID = "m-a-p/MERT-v1-95M"
_DRIVE_MODELS = Path("/Volumes/ExFAT 2/infinity-engine/models")
MODEL_DIR = (_DRIVE_MODELS / "mert") if _DRIVE_MODELS.exists() else None

TARGET_SR = 24000
CHUNK_S = 30.0
CONTEXT_S = 1.0          # overlap kept (then discarded) on each side of an internal chunk boundary
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


def _cache_tag() -> str:
    """`<model-id-slug>-tf<transformers version>`: the cache key's model half."""
    import transformers as _tf

    slug = re.sub(r"[^A-Za-z0-9]+", "-", MODEL_ID).strip("-")
    return f"{slug}-tf{_tf.__version__}"


def _cache_path(sha1: str) -> Path:
    return CACHE_DIR / f"{sha1}.{_cache_tag()}.mert.npy"


def _load():
    """Load the model + feature extractor once per process; cache in globals.

    The extractor's own `do_normalize` is turned off: it normalizes whatever
    array it is given to zero mean / unit variance, computed from THAT array
    alone — so the same underlying samples would get a different scale
    depending on whether they arrived as part of a 32 s chunk or a whole
    song. `_normalize` applies that same formula once, over the whole file,
    before chunking, so chunked and continuous processing see identical
    values for identical samples.
    """
    global _MODEL, _EXTRACTOR, _DEVICE
    if _MODEL is not None:
        return _MODEL, _EXTRACTOR, _DEVICE

    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    path = _model_path()
    model = AutoModel.from_pretrained(path, trust_remote_code=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, trust_remote_code=True)
    extractor.do_normalize = False
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()

    _MODEL, _EXTRACTOR, _DEVICE = model, extractor, device
    return _MODEL, _EXTRACTOR, _DEVICE


def _normalize(y: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance — the same formula the extractor's own
    (now disabled) `do_normalize` uses, applied once over the whole file."""
    y64 = np.asarray(y, dtype=np.float64)
    return ((y64 - y64.mean()) / np.sqrt(y64.var() + 1e-7)).astype(np.float32)


def _load_audio(wav: Path) -> np.ndarray:
    import librosa

    y, _sr = librosa.load(str(wav), sr=TARGET_SR, mono=True)
    return np.asarray(y, dtype=np.float32)


def _run_chunk(chunk: np.ndarray) -> np.ndarray:
    """One ALREADY-NORMALIZED chunk (see `_normalize`) -> (T, 768): hidden
    states averaged over every layer."""
    import torch

    model, extractor, device = _load()
    inputs = extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        out = model(input_values, output_hidden_states=True)
    layers = torch.stack(out.hidden_states, dim=0).squeeze(1)   # (n_layers+1, T, 768)
    avg = layers.mean(dim=0)                                     # (T, 768)
    return avg.to("cpu").float().numpy()


def _frames_for_audio(y: np.ndarray) -> np.ndarray:
    """Chunk `y` (24 kHz mono) through the model with overlap-and-discard.

    Each chunk [start, end) is run together with up to `CONTEXT_S` of extra
    audio on each side (clamped at the song's own start/end, where there is
    no neighboring audio to borrow). The model's actual output length for
    that padded slice gives each of its frames a nominal time
    (`pad_start_time + frame_idx / FRAME_RATE`); only the frames landing
    inside the chunk's own [start, end) are kept, so frame `i` of the
    stitched result is always `i / FRAME_RATE` seconds into `y`, and chunk
    boundaries never accumulate drift.
    """
    n = y.size
    if n == 0:
        return np.zeros((0, 768), dtype=np.float32)
    y = _normalize(y)

    chunk_len = int(round(CHUNK_S * TARGET_SR))
    context_len = int(round(CONTEXT_S * TARGET_SR))

    parts = []
    start = 0
    while start < n:
        end = min(start + chunk_len, n)
        pad_start = max(start - context_len, 0)
        pad_end = min(end + context_len, n)
        out = _run_chunk(y[pad_start:pad_end])
        if out.shape[0] > 0:
            frame_times = pad_start / TARGET_SR + np.arange(out.shape[0]) / FRAME_RATE
            own_a, own_b = start / TARGET_SR, end / TARGET_SR
            keep = (frame_times >= own_a - 1e-6) & (frame_times < own_b - 1e-6)
            parts.append(out[keep])
        start = end

    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 768), dtype=np.float32)


def frames(wav: Path) -> np.ndarray:
    """MERT frame embeddings for `wav`: (T, 768) at ~75 Hz, cached by sha1 + model identity."""
    wav = Path(wav)
    sha1 = _sha1_file(wav)
    cache = _cache_path(sha1)
    if cache.exists():
        return np.load(cache)

    y = _load_audio(wav)
    out = _frames_for_audio(y)

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
