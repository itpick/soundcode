"""Sing the lead vocal of a .sc in the original singer's voice.

score (sing_score) -> DiffSinger (explicit f0) -> Seed-VC (reference = the
original lead stem). Cached: Seed-VC takes minutes per 30 s.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import diffsinger, seedvc
from . import sing_score as ss
from .sing_score import SingError


def voice_ref(doc, explicit: Path | None) -> Path:
    if explicit is not None:
        if not Path(explicit).exists():
            raise SingError(f"--voice-ref {explicit} does not exist")
        return Path(explicit)
    stem = Path(doc.header.get("source", "")).stem
    for name in ("lead_vocals.wav", "vocals.wav"):
        p = Path("out") / "stems" / stem / name
        if stem and p.exists():
            return p
    raise SingError("no voice reference found; pass --voice-ref <wav> (the original singer)")


def _stream_text(doc, name: str) -> str:
    s = doc.stream(name)
    if s is None:
        return ""
    return json.dumps([s.fields, s.meta, s.statements, [e.raw for e in s.events]],
                      sort_keys=True, default=str)


def cache_key(doc, ref: Path, settings: dict) -> str:
    h = hashlib.sha1()
    for name in (ss.vocal_stream(doc), "text.vox", "contour.vox", "grid"):
        h.update(_stream_text(doc, name).encode())
    st = Path(ref).stat()
    h.update(f"{Path(ref).resolve()}|{st.st_mtime_ns}|{st.st_size}".encode())
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()[:16]


def sing(doc, ref: Path | None = None, cache: Path = Path("out/sing"),
         steps: int = 30) -> tuple[Path, list[str]]:
    import soundfile as sf

    ref = voice_ref(doc, ref)
    settings = {"steps": steps, "bank": diffsinger.BANK, "mode": "01CORE"}
    out = Path(cache) / f"{cache_key(doc, ref, settings)}.wav"
    score = ss.build(doc)
    if out.exists():
        return out, score.warnings
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".diffsinger.wav")
    sf.write(str(raw), diffsinger.render(score, mode=settings["mode"]), ss.SR)
    seedvc.convert(raw, ref, out, steps=steps)
    return out, score.warnings
