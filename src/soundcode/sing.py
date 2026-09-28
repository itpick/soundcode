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
    meta = {k: v for k, v in s.meta.items() if k != "level"}   # level is applied after, not sung
    return json.dumps([s.fields, meta, s.statements, [e.raw for e in s.events]],
                      sort_keys=True, default=str)


def cache_key(doc, ref: Path, settings: dict) -> str:
    h = hashlib.sha1()
    for name in (ss.vocal_stream(doc), "text.vox", "contour.vox", "grid"):
        h.update(_stream_text(doc, name).encode())
    st = Path(ref).stat()
    h.update(f"{Path(ref).resolve()}|{st.st_mtime_ns}|{st.st_size}".encode())
    h.update(f"duration={doc.duration}".encode())
    h.update(json.dumps({**settings, "ds_steps": 20, "ds_depth": 0.6, "vc": seedvc._FLAGS},
                        sort_keys=True).encode())
    return h.hexdigest()[:16]


DEFAULT_SINGER = "soulx"      # chosen by ear (2026-09-27) and voice/pitch/timing scores


def sing(doc, ref: Path | None = None, cache: Path = Path("out/sing"),
         steps: int = 50, durations: str = "model",
         singer: str | None = None) -> tuple[Path, list[str]]:
    singer = singer or DEFAULT_SINGER
    import soundfile as sf

    # No sung words, no singing: a vocal-family stream (matched via meta stem=
    # lead_vocals) with notes but no lyrics is a choir/pad line, not a voice to
    # clone onto -- it renders as its instrument instead (task 5a). Resolve the
    # vocal stream first so "no vocal stream at all" keeps its own message.
    ss.vocal_stream(doc)
    dur = doc.duration or 0.0
    if not any(w[0] < dur - 0.05 for w in ss.words(doc)):
        raise ss.NoVocalError("the vocal stream has no lyrics; rendered as its instrument")

    ref = voice_ref(doc, ref)
    settings = {"steps": steps, "bank": diffsinger.BANK, "mode": "01CORE", "durations": durations,
                "singer": singer}
    if singer == "soulx":
        from . import soulx
        settings["soulx"] = soulx.settings()
    out = Path(cache) / f"{cache_key(doc, ref, settings)}.wav"
    # SoulX only needs the score's warnings; the bank's duration model is DiffSinger's
    score = ss.build(doc, durations="heuristic" if singer == "soulx" else durations)
    if out.exists():
        try:
            import soundfile as _sf
            if _sf.info(str(out)).frames > 0:
                return out, score.warnings
        except Exception:                        # noqa: BLE001 — unreadable cache: redo it
            pass
        out.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    if singer == "soulx":
        from . import soulx
        try:
            y = soulx.render(doc, ref)
            part = out.with_suffix(".part.wav")
            sf.write(str(part), y, 44100)
            part.rename(out)
            return out, score.warnings
        except SingError as exc:
            score.warnings.append(f"SoulX-Singer failed ({exc}); sung with DiffSinger + Seed-VC")
            return sing(doc, ref, cache, steps, durations, singer="diffsinger")[0], score.warnings
    raw = out.with_suffix(".diffsinger.wav")
    sf.write(str(raw), diffsinger.render(score, mode=settings["mode"]), ss.SR)
    seedvc.convert(raw, ref, out, steps=steps)
    return out, score.warnings
