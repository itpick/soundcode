"""Demo site data: sizes of the original vs its sound code (spec 2026-09-28-demo-site-design).

Only NIN's *The Slip* clips (CC BY-NC-SA 3.0) may appear; River stays private."""

from __future__ import annotations

import gzip
from pathlib import Path

SONGS = [
    {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"},
    {"slug": "lights_in_the_sky-30s", "title": "Lights in the Sky", "clip": "audio/test/lights_in_the_sky-30s.mp3"},
    {"slug": "999999-30s", "title": "999,999", "clip": "audio/test/999999-30s.mp3"},
    {"slug": "corona_radiata-30s", "title": "Corona Radiata", "clip": "audio/test/corona_radiata-30s.mp3"},
]
ARTIST = "Nine Inch Nails"


def wav_bytes(frames: int) -> int:
    """16-bit stereo PCM WAV: the uncompressed baseline."""
    return frames * 2 * 2 + 44


def sc_sizes(text: str) -> tuple[int, int]:
    raw = text.encode()
    return len(raw), len(gzip.compress(raw, 9))


def kit_bytes(doc) -> int:
    s = doc.stream("perc.drums")
    if s is None or "kit" not in s.meta:
        return 0
    d = Path(s.meta["kit"])
    if not d.is_absolute() and getattr(doc, "path", None):
        d = Path(doc.path).parent / d
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0


def sung(doc) -> bool:
    from . import sing_score as ss
    try:
        ss.vocal_stream(doc)
    except ss.SingError:
        return False
    return bool(ss.words(doc))


def voice_bytes(doc, singer: str | None) -> int:
    """The singer's reference clip SoulX clones from (Seed-VC uses a clip of the same scale)."""
    from . import soulx
    return int(soulx.PROMPT_S * soulx.SR * 2) if singer and sung(doc) else 0


def singer_from_log(stderr: str, doc) -> str | None:
    if not sung(doc) or "rendering instruments only" in stderr:
        return None
    if "SoulX-Singer failed" in stderr or "sung with DiffSinger" in stderr:
        return "diffsinger"
    from . import sing
    return sing.DEFAULT_SINGER


def summary(doc) -> dict:
    from .expand import build_grid
    grid = build_grid(doc)
    tempo = round(grid.tempo[0][1]) if grid.tempo else None
    inst = {s.fields["inst"] for s in doc.of_kind("notes") if s.fields.get("inst")}
    if doc.of_kind("perc"):
        inst.add("drums")
    tv = doc.stream("text.vox")
    words = sum(len(e.text.split()) for e in (tv.events if tv else []) if e.text)
    return {"tempo": tempo, "instruments": sorted(inst), "words": words, "duration": doc.duration}
