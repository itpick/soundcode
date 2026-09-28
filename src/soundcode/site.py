"""Demo site data: sizes of the original vs its sound code (spec 2026-09-28-demo-site-design).

Only NIN's *The Slip* clips (CC BY-NC-SA 3.0) may appear; River stays private."""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import os
import shutil
import subprocess
import sys
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


CREDIT = "Nine Inch Nails — The Slip (2008), CC BY-NC-SA 3.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/"


def _frames(path: Path) -> int:
    import soundfile as sf
    try:
        return sf.info(str(path)).frames
    except Exception:                                   # noqa: BLE001 — mp3 without libsndfile support
        import librosa
        return int(round(librosa.get_duration(path=str(path)) * 44100))


def _stale(out: Path, src: Path, force: bool) -> bool:
    return force or not out.exists() or out.stat().st_mtime < src.stat().st_mtime


def _check(proc, slug: str, what: str):
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [str(proc.returncode)]
        raise RuntimeError(f"{slug}: {what} failed ({tail[0]})")


def _parse_file_cached(path: Path):
    from .parser import parse_file
    return parse_file(str(path))


def song_entry(song: dict, sc_path: Path, clip: Path, singer: str | None, media: dict) -> dict:
    doc = _parse_file_cached(sc_path)
    raw, gz = sc_sizes(sc_path.read_text())
    sizes = {"wav": wav_bytes(_frames(clip)), "mp3": clip.stat().st_size, "sc": raw, "sc_gz": gz,
             "kit": kit_bytes(doc), "voice": voice_bytes(doc, singer)}
    w = sizes["wav"]
    ratios = {"mp3": round(w / sizes["mp3"]), "sc": round(w / raw), "sc_gz": round(w / gz),
              "with_borrowed": round(w / (gz + sizes["kit"] + sizes["voice"]))}
    return {"slug": song["slug"], "title": song["title"], "artist": ARTIST, **media,
            "sizes": sizes, "ratios": ratios, "singer": singer, "summary": summary(doc)}


def build(root: Path, site_dir: Path, work: Path, run=subprocess.run, force: bool = False,
          songs: list[dict] = SONGS) -> dict:
    root, site_dir, work = Path(root), Path(site_dir), Path(work)
    media_staging = site_dir / ".media.part"
    media = site_dir / "media"
    if media_staging.exists():
        shutil.rmtree(media_staging)
    media_staging.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SOUNDCODE_SEEDVC_HOST": os.environ.get("SOUNDCODE_SEEDVC_HOST", "framepick")}
    cli = [sys.executable, "-m", "soundcode.cli"]
    entries = []
    try:
        for song in songs:
            slug, clip = song["slug"], root / song["clip"]
            sc, wav, log = work / f"{slug}.sc", work / f"{slug}.wav", work / f"{slug}.render.log"
            if _stale(sc, clip, force):
                _check(run([*cli, "encode", str(clip), "--title", song["title"], "--artist", ARTIST, "-o", str(sc)],
                           capture_output=True, text=True, env=env, cwd=root), slug, "encode")
            if _stale(wav, sc, force):
                proc = run([*cli, "render", str(sc), "--with-vocals", "-o", str(wav)],
                           capture_output=True, text=True, env=env, cwd=root)
                _check(proc, slug, "render")
                log.write_text(proc.stderr or "")
            singer = singer_from_log(log.read_text() if log.exists() else "", _parse_file_cached(sc))
            names = {"original": f"media/{slug}-original.mp3", "rebuild": f"media/{slug}-rebuild.mp3",
                     "sc": f"media/{slug}.sc"}
            shutil.copyfile(clip, media_staging / f"{slug}-original.mp3")
            shutil.copyfile(sc, media_staging / f"{slug}.sc")
            _check(run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k",
                        str(media_staging / f"{slug}-rebuild.mp3")], capture_output=True, text=True), slug, "mp3")
            entries.append(song_entry(song, sc, clip, singer, names))
        tot = {k: sum(e["sizes"][k] for e in entries) for k in ("wav", "mp3", "sc", "sc_gz")}
        tot["borrowed"] = sum(e["sizes"]["kit"] + e["sizes"]["voice"] for e in entries)
        data = {"built": _dt.date.today().isoformat(), "credit": CREDIT, "license_url": LICENSE_URL,
                "songs": entries, "totals": tot}
        tmp = site_dir / "data.json.part"
        tmp.write_text(json.dumps(data, indent=1))
        if media.exists():
            shutil.rmtree(media)
        media_staging.replace(media)
        tmp.replace(site_dir / "data.json")             # atomic: a failed build keeps the old page data
        return data
    except Exception:
        if media_staging.exists():
            shutil.rmtree(media_staging)
        raise
