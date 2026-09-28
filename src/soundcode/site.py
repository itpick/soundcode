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


# The encoder's -50 dBFS gate: a quieter stem is not a part of the song.
PART_GATE = 10 ** (-50 / 20)
# Stems are written with the separator's input gain (separate.HEADROOM) already
# undone, so stems + residual rebuild the original exactly (checked: max error
# 2e-7 on discipline-30s). No further gain, or the original parts would play
# 1/HEADROOM (+1.9 dB) louder than the original.
ORIGINAL_PART_GAIN = 1.0


def _peak(path: Path) -> float:
    import numpy as np
    import soundfile as sf
    y, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return float(np.abs(y).max()) if y.size else 0.0


def _mp3(run, src: Path, out: Path, slug: str, gain: float = 1.0, bitrate: str = "128k"):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if gain != 1.0:
        cmd += ["-filter:a", f"volume={gain}"]
    _check(run([*cmd, "-b:a", bitrate, str(out)], capture_output=True, text=True), slug, "mp3")


def export_parts(run, slug: str, stems_dir: Path, parts_dir: Path, staging: Path) -> list[dict]:
    """Encode each part of both sides into staging/<slug>/; one entry per part
    either side has, in PART_KEYS order, with null for a side that lacks it."""
    from .render_sf import PART_KEYS, PART_LABELS
    out = staging / slug
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for key in PART_KEYS:
        entry = {"key": key, "label": PART_LABELS[key], "original": None, "rebuild": None,
                 "original_note": None, "rebuild_note": None}
        stem = stems_dir / f"{key}.wav"
        exists = stem.is_file()
        try:
            loud = exists and _peak(stem) >= PART_GATE
        except Exception as exc:                        # noqa: BLE001 — unreadable stem: name the song
            raise RuntimeError(f"{slug}: could not read the {key} stem {stem} ({exc})") from exc
        if loud:
            _mp3(run, stem, out / f"original-{key}.mp3", slug, gain=ORIGINAL_PART_GAIN)
            entry["original"] = f"media/{slug}/original-{key}.mp3"
        else:
            entry["original_note"] = "silent in the original" if exists else "not separated"
        part = parts_dir / f"{key}.wav"
        if part.is_file():
            _mp3(run, part, out / f"rebuild-{key}.mp3", slug)
            entry["rebuild"] = f"media/{slug}/rebuild-{key}.mp3"
        else:
            entry["rebuild_note"] = "not in the rebuild"
        if entry["original"] or entry["rebuild"]:
            entries.append(entry)
    return entries


def _parse_file(path: Path):
    from .parser import parse_file
    return parse_file(str(path))


def song_entry(song: dict, sc_path: Path, clip: Path, singer: str | None, media: dict,
               parts: list[dict] | None = None) -> dict:
    doc = _parse_file(sc_path)
    raw, gz = sc_sizes(sc_path.read_text())
    sizes = {"wav": wav_bytes(_frames(clip)), "mp3": clip.stat().st_size, "sc": raw, "sc_gz": gz,
             "kit": kit_bytes(doc), "voice": voice_bytes(doc, singer)}
    w = sizes["wav"]
    ratios = {"mp3": round(w / sizes["mp3"]), "sc": round(w / raw), "sc_gz": round(w / gz),
              "with_borrowed": round(w / (gz + sizes["kit"] + sizes["voice"]))}
    parts_out = []
    for p in (parts or []):
        # An unsung song's rebuild "lead_vocals" part is a voice.choir GM line
        # (meta stem), not an actual singer -- "Vocals" would mislabel it.
        if singer is None and p["key"] == "lead_vocals" and p.get("rebuild") is not None:
            p = {**p, "label": "Choir / pad (vocal stem)"}
        parts_out.append(p)
    return {"slug": song["slug"], "title": song["title"], "artist": ARTIST, **media,
            "sizes": sizes, "ratios": ratios, "singer": singer, "summary": summary(doc),
            "parts": parts_out}


def build(root: Path, site_dir: Path, work: Path, run=subprocess.run, force: bool = False,
          songs: list[dict] = SONGS) -> dict:
    """Encode, render, and measure audio, writing site/ to mirror exactly the songs passed.

    site/media/ and data.json are replaced together; songs not in the songs parameter
    disappear from the page. Both are updated atomically: on any failure, both remain
    unchanged to prevent mismatches between media and metadata.

    Args:
        root: Project root (contains audio clip paths)
        site_dir: Output directory for site/media/ and site/data.json
        work: Working directory for encode/render intermediates
        run: subprocess.run (for testing: can pass a fake runner)
        force: Re-encode and re-render all songs, skipping cache
        songs: List of song dicts with slug, title, clip path (default: SONGS)

    Returns:
        Data dict with built date, credit, license, song entries, and totals.
    """
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
    from .separate import default_out_dir
    try:
        for song in songs:
            slug, clip = song["slug"], root / song["clip"]
            stems_dir = root / default_out_dir(clip)
            if not stems_dir.is_dir() or not any(stems_dir.glob("*.wav")):
                raise RuntimeError(f"{slug}: no separated stems in {stems_dir}; "
                                    f"run: soundcode separate {song['clip']}")
            sc, wav, log = work / f"{slug}.sc", work / f"{slug}.wav", work / f"{slug}.render.log"
            if _stale(sc, clip, force):
                _check(run([*cli, "encode", str(clip), "--title", song["title"], "--artist", ARTIST, "-o", str(sc)],
                           capture_output=True, text=True, env=env, cwd=root), slug, "encode")
            parts_dir = work / f"{slug}.parts"
            if _stale(wav, sc, force) or not parts_dir.is_dir() or not log.exists():
                if parts_dir.exists():
                    shutil.rmtree(parts_dir)            # no part left over from an older render
                proc = run([*cli, "render", str(sc), "--with-vocals", "-o", str(wav),
                            "--parts", str(parts_dir)],
                           capture_output=True, text=True, env=env, cwd=root)
                _check(proc, slug, "render")
                log.write_text(proc.stderr or "")
            singer = singer_from_log(log.read_text() if log.exists() else "", _parse_file(sc))
            names = {"original": f"media/{slug}-original.mp3", "rebuild": f"media/{slug}-rebuild.mp3",
                     "sc": f"media/{slug}.sc"}
            shutil.copyfile(clip, media_staging / f"{slug}-original.mp3")
            shutil.copyfile(sc, media_staging / f"{slug}.sc")
            _mp3(run, wav, media_staging / f"{slug}-rebuild.mp3", slug, bitrate="192k")
            parts = export_parts(run, slug, stems_dir, parts_dir, media_staging)
            entries.append(song_entry(song, sc, clip, singer, names, parts))
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
