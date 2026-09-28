"""Published lyrics (LRCLIB, free, no key) reconciled with what ASR heard.

The reference fixes mishearings ("wider" -> "whiter"); the ASR keeps the
performed timing and the ad-libs the reference leaves out. Responses are
cached under out/lyrics/ (gitignored: lyrics are copyrighted, private use).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = "https://lrclib.net/api"
UA = "infinity-engine/0.1 (private research)"
_LRC = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")


@dataclass(frozen=True)
class Line:
    t: float | None
    text: str


def normalise(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower()).strip("'")


def guess_title_artist(path: Path, title: str | None = None,
                       artist: str | None = None) -> tuple[str, str | None]:
    if title:
        return title, artist
    stem = Path(path).stem
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return t.strip(), a.strip()
    if "-" in stem:
        t, a = stem.rsplit("-", 1)
        a = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", a).strip()          # JordanFelix -> Jordan Felix
        return t.strip(), a or None
    return stem, artist


def _get(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _lines(rec: dict) -> list[Line]:
    if rec.get("syncedLyrics"):
        out = []
        for raw in rec["syncedLyrics"].splitlines():
            m = _LRC.match(raw.strip())
            if m and m.group(3).strip():
                out.append(Line(int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
        return out
    return [Line(None, ln.strip()) for ln in (rec.get("plainLyrics") or "").splitlines() if ln.strip()]


def lookup(title: str, artist: str | None, duration: float | None,
           cache: Path = Path("out/lyrics")) -> list[Line] | None:
    key = re.sub(r"[^a-z0-9]+", "-", f"{artist or ''}-{title}".lower()).strip("-")
    path = Path(cache) / f"{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        return _lines(rec) if rec else None
    try:
        q = {"track_name": title} | ({"artist_name": artist} if artist else {})
        rec = None
        if duration:
            try:
                rec = _get(f"{API}/get?" + urllib.parse.urlencode(q | {"duration": round(duration)}))
            except Exception:                                       # noqa: BLE001 - try search
                rec = None
        if not rec:
            found = _get(f"{API}/search?" + urllib.parse.urlencode(q)) or []
            found = [r for r in found if r.get("syncedLyrics") or r.get("plainLyrics")]
            if duration:
                found.sort(key=lambda r: abs((r.get("duration") or 0) - duration))
            rec = found[0] if found else None
    except Exception:                                               # noqa: BLE001 - offline, 429, 5xx
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec or {}))
    return _lines(rec) if rec else None


def ref_words(lines: list[Line], start: float, end: float) -> list[tuple[float | None, str]]:
    timed = any(ln.t is not None for ln in lines)
    out = []
    for ln in lines:
        if timed and (ln.t is None or not (start - 1.0 <= ln.t <= end)):
            continue
        out += [(ln.t, w) for w in (normalise(x) for x in ln.text.split()) if w]
    return out
