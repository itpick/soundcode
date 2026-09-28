"""Published lyrics (LRCLIB, free, no key) reconciled with what ASR heard.

The reference fixes mishearings ("wider" -> "whiter"); the ASR keeps the
performed timing and the ad-libs the reference leaves out. Responses are
cached under out/lyrics/ (gitignored: lyrics are copyrighted, private use).
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = "https://lrclib.net/api"
ASR_MODEL = os.environ.get("SOUNDCODE_ASR_MODEL", "large-v3-turbo")
_EXTERNAL_MODELS = Path("/Volumes/ExFAT 2/infinity-engine/models/whisper")


def asr_download_root() -> str | None:
    """Big ASR weights live on the external drive when it is mounted."""
    return str(_EXTERNAL_MODELS) if _EXTERNAL_MODELS.parent.is_dir() else None
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
    if title and artist:
        return title, artist
    t, a = _from_name(Path(path).stem)
    return (title or t), (artist or (a if not title or t.lower() == title.lower() else None))


def _from_name(stem: str) -> tuple[str, str | None]:
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return t.strip(), a.strip()
    if "-" in stem:
        t, a = stem.rsplit("-", 1)
        a = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", a).strip()          # JordanFelix -> Jordan Felix
        return t.strip(), a or None
    return stem, None


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
           cache: Path = Path("out/lyrics"), offline: bool = False) -> list[Line] | None:
    key = re.sub(r"[^a-z0-9]+", "-", f"{artist or ''}-{title}".lower()).strip("-")
    path = Path(cache) / f"{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        return _lines(rec) if rec else None
    if offline:
        return None
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
        out += [(ln.t, w) for w in (normalise(x) for x in ln.text.replace("-", " ").split()) if w]
    return out
def reconcile(asr: list[tuple[float, float, str, float]],
              ref: list[tuple[float | None, str]]):
    """Reference words with ASR timing. Returns (words, ratio); below 0.5 the
    reference is judged not to match this recording and ASR is kept as heard."""
    import difflib

    a = [normalise(w[2]) for w in asr]
    b = [w for _, w in ref]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ratio = sm.ratio()
    if ratio < 0.5 or not ref:
        return [(s, e, w, p, None) for s, e, w, p in asr], ratio
    out: list[tuple[float, float, str, float, str | None]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out += [(asr[i][0], asr[i][1], b[j1 + (i - i1)], max(asr[i][3], 0.9), None)
                    for i in range(i1, i2)]
        elif tag == "replace":
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                if k < i2 - i1 and k < j2 - j1:
                    s, e, heard, p = asr[i1 + k]
                    out.append((s, e, b[j1 + k], 0.7, normalise(heard) or None))
                elif k < j2 - j1:                                      # extra reference word
                    out.append((None, None, b[j1 + k], 0.5, None))
                else:                                                  # extra ASR word
                    s, e, heard, p = asr[i1 + k]
                    out.append((s, e, heard, p, None))
        elif tag == "delete":                                          # ASR only: ad-lib
            out += [(asr[i][0], asr[i][1], asr[i][2], asr[i][3], None) for i in range(i1, i2)]
        else:                                                          # insert: reference only
            out += [(None, None, b[j], 0.5, None) for j in range(j1, j2)]
    # reference words after the last word actually heard cannot be timed (and may
    # lie past the clip): drop them rather than invent a performance
    while out and out[-1][0] is None:
        out.pop()
    # time the reference-only words between their neighbours
    for k, w in enumerate(out):
        if w[0] is None:
            prev = next((x[1] for x in reversed(out[:k]) if x[1] is not None), 0.0)
            nxt = next((x[0] for x in out[k + 1:] if x[0] is not None), prev + 0.4)
            span = max(nxt - prev, 0.1)
            out[k] = (prev, prev + span, w[2], 0.5, None)
    return out, ratio
