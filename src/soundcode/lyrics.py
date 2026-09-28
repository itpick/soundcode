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


CONFIDENT = 0.9      # an ASR word this sure beats the published text (LRCLIB has typos too)


def _from_name(stem: str) -> tuple[str, str | None]:
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return t.strip(), a.strip()
    m = re.fullmatch(r"(.+)-([A-Z][a-z]+(?:[A-Z][a-z]+)+)", stem)   # Title-JordanFelix
    if m:
        return m.group(1).strip(), re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m.group(2))
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


def best_offset(asr: list[tuple], lines: list[Line], duration: float) -> float:
    """Where in the song a clip sits: the synced-line start whose window best
    matches what was heard (0.0 when nothing matches)."""
    starts = sorted({0.0, *(ln.t for ln in lines if ln.t is not None)})
    best, best_ratio = 0.0, 0.0
    for off in starts:
        ref = ref_words(lines, off, off + duration)
        if not ref:
            continue
        _, ratio = reconcile(asr, ref)
        if ratio > best_ratio + 1e-9:
            best, best_ratio = off, ratio
    return best if best_ratio >= 0.5 else 0.0


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
                    if p >= CONFIDENT:                    # sure of what was sung: keep it
                        out.append((s, e, normalise(heard), p, b[j1 + k]))
                    else:
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
    # time each run of reference-only words inside the gap around it (never overlapping)
    k = 0
    while k < len(out):
        if out[k][0] is not None:
            k += 1
            continue
        j = k
        while j < len(out) and out[j][0] is None:
            j += 1
        nxt = out[j][0] if j < len(out) else None
        prev = out[k - 1][1] if k > 0 else None
        n = j - k
        if prev is None:                                   # leading run: just before the first heard word
            end = nxt if nxt is not None else 0.4 * n
            prev = max(0.0, end - 0.4 * n)
        end = nxt if nxt is not None else prev + 0.4 * n
        step = max((end - prev) / n, 0.01)
        for m in range(n):
            a0 = prev + m * step
            out[k + m] = (a0, a0 + min(step, 0.6), out[k + m][2], 0.5, None)
        k = j
    return out, ratio


def reference_for(asr: list[tuple], lines: list[Line], duration: float,
                  offset: float | None) -> tuple[list[tuple[float | None, str]], float]:
    """Reference words for this clip: at `offset` when given, else wherever in
    the song the clip best matches (a clip cut from the middle of a song)."""
    off = offset if offset is not None else best_offset(asr, lines, duration)
    return ref_words(lines, off, off + duration), off
