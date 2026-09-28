"""The benchmark set, the cached prepare-and-score pipeline, and the run
history / README (spec 2026-09-28-benchmark-scorer, Task 6).

`run_bench` drives `prepare` (cut/separate/encode/render, each cached by the
mtime of its input) and `scorer.score_song` over every entry of a tier,
prints each song's `report.table` and writes its `report.html`, then -- only
once every song has scored -- appends a line to
`docs/results/benchmark/history.jsonl` and rewrites
`docs/results/benchmark/README.md` (atomically) with the latest table per
tier and the change from that tier's previous run.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import report, scorer

# --------------------------------------------------------------------------
# The benchmark set (design doc, "Benchmark set")
# --------------------------------------------------------------------------

_NIN = "Nine Inch Nails"

TIERS: dict[str, list[dict]] = {
    "A": [
        {"name": "river-30s", "source": "audio/test/river-30s.wav", "start": None, "end": None,
         "title": "The River", "artist": "Jordan Feliz"},
        {"name": "discipline-30s", "source": "audio/test/discipline-30s.mp3", "start": None, "end": None,
         "title": "Discipline", "artist": _NIN},
        {"name": "lights_in_the_sky-30s", "source": "audio/test/lights_in_the_sky-30s.mp3",
         "start": None, "end": None, "title": "Lights in the Sky", "artist": _NIN},
        {"name": "999999-30s", "source": "audio/test/999999-30s.mp3", "start": None, "end": None,
         "title": "999,999", "artist": _NIN},
        {"name": "corona_radiata-30s", "source": "audio/test/corona_radiata-30s.mp3",
         "start": None, "end": None, "title": "Corona Radiata", "artist": _NIN},
    ],
    "B": [
        {"name": "discipline-100s", "source": "audio/nin/discipline-100s.wav", "start": None, "end": None,
         "title": "Discipline", "artist": _NIN},
        {"name": "lights_in_the_sky-60s", "source": "audio/nin/lights_in_the_sky.mp3",
         "start": 30.0, "end": 90.0, "title": "Lights in the Sky", "artist": _NIN},
        {"name": "corona_radiata-60s", "source": "audio/nin/corona_radiata.mp3",
         "start": 0.0, "end": 60.0, "title": "Corona Radiata", "artist": _NIN},
        {"name": "999999-full", "source": "audio/nin/999999.mp3", "start": None, "end": None,
         "title": "999,999", "artist": _NIN},
    ],
    "C": [
        {"name": "discipline-full", "source": "audio/nin/discipline.mp3", "start": None, "end": None,
         "title": "Discipline", "artist": _NIN},
        {"name": "lights_in_the_sky-full", "source": "audio/nin/lights_in_the_sky.mp3",
         "start": None, "end": None, "title": "Lights in the Sky", "artist": _NIN},
        {"name": "corona_radiata-full", "source": "audio/nin/corona_radiata.mp3",
         "start": None, "end": None, "title": "Corona Radiata", "artist": _NIN},
        {"name": "river-full", "source": "audio/uploads/The River-JordanFelix.mp3", "start": None,
         "end": None, "title": "The River", "artist": "Jordan Feliz"},
    ],
}

HISTORY_PATH = Path("docs") / "results" / "benchmark" / "history.jsonl"
README_PATH = Path("docs") / "results" / "benchmark" / "README.md"


# --------------------------------------------------------------------------
# prepare: cut -> separate -> encode -> render, cached by mtime per stage
# --------------------------------------------------------------------------

def _stale(out: Path, src: Path, force: bool) -> bool:
    return force or not out.exists() or out.stat().st_mtime < src.stat().st_mtime


def _check(proc, name: str, stage: str) -> None:
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [str(proc.returncode)]
        raise RuntimeError(f"{name}: {stage} failed ({tail[0]})")


def _needs_cut(entry: dict) -> bool:
    """A B/C-tier entry whose name differs from its source file's stem (so the
    separated stems dir, keyed by that stem, lands under the entry's own
    name), or that needs trimming to a start/end -- gets copied/cut into
    `audio/bench/<name>.wav` first."""
    return (entry.get("start") is not None or entry.get("end") is not None
            or Path(entry["source"]).stem != entry["name"])


def prepare(entry: dict, root: Path, force: bool = False, run=subprocess.run) -> dict:
    """Cut (if needed) -> separate -> encode -> render one benchmark entry,
    each stage cached by the mtime of its input unless `force`. Returns
    `{name, original, stems_dir, sc, parts_dir, rebuild, timing}`, where
    `timing` is `{stage: seconds}` for every stage (0.0 when cached)."""
    root = Path(root)
    name = entry["name"]
    source = root / entry["source"]
    timing: dict[str, float] = {}

    def timed(stage: str, fn) -> None:
        t0 = time.monotonic()
        fn()
        timing[stage] = time.monotonic() - t0

    if _needs_cut(entry):
        original = root / "audio" / "bench" / f"{name}.wav"
        if _stale(original, source, force):
            def _cut():
                original.parent.mkdir(parents=True, exist_ok=True)
                cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
                if entry.get("start") is not None:
                    cmd += ["-ss", str(entry["start"])]
                if entry.get("end") is not None:
                    end = entry["end"] - (entry.get("start") or 0.0)
                    cmd += ["-t", str(end)]
                cmd += [str(original)]
                _check(run(cmd, capture_output=True, text=True), name, "cut")
            timed("cut", _cut)
        else:
            timing["cut"] = 0.0
    else:
        original = source
        timing["cut"] = 0.0

    stems_dir = root / "out" / "stems" / name

    def _separate():
        cmd = [sys.executable, "-m", "soundcode.cli", "separate", str(original), "-o", str(stems_dir)]
        _check(run(cmd, capture_output=True, text=True, cwd=root), name, "separate")
    if _stale(stems_dir, original, force) or not any(stems_dir.glob("*.wav")):
        timed("separate", _separate)
    else:
        timing["separate"] = 0.0

    bench_dir = root / "out" / "bench" / name
    sc = bench_dir / f"{name}.sc"

    def _encode():
        cmd = [sys.executable, "-m", "soundcode.cli", "encode", str(original), "-o", str(sc),
               "--title", entry["title"], "--artist", entry["artist"]]
        if entry.get("start") is not None:
            cmd += ["--offset", str(entry["start"])]
        _check(run(cmd, capture_output=True, text=True, cwd=root), name, "encode")
    if _stale(sc, original, force):
        timed("encode", _encode)
    else:
        timing["encode"] = 0.0

    parts_dir = bench_dir / "parts"
    rebuild = bench_dir / "rebuild.wav"
    env = {**os.environ, "SOUNDCODE_SEEDVC_HOST": os.environ.get("SOUNDCODE_SEEDVC_HOST", "framepick")}

    def _render():
        cmd = [sys.executable, "-m", "soundcode.cli", "render", str(sc), "--with-vocals",
               "--parts", str(parts_dir), "-o", str(rebuild)]
        _check(run(cmd, capture_output=True, text=True, cwd=root, env=env), name, "render")
    if _stale(rebuild, sc, force) or not parts_dir.is_dir() or not any(parts_dir.glob("*.wav")):
        timed("render", _render)
    else:
        timing["render"] = 0.0

    return {"name": name, "original": original, "stems_dir": stems_dir, "sc": sc,
            "parts_dir": parts_dir, "rebuild": rebuild, "timing": timing}


# --------------------------------------------------------------------------
# run history + README
# --------------------------------------------------------------------------

def slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (label or "").lower()).strip("-")
    return s or "run"


def _git_commit(root: Path, run) -> str:
    try:
        proc = run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=root)
    except Exception:                                       # noqa: BLE001 -- never fail the run over this
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unknown"


def _part_history(res: dict) -> dict:
    song = res.get("song")
    axes = (song or {}).get("axes") or {}
    return {"score": song.get("score") if song else None,
            "what": axes.get("what"), "sound": axes.get("sound"), "dyn": axes.get("dyn"),
            "sections": {s["label"]: s["score"] for s in res.get("sections", [])}}


def _song_history(result: dict) -> dict:
    return {"score": result.get("score"), "worst": result.get("worst"),
            "parts": {k: _part_history(v) for k, v in result.get("parts", {}).items()}}


def _read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def diff(prev_line: dict | None, cur_line: dict) -> list[tuple[str, str, float, float]]:
    """`(song, part, before, after)` for every song/part present with a
    non-None score in both lines. `part` is `"song"` for the overall song
    score. `prev_line=None` (no earlier run of this tier): empty."""
    if not prev_line:
        return []
    out: list[tuple[str, str, float, float]] = []
    prev_songs, cur_songs = prev_line.get("songs", {}), cur_line.get("songs", {})
    for name, cur in cur_songs.items():
        prev = prev_songs.get(name)
        if prev is None:
            continue
        if cur.get("score") is not None and prev.get("score") is not None:
            out.append((name, "song", prev["score"], cur["score"]))
        for part, cp in cur.get("parts", {}).items():
            pp = (prev.get("parts") or {}).get(part)
            if pp is None:
                continue
            if cp.get("score") is not None and pp.get("score") is not None:
                out.append((name, part, pp["score"], cp["score"]))
    return out


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.0f}"


def _fmt_delta(before, after) -> str:
    if before is None or after is None:
        return ""
    d = after - before
    return f" ({'+' if d >= 0 else ''}{d:.0f})"


def _latest_per_tier(lines: list[dict]) -> tuple[dict, dict]:
    """The last line, and the one before it, per tier -- scanned in file order
    so a later run always wins."""
    latest: dict[str, dict] = {}
    prev: dict[str, dict] = {}
    for line in lines:
        t = line["tier"]
        if t in latest:
            prev[t] = latest[t]
        latest[t] = line
    return latest, prev


def _render_readme(lines: list[dict]) -> str:
    latest, prev = _latest_per_tier(lines)
    out = ["# Benchmark results", "",
           "Regenerated after every `soundcode bench` run "
           "(spec 2026-09-28-benchmark-scorer). No raw audio, no lyrics.", ""]
    for tier in sorted(latest):
        cur = latest[tier]
        pv = prev.get(tier)
        changes = diff(pv, cur)
        deltas = {(s, p): (b, a) for s, p, b, a in changes}
        out.append(f"## Tier {tier}")
        out.append(f"_last run: {cur['date']} — {cur['label']} (commit {cur['commit']})_")
        out.append("")
        out.append("| song | part | what | sound | dyn | score | change |")
        out.append("|---|---|---|---|---|---|---|")
        for name, song in cur.get("songs", {}).items():
            b, a = deltas.get((name, "song"), (None, None))
            out.append(f"| {name} | **song** | | | | {_fmt(song.get('score'))} |{_fmt_delta(b, a)} |")
            for part, pd in song.get("parts", {}).items():
                b, a = deltas.get((name, part), (None, None))
                out.append(f"| {name} | {part} | {_fmt(pd.get('what'))} | {_fmt(pd.get('sound'))} | "
                           f"{_fmt(pd.get('dyn'))} | {_fmt(pd.get('score'))} |{_fmt_delta(b, a)} |")
        out.append("")
        # largest gain / worst regression first: sort by delta (after - before),
        # descending for improvements, ascending (most negative first) for regressions
        improved = sorted((c for c in changes if c[3] - c[2] >= 5), key=lambda c: c[2] - c[3])
        regressed = sorted((c for c in changes if c[3] - c[2] <= -5), key=lambda c: c[3] - c[2])
        out.append("### Improved (≥ +5)")
        out += ([f"- {s} {p}: {b:.0f} → {a:.0f} ({a - b:+.0f})" for s, p, b, a in improved]
               if improved else ["_none_"])
        out.append("")
        out.append("### Regressed (≤ −5)")
        out += ([f"- {s} {p}: {b:.0f} → {a:.0f} ({a - b:+.0f})" for s, p, b, a in regressed]
               if regressed else ["_none_"])
        out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# run_bench
# --------------------------------------------------------------------------

def run_bench(tier: str, label: str, root: Path, force: bool = False, run=subprocess.run) -> dict:
    """Prepare and score every entry of `tier` (or every tier, for `"all"`);
    only once every song has scored does it append to the run history and
    rewrite the README -- an error partway through (a failing pipeline stage,
    or `score_song` raising) leaves both untouched."""
    root = Path(root)
    tiers = list(TIERS) if tier == "all" else [tier]
    for t in tiers:
        if t not in TIERS:
            raise ValueError(f"unknown tier {t!r}")

    when = _dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    run_dir = root / "out" / "bench" / "runs" / f"{when}-{slug(label)}"
    cache_dir = (root / "out" / "bench" / "cache").resolve()
    commit = _git_commit(root, run)

    history_path = root / HISTORY_PATH
    prev_by_tier, _ = _latest_per_tier(_read_history(history_path))

    new_lines: list[dict] = []
    for t in tiers:
        songs: dict[str, dict] = {}
        timing: dict[str, dict] = {}
        for entry in TIERS[t]:
            paths = prepare(entry, root, force=force, run=run)
            out_dir = run_dir / entry["name"]
            result = scorer.score_song(paths["original"], paths["stems_dir"], paths["sc"],
                                       paths["parts_dir"], paths["rebuild"], out_dir,
                                       cache_dir=cache_dir)
            print(report.table(result))
            report.html(result, out_dir)
            songs[entry["name"]] = result
            timing[entry["name"]] = paths["timing"]

        line = {"date": when, "label": label, "commit": commit, "tier": t,
               "songs": {name: _song_history(res) for name, res in songs.items()}, "timing": timing}
        changes = diff(prev_by_tier.get(t), line)
        if changes:
            print(f"tier {t}: change from the previous run of this tier")
            for name, part, before, after in changes:
                print(f"  {name} {part} {before:.0f} → {after:.0f} ({after - before:+.0f})")
        new_lines.append(line)

    # only after every song in every requested tier has scored: commit the run
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        for line in new_lines:
            f.write(json.dumps(line) + "\n")

    lines = _read_history(history_path)
    readme_path = root / README_PATH
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    part = readme_path.with_name(readme_path.name + ".part")
    part.write_text(_render_readme(lines), encoding="utf-8")
    part.replace(readme_path)

    return {"run_dir": run_dir, "lines": new_lines}
