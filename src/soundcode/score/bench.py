"""The benchmark set, the cached prepare-and-score pipeline, calibration, and
the run history / README (spec 2026-09-28-benchmark-scorer, Tasks 6-7).

`run_bench` drives `prepare` (cut/separate/encode/render, each cached by the
mtime of its input) and `scorer.score_song` over every entry of a tier,
prints each song's `report.table` and writes its `report.html`, then -- only
once every song has scored -- appends a line to
`docs/results/benchmark/history.jsonl` and rewrites
`docs/results/benchmark/README.md` (atomically) with the latest table per
tier and the change from that tier's previous run.

`calibrate` re-measures `anchors.json`'s floor/ceiling from tier A, reusing
`scorer.part_metrics` -- the same per-part metric derivation `score_song`
itself uses -- against a second separation (ceiling) and a different tier-A
song's same stem (floor).
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

import numpy as np

from . import anchors, report, scorer

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


def _run_cut(entry: dict, root: Path, force: bool, run) -> tuple[Path, float]:
    """The original audio for `entry`: cut/trimmed into
    `audio/bench/<name>.wav` when its name differs from the source file's
    stem or it needs trimming to a start/end (`_needs_cut`), else the
    source file itself untouched. Cached by mtime unless `force`. Returns
    `(path, seconds)` (`seconds` is 0.0 when cached or no cut is needed)."""
    name = entry["name"]
    source = root / entry["source"]
    if not _needs_cut(entry):
        return source, 0.0
    original = root / "audio" / "bench" / f"{name}.wav"
    if not _stale(original, source, force):
        return original, 0.0
    t0 = time.monotonic()
    original.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
    if entry.get("start") is not None:
        cmd += ["-ss", str(entry["start"])]
    if entry.get("end") is not None:
        end = entry["end"] - (entry.get("start") or 0.0)
        cmd += ["-t", str(end)]
    cmd += [str(original)]
    _check(run(cmd, capture_output=True, text=True), name, "cut")
    return original, time.monotonic() - t0


def _run_separate(name: str, original: Path, stems_dir: Path, root: Path, force: bool, run) -> float:
    """Separate `original` into `stems_dir` (the `soundcode separate` CLI),
    cached by mtime unless `force`. Returns seconds (0.0 when cached)."""
    if not (_stale(stems_dir, original, force) or not any(stems_dir.glob("*.wav"))):
        return 0.0
    t0 = time.monotonic()
    cmd = [sys.executable, "-m", "soundcode.cli", "separate", str(original), "-o", str(stems_dir)]
    _check(run(cmd, capture_output=True, text=True, cwd=root), name, "separate")
    return time.monotonic() - t0


def prepare(entry: dict, root: Path, force: bool = False, run=subprocess.run) -> dict:
    """Cut (if needed) -> separate -> encode -> render one benchmark entry,
    each stage cached by the mtime of its input unless `force`. Returns
    `{name, original, stems_dir, sc, parts_dir, rebuild, timing}`, where
    `timing` is `{stage: seconds}` for every stage (0.0 when cached)."""
    root = Path(root)
    name = entry["name"]
    timing: dict[str, float] = {}

    def timed(stage: str, fn) -> None:
        t0 = time.monotonic()
        fn()
        timing[stage] = time.monotonic() - t0

    original, timing["cut"] = _run_cut(entry, root, force, run)

    stems_dir = root / "out" / "stems" / name
    timing["separate"] = _run_separate(name, original, stems_dir, root, force, run)

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


# --------------------------------------------------------------------------
# calibrate: re-measure anchors.json from tier A (Task 7)
# --------------------------------------------------------------------------

_MIN_ANCHOR_SPREAD = 1e-6   # |ceiling - floor| below this: too degenerate to use


def _median(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def _collect_metric_samples(pairs: list[tuple[Path, Path]], cache_dir: Path) -> dict[str, dict[str, list[float]]]:
    """`{part_type: {metric: [values]}}` over every `(ref_dir, est_dir)`
    pair, one `scorer.part_metrics` call per part present (as `<key>.wav`)
    in both directories. A pair/part that errors (e.g. too short/silent to
    feature) is skipped with a warning printed, rather than failing the
    whole calibration run."""
    out: dict[str, dict[str, list[float]]] = {}
    for ref_dir, est_dir in pairs:
        for key, part_type in scorer.PART_TYPE.items():
            ref_path, est_path = ref_dir / f"{key}.wav", est_dir / f"{key}.wav"
            if not (ref_path.is_file() and est_path.is_file()):
                continue
            drum_cache = cache_dir if key == "drums" else None
            try:
                metrics = scorer.part_metrics(ref_path, est_path, key, drum_cache=drum_cache)
            except Exception as exc:                       # noqa: BLE001 -- one bad pair shouldn't sink calibration
                print(f"calibrate: {ref_path} vs {est_path} ({key}): {exc}", file=sys.stderr)
                continue
            bucket = out.setdefault(part_type, {})
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    bucket.setdefault(name, []).append(float(value))
    return out


def _direction(ceiling: float, floor: float) -> int:
    if ceiling > floor:
        return 1
    if ceiling < floor:
        return -1
    return 0


def _update_anchor(part_type: str, metric: str, old: dict, ceiling: float | None,
                   floor: float | None, warnings: list[str]) -> dict:
    """`old` (the current anchor entry) with `ceiling`/`floor` replaced by
    the newly measured ones -- unless there isn't enough data, the new
    spread is too small to be useful, or the newly measured direction
    (better = higher vs. better = lower) flips against `old`'s. Each of
    those leaves `old` untouched and appends a warning; weight and axis are
    always kept as they are."""
    label = f"{part_type}.{metric}"
    if ceiling is None or floor is None:
        warnings.append(f"{label}: not enough data to measure (ceiling={ceiling}, floor={floor}) "
                        f"-- keeping floor={old['floor']}, ceiling={old['ceiling']}")
        return dict(old)
    if abs(ceiling - floor) < _MIN_ANCHOR_SPREAD:
        warnings.append(f"{label}: degenerate, ceiling≈floor≈{ceiling:.4g} "
                        f"-- keeping floor={old['floor']}, ceiling={old['ceiling']}")
        return dict(old)
    old_dir, new_dir = _direction(old["ceiling"], old["floor"]), _direction(ceiling, floor)
    if old_dir != 0 and new_dir != old_dir:
        warnings.append(f"{label}: direction flipped (was {old['floor']:g} -> {old['ceiling']:g}, "
                        f"measured {floor:g} -> {ceiling:g}) -- keeping the existing anchor")
        return dict(old)
    return {**old, "floor": floor, "ceiling": ceiling}


def calibrate(root: Path, run=subprocess.run, anchors_path: Path | None = None) -> dict:
    """Re-measure `anchors.json`'s floor/ceiling per part type and metric,
    from tier A (design doc, "From metrics to 0-100"):

    - **ceiling:** the median, over tier-A songs, of the metric between
      each song's original stem and a *second* separation of the same
      clip -- separated again into `out/bench/calib/<name>/`, since
      demucs/roformer vary by about ±1 dB run to run;
    - **floor:** the median, over every ordered pair of distinct tier-A
      songs (a != b), of the metric between song a's stem and song b's
      same stem.

    Both use `scorer.part_metrics` -- the same per-part metric derivation
    `score_song` itself uses -- so calibration measures exactly what
    scoring measures. Weights and axes are untouched.

    An anchor is left as it is, with a warning, when there isn't enough
    data to measure it, when the newly measured `|ceiling - floor|` is too
    small to be useful, or when its direction (better = higher vs. better
    = lower) flips against the anchor's existing direction. `anchors.json`
    (or `anchors_path`, for tests) is rewritten atomically.

    Returns `{anchors, warnings, ceilings, floors}`: the new anchors dict
    as written, the list of warnings, and the raw per-(part_type, metric)
    measured medians (before the degenerate/flip guard) for the caller to
    log or fold into a commit message.
    """
    root = Path(root)
    entries = TIERS["A"]
    cache_dir = (root / "out" / "bench" / "cache").resolve()
    anchors_path = Path(anchors_path) if anchors_path is not None else anchors.ANCHORS_PATH

    originals: dict[str, Path] = {}
    stems_dirs: dict[str, Path] = {}
    for entry in entries:
        name = entry["name"]
        original, _ = _run_cut(entry, root, False, run)
        stems_dir = root / "out" / "stems" / name
        _run_separate(name, original, stems_dir, root, False, run)
        originals[name] = original
        stems_dirs[name] = stems_dir

    calib_dirs: dict[str, Path] = {}
    for entry in entries:
        name = entry["name"]
        calib_dir = root / "out" / "bench" / "calib" / name
        cmd = [sys.executable, "-m", "soundcode.cli", "separate", str(originals[name]), "-o", str(calib_dir)]
        _check(run(cmd, capture_output=True, text=True, cwd=root), name, "calibrate-separate")
        calib_dirs[name] = calib_dir

    ceiling_pairs = [(stems_dirs[e["name"]], calib_dirs[e["name"]]) for e in entries]
    floor_pairs = [(stems_dirs[a["name"]], stems_dirs[b["name"]])
                   for a in entries for b in entries if a["name"] != b["name"]]

    ceilings = _collect_metric_samples(ceiling_pairs, cache_dir)
    floors = _collect_metric_samples(floor_pairs, cache_dir)

    current = json.loads(anchors_path.read_text())
    new_anchors: dict[str, dict] = {}
    warnings: list[str] = []
    measured_ceilings: dict[str, dict] = {}
    measured_floors: dict[str, dict] = {}
    for part_type, part_anchors in current.items():
        new_anchors[part_type] = {}
        for metric, entry in part_anchors.items():
            c = _median(ceilings.get(part_type, {}).get(metric, []))
            f = _median(floors.get(part_type, {}).get(metric, []))
            measured_ceilings.setdefault(part_type, {})[metric] = c
            measured_floors.setdefault(part_type, {})[metric] = f
            new_anchors[part_type][metric] = _update_anchor(part_type, metric, entry, c, f, warnings)

    tmp = anchors_path.with_name(anchors_path.name + ".part")
    tmp.write_text(json.dumps(new_anchors, indent=2) + "\n", encoding="utf-8")
    tmp.replace(anchors_path)

    return {"anchors": new_anchors, "warnings": warnings,
            "ceilings": measured_ceilings, "floors": measured_floors}
