"""Combining metrics into 0-100 scores: per slice, per part, per song.

Spec 2026-09-28-benchmark-scorer, Task 4, "From metrics to 0-100". This
module only combines already-computed metric dicts (`score_slice`'s input
is the flat `{metric_name: value|None}` that Task 5's `score_song` derives
per slice, e.g. `level_diff_db_abs = |level_diff_db|`, `lag_ms_abs =
|lag_ms|`) — it never touches audio.
"""

from __future__ import annotations

from . import anchors

# part string -> anchors.json key (the metric set and floor/ceiling/weight/
# axis to score that part against). `residual` is deliberately absent: it
# is never scored (design doc).
PART_TYPE: dict[str, str] = {
    "piano": "pitched",
    "guitar": "pitched",
    "other": "pitched",
    "bass": "bass",
    "drums": "drums",
    "lead_vocals": "vocal",
    "backing_vocals": "vocal",
    "mix": "mix",
}

# Part score = 0.4 What + 0.4 Sound + 0.2 Dyn, renormalised over the axes
# that aren't None (design doc, "Combining").
AXIS_WEIGHTS: dict[str, float] = {"what": 0.4, "sound": 0.4, "dyn": 0.2}

AXES = ("what", "sound", "dyn")


def score_slice(metrics: dict, part_type: str) -> dict:
    """Score one slice's flat metrics dict against `part_type`'s anchors.

    Returns `{axes: {what, sound, dyn}, score, metrics: {name: {raw, score}}}`.
    A metric name with no anchor entry for `part_type` is ignored (not an
    error) — it is simply left out of `metrics` and never affects an axis.
    An axis with no non-None metric score is `None` (excluded, not 0); the
    overall score is the `AXIS_WEIGHTS`-weighted mean of the axes that
    aren't None, renormalised over just those axes, or `None` if every axis
    is None.
    """
    part_anchors = anchors.ANCHORS.get(part_type, {})
    out_metrics: dict[str, dict] = {}
    axis_pairs: dict[str, list[tuple[float, float]]] = {axis: [] for axis in AXES}

    for name, raw in metrics.items():
        entry = part_anchors.get(name)
        if entry is None:
            continue
        score = anchors.to_score(raw, entry)
        out_metrics[name] = {"raw": raw, "score": score}
        if score is not None:
            axis_pairs[entry["axis"]].append((score, float(entry.get("weight", 1.0))))

    axes: dict[str, float | None] = {}
    for axis in AXES:
        pairs = axis_pairs[axis]
        total_w = sum(w for _, w in pairs)
        axes[axis] = (sum(s * w for s, w in pairs) / total_w) if total_w > 0 else None

    num = 0.0
    den = 0.0
    for axis, w in AXIS_WEIGHTS.items():
        if axes[axis] is not None:
            num += axes[axis] * w
            den += w
    score = (num / den) if den > 0 else None

    return {"axes": axes, "score": score, "metrics": out_metrics}


def score_part(
    song: dict,
    sections: list[tuple[str, dict]],
    windows: list[tuple[str, dict]],
    part_type: str,
    *,
    silent: bool = False,
    missing: bool = False,
) -> dict:
    """Combine one part's song/section/window metric dicts into its scored form.

    `song`, and each `(label, metrics)` pair in `sections`/`windows`, are
    flat metric dicts as `score_slice` takes. Returns `{song, sections:[…],
    windows:[…], worst: {label, score} | None, missing, silent}`.

    `silent` (the original is below the activity gate for the whole song):
    not scored, not a failure — `song` is `None`, `sections`/`windows` are
    empty, `worst` is `None`.
    `missing` (an active original part has no rebuilt counterpart): scores
    0 rather than being left unscored, so it drags the song average down.
    """
    if silent:
        return {"song": None, "sections": [], "windows": [], "worst": None,
                "missing": False, "silent": True}

    if missing:
        zero_slice = {"axes": {axis: None for axis in AXES}, "score": 0.0, "metrics": {}}
        return {"song": zero_slice, "sections": [], "windows": [], "worst": None,
                "missing": True, "silent": False}

    song_scored = score_slice(song, part_type)
    sections_scored = [{"label": label, **score_slice(m, part_type)} for label, m in sections]
    windows_scored = [{"label": label, **score_slice(m, part_type)} for label, m in windows]

    worst: dict | None = None
    for entry in sections_scored:
        if entry["score"] is None:
            continue
        if worst is None or entry["score"] < worst["score"]:
            worst = {"label": entry["label"], "score": entry["score"]}

    return {"song": song_scored, "sections": sections_scored, "windows": windows_scored,
            "worst": worst, "missing": False, "silent": False}


def _mix_score(mix) -> float | None:
    """Accept either a plain score, or a `score_part`-shaped dict (its
    `song.score`), since the mix is scored the same way as any other part."""
    if isinstance(mix, dict):
        song = mix.get("song")
        return song.get("score") if song else None
    return mix


def song_score(parts: dict, energy: dict, mix) -> float | None:
    """The energy-share-weighted mean of the non-silent parts, blended 50/50
    with the mix score. A missing active part counts as 0 but still keeps
    its energy weight; a silent part is excluded entirely, including its
    weight, so the remaining parts' shares are renormalised over what's
    left. `None` if there is nothing to average on either side."""
    total = 0.0
    total_w = 0.0
    for name, result in parts.items():
        if result.get("silent"):
            continue
        w = float(energy.get(name, 0.0))
        song = result.get("song")
        score = 0.0 if result.get("missing") else (song.get("score") if song else None)
        if score is None:
            score = 0.0
        total += w * score
        total_w += w
    parts_score = (total / total_w) if total_w > 0 else None

    mix_score = _mix_score(mix)

    if parts_score is None and mix_score is None:
        return None
    if parts_score is None:
        return mix_score
    if mix_score is None:
        return parts_score
    return 0.5 * parts_score + 0.5 * mix_score
