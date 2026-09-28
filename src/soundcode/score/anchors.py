"""Anchors: per-(part type, metric) floor/ceiling/weight/axis, and the
0-100 mapping.

Spec 2026-09-28-benchmark-scorer, Task 4, "From metrics to 0-100". Anchors
live in `anchors.json` (committed) so `soundcode bench --calibrate` (a later
task) can rewrite them without touching code. `floor` is the value seen
against a different song's same part (worst case); `ceiling` is the value
seen against a second separation run of the same song (best case). Lower-is-
better metrics are expressed the same way, with `floor > ceiling`.
"""

from __future__ import annotations

import json
from pathlib import Path

ANCHORS_PATH = Path(__file__).parent / "anchors.json"
ANCHORS: dict[str, dict[str, dict]] = json.loads(ANCHORS_PATH.read_text())


def to_score(value: float | None, a: dict) -> float | None:
    """`100 * clamp((value - floor) / (ceiling - floor), 0, 1)`; None if
    `value` is None. `a` is an anchor entry (at least `floor`, `ceiling`)."""
    if value is None:
        return None
    floor, ceiling = a["floor"], a["ceiling"]
    frac = (value - floor) / (ceiling - floor)
    return 100.0 * min(max(frac, 0.0), 1.0)
