"""Benchmark scorer: rate a rebuild against the original, per part.

Spec 2026-09-28-benchmark-scorer. `slices` cuts a song into sections,
20 s windows and one whole-song span; `align` measures whether a rebuild's
audio is on time against those slices; `metrics`, `drums` and `embed`
measure each slice; `scorer` maps them to 0-100 and `score_song` runs the
whole song; `report` prints the table and writes the HTML report.
"""

from .scorer import score_song

__all__ = ["score_song"]
