"""Benchmark scorer: rate a rebuild against the original, per part.

Spec 2026-09-28-benchmark-scorer. `slices` cuts a song into sections,
20 s windows and one whole-song span; `align` measures whether a rebuild's
audio is on time against those slices.
"""
