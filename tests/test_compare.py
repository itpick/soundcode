"""soundcode compare (spec Part A1)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import compare as cmp  # noqa: E402

SR = cmp.SR


def tone(freq, secs=2.0, amp=0.3, start=0.0, total=None):
    total = total or secs + start
    y = np.zeros(int(total * SR), np.float32)
    t = np.arange(int(secs * SR)) / SR
    y[int(start * SR):int(start * SR) + t.size] = amp * np.sin(2 * np.pi * freq * t)
    return y


def clicks(times, total=4.0):
    y = np.zeros(int(total * SR), np.float32)
    for t in times:
        i = int(t * SR)
        y[i:i + 200] = np.hanning(200)
    return y


def test_level_of_silence_is_none():
    assert cmp.level_db(np.zeros(100, np.float32)) is None
    assert cmp.level_db(np.ones(100, np.float32)) == pytest.approx(0.0)


def test_note_f1_identical_is_one_and_octave_agnostic_forgives_octaves():
    iv = np.array([[0.0, 0.5], [0.5, 1.0]])
    hz = np.array([440.0, 523.25])
    assert cmp.note_f1(iv, hz, iv, hz) == pytest.approx(1.0)
    assert cmp.note_f1(iv, hz, iv, hz * 2) == pytest.approx(0.0)
    assert cmp.note_f1(iv, hz, iv, hz * 2, octave_agnostic=True) == pytest.approx(1.0)


def test_note_f1_empty_sides():
    e = np.zeros((0, 2))
    assert cmp.note_f1(e, np.zeros(0), e, np.zeros(0)) is None
    assert cmp.note_f1(np.array([[0.0, 0.5]]), np.array([440.0]), e, np.zeros(0)) == 0.0


def test_chroma_blocks_same_pitch_high_different_low_silent_none():
    a, b = tone(440, 4.0), tone(440, 4.0)
    c = tone(311.13, 4.0)
    blocks = [(0.0, 2.0), (2.0, 4.0)]
    assert min(cmp.chroma_blocks(a, b, blocks)) > 0.95
    assert max(cmp.chroma_blocks(a, c, blocks)) < 0.6
    z = np.zeros_like(a)
    assert cmp.chroma_blocks(a, z, blocks) == [None, None]


def test_onset_f1_matches_same_clicks():
    ref = clicks([0.5, 1.0, 1.5, 2.0, 2.5])
    assert cmp.onset_f1(ref, ref) == pytest.approx(1.0)
    assert cmp.onset_f1(ref, clicks([3.5])) < 0.3


def test_energy_corr_identical_is_one_and_silence_is_none():
    y = tone(440, 1.0, start=1.0, total=3.0)
    assert cmp.energy_corr(y, y) == pytest.approx(1.0)
    assert cmp.energy_corr(y, np.zeros_like(y)) is None


def test_blocks_fall_back_to_two_seconds_without_grid():
    from soundcode.parser import parse
    doc = parse("%sc 0.3\n@duration 5.0\n")
    assert cmp.blocks_from_grid(doc, 5.0) == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]


def test_json_safe_strips_non_finite():
    out = cmp.json_safe({"a": math.inf, "b": [math.nan, 1.0], "c": {"d": -math.inf}})
    assert json.dumps(out, allow_nan=False) == '{"a": null, "b": [null, 1.0], "c": {"d": null}}'
