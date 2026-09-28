"""Slices and alignment (spec 2026-09-28-benchmark-scorer)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundcode.parser import parse  # noqa: E402
from soundcode.score import align, slices  # noqa: E402

GRID = "%sc 0.3\n@duration 40.0\n\n:grid\nmeter @0.000 4/4\nanchor bar 1 @0.000\ntempo @0.000 120\n"


def clicks(sr, times, n):
    y = np.zeros(n, np.float32)
    for t in times:
        i = int(t * sr)
        y[i:i + 200] = np.hanning(200)
    return y


def test_windows_cover_the_song_with_hop():
    w = slices.windows(45.0)
    assert [(s.a, s.b) for s in w][:3] == [(0.0, 20.0), (10.0, 30.0), (20.0, 40.0)]
    assert w[-1].b == 45.0 and all(s.kind == "window" for s in w)


def test_sections_fall_back_to_8_bar_windows_when_struct_is_weak():
    doc = parse(GRID)                                   # no :struct → 8 bars at 120 bpm 4/4 = 16 s
    s = slices.sections(doc, 40.0)
    assert [(x.a, x.b) for x in s] == [(0.0, 16.0), (16.0, 32.0), (32.0, 40.0)]
    assert s[0].label == "bars 1-8"


def test_sections_use_struct_when_it_has_two_labels():
    doc = parse(GRID + "\n:struct\nintro 1-4 inst energy=0.2\nverse 5-12 vocal energy=0.5\n")
    s = slices.sections(doc, 40.0)
    assert [x.label for x in s][:2] == ["intro", "verse"]
    assert s[0].a == 0.0 and s[0].b == pytest.approx(8.0)


def test_sections_leave_a_gap_uncovered_between_struct_segments():
    doc = parse(GRID + "\n:struct\nintro 1-4 inst energy=0.2\nverse 7-10 vocal energy=0.5\n")
    s = slices.sections(doc, 40.0)
    assert s[0].label == "intro" and s[0].b == pytest.approx(8.0)    # bar 5's time
    assert s[1].label == "verse" and s[1].a == pytest.approx(12.0)   # bar 7's time


def test_active_gate():
    sr = 8000
    y = np.zeros(sr * 4, np.float32)
    y[sr:sr + 800] = 0.1                                # -20 dBFS burst in second 1
    assert slices.active(y, sr, 1.0, 2.0) and not slices.active(y, sr, 2.0, 4.0)


def test_lag_and_drift():
    sr = 16000
    t = np.arange(0.5, 19.5, 0.37)
    ref = clicks(sr, t, sr * 20)
    late = clicks(sr, t + 0.2, sr * 20)
    assert align.lag_ms(ref, ref, sr, 0, 20) == pytest.approx(0, abs=10)
    assert align.lag_ms(ref, late, sr, 0, 20) == pytest.approx(200, abs=10)
    d = align.drift(ref, late, sr, [slices.Slice("window", "0:00", 0.0, 20.0)])
    assert d[0]["drift"] and d[0]["lag_ms"] == pytest.approx(200, abs=10)
    assert align.lag_ms(np.zeros(sr * 20, np.float32), ref, sr, 0, 20) is None


def test_drift_matches_per_window_lag_ms():
    # Irregular click spacing avoids the periodic-autocorrelation ambiguity a
    # uniform click train has over a +/-1.5 s lag search.
    sr = 16000
    dur = 60.0
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.2, 0.6, size=200))
    t = t[t < dur - 1.0]
    ref = clicks(sr, t, int(sr * dur))
    late = clicks(sr, t + 0.2, int(sr * dur))
    wins = slices.windows(dur)
    d = align.drift(ref, late, sr, wins)
    assert len(d) > 2
    for row, w in zip(d, wins):
        expected = align.lag_ms(ref, late, sr, w.a, w.b)
        assert row["lag_ms"] == pytest.approx(expected, abs=15)
        assert row["lag_ms"] == pytest.approx(200, abs=15)
