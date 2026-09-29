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
    y[sr:sr + 800] = 0.1                                # -20 dBFS burst fills all of second 1
    assert slices.active(y, sr, 1.0, 2.0) and not slices.active(y, sr, 2.0, 4.0)


def test_single_click_in_long_silence_is_not_active():
    # Amendment 2: one loud 100 ms frame among ~200 in a 20 s slice is a
    # false alarm (the Discipline keys-intro bug), not "active".
    sr = 8000
    dur = 20.0
    n = int(sr * dur)
    y = np.zeros(n, np.float32)
    y[sr:sr + int(0.1 * sr)] = 1.0                      # one very loud 100 ms click
    assert not slices.active(y, sr, 0.0, dur)


def test_thirty_percent_loud_enough_is_active():
    # 30% of the slice's 100 ms frames at -30 dBFS clears both the 10%
    # min-fraction rule and the gate.
    sr = 8000
    dur = 20.0
    frame = int(0.1 * sr)
    n_frames = int(round(dur / 0.1))
    amp = 10 ** (-30 / 20)
    y = np.zeros(n_frames * frame, np.float32)
    for i in range(n_frames):
        if i % 10 < 3:                                  # 3 of every 10 frames: 30%
            y[i * frame:(i + 1) * frame] = amp
    assert slices.active(y, sr, 0.0, dur)


def test_sustained_quiet_below_gate_is_not_active():
    # Every frame is at -55 dBFS: below the -50 dBFS gate, so 0% of frames
    # qualify, even though the slice isn't below the -60 dBFS floor either.
    sr = 8000
    dur = 20.0
    n = int(sr * dur)
    amp = 10 ** (-55 / 20)
    y = np.full(n, amp, np.float32)
    assert not slices.active(y, sr, 0.0, dur)


def test_active_constants_match_the_amendment():
    assert slices.GATE_DB == -50.0
    assert slices.MIN_ACTIVE_FRACTION == pytest.approx(0.10)
    assert slices.SLICE_FLOOR_DB == -60.0


def test_lag_and_drift():
    # Amendment 3: the lag is the nearest correlation peak within 0.05 of the
    # max, so on a strictly periodic train a delay is only recoverable when
    # it is under half the period (a 0.37 s train delayed 200 ms reads as
    # -170 ms, correctly by that rule). 0.49 s keeps 200 ms the nearest peak.
    sr = 16000
    t = np.arange(0.5, 19.5, 0.49)
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


# --- Amendment 3: nearest-peak lag, FFT cross-correlation --------------------

def _brute_pearson(o_ref, o_est, max_lag):
    """The pre-amendment Pearson loop, as a reference: {lag: corr} over every
    lag with >= 2 overlapping frames of non-zero variance."""
    m = min(o_ref.size, o_est.size)
    o_ref, o_est = o_ref[:m], o_est[:m]
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r, e = o_ref[: m - lag], o_est[lag:]
        else:
            r, e = o_ref[-lag:], o_est[: m + lag]
        if r.size < 2 or np.std(r) == 0.0 or np.std(e) == 0.0:
            continue
        out[lag] = float(np.corrcoef(r, e)[0, 1])
    return out


def _nearest_peak_reference(corr: dict, tol=0.05):
    lags = sorted(corr)
    best = max(corr.values())

    def is_peak(lag):
        c = corr[lag]
        return all(corr.get(n, -np.inf) <= c for n in (lag - 1, lag + 1))
    cands = [lag for lag in lags if corr[lag] >= best - tol and is_peak(lag)]
    return min(cands, key=lambda lag: (abs(lag), -corr[lag]))


def test_periodic_click_train_20ms_late_is_20_not_a_beat_multiple():
    sr = 16000
    t = np.arange(0.5, 19.5, 0.49)
    ref = clicks(sr, t, sr * 20)
    late = clicks(sr, t + 0.02, sr * 20)
    assert align.lag_ms(ref, late, sr, 0, 20) == pytest.approx(20, abs=10)
    d = align.drift(ref, late, sr, [slices.Slice("window", "0:00", 0.0, 20.0)])
    assert d[0]["lag_ms"] == pytest.approx(20, abs=10)
    assert not d[0]["drift"]          # 20 ms is under the 30 ms drift threshold


def test_xcorr_matches_a_brute_force_pearson_loop():
    rng = np.random.default_rng(11)
    for _ in range(5):
        o_ref = rng.random(600) ** 4
        o_est = rng.random(600) ** 4
        lags, corr = align._xcorr(o_ref, o_est, 150)
        ref = _brute_pearson(o_ref, o_est, 150)
        got = {int(lag): float(c) for lag, c in zip(lags, corr) if np.isfinite(c)}
        assert set(got) == set(ref)
        for lag, c in ref.items():
            assert got[lag] == pytest.approx(c, abs=1e-6)


def test_fft_lag_agrees_with_brute_force_on_random_envelopes_within_10ms():
    rng = np.random.default_rng(5)
    for shift in (-40, -7, 0, 3, 25, 90):
        base = rng.random(900) ** 6
        o_ref = base[100:700]
        o_est = base[100 - shift:700 - shift]
        got = align._search_lag(o_ref, o_est, 10.0, 150)
        want = _nearest_peak_reference(_brute_pearson(o_ref, o_est, 150)) * 10.0
        assert got == pytest.approx(want, abs=10)
        assert got == pytest.approx(shift * 10.0, abs=10)


def _curve(peaks: dict, max_lag=150):
    """A correlation curve over -max_lag..max_lag: triangular peaks of the
    given heights at the given lags on a 0 floor."""
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(lags.size)
    for at, h in peaks.items():
        corr = np.maximum(corr, h - 0.1 * np.abs(lags - at))
    return lags, corr


def test_search_lag_prefers_the_nearest_peak_within_0_05_of_the_max(monkeypatch):
    assert align.LAG_PEAK_TOLERANCE == pytest.approx(0.05)
    env = np.random.default_rng(0).random(400)
    # a far peak (+120 frames) that wins by a hair (0.02) loses to the near one (+1)
    monkeypatch.setattr(align, "_xcorr", lambda r, e, m: _curve({1: 0.96, 120: 0.98, -3: 0.90}))
    assert align._search_lag(env, env, 10.0, 150) == pytest.approx(10.0)
    # ... but not when the near peak is more than 0.05 below the max
    monkeypatch.setattr(align, "_xcorr", lambda r, e, m: _curve({1: 0.92, 120: 0.98}))
    assert align._search_lag(env, env, 10.0, 150) == pytest.approx(1200.0)
    # a shoulder on the way up to a far peak is not a candidate, only peaks are
    monkeypatch.setattr(align, "_xcorr", lambda r, e, m: _curve({40: 0.99}))
    assert align._search_lag(env, env, 10.0, 150) == pytest.approx(400.0)
