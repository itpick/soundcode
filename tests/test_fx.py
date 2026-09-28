"""Production matching (spec 2026-09-27-production-match)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import fx  # noqa: E402
from soundcode.parser import parse  # noqa: E402

SR = 44100


def noise(secs=3.0, seed=0):
    return np.random.default_rng(seed).standard_normal(int(secs * SR)).astype(np.float32) * 0.1


def tilt(y, db_per_oct):
    """Apply a spectral tilt of db_per_oct around 1 kHz."""
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    g = 10 ** (db_per_oct * np.log2(np.maximum(f, 20) / 1000) / 20)
    return np.fft.irfft(Y * g, len(y)).astype(np.float32)


def test_band_db_sees_a_tilt():
    flat, dark = fx.band_db(noise(), SR), fx.band_db(tilt(noise(), -6), SR)
    assert len(flat) == 31
    hi = fx.BANDS > 4000
    assert dark[hi].mean() < flat[hi].mean() - 10


def test_band_db_of_silence_is_zeros():
    assert np.allclose(fx.band_db(np.zeros(SR, np.float32), SR), 0)


def test_rt60_of_an_exponential_decay():
    t = np.arange(int(2.0 * SR)) / SR
    y = np.zeros_like(t, dtype=np.float32)
    burst = noise(0.3, 1)
    y[:burst.size] = burst
    tail = noise(1.7, 2) * np.exp(-6.91 * (t[burst.size:] - 0.3) / 0.8)[: int(1.7 * SR)]  # rt60 0.8 s
    y[burst.size:burst.size + tail.size] = tail * (np.abs(burst[-2000:]).mean() / 0.08)
    f = fx.measure(np.stack([y, y]), SR, offsets=[0.3])
    assert 0.5 < f.rt60 < 1.2
    assert f.wet > 0.2


def test_width_and_pan():
    l, r = noise(seed=1), noise(seed=2)
    wide = fx.measure(np.stack([l, r]), SR, [])
    mono_left = fx.measure(np.stack([l, l * 0.25]), SR, [])
    assert wide.width > 0.8 and mono_left.width < 0.2
    assert mono_left.pan < -0.5


def test_measure_on_silence_or_a_blip_gives_defaults_not_nan():
    f = fx.measure(np.zeros((2, SR), np.float32), SR, [0.5])
    assert all(np.isfinite([f.rt60, f.wet, f.width, f.pan, f.crest]))
    assert fx.parse_fx(parse("%sc 0.3\n\n:notes.x\n" + fx.fx_line(f) + "\n").stream("notes.x"))


def test_fx_line_round_trips_and_tolerates_garbage():
    f = fx.Fx(eq=list(range(-15, 16)), rt60=0.62, wet=0.18, width=0.35, pan=-0.1, crest=14.2)
    doc = parse("%sc 0.3\n\n:notes.piano\n" + fx.fx_line(f) + "\n")
    g = fx.parse_fx(doc.stream("notes.piano"))
    assert g.eq == f.eq and g.rt60 == pytest.approx(0.62) and g.pan == pytest.approx(-0.1)
    bad = parse("%sc 0.3\n\n:notes.p\nfx  rt60=banana  wet=0.3  eq=1,2\n").stream("notes.p")
    h = fx.parse_fx(bad)
    assert h.wet == pytest.approx(0.3) and h.rt60 == fx.Fx(eq=[]).rt60 and h.eq == []
