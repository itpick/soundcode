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


# --- applying fx -------------------------------------------------------------------------------

def test_eq_match_moves_the_spectrum_toward_the_target():
    src = np.stack([noise(seed=3)] * 2, 1)
    target = [int(round(v)) for v in fx.band_db(tilt(noise(seed=4), -6), SR)]
    before = np.abs(fx.band_db(src.T, SR) - target)[fx.BANDS < 10000].mean()
    out = fx.eq_match(src, SR, target)
    after = np.abs(fx.band_db(out.T, SR) - target)[fx.BANDS < 10000].mean()
    assert after < before * 0.5


def test_eq_never_boosts_empty_bands():
    t = np.arange(SR * 2) / SR
    sine = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32)
    out = fx.eq_match(sine, SR, [0] * 31)                   # flat target vs a single sine
    assert np.abs(out).max() < 4 * np.abs(sine).max()


def test_reverb_and_width_change_the_signal_but_not_its_start():
    t = np.arange(SR) / SR
    y = np.zeros((SR * 2, 2), np.float32)
    y[:SR, 0] = y[:SR, 1] = 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 0.2)
    out = fx.apply(y, SR, fx.Fx(eq=[], rt60=1.5, wet=0.6, width=0.8, pan=0.0))
    assert np.abs(out[int(0.4 * SR):int(0.8 * SR)]).mean() > np.abs(y[int(0.4 * SR):int(0.8 * SR)]).mean() + 1e-3
    onset = lambda a: int(np.argmax(np.abs(a[:, 0]) > 1e-3))  # noqa: E731
    assert abs(onset(out) - onset(y)) < int(0.002 * SR)


def test_render_without_fx_lines_is_unchanged():
    from soundcode import render_sf
    sf2 = Path(__file__).resolve().parents[1] / "models" / "soundfonts" / "GeneralUser-GS.sf2"
    if not sf2.exists():
        pytest.skip("no SoundFont")
    doc = parse("%sc 0.3\n@duration 2.0\n\n:notes.keys inst=keys.piano\n@0.0 C4 0.5s 100\n")
    a = render_sf.render_streams(doc, sf2=sf2)
    b = render_sf.render_streams(doc, sf2=sf2, no_fx=True)
    np.testing.assert_array_equal(a["notes.keys"], b["notes.keys"])


# --- encoder attaches fx ---------------------------------------------------------------------

def test_attach_fx_adds_a_line_to_stem_backed_stages(tmp_path):
    import soundfile as sf
    from soundcode import encode as enc
    p = tmp_path / "piano.wav"
    sf.write(str(p), np.stack([noise(seed=5), noise(seed=6)], 1), SR)
    st = enc.Stage("notes.piano", src="x", ok=True, stem="piano")
    st.lines = ["1:1.000  C4  1.000b 90"]
    other = enc.Stage("notes.x", src="x", ok=True)
    other.lines = ["1:1.000  C4  1.000b 90"]
    enc.attach_fx([st, other], {"piano": p}, SR)
    assert st.lines[0].startswith("fx      eq=")
    assert other.lines == ["1:1.000  C4  1.000b 90"]
    text = "\n".join(enc._stage_lines(st))
    assert fx.parse_fx(parse("%sc 0.3\n\n" + text).stream("notes.piano")) is not None
