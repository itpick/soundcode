"""basic-pitch reports +1 bend bin on in-tune notes; the encoder must not."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import encode as enc  # noqa: E402


def test_bend_zero_point_is_one_bin():
    assert enc.bend_to_cents([1, 1, 1]) == pytest.approx(0.0)
    assert enc.bend_to_cents([2, 2]) == pytest.approx(100 / 3)
    assert enc.bend_to_cents(None) == 0.0


basic_pitch = pytest.importorskip("basic_pitch")


def _tone_cents(tmp_path, hz):
    from basic_pitch.inference import predict
    sr = 22050
    t = np.arange(int(sr * 1.5)) / sr
    y = 0.4 * np.sin(2 * np.pi * hz * t) * np.minimum(1, (1.5 - t) * 10)
    p = tmp_path / f"{hz:.2f}.wav"
    sf.write(str(p), y, sr)
    _, _, ev = predict(str(p))
    start, end, midi, amp, bend = max(ev, key=lambda e: e[1] - e[0])
    true = 1200 * np.log2(hz / 440.0) + 6900
    return midi * 100 + enc.bend_to_cents(bend) - true


@pytest.mark.parametrize("hz", [110.0, 261.63, 440.0])
def test_in_tune_tones_transcribe_in_tune(tmp_path, hz):
    assert abs(_tone_cents(tmp_path, hz)) <= 10


def test_a_sharp_tone_still_reads_sharp(tmp_path):
    hz = 440.0 * 2 ** (30 / 1200)
    assert 15 <= _tone_cents(tmp_path, hz) + 30 <= 45
