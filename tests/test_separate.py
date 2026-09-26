"""Stem separation (Milestone 1). Models are never loaded here: a fake
backend stands in for audio-separator, so these tests run in milliseconds."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import separate as sep  # noqa: E402

SR = sep.SR


def tone(freq: float, secs: float = 1.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * secs)) / SR
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, x])


# --- audio helpers ------------------------------------------------------------

def test_read_stereo_turns_mono_into_two_channels(tmp_path):
    mono = tone(440)[0]
    sf.write(tmp_path / "m.wav", mono, SR)
    y = sep.read_stereo(tmp_path / "m.wav")
    assert y.shape == (2, SR)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y[0], y[1])


def test_fit_length_pads_and_trims():
    y = tone(440, 0.5)
    assert sep.fit_length(y, SR).shape == (2, SR)
    assert sep.fit_length(y, 100).shape == (2, 100)
    assert np.all(sep.fit_length(y, SR)[:, len(y[0]):] == 0)


def test_rms_db_of_silence_is_minus_inf():
    assert sep.rms_db(np.zeros((2, 10), np.float32)) == -math.inf
    # a full-scale square wave is 0 dBFS
    assert sep.rms_db(np.ones((2, 10), np.float32)) == pytest.approx(0.0)


# --- labelling audio-separator outputs --------------------------------------

def test_label_outputs_uses_the_last_parenthesised_group():
    paths = [
        Path("/w/mix_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755.wav"),
        Path("/w/Song (Live)_(Instrumental)_model_bs_roformer.wav"),
    ]
    labels = sep.label_outputs(paths)
    assert labels == {"vocals": paths[0], "instrumental": paths[1]}


def test_label_outputs_ignores_unlabelled_files():
    assert sep.label_outputs([Path("/w/readme.wav")]) == {}


# --- sum check ----------------------------------------------------------------

def test_sum_check_passes_when_stems_rebuild_the_mix():
    a, b = tone(220), tone(330)
    report = sep.sum_check(a + b, {"a": a, "b": b})
    assert report.ok
    assert report.level_diff_db == pytest.approx(0.0, abs=1e-6)
    assert report.residual_db == -math.inf


def test_sum_check_fails_when_a_stem_is_missing():
    a, b = tone(220), tone(330)
    report = sep.sum_check(a + b, {"a": a})
    assert not report.ok
    assert report.residual_db > -15.0


def test_sum_check_on_silence_does_not_crash():
    z = np.zeros((2, SR), np.float32)
    report = sep.sum_check(z, {"a": z})
    assert report.ok
    assert report.level_diff_db == 0.0
