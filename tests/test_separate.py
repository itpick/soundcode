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


# --- pipeline with a fake backend --------------------------------------------

import json  # noqa: E402


class FakeBackend:
    """Splits by fixed gains so the expected stems are known exactly.

    Each output is the input times a gain, and the gains of one pass sum to
    1.0, so the stems rebuild the mix. `drift` makes outputs 7 samples
    longer, as real models sometimes are."""

    GAINS = {
        "vocals": {"vocals": 0.4, "instrumental": 0.6},
        "karaoke": {"vocals": 0.75, "instrumental": 0.25},
        "demucs": {"drums": 0.3, "bass": 0.2, "guitar": 0.2, "piano": 0.1,
                   "other": 0.15, "vocals": 0.05},
    }

    def __init__(self, drift: int = 0, fail_on: str | None = None):
        self.drift, self.fail_on, self.calls = drift, fail_on, []

    def run(self, model, audio, out_dir):
        self.calls.append((model, Path(audio).name))
        if self.fail_on and self.fail_on in model:
            raise RuntimeError("model download failed")
        kind = ("karaoke" if "karaoke" in model else
                "demucs" if "demucs" in model else "vocals")
        y, _ = sf.read(str(audio), always_2d=True)
        y = y.T.astype(np.float32)
        if self.drift:
            y = np.pad(y, ((0, 0), (0, self.drift)))
        out = {}
        for label, g in self.GAINS[kind].items():
            p = Path(out_dir) / f"{Path(audio).stem}_({label.title()})_{model}.wav"
            sf.write(str(p), (y * g).T, SR, subtype="FLOAT")
            out[label] = p
        return out


def test_separate_writes_all_stems_mixes_and_manifest(tmp_path):
    src = tmp_path / "Song (Live).wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    res = sep.separate(src, tmp_path / "out", backend=FakeBackend())

    assert set(res.stems) == set(sep.STEMS)
    assert set(res.mixes) == {"vocals", "instrumental", "residual"}
    for p in [*res.stems.values(), *res.mixes.values()]:
        assert p.exists() and p.parent == tmp_path / "out"
    assert res.report.ok

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["stems"]["lead_vocals"] == "lead_vocals.wav"
    assert [m["model"] for m in manifest["passes"]] == [p.model for p in sep.PASSES]


def test_separate_routes_each_pass_to_the_right_input(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    fake = FakeBackend()
    sep.separate(src, tmp_path / "out", backend=fake)
    # pass 1 sees the copied mix, pass 2 the vocals, pass 3 the instrumental
    assert fake.calls[0][1] == "mix.wav"
    assert fake.calls[1][1] == "vocals.wav"
    assert fake.calls[2][1] == "instrumental.wav"


def test_separate_gains_are_correct_and_demucs_vocal_bleed_goes_to_other(tmp_path):
    src = tmp_path / "a.wav"
    mix = tone(220, 0.5)
    sf.write(str(src), mix.T, SR)
    res = sep.separate(src, tmp_path / "out", backend=FakeBackend())
    lead = sep.read_stereo(res.stems["lead_vocals"])
    other = sep.read_stereo(res.stems["other"])
    np.testing.assert_allclose(lead, mix * 0.4 * 0.75, atol=1e-5)
    np.testing.assert_allclose(other, mix * 0.6 * (0.15 + 0.05), atol=1e-5)


def test_separate_fixes_length_drift_and_mono_input(tmp_path):
    src = tmp_path / "mono.wav"
    sf.write(str(src), tone(220, 0.5)[0], SR)
    res = sep.separate(src, tmp_path / "out", backend=FakeBackend(drift=7))
    for p in res.stems.values():
        assert sep.read_stereo(p).shape == (2, SR // 2)


def test_separate_silence_writes_valid_json(tmp_path):
    src = tmp_path / "silent.wav"
    sf.write(str(src), np.zeros((SR // 4, 2), np.float32), SR)
    sep.separate(src, tmp_path / "out", backend=FakeBackend())
    text = (tmp_path / "out" / "manifest.json").read_text()
    assert "Infinity" not in text and "NaN" not in text
    json.loads(text)


def test_rerun_replaces_owned_files_and_keeps_others(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    (out / "drums.wav").write_bytes(b"stale")
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    sep.separate(src, out, backend=FakeBackend())
    assert (out / "notes.txt").read_text() == "mine"
    assert sep.read_stereo(out / "drums.wav").shape == (2, SR // 2)


def test_backend_failure_raises_separation_error_naming_the_model(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    with pytest.raises(sep.SeparationError, match="karaoke"):
        sep.separate(src, tmp_path / "out", backend=FakeBackend(fail_on="karaoke"))


# --- CLI -----------------------------------------------------------------------

from soundcode import cli  # noqa: E402


def test_default_out_dir_is_under_out_stems():
    assert sep.default_out_dir("audio/test/Song (Live).mp3") == Path("out/stems/Song (Live)")


def test_cli_separate_prints_levels_and_exits_zero(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    monkeypatch.setattr(sep, "AudioSeparatorBackend", FakeBackend)
    code = cli.main(["separate", str(src), "-o", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0
    assert "lead_vocals" in out and "sum check: OK" in out


def test_cli_separate_reports_backend_failure_and_exits_two(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    monkeypatch.setattr(sep, "AudioSeparatorBackend",
                        lambda: FakeBackend(fail_on="roformer_ep_317"))
    code = cli.main(["separate", str(src), "-o", str(tmp_path / "out")])
    assert code == 2
    assert "separation failed" in capsys.readouterr().err


# --- encoder integration -------------------------------------------------------

from soundcode import encode as enc  # noqa: E402


def test_encoder_stems_maps_lead_vocals_to_vocals(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    res = sep.separate(src, tmp_path / "out", backend=FakeBackend())
    stems = enc.encoder_stems(res, [])
    assert stems["vocals"] == res.stems["lead_vocals"]
    assert {"drums", "bass", "guitar", "piano", "other"} <= set(stems)


def test_separate_stems_records_failure_and_returns_empty(tmp_path, monkeypatch):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    monkeypatch.setattr(sep, "AudioSeparatorBackend", lambda: FakeBackend(fail_on="htdemucs"))
    enc.STEM_FAILURE.clear()
    assert enc.separate_stems(str(src), tmp_path / "wd") == {}
    assert "htdemucs" in enc.STEM_FAILURE[0]


# --- listening server ------------------------------------------------------------

from soundcode import server  # noqa: E402


def test_server_lists_stems_and_hides_work_files(tmp_path, monkeypatch):
    d = tmp_path / "out" / "stems" / "song"
    (d / "_work" / "pass1").mkdir(parents=True)
    for name in ("lead_vocals.wav", "instrumental.wav", "_work/mix.wav",
                 "_work/pass1/x_(Vocals)_m.wav"):
        (d / name).write_bytes(b"RIFF")
    monkeypatch.setenv("SOUNDCODE_ROOT", str(tmp_path))
    tracks = server._discover_tracks()
    labels = {t["label"]: t["kind"] for t in tracks}
    assert labels == {"stems/song/lead_vocals": "stem",
                      "stems/song/instrumental": "stem"}


# --- loud masters ------------------------------------------------------------------

class NormalisingBackend(FakeBackend):
    """Like audio-separator's default: input whose peak exceeds 0.9 is scaled
    down to 0.9 before separating, and the outputs stay at that lower level."""

    def run(self, model, audio, out_dir):
        y, _ = sf.read(str(audio), always_2d=True)
        peak = float(np.abs(y).max())
        if peak > 0.9:
            sf.write(str(audio), y * (0.9 / peak), SR, subtype="FLOAT")
        return super().run(model, audio, out_dir)


def test_loud_master_over_full_scale_still_rebuilds_the_mix(tmp_path):
    src = tmp_path / "loud.wav"
    sf.write(str(src), tone(220, 0.5, amp=1.1).T, SR, subtype="FLOAT")
    res = sep.separate(src, tmp_path / "out", backend=NormalisingBackend())
    assert res.report.ok
    assert abs(res.report.level_diff_db) < 0.01


# --- final-review fixes ------------------------------------------------------------

def test_refuses_to_overwrite_its_own_input(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    src = out / "vocals.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    with pytest.raises(sep.SeparationError, match="overwrite"):
        sep.separate(src, out, backend=FakeBackend())
    assert src.exists()


def test_unreadable_input_leaves_previous_run_intact(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    out = tmp_path / "out"
    sep.separate(src, out, backend=FakeBackend())
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(Exception):
        sep.separate(bad, out, backend=FakeBackend())
    assert (out / "drums.wav").exists()


def test_failed_rerun_removes_the_old_manifest(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    out = tmp_path / "out"
    sep.separate(src, out, backend=FakeBackend())
    with pytest.raises(sep.SeparationError):
        sep.separate(src, out, backend=FakeBackend(fail_on="karaoke"))
    assert not (out / "manifest.json").exists()


class OutputNormalisingBackend(FakeBackend):
    """Like audio-separator: any output peaking over 0.9 is scaled to 0.9.
    Outputs are first boosted 2.5x, since a real stem can peak above the mix
    it came from (the mix partly cancels it)."""

    def run(self, model, audio, out_dir):
        out = super().run(model, audio, out_dir)
        for p in out.values():
            y, _ = sf.read(str(p), always_2d=True)
            y = y * 2.5
            peak = float(np.abs(y).max())
            sf.write(str(p), y * (0.9 / peak if peak > 0.9 else 1.0), SR,
                     subtype="FLOAT")
        return out


def test_output_rescale_is_flagged_in_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(sep, "HEADROOM", 1.0)   # force outputs over 0.9
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5, amp=0.8).T, SR, subtype="FLOAT")
    res = sep.separate(src, tmp_path / "out", backend=OutputNormalisingBackend())
    assert any("rescaled" in w for w in res.warnings)


def test_headroom_leaves_room_below_the_models_0_9_ceiling():
    assert sep.HEADROOM <= 0.5


class SilentLeadBackend(FakeBackend):
    GAINS = {**FakeBackend.GAINS, "karaoke": {"vocals": 0.0, "instrumental": 1.0}}


def test_encoder_uses_full_vocals_when_lead_stem_is_silent(tmp_path):
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    res = sep.separate(src, tmp_path / "out", backend=SilentLeadBackend())
    notes: list[str] = []
    stems = enc.encoder_stems(res, notes)
    assert stems["vocals"] == res.mixes["vocals"]
    assert any("lead" in n for n in notes)


def test_separate_stems_is_fail_soft_on_any_error(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("No space left on device")
    monkeypatch.setattr(sep, "separate", boom)
    enc.STEM_FAILURE.clear()
    assert enc.separate_stems("x.wav", tmp_path) == {}
    assert "No space left" in enc.STEM_FAILURE[0]


def test_failed_sum_check_is_noted(tmp_path, monkeypatch):
    class Lossy(FakeBackend):
        GAINS = {**FakeBackend.GAINS, "vocals": {"vocals": 0.2, "instrumental": 0.3}}
    monkeypatch.setattr(sep, "AudioSeparatorBackend", Lossy)
    src = tmp_path / "a.wav"
    sf.write(str(src), tone(220, 0.5).T, SR)
    enc.STEM_NOTES.clear()
    assert enc.separate_stems(str(src), tmp_path / "wd")
    assert any("sum check" in n for n in enc.STEM_NOTES)
