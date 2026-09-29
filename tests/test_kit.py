"""The song's own drum kit (spec 2026-09-27-production-match §2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import kit  # noqa: E402
from soundcode.parser import parse, parse_file  # noqa: E402

SR = 44100


def test_isolated_hits_skip_crowded_ones_and_prefer_typical_velocity():
    hits = [(0.0, "kick", 100), (0.05, "hat", 60), (1.0, "kick", 90), (2.0, "kick", 40),
            (3.0, "kick", 95), (3.1, "snare", 80)]
    got = kit.isolated(hits)
    assert got["kick"][0] in (1.0,)                       # 0.0 crowded by the hat; 3.0 by the snare
    assert 2.0 in got["kick"] and 0.0 not in got["kick"]
    assert "hat" not in got                               # its only hit is crowded


def test_build_cuts_with_fades_and_play_round_robins(tmp_path):
    y = np.zeros((SR * 4, 2), np.float32)
    for t in (0.5, 1.5, 2.5):
        i = int(t * SR)
        y[i:i + 2000] = np.hanning(4000)[2000:, None] * 0.8
    stem = tmp_path / "drums.wav"
    sf.write(str(stem), y, SR)
    made = kit.build(stem, [(0.5, "clap", 100), (1.5, "clap", 100), (2.5, "clap", 100)], tmp_path / "kit")
    assert len(made["clap"]) == 3
    s0, _ = sf.read(str(made["clap"][0]))
    assert abs(s0[0]).max() < 0.05                         # fade-in, starts 5 ms early
    samples = kit.load(tmp_path / "kit", SR)
    out, missing = kit.play([(0.1, "clap", 127), (0.6, "clap", 127), (1.1, "kick", 100)], samples, SR * 2)
    assert missing == [(1.1, "kick", 100)]
    assert np.abs(out[int(0.1 * SR):int(0.2 * SR)]).max() > 0.3


def test_kit_resamples_on_load(tmp_path):
    (tmp_path / "k").mkdir()
    sf.write(str(tmp_path / "k" / "snare_0.wav"), np.zeros((4410, 2), np.float32), SR)
    got = kit.load(tmp_path / "k", 22050)
    assert got["snare"][0].shape == (2205, 2)


def test_parse_file_records_the_path(tmp_path):
    p = tmp_path / "s.sc"
    p.write_text("%sc 0.3\n")
    assert parse_file(str(p)).path == p


def test_render_uses_the_kit_and_falls_back_when_it_is_gone(tmp_path):
    import pytest
    from soundcode import render_sf
    sf2 = Path(__file__).resolve().parents[1] / "models" / "soundfonts" / "GeneralUser-GS.sf2"
    if not sf2.exists():
        pytest.skip("no SoundFont")
    (tmp_path / "s.kit").mkdir()
    t = np.arange(4410) / SR
    sf.write(str(tmp_path / "s.kit" / "clap_0.wav"),
             np.stack([np.sin(2 * np.pi * 3000 * t)] * 2, 1).astype(np.float32) * 0.5, SR)
    sc = tmp_path / "s.sc"
    sc.write_text("%sc 0.3\n@duration 2.0\n\n:perc.drums inst=drums.kit\nmeta kit=s.kit\n"
                  "@0.5 clap 127\n@1.0 kick 100\n")
    doc = parse_file(str(sc))
    y = render_sf.render_streams(doc, sf2=sf2)["perc.drums"]
    assert np.abs(y[int(0.5 * SR):int(0.55 * SR)]).max() > 0.2        # the kit clap
    assert np.abs(y[int(1.0 * SR):int(1.2 * SR)]).max() > 0.01        # GM kick fallback
    import shutil
    shutil.rmtree(tmp_path / "s.kit")
    y2 = render_sf.render_streams(parse_file(str(sc)), sf2=sf2)["perc.drums"]
    assert np.abs(y2).max() > 0.01                                     # full GM fallback, no crash


def test_simultaneous_hits_crowd_each_other():
    got = kit.isolated([(1.0, "kick", 100), (1.0, "hat", 60), (3.0, "kick", 90)])
    assert got.get("kick") == [3.0] and "hat" not in got


def test_render_reports_a_missing_kit(tmp_path, capsys):
    import pytest
    from soundcode import render_sf
    sf2 = Path(__file__).resolve().parents[1] / "models" / "soundfonts" / "GeneralUser-GS.sf2"
    if not sf2.exists():
        pytest.skip("no SoundFont")
    sc = tmp_path / "s.sc"
    sc.write_text("%sc 0.3\n@duration 1.0\n\n:perc.drums inst=drums.kit\nmeta kit=gone.kit\n@0.2 kick 100\n")
    render_sf.render(parse_file(str(sc)), 44100, sf2=sf2)
    assert "gone.kit" in capsys.readouterr().err


def test_kit_load_ignores_appledouble_companion_files(tmp_path):
    """kit.load() ignores AppleDouble ._*.wav files (exFAT quirk)."""
    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()
    # Create real WAV files
    sf.write(str(kit_dir / "hat_0.wav"), np.zeros((4410, 2), np.float32), SR)
    sf.write(str(kit_dir / "kick_0.wav"), np.zeros((4410, 2), np.float32), SR)
    # Create AppleDouble companion files (garbage that should be ignored)
    (kit_dir / "._hat_0.wav").write_bytes(b"AppleDouble companion junk")
    (kit_dir / "._kick_0.wav").write_bytes(b"AppleDouble companion junk")

    # kit.load should ignore the ._*.wav files and load only real WAV files
    samples = kit.load(kit_dir, SR)

    assert "hat" in samples
    assert "kick" in samples
    assert len(samples["hat"]) == 1
    assert len(samples["kick"]) == 1
    assert samples["hat"][0].shape == (4410, 2)
    assert samples["kick"][0].shape == (4410, 2)
