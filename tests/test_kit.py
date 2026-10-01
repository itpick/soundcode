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
K40_AT_0P4S = -np.log(10 ** (-40 / 20)) / 0.4    # exp decay rate: -40 dB at 0.4 s
K40_AT_1P2S = -np.log(10 ** (-40 / 20)) / 1.2    # a slower, more cymbal/snare-like ring


def _decaying_hit(onset_s: float, dur_s: float, sr: int, k: float, seed: int) -> tuple[int, np.ndarray]:
    """A broadband decaying one-shot: stereo noise times exp(-k*t)."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, 2)).astype(np.float32)
    env = np.exp(-k * t).astype(np.float32)[:, None]
    return int(onset_s * sr), noise * env


def _add(y: np.ndarray, i: int, seg: np.ndarray) -> None:
    m = min(len(seg), len(y) - i)
    if m > 0:
        y[i:i + m] += seg[:m]


def _rms_db(a: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


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


def test_isolated_prefers_the_long_free_gap_over_typical_velocity():
    """A hit followed by a full target-length gap is kept over one closer to
    the voice's median velocity but boxed in by the next hit (spec: 'prefer
    hits followed by a gap of at least the voice's target length')."""
    hits = [(0.5, "snare", 100), (0.8, "hat", 60),     # 0.5: only a 0.3 s gap
            (2.0, "snare", 40), (2.8, "hat", 60)]       # 2.0: a 0.8 s gap (>= 0.6 s target)
    assert kit.isolated(hits)["snare"] == [2.0]


def test_isolated_falls_back_to_the_longest_gap_when_none_is_long_enough():
    """When no hit of a voice has a clean target-length gap (e.g. hats on
    every eighth, no pause anywhere), fall back to the longest gap there is."""
    hits = [(0.0, "kick", 100), (0.3, "kick", 90), (0.6, "kick", 80),
            (0.65, "snare", 50)]
    # kick@0.0 -> gap 0.3; kick@0.3 -> gap 0.3; none reach the 0.5 s target,
    # so the longest (either, both 0.3) wins over picking by velocity alone.
    got = kit.isolated(hits)
    assert set(got["kick"]) <= {0.0, 0.3}
    assert 0.6 not in got.get("kick", [])        # crowded by the snare 50 ms later


def test_voice_target_lengths_match_the_brief():
    assert kit.target_length_s("kick") == 0.5
    assert kit.target_length_s("snare") == 0.6
    assert kit.target_length_s("clap") == 0.6
    assert kit.target_length_s("hat") == 0.25
    assert kit.target_length_s("hat.open") == 1.5
    assert kit.target_length_s("crash") == 1.5
    assert kit.target_length_s("ride") == 1.5
    assert kit.target_length_s("tom.hi") == 0.8
    assert kit.target_length_s("tom.floor.lo") == 0.8
    assert kit.target_length_s("cowbell") == 0.6       # anything else


def test_snare_sample_rings_past_the_hat_grid_without_a_discontinuity(tmp_path):
    """A snare boxed in by a hat 0.3 s later (closer than its 0.6 s target)
    used to be chopped mid-ring; it should now be extended to the target
    length with a decay that continues smoothly past the real cut."""
    dur = 2.5
    y = np.zeros((int(dur * SR), 2), np.float32)
    hits = []
    # a dense hat grid (every 0.23 s, per the brief) with one gap left around
    # the snare -- exactly the "snare-only bars" case the fix targets.
    for i, h in enumerate(list(np.arange(0.0, 0.70, 0.23)) + list(np.arange(1.3, dur, 0.23))):
        i0, seg = _decaying_hit(float(h), 0.1, SR, k=40.0, seed=100 + i)
        _add(y, i0, seg)
        hits.append((round(float(h), 3), "hat", 70))
    onset = 1.0
    i0, seg = _decaying_hit(onset, 1.0, SR, k=K40_AT_1P2S, seed=1)
    _add(y, i0, seg)
    hits.append((onset, "snare", 100))

    stem = tmp_path / "drums.wav"
    sf.write(str(stem), y, SR)
    made = kit.build(stem, hits, tmp_path / "kit")
    assert "snare" in made and len(made["snare"]) == 1
    out, sr = sf.read(str(made["snare"][0]))
    assert sr == SR
    assert len(out) >= int(0.5 * SR)                 # extended well past the 0.3 s cut

    nxt_hat = min(t for t, v, _ in hits if v == "hat" and t > onset)
    target_s = kit.target_length_s("snare")
    join = (int(min(nxt_hat - kit.PRE_S, onset + target_s) * SR) -
           int(max(0, onset - kit.PRE_S) * SR))
    w = int(0.05 * SR)
    before, after = out[join - w:join], out[join:join + w]
    assert abs(_rms_db(before) - _rms_db(after)) < 3.0


def test_already_quiet_tail_is_not_force_extended(tmp_path):
    """A hit that has already decayed below the -40 dB floor by the time it
    is cut isn't padded out with manufactured ringing -- chopping silence
    isn't audible, so the short sample is kept as-is."""
    dur = 1.0
    y = np.zeros((int(dur * SR), 2), np.float32)
    i0, seg = _decaying_hit(0.0, 1.0, SR, k=K40_AT_0P4S, seed=7)
    _add(y, i0, seg)
    # the next hit (any voice) comes 0.5 s later -- the kick's own target --
    # but by then its envelope has long passed -40 dB (reached at 0.4 s).
    hits = [(0.0, "kick", 100), (0.5, "hat", 70)]

    stem = tmp_path / "drums.wav"
    sf.write(str(stem), y, SR)
    made = kit.build(stem, hits, tmp_path / "kit")
    assert "kick" in made
    out, sr = sf.read(str(made["kick"][0]))
    target_n = int(kit.target_length_s("kick") * SR)
    assert len(out) < target_n                   # not padded out to the target


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
