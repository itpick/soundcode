"""Singing thin slice (spec 2026-09-27)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import contour as ct  # noqa: E402
from soundcode.parser import parse  # noqa: E402


def test_phrases_resample_to_20ms_and_split_on_long_gaps():
    t = np.arange(0, 2.0, 0.01)                       # 10 ms frames
    cents = np.full_like(t, 6900.0)
    voiced = np.ones_like(t, bool)
    voiced[(t >= 0.50) & (t < 0.53)] = False          # 30 ms: filled
    voiced[(t >= 1.00) & (t < 1.20)] = False          # 200 ms: split
    ph = ct.phrases(t, cents, voiced)
    assert [round(s, 2) for s, _ in ph] == [0.0, 1.2]
    assert len(ph[0][1]) == 50 and all(v == 6900 for v in ph[0][1])


def test_vibrato_survives_at_50hz():
    t = np.arange(0, 1.0, 0.01)
    cents = 6900 + 50 * np.sin(2 * np.pi * 5.5 * t)   # 5.5 Hz, +/-50 c
    (start, vals), = ct.phrases(t, cents, np.ones_like(t, bool))
    assert max(vals) - min(vals) >= 90


def test_contour_lines_round_trip_through_the_parser():
    ph = [(12.34, [6912, 6915, 6920]), (14.02, list(range(6700, 6760)))]
    text = "%sc 0.3\n\n:contour.vox rate=50\n" + "\n".join(ct.contour_lines(ph)) + "\n"
    segs = ct.read_contour(parse(text))
    assert [round(s, 2) for s, _ in segs] == [12.34, 14.02, 15.02]       # 60 values -> 50 + 10
    assert list(segs[0][1]) == [6912, 6915, 6920]
    assert len(segs[1][1]) == 50 and len(segs[2][1]) == 10


def test_no_voiced_frames_gives_no_phrases():
    t = np.arange(0, 1.0, 0.01)
    assert ct.phrases(t, np.zeros_like(t), np.zeros_like(t, bool)) == []


def test_stage_contour_on_a_gated_stem_writes_nothing(tmp_path):
    import soundfile as sf
    from soundcode import encode as enc
    sr = 16000
    p = tmp_path / "lead_vocals.wav"
    sf.write(str(p), np.zeros(sr * 4, np.float32), sr)
    st = enc.stage_contour(p, "lead_vocals", np.random.default_rng(0).standard_normal(sr * 4) * 0.1, sr)
    assert not st.ok


# --- lyrics as performed ------------------------------------------------------------------

from soundcode import encode as enc  # noqa: E402

GRID = {"downbeat": 1.0, "bar_dur": 2.0}


def test_text_events_keep_their_duration():
    doc = parse('%sc 0.3\n\n:text.vox\n1:2.125 "river" 0.500b ?0.71\n@0.400 "I" 0.200s\n')
    evs = doc.stream("text.vox").events
    assert [(e.text, e.dur) for e in evs] == [("river", "0.500b"), ("I", "0.200s")]


def test_lyric_cells_use_performed_timing_and_durations():
    words = [(1.5625, 1.9375, "river", 0.9), (0.4, 0.6, "I", 0.5)]
    cells = enc.lyric_cells(words, GRID)
    assert cells == ['@0.400 "I" 0.200s ?0.50', '1:2.125 "river" 0.750b']   # chronological


def test_colliding_words_are_nudged_20ms():
    words = [(1.5, 1.8, "a", 0.9), (1.5, 1.9, "b", 0.9)]
    cells = enc.lyric_cells(words, GRID)
    assert cells[1].startswith("1:2.040")                         # +20 ms at 0.5 s per beat


# --- vocal score --------------------------------------------------------------------------

from soundcode import sing_score as ss  # noqa: E402

SONG = """%sc 0.3
@duration 4.0

:grid
meter @0.000 4/4
anchor bar 1 @0.000
tempo @0.000 120

:notes.lead inst=voice.lead
meta stem=lead_vocals
1:1.000  C4  1.000b 90
1:2.000  D4  1.000b 90
1:3.000  E4  2.000b 90

:text.vox
1:1.000 "hello" 1.000b | 1:3.000 "river" 2.000b
"""


def test_vocal_stream_and_notes():
    doc = parse(SONG)
    assert ss.vocal_stream(doc) == "notes.lead"
    assert ss.vocal_notes(doc, "notes.lead") == [(0.0, 0.5, 60), (0.5, 1.0, 62), (1.0, 2.0, 64)]


def test_words_and_g2p():
    doc = parse(SONG)
    assert [(round(a, 2), round(b, 2), w) for a, b, w in ss.words(doc)] == [(0.0, 0.5, "hello"), (1.0, 2.0, "river")]
    ph, hit = ss.g2p("river")
    assert hit and ph[0] == "en/r" and "en/er" in ph


def test_unknown_word_falls_back_and_warns():
    ph, hit = ss.g2p("zzxqv")
    assert not hit and ph                                   # letter-based guess, never empty


def test_score_frames_cover_the_song_and_f0_follows_notes():
    sc = ss.build(parse(SONG))
    assert sum(sc.frames) == sc.n_frames == int(round(4.0 * ss.SR / ss.HOP))
    t = (np.arange(sc.n_frames) + 0.5) * ss.HOP / ss.SR
    mid_c4 = sc.f0_hz[(t > 0.1) & (t < 0.4)]
    assert np.all(np.abs(1200 * np.log2(mid_c4 / 261.63)) < 5)
    assert "SP" in sc.phonemes and sc.phonemes.count("en/ow") == 1


def test_contour_overrides_notes_where_present():
    doc = parse(SONG + "\n:contour.vox rate=50\nf0  @0.100  " + " ".join(["6030"] * 10) + "\n")
    sc = ss.build(doc)
    t = (np.arange(sc.n_frames) + 0.5) * ss.HOP / ss.SR
    seg = sc.f0_hz[(t > 0.12) & (t < 0.28)]
    assert np.all(np.abs(1200 * np.log2(seg / 261.63) - 30) < 3)        # C4 + 30 c from contour


def test_melisma_and_extra_words():
    doc = parse(SONG.replace('| 1:3.000 "river" 2.000b', '| 1:2.500 "river" 0.500b | 1:2.750 "run" 0.250b'))
    sc = ss.build(doc)
    assert sum(sc.frames) == sc.n_frames
    assert not any("crash" in w for w in sc.warnings)


# --- DiffSinger engine -------------------------------------------------------------------

from soundcode import diffsinger as ds  # noqa: E402


def test_missing_bank_is_one_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCODE_DIFFSINGER", str(tmp_path / "nope"))
    with pytest.raises(ss.SingError, match="nope"):
        ds.bank_dir()


def _bank_present():
    try:
        ds.bank_dir(), ds.vocoder_path()
        return True
    except ss.SingError:
        return False


@pytest.mark.skipif(not _bank_present(), reason="DiffSinger bank not installed")
def test_diffsinger_sings_two_bars_at_the_requested_pitch():
    import librosa
    sc = ss.build(parse(SONG))
    y = ds.render(sc)
    assert abs(len(y) / ss.SR - 4.0) < 0.1 and np.abs(y).max() > 0.1
    f0, v, _ = librosa.pyin(y, fmin=100, fmax=600, sr=ss.SR, frame_length=2048)
    t = librosa.times_like(f0, sr=ss.SR)
    seg = f0[(t > 1.3) & (t < 1.8) & v]                    # the E4 note
    assert len(seg) and abs(np.median(1200 * np.log2(seg / 329.63))) < 50


# --- sing orchestrator + render --with-vocals ---------------------------------------------

from soundcode import render_sf, sing  # noqa: E402


def test_voice_ref_defaults_to_the_songs_lead_stem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "out" / "stems" / "river-30s"
    d.mkdir(parents=True)
    (d / "lead_vocals.wav").write_bytes(b"RIFF")
    doc = parse("%sc 0.3\n@source river-30s.wav\n")
    assert sing.voice_ref(doc, None) == Path("out/stems/river-30s/lead_vocals.wav")
    with pytest.raises(ss.SingError, match="--voice-ref"):
        sing.voice_ref(parse("%sc 0.3\n@source other.wav\n"), None)


def test_cache_key_changes_with_every_input(tmp_path):
    ref = tmp_path / "r.wav"
    ref.write_bytes(b"x")
    k = sing.cache_key(parse(SONG), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG.replace("E4", "F4")), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG.replace('"river"', '"rover"')), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG + "\n:contour.vox rate=50\nf0  @0.1  6000\n"), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG), ref, {"steps": 50})


def test_sing_uses_the_cache_and_chains_the_engines(tmp_path, monkeypatch):
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(4410, np.float32), 44100)
    calls = []
    monkeypatch.setattr(sing.diffsinger, "render", lambda score, **k: calls.append("ds") or np.zeros(44100, np.float32))
    def fake_convert(src, r, out, steps=30):
        calls.append("vc"); sf.write(str(out), np.zeros(44100, np.float32), 44100); return out
    monkeypatch.setattr(sing.seedvc, "convert", fake_convert)
    p1, _ = sing.sing(parse(SONG), ref, cache=tmp_path / "c")
    p2, _ = sing.sing(parse(SONG), ref, cache=tmp_path / "c")
    assert p1 == p2 and p1.exists() and calls == ["ds", "vc"]


def test_render_with_vocals_mixes_the_sung_stream_level_matched(tmp_path, monkeypatch):
    import soundfile as sf
    sung = tmp_path / "sung.wav"
    t = np.arange(44100 * 4) / 44100
    sf.write(str(sung), (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 44100)
    monkeypatch.setattr(sing, "sing", lambda doc, ref=None, **k: (sung, []))
    doc = parse(SONG.replace("meta stem=lead_vocals", "meta stem=lead_vocals level=-30.0"))
    keys = np.zeros((44100 * 5, 2), np.float32)
    monkeypatch.setattr(render_sf, "render_streams", lambda d, sr=None, sf2=None, **k: {"notes.lead": keys})
    without = render_sf.render(doc, 44100)
    with_v = render_sf.render(doc, 44100, with_vocals=True)
    assert np.abs(without).max() == 0 and np.abs(with_v).max() > 0.1


def test_seedvc_runs_on_the_remote_gpu_host_when_configured(tmp_path, monkeypatch):
    """SOUNDCODE_SEEDVC_HOST: copy inputs over, run there, copy the result back."""
    import subprocess
    from soundcode import seedvc
    monkeypatch.setenv("SOUNDCODE_SEEDVC_HOST", "gpubox")
    src, ref, out = tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "o" / "v.wav"
    src.write_bytes(b"s"); ref.write_bytes(b"r")
    cmds = []
    def fake_run(cmd, **kw):
        cmds.append(cmd)
        if cmd[0] == "scp" and not cmd[-1].startswith("gpubox:"):      # fetch back
            Path(cmd[-1]).write_bytes(b"converted")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(seedvc.subprocess, "run", fake_run)
    got = seedvc.convert(src, ref, out, steps=30)
    assert got == out.resolve() and out.read_bytes() == b"converted"
    kinds = [c[0] for c in cmds]
    assert kinds[:2] == ["ssh", "scp"] and "ssh" in kinds[2:] and kinds[-1] == "ssh"   # mkdir, push, run, fetch, cleanup
    run = next(c for c in cmds[2:] if c[0] == "ssh")
    assert "inference.py" in run[-1] and "--diffusion-steps 30" in run[-1]


def test_remote_failure_is_a_sing_error(tmp_path, monkeypatch):
    import subprocess
    from soundcode import seedvc
    monkeypatch.setenv("SOUNDCODE_SEEDVC_HOST", "gpubox")
    (tmp_path / "s.wav").write_bytes(b"s"); (tmp_path / "r.wav").write_bytes(b"r")
    monkeypatch.setattr(seedvc.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 255, "", "ssh: connect to host gpubox: timed out"))
    with pytest.raises(ss.SingError, match="gpubox"):
        seedvc.convert(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "v.wav")


def test_f0_extraction_is_deterministic(tmp_path):
    """torchcrepe's default Viterbi decoder is random (~9 c between identical
    calls, even seeded); the contour must be repeatable."""
    import soundfile as sf
    sr = 16000
    t = np.arange(sr * 2) / sr
    y = (0.3 * np.sin(2 * np.pi * np.cumsum(220 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))) / sr)).astype(np.float32)
    p = tmp_path / "v.wav"
    sf.write(str(p), y, sr)
    a, b = ct.extract(p), ct.extract(p)
    np.testing.assert_array_equal(a[1], b[1])


# --- vocal level matching (Task 7 finding) -----------------------------------------------------

def test_sung_level_ignores_the_noise_floor():
    """Seed-VC output has a faint floor in every block; level matching must
    measure only where the voice is (within 30 dB of its loudest block)."""
    sr = 44100
    y = np.full(sr * 8, 10 ** (-80 / 20), np.float32)                 # -80 dB floor everywhere
    t = np.arange(sr * 2) / sr
    y[:sr * 2] = 0.3 * np.sin(2 * np.pi * 220 * t)                    # 2 s of singing
    out = render_sf.match_level(y, sr, -30.0)
    sung = out[:sr * 2]
    assert abs(20 * np.log10(np.sqrt(np.mean(sung.astype(np.float64) ** 2))) - (-30.0)) < 0.5


def test_compare_level_matches_the_sung_vocal(tmp_path):
    import soundfile as sf
    from soundcode import compare as cmp
    sr = 44100
    t = np.arange(sr * 2) / sr
    y = np.full(sr * 4, 1e-4, np.float32)
    y[:sr * 2] = 0.8 * np.sin(2 * np.pi * 220 * t)
    wav = tmp_path / "sung.wav"
    sf.write(str(wav), y, sr)
    doc = parse(SONG.replace("meta stem=lead_vocals", "meta stem=lead_vocals level=-30.0"))
    m = cmp.sung_stream(doc, wav, cmp.SR * 4)
    seg = m[:cmp.SR * 2]
    assert abs(20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) - (-30.0)) < 0.5


# --- final-review fixes ------------------------------------------------------------------------

SAFE = {"en/" + p for p in ("aa ae ah ao aw ax ay b ch d dh eh er ey f g hh ih iy jh k l m n ng "
                             "ow oy p r s sh t th uh uw v w y z zh").split()}


def test_letter_fallback_only_emits_bank_phonemes():
    for w in ("cuz", "doncha", "jinx", "xq", "qwerty"):
        ph, hit = ss.g2p(w)
        assert set(ph) <= SAFE, (w, ph)


def test_diffsinger_substitutes_unknown_phonemes_instead_of_dying(monkeypatch):
    sc = ss.build(parse(SONG))
    sc.phonemes[1] = "en/zzz"
    if not _bank_present():
        pytest.skip("no bank")
    y = ds.render(sc)
    assert np.abs(y).max() > 0.1 and any("zzz" in w for w in sc.warnings)


def test_failed_remote_download_does_not_poison_the_cache(tmp_path, monkeypatch):
    import subprocess
    from soundcode import seedvc
    monkeypatch.setenv("SOUNDCODE_SEEDVC_HOST", "gpubox")
    (tmp_path / "s.wav").write_bytes(b"s"); (tmp_path / "r.wav").write_bytes(b"r")
    out = tmp_path / "v.wav"
    def fake_run(cmd, **kw):
        if cmd[0] == "scp" and not cmd[-1].startswith("gpubox:"):
            Path(cmd[-1]).write_bytes(b"trunc")
            return subprocess.CompletedProcess(cmd, 1, "", "Connection closed")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(seedvc.subprocess, "run", fake_run)
    with pytest.raises(ss.SingError):
        seedvc.convert(tmp_path / "s.wav", tmp_path / "r.wav", out)
    assert not out.exists()


def test_timeouts_become_sing_errors_and_scp_is_batch(tmp_path, monkeypatch):
    import subprocess
    from soundcode import seedvc
    monkeypatch.setenv("SOUNDCODE_SEEDVC_HOST", "gpubox")
    (tmp_path / "s.wav").write_bytes(b"s"); (tmp_path / "r.wav").write_bytes(b"r")
    seen = []
    def fake_run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "scp":
            assert "BatchMode=yes" in " ".join(cmd)
        if cmd[0] == "ssh" and "inference.py" in cmd[-1]:
            raise subprocess.TimeoutExpired(cmd, 1800)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(seedvc.subprocess, "run", fake_run)
    with pytest.raises(ss.SingError, match="timed out"):
        seedvc.convert(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "v.wav")


def test_no_vocal_song_renders_instruments_with_a_note(monkeypatch, capsys):
    doc = parse("%sc 0.3\n@duration 2.0\n\n:notes.keys inst=keys.piano\n@0.0 C4 0.5s 100\n")
    keys = np.full((44100 * 2, 2), 0.1, np.float32)
    monkeypatch.setattr(render_sf, "render_streams", lambda d, sr=None, sf2=None, **k: {"notes.keys": keys})
    y = render_sf.render(doc, 44100, with_vocals=True)
    assert np.abs(y).max() > 0 and "no lead vocal" in capsys.readouterr().err


def test_notes_without_words_are_sung_on_ah():
    sc = ss.build(parse(SONG.replace(' | 1:3.000 "river" 2.000b', "")))
    t_end = np.cumsum(sc.frames) * ss.HOP / ss.SR
    t_start = t_end - np.array(sc.frames) * ss.HOP / ss.SR
    at = lambda t: sc.phonemes[int(np.searchsorted(t_end, t))]  # noqa: E731
    assert at(1.5) in ("en/aa", "en/ah")                  # the E4 note, no word: sung "ah"
    assert any("'ah'" in w for w in sc.warnings)


def test_contour_line_boundaries_leave_no_note_pitch_frames():
    vals = " ".join(["6030"] * 50)
    doc = parse(SONG + f"\n:contour.vox rate=50\nf0  @0.000  {vals}\nf0  @1.000  " + " ".join(["6030"] * 10) + "\n")
    sc = ss.build(doc)
    t = (np.arange(sc.n_frames) + 0.5) * ss.HOP / ss.SR
    seg = sc.f0_hz[(t > 0.02) & (t < 1.18)]
    assert np.all(np.abs(1200 * np.log2(seg / 261.63) - 30) < 3)


def test_cache_key_includes_duration_and_ignores_level(tmp_path):
    ref = tmp_path / "r.wav"; ref.write_bytes(b"x")
    k = sing.cache_key(parse(SONG), ref, {"steps": 30})
    assert k != sing.cache_key(parse(SONG.replace("@duration 4.0", "@duration 8.0")), ref, {"steps": 30})
    assert k == sing.cache_key(parse(SONG.replace("meta stem=lead_vocals", "meta stem=lead_vocals level=-30.0")), ref, {"steps": 30})


def test_compare_with_vocals_errors_are_one_line(tmp_path, monkeypatch, capsys):
    from soundcode import cli, compare as cmp
    def boom(*a, **k):
        raise ss.SingError("DiffSinger bank missing at /nope")
    monkeypatch.setattr(cmp, "run", boom)
    assert cli.main(["compare", "a.wav", "b.sc", "--with-vocals", "-o", str(tmp_path)]) == 2
    assert "bank missing" in capsys.readouterr().err


def test_sung_vocal_gets_the_lead_streams_fx(tmp_path):
    import soundfile as sf
    sr = 44100
    t = np.arange(sr * 2) / sr
    y = np.zeros(sr * 3, np.float32)
    y[:sr * 2] = 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 0.3)
    wav = tmp_path / "sung.wav"
    sf.write(str(wav), y, sr)
    base = "%sc 0.3\n\n:notes.lead inst=voice.lead\nmeta stem=lead_vocals level=-30.0\n"
    dry = render_sf._load_stream(wav, sr, sr * 3, parse(base).stream("notes.lead"))
    wet = render_sf._load_stream(wav, sr, sr * 3,
                                 parse(base + "fx  rt60=1.5s  wet=0.35  width=0.5\n").stream("notes.lead"))
    tail = slice(int(0.6 * sr), int(1.2 * sr))
    assert np.abs(wet[tail]).mean() > 3 * np.abs(dry[tail]).mean()


def test_model_durations_keep_word_boundaries_and_total_length():
    if not _bank_present():
        pytest.skip("no bank")
    heur = ss.build(parse(SONG))
    mod = ss.build(parse(SONG), durations="model")
    assert sum(mod.frames) == mod.n_frames == heur.n_frames
    assert mod.phonemes == heur.phonemes
    assert mod.frames != heur.frames                        # the model actually changed timing
    # word boundaries preserved: cumulative frames at each word start are equal
    b_h = np.cumsum([0] + heur.frames)
    b_m = np.cumsum([0] + mod.frames)
    for i0, i1 in mod.spans:
        assert b_h[i0] == b_m[i0] and b_h[i1] == b_m[i1]
