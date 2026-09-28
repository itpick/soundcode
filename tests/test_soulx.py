"""SoulX-Singer adapter (spec 2026-09-27-lyrics-and-voice §2).

Format pinned from SoulX-Singer@81aeb3a preprocess/tools/midi_parser.py:
note_type 1 = rest <SP>, 2 = a word's first note, 3 = continuation note of the
previous word (melisma); f0 at 24 kHz / hop 480 = 50 Hz; time in ms; durations in s."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from soundcode import sing, soulx  # noqa: E402
from soundcode import sing_score as ss  # noqa: E402
from soundcode.parser import parse  # noqa: E402
from test_sing import SONG  # noqa: E402


def test_metadata_matches_the_pinned_format():
    segs = soulx.metadata(parse(SONG), 0.0, 4.0)
    assert len(segs) == 1
    s = segs[0]
    n = len(s["note_pitch"].split())
    assert len(s["text"].split()) == len(s["duration"].split()) == len(s["note_type"].split()) == n
    types, pitches, text = s["note_type"].split(), s["note_pitch"].split(), s["text"].split()
    assert types[text.index("river")] == "2" and pitches[text.index("river")] == "64"
    assert all(p == "0" for t, p in zip(text, pitches) if t == "<SP>")
    assert "en_R-IH1-V-ER0" in s["phoneme"].split()
    assert s["time"] == [0, 4000] and s["language"] == "English"
    assert abs(sum(float(d) for d in s["duration"].split()) - 4.0) < 0.02
    assert len(s["f0"].split()) == round(4.0 * soulx.F0_RATE)


def test_melisma_notes_are_continuations():
    doc = parse(SONG.replace('1:1.000 "hello" 1.000b | 1:3.000 "river" 2.000b', '1:1.000 "hello" 2.000b'))
    s = soulx.metadata(doc, 0.0, 4.0)[0]
    types, text = s["note_type"].split(), s["text"].split()
    assert types[:2] == ["2", "3"] and text[:2] == ["hello", "hello"]


def test_long_songs_split_at_rests_under_15s():
    notes = "\n".join(f"@{t:.2f} C4 0.5s 90" for k in range(5) for t in np.arange(k * 8.0, k * 8.0 + 6.0, 0.5))
    words = " | ".join(f'@{t:.2f} "la" 0.5s' for k in range(5) for t in np.arange(k * 8.0, k * 8.0 + 6.0, 0.5))
    doc = parse(f"%sc 0.3\n@duration 40.0\n\n:notes.lead inst=voice.lead\n{notes}\n\n:text.vox\n{words}\n")
    segs = soulx.metadata(doc, 0.0, 40.0)
    assert len(segs) >= 3 and all(s["time"][1] - s["time"][0] <= 15000 for s in segs)
    covered = sum(sum(1 for t in s["note_type"].split() if t != "1") for s in segs)
    assert covered == 60


def test_soulx_failure_falls_back_to_diffsinger(tmp_path, monkeypatch):
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(44100, np.float32), 44100)
    monkeypatch.setattr(soulx, "render", lambda *a, **k: (_ for _ in ()).throw(ss.SingError("CUDA OOM")))
    monkeypatch.setattr(sing.diffsinger, "render", lambda score, **k: np.zeros(44100, np.float32))
    def fake_convert(src, r, out, steps=30):
        sf.write(str(out), np.zeros(44100, np.float32), 44100); return out
    monkeypatch.setattr(sing.seedvc, "convert", fake_convert)
    p, warns = sing.sing(parse(SONG), ref, cache=tmp_path / "c", singer="soulx")
    assert p.exists() and any("soulx" in w.lower() and "CUDA OOM" in w for w in warns)


def test_soulx_does_not_need_the_diffsinger_bank(tmp_path, monkeypatch):
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(44100, np.float32), 44100)
    monkeypatch.setenv("SOUNDCODE_DIFFSINGER", str(tmp_path / "nope"))
    monkeypatch.setattr(soulx, "render", lambda *a, **k: np.zeros(44100, np.float32))
    p, _ = sing.sing(parse(SONG), ref, cache=tmp_path / "c", singer="soulx")
    assert p.exists()


def test_long_unbroken_singing_is_cut_at_a_rest_not_mid_note():
    notes = "\n".join(f"@{t:.2f} C4 0.45s 90" for t in np.arange(0.0, 20.0, 0.5))    # 50 ms gaps only
    words = " | ".join(f'@{t:.2f} "la" 0.45s' for t in np.arange(0.0, 20.0, 0.5))
    doc = parse(f"%sc 0.3\n@duration 20.0\n\n:notes.lead inst=voice.lead\n{notes}\n\n:text.vox\n{words}\n")
    segs = soulx.metadata(doc, 0.0, 20.0)
    assert all(s["time"][1] - s["time"][0] <= 15000 for s in segs)
    for s in segs:
        assert all(float(d) >= 0.01 for d in s["duration"].split())
    total_notes = sum(1 for s in segs for t in s["note_type"].split() if t != "1")
    assert total_notes == 40                                                      # no note split in two


# --- every word is sung (word-driven note alignment) --------------------------------------

def test_a_long_note_spanning_two_words_is_split_so_both_are_sung():
    doc = parse(SONG.replace("1:3.000  E4  2.000b 90", "1:3.000  E4  2.000b 90")
                .replace('1:1.000 "hello" 1.000b | 1:3.000 "river" 2.000b',
                         '1:1.000 "hello" 1.000b | 1:3.000 "down" 1.000b | 1:4.000 "river" 1.000b'))
    s = soulx.metadata(doc, 0.0, 4.0)[0]
    words = [t for t, ty in zip(s["text"].split(), s["note_type"].split()) if ty == "2"]
    assert [w for w in words if w != "ah"] == ["hello", "down", "river"]


def test_a_word_between_notes_still_gets_a_note():
    # the E4 note is shortened to 1.0-1.1 s; "gap" is sung at 2.9-3.5 s where no note exists
    doc2 = parse(SONG.replace("1:3.000  E4  2.000b 90", "1:3.000  E4  0.200b 90")
                 .replace('1:1.000 "hello" 1.000b | 1:3.000 "river" 2.000b',
                          '1:1.000 "hello" 1.000b | @2.9 "gap" 0.6s'))
    s = soulx.metadata(doc2, 0.0, 4.0)[0]
    text, types, pitch = s["text"].split(), s["note_type"].split(), s["note_pitch"].split()
    k = text.index("gap")
    assert types[k] == "2" and int(pitch[k]) > 0


def test_river_sends_every_word(tmp_path):
    p = Path(__file__).resolve().parents[1] / "out" / "sc" / "lv" / "river-30s.sc"
    if not p.exists():
        pytest.skip("no River .sc")
    from soundcode.parser import parse_file
    d = parse_file(str(p))
    lyric = [w for a, _, w in ss.words(d) if a < d.duration - 0.05]
    segs = soulx.metadata(d, 0.0, d.duration)
    sung = [t for s in segs for t, ty in zip(s["text"].split(), s["note_type"].split()) if ty == "2"]
    assert [w for w in sung if w != "ah"] == lyric


def test_render_seeds_and_overrides_the_diffusion_settings(tmp_path, monkeypatch):
    import subprocess
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(44100 * 4, np.float32), 44100)
    cmds = []

    def fake_run(cmd, **k):
        cmds.append(cmd)
        if cmd[0] == "scp" and cmd[-1].endswith("generated.wav"):
            sf.write(cmd[-1], np.zeros(24000, np.float32), 24000)
        if cmd[0] == "scp" and any(str(a).endswith("run.py") for a in cmd):
            run_py = next(Path(a) for a in cmd if str(a).endswith("run.py"))
            cmds.append(["run.py", run_py.read_text()])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(soulx.subprocess, "run", fake_run)
    soulx.render(parse(SONG), ref, n_steps=64, cfg=5)
    remote = next(c[-1] for c in cmds if c[0] == "ssh" and "run.py" in c[-1])
    assert "run.py 0 " in remote                       # default seed 0 → repeatable renders
    assert "n_steps:\\).*/\\1 64/" in remote and "cfg:\\).*/\\1 5.0/" in remote
    assert "--config ~/" in remote                      # the job's edited copy, not the shipped config
    assert "torch.manual_seed(s)" in next(c[1] for c in cmds if c[0] == "run.py")

    cmds.clear()
    soulx.render(parse(SONG), ref, seed=None)
    remote = next(c[-1] for c in cmds if c[0] == "ssh" and "run.py" in c[-1])
    assert "run.py -1 " in remote and "--config soulxsinger/config/soulxsinger.yaml" in remote



def test_soulx_settings_are_part_of_the_vocal_cache_key(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(soulx, "render", lambda *a, **k: calls.append(1) or np.zeros(44100, np.float32))
    ref = tmp_path / "ref.wav"
    import soundfile as sf
    sf.write(str(ref), np.zeros(44100, np.float32), 44100)
    doc = parse(SONG)
    a, _ = sing.sing(doc, ref=ref, cache=tmp_path, singer="soulx")
    monkeypatch.setattr(soulx, "PROMPT_S", soulx.PROMPT_S + 4)
    b, _ = sing.sing(doc, ref=ref, cache=tmp_path, singer="soulx")
    assert a != b and len(calls) == 2


def test_render_never_transposes_the_melody(tmp_path, monkeypatch):
    # SoulX's --auto_shift moves every segment to the voice prompt's median pitch, each by its
    # own amount: full-length Discipline was sung 2-9 semitones off, section by section.
    import subprocess
    import soundfile as sf
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(44100 * 4, np.float32), 44100)
    remote = []

    def fake_run(cmd, **k):
        if cmd[0] == "ssh" and "run.py" in cmd[-1]:
            remote.append(cmd[-1])
        if cmd[0] == "scp" and cmd[-1].endswith("generated.wav"):
            sf.write(cmd[-1], np.zeros(24000, np.float32), 24000)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(soulx.subprocess, "run", fake_run)
    soulx.render(parse(SONG), ref)
    assert remote and "--auto_shift" not in remote[0] and "--pitch_shift 0" in remote[0]
