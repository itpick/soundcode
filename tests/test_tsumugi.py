"""tsumugi subprocess wrapper (spec Part C)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import tsumugi as ts  # noqa: E402


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    py = tmp_path / "tsumugi" / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setenv("SOUNDCODE_TSUMUGI", str(tmp_path / "tsumugi"))
    return tmp_path / "tsumugi"


def test_missing_install_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCODE_TSUMUGI", str(tmp_path / "nope"))
    with pytest.raises(ts.TsumugiError, match="install_tsumugi.sh"):
        ts.home()


def test_transcribe_builds_the_cli_call(fake_home, monkeypatch, tmp_path):
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"], seen["cwd"] = cmd, kw.get("cwd")
        Path(cmd[cmd.index("--output-midi") + 1]).write_bytes(b"MThd")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    out = ts.transcribe(tmp_path / "piano.wav", tmp_path / "piano.mid", "default")
    assert out.exists()
    cmd = seen["cmd"]
    assert cmd[0].endswith(".venv/bin/python") and cmd[1:3] == ["-m", "instrument_agnostic_amt.amt.cli.infer"]
    assert cmd[cmd.index("--type") + 1] == "default"
    assert "--device" in cmd and "--disable-tqdm" in cmd
    assert seen["cwd"] == str(fake_home)


def test_nonzero_exit_names_the_step(fake_home, monkeypatch, tmp_path):
    monkeypatch.setattr(ts.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom\nRuntimeError: bad audio"))
    with pytest.raises(ts.TsumugiError, match="amt.cli.infer: RuntimeError: bad audio"):
        ts.transcribe(tmp_path / "x.wav", tmp_path / "x.mid", "default")


def test_refine_reads_global_candidates(fake_home, monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        report = {"global_candidates": [{"class_name": "piano", "probability": 0.73},
                                        {"class_name": "electric_piano", "probability": 0.27}]}
        Path(cmd[cmd.index("--output-json") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    got = ts.refine(tmp_path / "g.wav", tmp_path / "g.mid", "piano", tmp_path / "g.json")
    assert got == [("piano", 0.73), ("electric_piano", 0.27)]


def test_refine_bad_json_is_a_tsumugi_error(fake_home, monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        Path(cmd[cmd.index("--output-json") + 1]).write_text("{not json")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    with pytest.raises(ts.TsumugiError, match="refine"):
        ts.refine(tmp_path / "g.wav", tmp_path / "g.mid", "piano", tmp_path / "g.json")


def test_stem_model_table_covers_every_separation_stem():
    from soundcode.separate import STEMS
    assert set(STEMS) <= set(ts.STEM_MODEL)


def test_paths_are_passed_absolute_because_tsumugi_runs_in_its_own_dir(fake_home, monkeypatch, tmp_path):
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        Path(cmd[cmd.index("--output-midi") + 1]).write_bytes(b"MThd")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    ts.transcribe(Path("piano.wav"), Path("piano.mid"), "default")
    cmd = seen["cmd"]
    assert Path(cmd[cmd.index("--audio") + 1]).is_absolute()
    assert Path(cmd[cmd.index("--output-midi") + 1]).is_absolute()


# --- MIDI -> .sc lines --------------------------------------------------------------------

import pretty_midi  # noqa: E402
from soundcode import tsumugi_sc as tsc  # noqa: E402

GRID = {"downbeat": 1.0, "bar_dur": 2.0}          # 120 bpm, 4/4, bar 1 at 1.0 s


def _midi(tmp_path, tracks):
    pm = pretty_midi.PrettyMIDI()
    for program, is_drum, notes in tracks:
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
        for s, e, p, v in notes:
            inst.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=e))
        pm.instruments.append(inst)
    path = tmp_path / "t.mid"
    pm.write(str(path))
    return path


def test_read_tracks_labels_by_program_and_drums(tmp_path):
    p = _midi(tmp_path, [(5, False, [(1.0, 1.5, 60, 90)]), (0, True, [(1.0, 1.1, 39, 100)])])
    tracks = tsc.read_tracks(p)
    assert [(t.klass, t.inst) for t in tracks] == [("electric_piano", "keys.ep"),
                                                   ("drums", "drums.kit.standard")]


def test_positions_keep_performed_timing_and_pickups():
    assert tsc.position(1.0, GRID) == "1:1.000"
    assert tsc.position(1.0 + 0.5 + 0.0625, GRID) == "1:2.125"     # a 32nd after beat 2
    assert tsc.position(3.0, GRID) == "2:1.000"
    assert tsc.position(0.4, GRID) == "@0.400"                     # before the first downbeat


def test_note_lines_pitched_and_drums():
    t = tsc.Track("piano", "keys.piano", [(1.5625, 1.9375, 60, 88)])
    assert tsc.note_lines(t, GRID) == ["1:2.125  C4  0.750b 88"]
    d = tsc.Track("drums", "drums.kit.standard", [(3.0, 3.1, 39, 96)])
    assert tsc.note_lines(d, GRID) == ["2:1.000 clap 96"]


def test_bleed_tracks_are_dropped_with_a_reason():
    big = tsc.Track("piano", "keys.piano", [(i, i + 0.5, 60, 90) for i in range(200)])
    tiny = tsc.Track("strings", "strings.ensemble", [(0, 1, 60, 90), (1, 2, 62, 90)])
    rare = tsc.Track("organ", "organ.drawbar", [(i, i + 0.5, 60, 90) for i in range(3)])
    kept, why = tsc.drop_bleed([big, tiny, rare])
    assert [t.klass for t in kept] == ["piano"]
    assert any("strings" in w and "2 notes" in w for w in why)
    assert any("organ" in w for w in why)                          # 3 of 205 < 2 %


def test_empty_midi_gives_no_tracks(tmp_path):
    assert tsc.read_tracks(_midi(tmp_path, [])) == []


def test_stream_names_do_not_clash():
    taken: set[str] = set()
    assert tsc.stream_name("keys.piano", "piano", taken) == "notes.piano"
    assert tsc.stream_name("keys.piano", "guitar", taken) == "notes.piano.guitar"
    assert tsc.stream_name("keys.ep", "guitar", taken) == "notes.ep"


# --- inventory voting ------------------------------------------------------------------------

from soundcode import gm, inventory as inv  # noqa: E402


def _t(klass, onsets, pitch=60):
    return tsc.Track(klass, gm.TSUMUGI[klass]["inst"], [(o, o + 0.3, pitch, 90) for o in onsets])


def test_mix_votes_count_matching_notes_by_family():
    stem = [_t("distorted_guitar", [1.0, 2.0, 3.0, 4.0])]
    mix = [_t("electric_piano", [1.02, 2.01, 3.03]), _t("strings", [4.0])]
    assert inv.mix_votes(stem, mix) == {"keys": 3, "strings": 1}


def test_guitar_stem_that_the_mix_calls_keys_is_reassigned():
    d = inv.decide("guitar", "gtr", [("distorted_guitar", 0.97)], {"keys": 8, "gtr": 1, "strings": 1})
    assert d.family == "keys" and d.reassigned
    assert "reassigned" in d.warn and "0.80" in d.warn


def test_agreement_keeps_the_stem_family_with_combined_confidence():
    d = inv.decide("piano", "keys", [("piano", 0.97)], {"keys": 9, "gtr": 1})
    assert d.family == "keys" and not d.reassigned
    assert abs(d.conf - 0.97 * 0.9) < 1e-9


def test_too_few_votes_leaves_refinement_in_charge():
    d = inv.decide("bass", "bass", [("electric_bass", 0.88)], {"keys": 3})
    assert d.family == "bass" and not d.reassigned and d.conf == 0.88
    assert "few mix votes" in d.warn
