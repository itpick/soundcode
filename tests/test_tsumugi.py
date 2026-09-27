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
