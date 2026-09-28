"""`encode()`'s temp workdir lifecycle (round-1 finding 4): without
`--workdir`, `encode` used to `tempfile.mkdtemp()` a full stem set (GBs for
a full song) and never remove it -- twelve abandoned `sc-*` dirs filled the
disk. `encode` must remove a workdir it created itself once the .sc is
written, but must never touch one the caller supplied.

Every stage `encode()` calls is faked here so a full `encode()` run costs
microseconds -- none of it needs real audio.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import encode as enc  # noqa: E402
from soundcode import lyrics  # noqa: E402


def _fake_stage(name, **kw):
    return enc.Stage(name, ok=True, **kw)


def _patch_heavy_stages(monkeypatch):
    monkeypatch.setattr(enc, "load_audio",
                        lambda path, sr=44100: (np.zeros((2, sr * 2), np.float32), sr))
    monkeypatch.setattr(enc, "separate_stems", lambda path, workdir: {})
    monkeypatch.setattr(enc, "stage_grid",
                        lambda y, sr, duration: (_fake_stage("grid"),
                                                 {"tempo": 120.0, "downbeat": 0.0, "bar_dur": 2.0}))
    monkeypatch.setattr(enc, "stage_struct", lambda y, sr, grid, duration: _fake_stage("struct"))
    monkeypatch.setattr(enc, "stage_harmony", lambda y, sr, grid: _fake_stage("harmony"))
    monkeypatch.setattr(enc, "stage_contour",
                        lambda stem, stem_name, mix, sr: _fake_stage("contour.vocals"))
    monkeypatch.setattr(enc, "stage_percussion", lambda stem, y, sr, grid: _fake_stage("perc.drums"))
    monkeypatch.setattr(enc, "stage_notes_poly",
                        lambda stem, name, grid, mix=None, sr=None, stem_name=None:
                            _fake_stage(f"notes.{name}"))
    monkeypatch.setattr(enc, "stage_lyrics",
                        lambda stem, sr, grid, ref_lines, duration, offset: _fake_stage("text.vox"))
    monkeypatch.setattr(enc, "stage_mix", lambda y, sr: _fake_stage("mix"))
    monkeypatch.setattr(enc, "_estimate_key", lambda y, sr: "C")
    monkeypatch.setattr(lyrics, "guess_title_artist",
                        lambda src, title, artist: (title or "Test Song", artist or "Test Artist"))
    monkeypatch.setattr(lyrics, "lookup", lambda title, artist, duration: None)


def _spy_mkdtemp(monkeypatch):
    """Records every dir `encode()` asks `tempfile.mkdtemp` for, while still
    letting the real call through (so `encode`'s own `wd.mkdir` etc. work)."""
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*a, **kw):
        d = real_mkdtemp(*a, **kw)
        created.append(d)
        return d

    monkeypatch.setattr(enc.tempfile, "mkdtemp", spy)
    return created


def test_encode_without_workdir_removes_its_own_temp_dir(tmp_path, monkeypatch):
    _patch_heavy_stages(monkeypatch)
    monkeypatch.delenv("SOUNDCODE_KEEP_WORK", raising=False)
    created = _spy_mkdtemp(monkeypatch)
    out = tmp_path / "song.sc"

    enc.encode("song.wav", out_path=str(out))

    assert out.exists()
    assert len(created) == 1
    assert not Path(created[0]).exists()


def test_encode_keeps_its_temp_dir_when_soundcode_keep_work_is_set(tmp_path, monkeypatch):
    _patch_heavy_stages(monkeypatch)
    monkeypatch.setenv("SOUNDCODE_KEEP_WORK", "1")
    created = _spy_mkdtemp(monkeypatch)
    out = tmp_path / "song.sc"

    enc.encode("song.wav", out_path=str(out))

    assert out.exists()
    assert len(created) == 1
    assert Path(created[0]).exists()
    shutil.rmtree(created[0], ignore_errors=True)   # kept on purpose by the test; ours to clean up


def test_encode_never_removes_a_caller_supplied_workdir(tmp_path, monkeypatch):
    _patch_heavy_stages(monkeypatch)
    monkeypatch.delenv("SOUNDCODE_KEEP_WORK", raising=False)
    created = _spy_mkdtemp(monkeypatch)
    workdir = tmp_path / "work"
    workdir.mkdir()
    out = tmp_path / "song.sc"

    enc.encode("song.wav", out_path=str(out), workdir=str(workdir))

    assert out.exists()
    assert workdir.exists()          # caller-owned: encode never deletes it
    assert created == []             # no tempfile.mkdtemp at all when given a workdir


def test_encode_cleans_up_even_when_a_stage_raises(tmp_path, monkeypatch):
    """The temp dir must go even when `encode` fails partway through --
    otherwise every failed run leaks one too."""
    _patch_heavy_stages(monkeypatch)
    monkeypatch.delenv("SOUNDCODE_KEEP_WORK", raising=False)
    created = _spy_mkdtemp(monkeypatch)

    def boom(y, sr, grid):
        raise RuntimeError("boom")

    monkeypatch.setattr(enc, "stage_harmony", boom)

    try:
        enc.encode("song.wav", out_path=str(tmp_path / "song.sc"))
    except RuntimeError:
        pass

    assert len(created) == 1
    assert not Path(created[0]).exists()
