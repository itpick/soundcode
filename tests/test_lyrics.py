"""Lyrics: LRCLIB lookup and reconciliation (spec 2026-09-27-lyrics-and-voice)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import lyrics as ly  # noqa: E402

LRC = "[00:01.00] Walk me down to the harbor\n[00:05.50] Lights are low and the tide is high\n[00:40.00] Walk me down to the harbor\n"


class FakeResp:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def read(self):
        return json.dumps(self.body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_guess_title_artist_from_file_names():
    assert ly.guess_title_artist(Path("The River-JordanFelix.mp3")) == ("The River", "Jordan Felix")
    assert ly.guess_title_artist(Path("Some Artist - A Song.wav")) == ("A Song", "Some Artist")
    assert ly.guess_title_artist(Path("x.wav"), "T", "A") == ("T", "A")


def test_lookup_parses_synced_lines_and_caches(tmp_path, monkeypatch):
    calls = []
    def fake_open(req, timeout=20):
        calls.append(req.full_url)
        return FakeResp({"trackName": "Harbor", "artistName": "Nobody", "duration": 60.0,
                         "syncedLyrics": LRC, "plainLyrics": "x"})
    monkeypatch.setattr(ly.urllib.request, "urlopen", fake_open)
    lines = ly.lookup("Harbor", "Nobody", 60.0, cache=tmp_path)
    assert lines[0] == ly.Line(1.0, "Walk me down to the harbor")
    assert ly.lookup("Harbor", "Nobody", 60.0, cache=tmp_path) == lines
    assert len(calls) == 1                                   # second call from the cache


def test_lookup_failure_is_none_not_a_crash(tmp_path, monkeypatch):
    import urllib.error
    def boom(req, timeout=20):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
    monkeypatch.setattr(ly.urllib.request, "urlopen", boom)
    assert ly.lookup("Harbor", "Nobody", 60.0, cache=tmp_path) is None


def test_ref_words_keep_only_the_clip_window():
    lines = [ly.Line(1.0, "Walk me down to the harbor"), ly.Line(5.5, "Lights are low, and the tide is high!"),
             ly.Line(40.0, "Walk me down to the harbor")]
    words = [w for _, w in ly.ref_words(lines, 0.0, 30.0)]
    assert words[:3] == ["walk", "me", "down"] and "high" in words and words.count("harbor") == 1
