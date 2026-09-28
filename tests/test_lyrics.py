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


# --- reconcile -------------------------------------------------------------------------------

ASR = [(1.0, 1.3, "walk", 0.9), (1.3, 1.5, "me", 0.95), (1.5, 1.8, "down", 0.9), (1.8, 2.0, "to", 0.9),
       (2.0, 2.2, "the", 0.9), (2.2, 2.9, "harder", 0.6),                  # misheard 'harbor'
       (3.0, 3.3, "yeah", 0.7)]                                           # ad-lib
REF = [(1.0, w) for w in "walk me down to the harbor".split()]


def test_reconcile_fixes_mishearings_and_keeps_adlibs():
    words, ratio = ly.reconcile(ASR, REF)
    texts = [w[2] for w in words]
    assert texts == ["walk", "me", "down", "to", "the", "harbor", "yeah"]
    harbor = words[5]
    assert harbor[0] == 2.2 and harbor[4] == "harder"                     # ASR timing, alt kept
    assert words[6][3] == 0.7 and ratio > 0.7


def test_missed_reference_words_are_interpolated_and_marked():
    asr = [w for w in ASR if w[2] != "to"]
    words, _ = ly.reconcile(asr, REF)
    to = next(w for w in words if w[2] == "to")
    assert 1.8 <= to[0] <= 2.0 and to[3] == 0.5


def test_wrong_recording_falls_back_to_asr():
    ref = [(1.0, w) for w in "completely different words about nothing here".split()]
    words, ratio = ly.reconcile(ASR, ref)
    assert ratio < 0.5 and [w[2] for w in words] == [w[2] for w in ASR]


def test_repeated_chorus_stays_monotonic():
    asr = [(float(i), float(i) + 0.4, w, 0.9) for i, w in enumerate("la la land la la land".split())]
    ref = [(None, w) for w in "la la land la la land".split()]
    words, _ = ly.reconcile(asr, ref)
    assert [w[0] for w in words] == sorted(w[0] for w in words) and len(words) == 6


def test_lyric_cells_carry_the_heard_word_as_alt():
    from soundcode import encode as enc
    from soundcode.parser import parse
    grid = {"downbeat": 1.0, "bar_dur": 2.0}
    cells = enc.lyric_cells([(1.5, 1.9, "harbor", 0.7, "harder")], grid)
    assert cells == ['1:2.000 "harbor" 0.800b ?0.70 alt="harder"']
    ev = parse('%sc 0.3\n\n:text.vox\n' + cells[0] + "\n").stream("text.vox").events[0]
    assert ev.text == "harbor" and ev.alt == "harder" and ev.dur == "0.800b"


def test_hyphenated_vocables_split_and_alts_are_clean():
    words = [w for _, w in ly.ref_words([ly.Line(1.0, "Deep water, oh-oh-oh-oh")], 0, 30)]
    assert words == ["deep", "water", "oh", "oh", "oh", "oh"]
    out, _ = ly.reconcile([(1.0, 1.4, "delight.", 0.6)], [(1.0, "life")])
    assert out[0][4] in (None, "delight")


def test_reference_words_after_the_last_heard_word_are_dropped():
    asr = [(1.0, 1.3, "walk", 0.9), (1.3, 1.5, "me", 0.9)]
    ref = [(1.0, w) for w in "walk me oh oh oh".split()]
    words, _ = ly.reconcile(asr, ref)
    assert [w[2] for w in words] == ["walk", "me"]


# --- final-review fixes ------------------------------------------------------------------------

def test_a_confident_asr_word_beats_the_reference():
    heard = [("walk", 0.9), ("me", 0.9), ("down", 0.9), ("meeting", 0.97), ("to", 0.9),
             ("the", 0.9), ("harder", 0.5)]
    asr = [(1.0 + 0.3 * i, 1.3 + 0.3 * i, w, p) for i, (w, p) in enumerate(heard)]
    ref = [(1.0, w) for w in "walk me down eating to the harbor".split()]
    words, _ = ly.reconcile(asr, ref)
    got = {w[2]: w[4] for w in words}
    assert got["meeting"] == "eating" and got["harbor"] == "harder" and "eating" not in got


def test_runs_of_missed_words_share_their_gap_and_never_overlap():
    asr = [(1.0, 1.3, "walk", 0.9), (1.3, 1.5, "me", 0.9), (4.0, 4.3, "down", 0.9), (4.3, 4.6, "now", 0.9)]
    ref = [(1.0, w) for w in "walk me oh oh oh down now".split()]
    words, _ = ly.reconcile(asr, ref)
    starts = [w[0] for w in words]
    assert starts == sorted(starts)
    ohs = [w for w in words if w[2] == "oh"]
    assert all(1.5 <= w[0] < w[1] <= 4.0 for w in ohs)


def test_a_leading_run_is_anchored_before_the_first_heard_word():
    asr = [(5.0, 5.3, "down", 0.9)]
    ref = [(None, w) for w in "walk me down".split()]
    words, _ = ly.reconcile(asr, ref)
    assert words[0][2] == "walk" and words[0][0] >= 3.5 and words[1][1] <= 5.0


def test_file_names_only_give_an_artist_with_a_spaced_dash():
    assert ly.guess_title_artist(Path("river-30s.wav")) == ("river-30s", None)
    assert ly.guess_title_artist(Path("The River - Jordan Feliz.mp3"))[1] in ("The River", "Jordan Feliz")


def test_best_window_finds_a_mid_song_clip():
    lines = [ly.Line(0.0, "intro words here"), ly.Line(30.0, "walk me down to the harbor"),
             ly.Line(34.0, "lights are low tonight")]
    asr = [(0.5 + i * 0.4, 0.8 + i * 0.4, w, 0.9) for i, w in enumerate("walk me down to the harbor lights are low".split())]
    off = ly.best_offset(asr, lines, duration=10.0)
    assert 29.0 <= off <= 30.5


def test_reference_for_uses_a_given_offset_or_finds_one():
    lines = [ly.Line(0.0, "intro words here"), ly.Line(30.0, "walk me down to the harbor")]
    asr = [(0.5 + i * 0.4, 0.8 + i * 0.4, w, 0.9) for i, w in enumerate("walk me down to the harbor".split())]
    ref, off = ly.reference_for(asr, lines, duration=10.0, offset=None)
    assert off == 30.0 and [w for _, w in ref][:2] == ["walk", "me"]
    ref2, off2 = ly.reference_for(asr, lines, duration=10.0, offset=0.0)
    assert off2 == 0.0 and [w for _, w in ref2][0] == "intro"
