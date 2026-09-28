"""Demo site measures (spec 2026-09-28-demo-site-design)."""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import site  # noqa: E402
from soundcode.parser import parse  # noqa: E402

SC = """%sc 0.3
@title "t"
@duration 4.0

:grid
meter @0.000 4/4
anchor bar 1 @0.000
tempo @0.000 110

:notes.piano inst=keys.piano
1:1.000  C4  1.000b 90

:notes.lead inst=voice.lead
meta stem=lead_vocals
1:1.000  E4  1.000b 90

:perc.drums
meta kit=t.kit
1:1.000  kick  0.100b 100

:text.vox
1:1.000 "a <b> & c" 1.000b | 1:3.000 "river" 1.000b
"""


def test_only_the_cc_licensed_slip_clips_can_be_songs():
    allowed = {"discipline-30s.mp3", "lights_in_the_sky-30s.mp3", "999999-30s.mp3", "corona_radiata-30s.mp3"}
    assert {Path(s["clip"]).name for s in site.SONGS} == allowed
    assert not any("river" in s["clip"].lower() for s in site.SONGS)


def test_sizes():
    assert site.wav_bytes(44100 * 30) == 44100 * 30 * 4 + 44
    raw, gz = site.sc_sizes("abc " * 1000)
    assert raw == 4000 and gz == len(gzip.compress(("abc " * 1000).encode(), 9))


def test_kit_bytes_sums_the_folder_and_is_zero_when_missing(tmp_path):
    p = tmp_path / "t.sc"
    p.write_text(SC)
    from soundcode.parser import parse_file
    doc = parse_file(str(p))
    assert site.kit_bytes(doc) == 0                      # folder missing: no crash
    (tmp_path / "t.kit").mkdir()
    (tmp_path / "t.kit" / "kick_0.wav").write_bytes(b"x" * 100)
    (tmp_path / "t.kit" / "snare_0.wav").write_bytes(b"x" * 50)
    assert site.kit_bytes(doc) == 150


def test_voice_bytes_only_when_sung():
    from soundcode import soulx
    doc = parse(SC)
    assert site.sung(doc)
    assert site.voice_bytes(doc, "soulx") == int(soulx.PROMPT_S * soulx.SR * 2)
    assert site.voice_bytes(doc, None) == 0
    instrumental = parse(SC.split(":notes.lead")[0])
    assert not site.sung(instrumental)


def test_singer_from_the_render_log():
    doc = parse(SC)
    assert site.singer_from_log("rendered 5 events", doc) == "soulx"
    assert site.singer_from_log("with-vocals: SoulX-Singer failed (x); sung with DiffSinger + Seed-VC", doc) == "diffsinger"
    assert site.singer_from_log("with-vocals: no lead vocal; rendering instruments only", doc) is None
    assert site.singer_from_log("", parse(SC.split(":notes.lead")[0])) is None


def test_summary():
    s = site.summary(parse(SC))
    assert s["tempo"] == 110
    assert s["instruments"] == ["drums", "keys.piano", "voice.lead"]
    assert s["words"] == 5              # "a", "<b>", "&", "c", "river"
    assert s["duration"] == 4.0
