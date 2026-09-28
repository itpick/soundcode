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


def _fake_run(root):
    """Stands in for the CLI and ffmpeg: writes the files each call would produce."""
    import subprocess
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        out = Path(cmd[cmd.index("-o") + 1]) if "-o" in cmd else Path(cmd[-1])
        if "encode" in cmd:
            out.write_text(SC)
        elif "render" in cmd:
            out.write_bytes(b"RIFF")
            if "--parts" in cmd:
                parts = Path(cmd[cmd.index("--parts") + 1])
                parts.mkdir(parents=True, exist_ok=True)
                (parts / "piano.wav").write_bytes(b"RIFF")
                (parts / "lead_vocals.wav").write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, "", "with-vocals: SoulX-Singer failed (x); sung with DiffSinger + Seed-VC")
        else:                                            # ffmpeg ... <out.mp3>
            out.write_bytes(b"ID3" + b"\0" * 997)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run, calls


def test_build_writes_data_and_media(tmp_path, monkeypatch):
    import json
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    (root / song["clip"]).write_bytes(b"x" * 480_000)
    monkeypatch.setattr(site, "_frames", lambda p: 44100 * 30)
    run, calls = _fake_run(root)
    data = site.build(root, root / "site", root / "work", run=run, songs=[song])
    on_disk = json.loads((root / "site" / "data.json").read_text())
    assert on_disk == data
    s = data["songs"][0]
    assert s["sizes"]["wav"] == 44100 * 30 * 4 + 44 and s["sizes"]["mp3"] == 480_000
    assert s["sizes"]["sc"] == len(SC.encode()) and s["singer"] == "diffsinger"
    assert s["ratios"]["sc_gz"] == round(s["sizes"]["wav"] / s["sizes"]["sc_gz"])
    for f in (s["original"], s["rebuild"], s["sc"]):
        assert (root / "site" / f).exists()
    assert "CC BY-NC-SA 3.0" in data["credit"]

    n = len(calls)
    site.build(root, root / "site", root / "work", run=run, songs=[song])   # cached: no re-encode/render
    assert not any("encode" in c or "render" in c for c in calls[n:])


def test_a_failed_song_names_itself_and_keeps_the_old_data(tmp_path, monkeypatch):
    import subprocess
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    (root / song["clip"]).write_bytes(b"x")
    (root / "site").mkdir()
    (root / "site" / "data.json").write_text('{"old": true}')

    def run(cmd, **k):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")
    with pytest.raises(RuntimeError, match="discipline-30s"):
        site.build(root, root / "site", root / "work", run=run, songs=[song])
    assert (root / "site" / "data.json").read_text() == '{"old": true}'


def test_partial_failure_preserves_existing_media_and_data(tmp_path, monkeypatch):
    import json
    import subprocess
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song1 = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    song2 = {"slug": "lights-30s", "title": "Lights", "clip": "audio/test/lights-30s.mp3"}
    (root / song1["clip"]).write_bytes(b"x" * 480_000)
    (root / song2["clip"]).write_bytes(b"y" * 480_000)

    # Create existing site with song1's media
    (root / "site" / "media").mkdir(parents=True)
    original_media = b"ORIGINAL_SONG1"
    (root / "site" / "media" / "discipline-30s-original.mp3").write_bytes(original_media)
    old_data = {"old": "data", "songs": [{"slug": "discipline-30s"}]}
    (root / "site" / "data.json").write_text(json.dumps(old_data))

    monkeypatch.setattr(site, "_frames", lambda p: 44100 * 30)

    def run(cmd, **k):
        out = Path(cmd[cmd.index("-o") + 1]) if "-o" in cmd else Path(cmd[-1])
        if "encode" in cmd:
            out.write_text(SC)
        elif "render" in cmd:
            if "lights" in str(out):  # Song 2's render fails
                return subprocess.CompletedProcess(cmd, 1, "", "render failed")
            out.write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        else:  # ffmpeg
            if "lights" not in str(out):  # Only song1's ffmpeg succeeds
                out.write_bytes(b"ID3" + b"\0" * 997)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with pytest.raises(RuntimeError, match="lights-30s"):
        site.build(root, root / "site", root / "work", run=run, songs=[song1, song2])

    # Verify existing media is untouched (byte-identical)
    assert (root / "site" / "media" / "discipline-30s-original.mp3").read_bytes() == original_media
    # Verify data.json is untouched
    assert json.loads((root / "site" / "data.json").read_text()) == old_data
    # Verify no staging directory left behind
    assert not (root / "site" / ".media.part").exists()


def test_build_with_subset_replaces_entire_site(tmp_path, monkeypatch):
    import json
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song1 = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    song2 = {"slug": "lights-30s", "title": "Lights", "clip": "audio/test/lights-30s.mp3"}
    (root / song1["clip"]).write_bytes(b"x" * 480_000)
    (root / song2["clip"]).write_bytes(b"y" * 480_000)

    monkeypatch.setattr(site, "_frames", lambda p: 44100 * 30)
    run, calls = _fake_run(root)

    # First build: both songs
    data1 = site.build(root, root / "site", root / "work", run=run, songs=[song1, song2])
    assert len(data1["songs"]) == 2
    assert (root / "site" / "media" / "discipline-30s-original.mp3").exists()
    assert (root / "site" / "media" / "lights-30s-original.mp3").exists()
    on_disk1 = json.loads((root / "site" / "data.json").read_text())
    assert len(on_disk1["songs"]) == 2

    # Second build: only song1 → song2 disappears from site/
    data2 = site.build(root, root / "site", root / "work", run=run, songs=[song1])
    assert len(data2["songs"]) == 1
    assert data2["songs"][0]["slug"] == "discipline-30s"
    # song2's media gone
    assert not (root / "site" / "media" / "lights-30s-original.mp3").exists()
    # song1's media still there
    assert (root / "site" / "media" / "discipline-30s-original.mp3").exists()
    # data.json reflects only song1
    on_disk2 = json.loads((root / "site" / "data.json").read_text())
    assert len(on_disk2["songs"]) == 1
    assert on_disk2["songs"][0]["slug"] == "discipline-30s"


def test_page_reads_data_json_safely():
    html = (Path(__file__).resolve().parents[1] / "site" / "index.html").read_text()
    assert "fetch('data.json')" in html
    assert "python -m http.server -d site" in html          # file:// fallback message
    assert "innerHTML" not in html                          # data shown with textContent only
    for key in ("sizes", "ratios", "sc_gz", "with_borrowed", "singer", "license_url", "totals"):
        assert key in html
    assert "<script src=" not in html                        # no external scripts
    sc_fetch = html[html.index("fetch(song.sc)"):html.index("fetch(song.sc)") + 200]
    assert "r.ok" in sc_fetch                                 # a failed .sc fetch is caught, not rendered raw
    assert "four clips" not in html                           # no counts hardcoded outside data.json
    # part toggles (Task 4): an in-sync Web Audio transport with an Original/Rebuild switch
    for needle in ("decodeAudioData", "createGain", "Original", "Rebuild", "aria-pressed",
                   "not in the rebuild", "not separated", "Loading parts"):
        assert needle in html
    # the seek thumb follows playback unless the user is dragging it (focus is not dragging)
    assert "pointerdown" in html and 'seek.addEventListener("change"' in html
    assert "activeElement" not in html


def test_build_lists_parts_from_both_sides(tmp_path, monkeypatch):
    import json

    import numpy as np
    import soundfile as sf
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    (root / song["clip"]).write_bytes(b"x" * 480_000)
    stems = root / "out" / "stems" / "discipline-30s"
    stems.mkdir(parents=True)
    for key in ("piano", "drums", "lead_vocals"):
        sf.write(str(stems / f"{key}.wav"), np.full((22050, 2), 0.1, np.float32), 44100)
    sf.write(str(stems / "bass.wav"), np.zeros((22050, 2), np.float32), 44100)     # silent: left out
    monkeypatch.setattr(site, "_frames", lambda p: 44100 * 30)
    monkeypatch.chdir(root)
    run, calls = _fake_run(root)
    data = site.build(root, root / "site", root / "work", run=run, songs=[song])
    parts = data["songs"][0]["parts"]
    assert [p["key"] for p in parts] == ["lead_vocals", "piano", "drums"]
    assert [p["label"] for p in parts] == ["Vocals", "Keys", "Drums"]
    by = {p["key"]: p for p in parts}
    assert by["drums"]["rebuild"] is None                     # the rebuild has no drums
    assert by["piano"]["original"] == "media/discipline-30s/original-piano.mp3"
    assert by["piano"]["rebuild"] == "media/discipline-30s/rebuild-piano.mp3"
    for p in parts:
        for side in ("original", "rebuild"):
            if p[side] is not None:
                assert (root / "site" / p[side]).exists()
    assert "river" not in json.dumps(data).lower()
    assert "river" not in " ".join(str(f) for f in (root / "site").rglob("*")).lower()
    render = next(c for c in calls if "render" in c)
    assert render[render.index("--parts") + 1] == str(root / "work" / "discipline-30s.parts")

    n = len(calls)
    import shutil
    shutil.rmtree(root / "work" / "discipline-30s.parts")      # parts missing: the render is stale
    site.build(root, root / "site", root / "work", run=run, songs=[song])
    assert any("render" in c for c in calls[n:])


def test_an_unreadable_stem_names_its_song(tmp_path, monkeypatch):
    import json
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    (root / song["clip"]).write_bytes(b"x" * 480_000)
    stems = root / "out" / "stems" / "discipline-30s"
    stems.mkdir(parents=True)
    (stems / "piano.wav").write_bytes(b"not a wav")
    (root / "site").mkdir()
    (root / "site" / "data.json").write_text('{"old": true}')
    monkeypatch.setattr(site, "_frames", lambda p: 44100 * 30)
    run, calls = _fake_run(root)
    with pytest.raises(RuntimeError, match=r"^discipline-30s: .*piano"):
        site.build(root, root / "site", root / "work", run=run, songs=[song])
    assert json.loads((root / "site" / "data.json").read_text()) == {"old": True}
    assert not (root / "site" / ".media.part").exists()
