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


def test_choir_stream_with_no_words_is_not_a_singer(tmp_path):
    """A vocal-family stream (meta stem=lead_vocals) with notes but no
    :text.vox -- e.g. a quiet choir/pad line ASR found nothing in -- must not
    be reported as sung (task 5a: no singing without words)."""
    choir = SC.replace(
        ":notes.lead inst=voice.lead\nmeta stem=lead_vocals\n1:1.000  E4  1.000b 90\n",
        ":notes.choir inst=voice.choir\nmeta stem=lead_vocals\n1:1.000  E4  1.000b 90\n",
    ).split(":text.vox")[0]
    doc = parse(choir)
    assert not site.sung(doc)
    assert site.voice_bytes(doc, "soulx") == 0
    assert site.singer_from_log("with-vocals: the vocal stream has no lyrics; "
                                "rendered as its instrument; rendering instruments only", doc) is None


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


def _stems(root, slug):
    """A minimal stems dir satisfying build()'s separated-stems prerequisite (task 1)."""
    d = root / "out" / "stems" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "placeholder.wav").write_bytes(b"RIFF")          # name not in PART_KEYS: export_parts ignores it
    return d


def test_build_writes_data_and_media(tmp_path, monkeypatch):
    import json
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    (root / song["clip"]).write_bytes(b"x" * 480_000)
    _stems(root, song["slug"])
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
    _stems(root, song["slug"])
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
    _stems(root, song1["slug"])
    _stems(root, song2["slug"])

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
    _stems(root, song1["slug"])
    _stems(root, song2["slug"])

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
    assert 'seek.addEventListener("blur"' in html               # Tab away never leaves it "dragging"
    # Task 6: the file:// fallback message only when opened as file:; another
    # failure (e.g. a 404 on a real server) gets its own status/error message.
    assert "location.protocol" in html
    assert "Could not load data.json" in html
    # Task 7: the kit is not shipped -- say so next to the download, only when sizes.kit > 0.
    assert "Drum kit (hits cut from the recording) not included" in html
    assert "sizes.kit" in html


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


def test_missing_stems_fails_before_encoding_and_names_the_command(tmp_path):
    """Task 1: build() needs out/stems/<slug>/ (from `soundcode separate`); encode
    alone does not create it. A fresh checkout must fail loudly, not silently
    ship an instrumental with every original part null."""
    import json
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    (root / song["clip"]).write_bytes(b"x" * 480_000)
    (root / "site").mkdir()
    (root / "site" / "data.json").write_text('{"old": true}')
    run, calls = _fake_run(root)
    with pytest.raises(RuntimeError, match=r"discipline-30s: no separated stems in .*"
                                            r"; run: soundcode separate audio/test/discipline-30s\.mp3"):
        site.build(root, root / "site", root / "work", run=run, songs=[song])
    assert not calls                                          # never reached encode
    assert json.loads((root / "site" / "data.json").read_text()) == {"old": True}
    assert not (root / "site" / ".media.part").exists()


def test_an_empty_stems_dir_also_fails(tmp_path):
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    (root / song["clip"]).write_bytes(b"x" * 480_000)
    (root / "out" / "stems" / "discipline-30s").mkdir(parents=True)   # dir exists, no *.wav
    run, calls = _fake_run(root)
    with pytest.raises(RuntimeError, match="no separated stems"):
        site.build(root, root / "site", root / "work", run=run, songs=[song])


def test_instrumental_lead_vocals_part_is_labelled_choir_not_vocals(tmp_path, monkeypatch):
    """Task 2: an instrumental's rebuild lead_vocals part is a voice.choir GM
    line (meta stem), not an actual singer -- it must not read "Vocals"."""
    p = tmp_path / "t.sc"
    p.write_text(SC)
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"x" * 100)
    monkeypatch.setattr(site, "_frames", lambda c: 44100)
    parts = [{"key": "lead_vocals", "label": "Vocals", "original": None,
              "rebuild": "media/t/rebuild-lead_vocals.mp3"}]

    instrumental = site.song_entry({"slug": "t", "title": "T"}, p, clip, None, {}, parts)
    assert instrumental["parts"][0]["label"] == "Choir / pad (vocal stem)"

    sung = site.song_entry({"slug": "t", "title": "T"}, p, clip, "soulx", {}, parts)
    assert sung["parts"][0]["label"] == "Vocals"                 # unchanged when actually sung
    assert parts[0]["label"] == "Vocals"                         # the input list is not mutated


def test_export_parts_notes_explain_a_missing_side(tmp_path):
    """Task 3: per-part, per-side reasons for a disabled chip -- 'silent in the
    original' (stem exists, gated out) is distinct from 'not separated'
    (stem never existed); a missing rebuild is 'not in the rebuild'."""
    import numpy as np
    import soundfile as sf
    stems = tmp_path / "stems"
    stems.mkdir()
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    staging = tmp_path / "staging"
    sf.write(str(stems / "piano.wav"), np.full((1000, 2), 0.1, np.float32), 44100)    # loud: kept
    sf.write(str(stems / "drums.wav"), np.zeros((1000, 2), np.float32), 44100)         # silent: gated
    sf.write(str(parts_dir / "drums.wav"), np.full((1000, 2), 0.1, np.float32), 44100)
    sf.write(str(parts_dir / "guitar.wav"), np.full((1000, 2), 0.1, np.float32), 44100)  # no stem at all

    run, _ = _fake_run(tmp_path)
    result = site.export_parts(run, "t", stems, parts_dir, staging)
    by = {p["key"]: p for p in result}

    assert by["piano"]["original"] is not None and by["piano"]["original_note"] is None
    assert by["piano"]["rebuild"] is None and by["piano"]["rebuild_note"] == "not in the rebuild"

    assert by["drums"]["original"] is None
    assert by["drums"]["original_note"] == "silent in the original"
    assert by["drums"]["rebuild"] is not None and by["drums"]["rebuild_note"] is None

    assert by["guitar"]["original"] is None
    assert by["guitar"]["original_note"] == "not separated"
    assert by["guitar"]["rebuild"] is not None


def test_missing_render_log_forces_a_rerender(tmp_path, monkeypatch):
    """Task 4: singer_from_log() falls back to the default singer when the log
    is missing -- a cached wav with a deleted log must not be trusted; force
    a re-render instead of misreporting the singer."""
    root = tmp_path
    (root / "audio" / "test").mkdir(parents=True)
    song = {"slug": "discipline-30s", "title": "Discipline", "clip": "audio/test/discipline-30s.mp3"}
    (root / song["clip"]).write_bytes(b"x" * 480_000)
    _stems(root, song["slug"])
    monkeypatch.setattr(site, "_frames", lambda p: 44100 * 30)
    run, calls = _fake_run(root)

    site.build(root, root / "site", root / "work", run=run, songs=[song])
    (root / "work" / "discipline-30s.render.log").unlink()        # log gone, wav + parts kept

    n = len(calls)
    site.build(root, root / "site", root / "work", run=run, songs=[song])
    assert any("render" in c for c in calls[n:])


def test_site_is_prepped_for_private_cloudflare_hosting():
    site_dir = Path(__file__).resolve().parents[1] / "site"
    headers = (site_dir / "_headers").read_text()
    assert "X-Robots-Tag: noindex" in headers and "Content-Security-Policy: default-src 'self'" in headers
    assert (site_dir / "robots.txt").read_text().strip().endswith("Disallow: /")
    readme = (site_dir / "README.md").read_text()
    assert "Cloudflare Access" in readme and "Build output directory: **`site`**" in readme
    # the page loads nothing from other origins, so the strict CSP cannot break it
    html = (site_dir / "index.html").read_text()
    assert "<script src=" not in html and "<link rel=\"stylesheet\"" not in html
    assert all(f.stat().st_size < 25 * 1024 * 1024 for f in site_dir.rglob("*") if f.is_file())
