"""`bench`, run history, README, and the score/bench CLI (spec
2026-09-28-benchmark-scorer, Task 6).

The pipeline (ffmpeg cut, separate/encode/render CLI calls) is faked, like
`tests/test_site.py`'s `_fake_run`; `scorer.score_song` is monkeypatched to
return a fixed, well-formed result dict rather than touching real audio.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode.score import bench  # noqa: E402


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

def _fake_run(calls=None):
    """Stands in for subprocess.run across ffmpeg, `soundcode.cli` and git:
    writes the files each real call would produce."""
    calls = calls if calls is not None else []

    def run(cmd, **k):
        calls.append(cmd)
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "abc1234\n", "")
        if cmd[0] == "ffmpeg":
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF....WAVEfmt ")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        sub = cmd[3]                                       # [py, -m, soundcode.cli, <sub>, ...]
        if sub == "separate":
            out = Path(cmd[cmd.index("-o") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "piano.wav").write_bytes(b"RIFF")
            (out / "lead_vocals.wav").write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if sub == "encode":
            out = Path(cmd[cmd.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("%sc 0.3\n@duration 4.0\n@source x.wav\n")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if sub == "render":
            out = Path(cmd[cmd.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF")
            parts = Path(cmd[cmd.index("--parts") + 1])
            parts.mkdir(parents=True, exist_ok=True)
            (parts / "piano.wav").write_bytes(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected cli sub-command: {cmd}")
    return run, calls


ENTRY_A = {"name": "song-a", "source": "audio/test/song-a.wav", "start": None, "end": None,
           "title": "Song A", "artist": "Artist"}
ENTRY_B = {"name": "song-b", "source": "audio/test/song-b.wav", "start": None, "end": None,
           "title": "Song B", "artist": "Artist"}
# a name that does not match its source file's stem -- exercises the cut/copy stage
ENTRY_RENAMED = {"name": "song-full", "source": "audio/nin/song.mp3", "start": None, "end": None,
                  "title": "Song", "artist": "Artist"}
ENTRY_EXCERPT = {"name": "song-60s", "source": "audio/nin/song.mp3", "start": 30.0, "end": 90.0,
                  "title": "Song", "artist": "Artist"}


def _mk_source(root: Path, entry: dict, data: bytes = b"x" * 1000):
    p = root / entry["source"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _mk_result(name: str, score: float, part_scores: dict[str, float]) -> dict:
    """A minimal, well-formed `score_song`-shaped result: enough for
    `report.table`/`report.html` and bench's history extraction, without any
    real audio."""
    parts = {}
    for key, s in part_scores.items():
        parts[key] = {
            "song": {"axes": {"what": s, "sound": s, "dyn": s}, "score": s, "metrics": {}},
            "sections": [{"label": "verse", "a": 0.0, "b": 1.0, "score": s, "axes": {}, "metrics": {}}],
            "windows": [], "worst": {"label": "verse", "score": s, "a": 0.0, "b": 1.0},
            "missing": False, "silent": False, "type": "vocal", "label": key,
            "files": {"original": None, "rebuild": None}, "energy": 1.0, "extra": False,
        }
    mix = {"song": {"axes": {"what": score, "sound": score, "dyn": score}, "score": score, "metrics": {}},
           "sections": [], "windows": [], "worst": None, "missing": False, "silent": False,
           "type": "mix", "label": "Mix", "files": {"original": None, "rebuild": None}}
    worst_part = min(part_scores, key=part_scores.get) if part_scores else None
    worst = ({"part": worst_part, "label": "verse", "score": part_scores[worst_part], "a": 0.0, "b": 1.0}
              if worst_part else None)
    return {"song": name, "duration": 4.0, "parts": parts, "mix": mix, "score": score, "worst": worst,
            "drift": {}, "energy": {k: 1.0 for k in part_scores}, "sections": [],
            "files": {"original": None, "sc": None, "stems": None, "parts": None, "rebuild_mix": None}}


# --------------------------------------------------------------------------
# TIERS
# --------------------------------------------------------------------------

def test_tiers_cover_a_b_c_with_the_spec_entries():
    assert set(bench.TIERS) == {"A", "B", "C"}
    names = {e["name"] for e in bench.TIERS["A"]}
    assert names == {"river-30s", "discipline-30s", "lights_in_the_sky-30s",
                      "999999-30s", "corona_radiata-30s"}
    b_names = {e["name"] for e in bench.TIERS["B"]}
    assert b_names == {"discipline-100s", "lights_in_the_sky-60s", "corona_radiata-60s", "999999-full"}
    c_names = {e["name"] for e in bench.TIERS["C"]}
    assert c_names == {"discipline-full", "lights_in_the_sky-full", "corona_radiata-full", "river-full"}
    for entries in bench.TIERS.values():
        for e in entries:
            assert set(e) == {"name", "source", "start", "end", "title", "artist"}
    river = next(e for e in bench.TIERS["A"] if e["name"] == "river-30s")
    assert river["source"] == "audio/test/river-30s.wav"
    assert river["title"] == "The River" and river["artist"] == "Jordan Feliz"
    excerpt = next(e for e in bench.TIERS["B"] if e["name"] == "lights_in_the_sky-60s")
    assert excerpt["start"] == 30.0 and excerpt["end"] == 90.0


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

def test_prepare_runs_every_stage_and_returns_its_paths_and_timing(tmp_path):
    _mk_source(tmp_path, ENTRY_A)
    run, calls = _fake_run()
    paths = bench.prepare(ENTRY_A, tmp_path, run=run)
    assert paths["stems_dir"] == tmp_path / "out" / "stems" / "song-a"
    assert paths["sc"] == tmp_path / "out" / "bench" / "song-a" / "song-a.sc"
    assert paths["parts_dir"] == tmp_path / "out" / "bench" / "song-a" / "parts"
    assert paths["rebuild"] == tmp_path / "out" / "bench" / "song-a" / "rebuild.wav"
    assert paths["original"].exists() and paths["sc"].exists()
    assert (paths["stems_dir"] / "piano.wav").exists()
    assert set(paths["timing"]) == {"cut", "separate", "encode", "render"}
    assert all(isinstance(v, float) for v in paths["timing"].values())
    assert any("separate" in c for c in calls)
    assert any("encode" in c for c in calls)
    assert any("render" in c for c in calls)


def test_prepare_uses_the_source_directly_when_the_name_already_matches(tmp_path):
    """No stem mismatch and no start/end: no ffmpeg cut is needed."""
    _mk_source(tmp_path, ENTRY_A)
    run, calls = _fake_run()
    paths = bench.prepare(ENTRY_A, tmp_path, run=run)
    assert paths["original"] == tmp_path / ENTRY_A["source"]
    assert not any(c[0] == "ffmpeg" for c in calls)


def test_prepare_cuts_to_audio_bench_when_the_name_does_not_match_the_source(tmp_path):
    _mk_source(tmp_path, ENTRY_RENAMED)
    run, calls = _fake_run()
    paths = bench.prepare(ENTRY_RENAMED, tmp_path, run=run)
    assert paths["original"] == tmp_path / "audio" / "bench" / "song-full.wav"
    assert any(c[0] == "ffmpeg" for c in calls)
    ffmpeg_call = next(c for c in calls if c[0] == "ffmpeg")
    assert "-ss" not in ffmpeg_call                         # no trim requested


def test_prepare_cuts_an_excerpt_with_start_and_end(tmp_path):
    _mk_source(tmp_path, ENTRY_EXCERPT)
    run, calls = _fake_run()
    paths = bench.prepare(ENTRY_EXCERPT, tmp_path, run=run)
    assert paths["original"] == tmp_path / "audio" / "bench" / "song-60s.wav"
    ffmpeg_call = next(c for c in calls if c[0] == "ffmpeg")
    assert "-ss" in ffmpeg_call and "30.0" in ffmpeg_call
    encode_call = next(c for c in calls if len(c) > 3 and c[3] == "encode")
    assert "--offset" in encode_call and "30.0" in encode_call


def test_prepare_is_cached_by_mtime_and_force_redoes_everything(tmp_path):
    _mk_source(tmp_path, ENTRY_A)
    run, calls = _fake_run()
    bench.prepare(ENTRY_A, tmp_path, run=run)
    n = len(calls)
    bench.prepare(ENTRY_A, tmp_path, run=run)               # nothing stale: no new calls
    assert len(calls) == n
    paths = bench.prepare(ENTRY_A, tmp_path, run=run, force=True)
    assert len(calls) > n
    assert paths["stems_dir"].exists()


def test_prepare_failing_stage_names_the_song_and_the_stage(tmp_path):
    _mk_source(tmp_path, ENTRY_A)

    def run(cmd, **k):
        if len(cmd) > 3 and cmd[3] == "encode":
            return subprocess.CompletedProcess(cmd, 1, "", "boom\nlast line here")
        return _fake_run()[0](cmd, **k)

    with pytest.raises(RuntimeError, match=r"song-a: encode failed \(last line here\)"):
        bench.prepare(ENTRY_A, tmp_path, run=run)


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------

def test_diff_lists_song_and_part_changes_present_in_both_lines():
    prev = {"tier": "A", "songs": {
        "a": {"score": 70.0, "worst": None, "parts": {"lead_vocals": {"score": 71.0}, "piano": {"score": 60.0}}},
        "gone": {"score": 50.0, "worst": None, "parts": {}},
    }}
    cur = {"tier": "A", "songs": {
        "a": {"score": 75.0, "worst": None, "parts": {"lead_vocals": {"score": 84.0}, "piano": {"score": 60.0}}},
        "new": {"score": 90.0, "worst": None, "parts": {}},
    }}
    d = bench.diff(prev, cur)
    assert ("a", "song", 70.0, 75.0) in d
    assert ("a", "lead_vocals", 71.0, 84.0) in d
    assert ("a", "piano", 60.0, 60.0) in d
    assert not any(song == "gone" or song == "new" for song, *_ in d)


def test_diff_with_no_previous_line_is_empty():
    assert bench.diff(None, {"tier": "A", "songs": {}}) == []


def _bullets(section: str) -> list[str]:
    return [ln[2:].split(":")[0].strip() for ln in section.splitlines() if ln.startswith("- ")]


def test_readme_sorts_the_worst_regression_and_biggest_improvement_first():
    prev = {"date": "d0", "label": "prev", "commit": "c0", "tier": "A", "songs": {
        "improve-small": {"score": 70.0, "worst": None, "parts": {}},
        "improve-big": {"score": 70.0, "worst": None, "parts": {}},
        "regress-small": {"score": 70.0, "worst": None, "parts": {}},
        "regress-big": {"score": 70.0, "worst": None, "parts": {}},
    }}
    cur = {"date": "d1", "label": "cur", "commit": "c1", "tier": "A", "songs": {
        "improve-small": {"score": 75.0, "worst": None, "parts": {}},    # +5
        "improve-big": {"score": 90.0, "worst": None, "parts": {}},      # +20
        "regress-small": {"score": 65.0, "worst": None, "parts": {}},    # -5
        "regress-big": {"score": 40.0, "worst": None, "parts": {}},      # -30
    }}
    readme = bench._render_readme([prev, cur])
    improved = readme.split("### Improved")[1].split("### Regressed")[0]
    regressed = readme.split("### Regressed")[1]
    assert _bullets(improved) == ["improve-big song", "improve-small song"]
    assert _bullets(regressed) == ["regress-big song", "regress-small song"]


# --------------------------------------------------------------------------
# run_bench: the two-run sequence
# --------------------------------------------------------------------------

def _patch_two_runs(monkeypatch):
    """First run: song-a=70 (lead_vocals=71), song-b=80. Second run:
    song-a=83 (lead_vocals=84, +13, improved), song-b=70 (regressed)."""
    calls = {"n": 0}

    def fake_score_song(original, stems_dir, sc, parts_dir, rebuild_mix, out_dir, *, cache_dir=None):
        assert Path(cache_dir).is_absolute()
        name = Path(sc).stem
        if calls["n"] == 0:
            result = {"song-a": _mk_result("song-a", 70.0, {"lead_vocals": 71.0}),
                      "song-b": _mk_result("song-b", 80.0, {"lead_vocals": 80.0})}[name]
        else:
            result = {"song-a": _mk_result("song-a", 83.0, {"lead_vocals": 84.0}),
                      "song-b": _mk_result("song-b", 70.0, {"lead_vocals": 70.0})}[name]
        return result

    monkeypatch.setattr(bench.scorer, "score_song", fake_score_song)
    return calls


def test_run_bench_two_run_sequence_writes_history_and_readme(tmp_path, monkeypatch, capsys):
    _mk_source(tmp_path, ENTRY_A)
    _mk_source(tmp_path, ENTRY_B)
    monkeypatch.setattr(bench, "TIERS", {"A": [ENTRY_A, ENTRY_B]})
    calls = _patch_two_runs(monkeypatch)
    run, _ = _fake_run()

    bench.run_bench("A", "first", tmp_path, run=run)
    out = capsys.readouterr().out
    assert "→" not in out                              # nothing to diff against yet
    history = tmp_path / "docs" / "results" / "benchmark" / "history.jsonl"
    lines = [json.loads(l) for l in history.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["label"] == "first" and lines[0]["tier"] == "A" and lines[0]["commit"] == "abc1234"
    assert lines[0]["songs"]["song-a"]["score"] == 70.0
    assert lines[0]["songs"]["song-a"]["parts"]["lead_vocals"]["score"] == 71.0
    assert set(lines[0]["timing"]["song-a"]) == {"cut", "separate", "encode", "render"}

    calls["n"] = 1
    bench.run_bench("A", "second", tmp_path, run=run)
    out = capsys.readouterr().out
    assert "song-a lead_vocals 71 → 84 (+13)" in out    # printed change, per the design doc
    assert "song-b" in out and "(-10)" in out
    lines = [json.loads(l) for l in history.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[1]["songs"]["song-a"]["score"] == 83.0

    readme = (tmp_path / "docs" / "results" / "benchmark" / "README.md").read_text()
    assert "+13" in readme or "+13.0" in readme          # song-a lead_vocals improved
    assert "-10" in readme or "-10.0" in readme           # song-b regressed
    assert "song-a" in readme and "song-b" in readme
    assert "Improved" in readme and "Regressed" in readme
    assert not (tmp_path / "docs" / "results" / "benchmark" / "README.md.part").exists()

    # per-song reports were written
    runs_dir = tmp_path / "out" / "bench" / "runs"
    run_dirs = sorted(runs_dir.iterdir())
    assert len(run_dirs) == 2
    assert (run_dirs[0] / "song-a" / "report.html").exists()


def test_run_bench_all_writes_one_history_line_per_tier(tmp_path, monkeypatch):
    _mk_source(tmp_path, ENTRY_A)
    _mk_source(tmp_path, ENTRY_B)
    monkeypatch.setattr(bench, "TIERS", {"A": [ENTRY_A], "B": [ENTRY_B]})
    monkeypatch.setattr(bench.scorer, "score_song",
                        lambda original, stems_dir, sc, *a, **k: {
                            "song-a": _mk_result("song-a", 70.0, {"lead_vocals": 71.0}),
                            "song-b": _mk_result("song-b", 80.0, {"lead_vocals": 80.0}),
                        }[Path(sc).stem])
    run, _ = _fake_run()
    bench.run_bench("all", "everything", tmp_path, run=run)
    history = tmp_path / "docs" / "results" / "benchmark" / "history.jsonl"
    lines = [json.loads(l) for l in history.read_text().splitlines()]
    assert {l["tier"] for l in lines} == {"A", "B"}
    assert {l["label"] for l in lines} == {"everything"}
    readme = (tmp_path / "docs" / "results" / "benchmark" / "README.md").read_text()
    assert "## Tier A" in readme and "## Tier B" in readme


def test_run_bench_failing_stage_raises_and_history_is_untouched(tmp_path, monkeypatch):
    _mk_source(tmp_path, ENTRY_A)
    monkeypatch.setattr(bench, "TIERS", {"A": [ENTRY_A]})
    monkeypatch.setattr(bench.scorer, "score_song",
                        lambda *a, **k: _mk_result("song-a", 70.0, {"lead_vocals": 71.0}))

    def run(cmd, **k):
        if len(cmd) > 3 and cmd[3] == "separate":
            return subprocess.CompletedProcess(cmd, 1, "", "disk full")
        return _fake_run()[0](cmd, **k)

    with pytest.raises(RuntimeError, match=r"song-a: separate failed \(disk full\)"):
        bench.run_bench("A", "x", tmp_path, run=run)
    history = tmp_path / "docs" / "results" / "benchmark" / "history.jsonl"
    assert not history.exists()


def test_run_bench_interrupted_by_score_song_leaves_history_and_readme_untouched(tmp_path, monkeypatch):
    _mk_source(tmp_path, ENTRY_A)
    _mk_source(tmp_path, ENTRY_B)
    monkeypatch.setattr(bench, "TIERS", {"A": [ENTRY_A, ENTRY_B]})
    run, _ = _fake_run()

    def fake_score_song(original, stems_dir, sc, parts_dir, rebuild_mix, out_dir, *, cache_dir=None):
        if Path(sc).stem == "song-b":
            raise RuntimeError("boom mid-run")
        return _mk_result("song-a", 70.0, {"lead_vocals": 71.0})

    monkeypatch.setattr(bench.scorer, "score_song", fake_score_song)
    history = tmp_path / "docs" / "results" / "benchmark" / "history.jsonl"
    readme = tmp_path / "docs" / "results" / "benchmark" / "README.md"

    with pytest.raises(RuntimeError, match="boom mid-run"):
        bench.run_bench("A", "x", tmp_path, run=run)
    assert not history.exists()
    assert not readme.exists()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_bench_calibrate_prints_not_yet(capsys):
    from soundcode import cli
    assert cli.main(["bench", "--calibrate"]) == 0
    assert "not yet" in capsys.readouterr().out


def test_cli_bench_dispatches_to_run_bench(tmp_path, monkeypatch):
    from soundcode import cli
    seen = {}
    monkeypatch.setattr(bench, "run_bench",
                        lambda tier, label, root, force=False, run=subprocess.run:
                        seen.update(tier=tier, label=label, force=force) or {"run_dir": tmp_path})
    monkeypatch.chdir(tmp_path)
    assert cli.main(["bench", "--tier", "B", "--label", "nightly", "--force"]) == 0
    assert seen == {"tier": "B", "label": "nightly", "force": True}


def test_cli_score_renders_scores_and_prints_the_table(tmp_path, monkeypatch):
    from soundcode import cli
    from soundcode.score import scorer as real_scorer

    original = tmp_path / "orig.wav"
    original.write_bytes(b"RIFF")
    sc = tmp_path / "song.sc"
    sc.write_text("%sc 0.3\n@duration 4.0\n")

    rendered = {}

    def fake_render_parts(doc, sr, **k):
        rendered["called"] = True
        import numpy as np
        return np.zeros((44100, 2), np.float32), {"piano": np.full((44100, 2), 0.1, np.float32)}

    monkeypatch.setattr("soundcode.render_sf.render_parts", fake_render_parts)
    result = _mk_result("song", 88.0, {"lead_vocals": 90.0})
    seen = {}

    def fake_score_song(*a, **k):
        seen.update(k)
        return result

    monkeypatch.setattr(real_scorer, "score_song", fake_score_song)
    out_dir = tmp_path / "score-out"
    assert cli.main(["score", str(original), str(sc), "--out", str(out_dir)]) == 0
    assert rendered.get("called")
    assert (out_dir / "report.html").exists()
    assert "cache_dir" in seen and Path(seen["cache_dir"]).is_absolute()
    assert Path(seen["cache_dir"]) == (Path("out") / "bench" / "cache").resolve()
