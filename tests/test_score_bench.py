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

def _fake_run(calls=None, separate_returncode=0, separate_stdout=None):
    """Stands in for subprocess.run across ffmpeg, `soundcode.cli` and git:
    writes the files each real call would produce. `separate_returncode` and
    `separate_stdout` can override the default separate behavior for testing."""
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
            if separate_returncode == 0 or separate_returncode == 1:
                # For code 0 or 1, create the stems
                (out / "piano.wav").write_bytes(b"RIFF")
                (out / "lead_vocals.wav").write_bytes(b"RIFF")
            stdout_msg = separate_stdout if separate_stdout is not None else ""
            return subprocess.CompletedProcess(cmd, separate_returncode, stdout_msg, "")
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


def test_prepare_passes_a_workdir_under_out_bench_to_encode(tmp_path):
    """Without this, `encode` makes its own `tempfile.mkdtemp` per call and
    never removes it when the caller gave no `--workdir` -- an unbounded
    leak across repeated bench/calibrate runs. `bench` must always pass
    one, under `out/bench/<name>` (so intermediates are cached there, on
    whatever drive `out/bench` lives on, and reused run to run)."""
    _mk_source(tmp_path, ENTRY_A)
    run, calls = _fake_run()
    bench.prepare(ENTRY_A, tmp_path, run=run)
    encode_call = next(c for c in calls if len(c) > 3 and c[3] == "encode")
    assert "--workdir" in encode_call
    workdir = encode_call[encode_call.index("--workdir") + 1]
    assert Path(workdir) == tmp_path / "out" / "bench" / "song-a" / "work"


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


def test_prepare_separate_sum_check_failure_is_a_warning_not_a_crash(tmp_path, capsys):
    """Exit code 1 from separate when stems are written + 'sum check: FAILED'
    in stdout is treated as a warning, not a crash. Prepare continues and
    marks the entry with sum_check_failed: True."""
    _mk_source(tmp_path, ENTRY_A)
    run, _ = _fake_run(separate_returncode=1,
                       separate_stdout="sum check: FAILED  (level diff +0.6 dB, residual -60.0 dB)  -> out")
    paths = bench.prepare(ENTRY_A, tmp_path, run=run)
    assert paths["sum_check_failed"] is True
    # The warning was printed to stderr
    out = capsys.readouterr()
    assert "song-a: separation sum check failed (stems kept)" in out.err
    # But prepare continues: encode, render still ran
    assert paths["sc"].exists()
    assert paths["rebuild"].exists()


def test_prepare_separate_code_1_without_stems_is_a_crash(tmp_path):
    """Exit code 1 from separate without stems present is still a crash."""
    _mk_source(tmp_path, ENTRY_A)
    default_run, _ = _fake_run()

    def run_no_stems(cmd, **k):
        if len(cmd) > 3 and cmd[3] == "separate":
            # Code 1 but don't create stems, and no "sum check: FAILED" message
            return subprocess.CompletedProcess(cmd, 1, "some error", "")
        return default_run(cmd, **k)

    with pytest.raises(RuntimeError, match=r"song-a: separate failed"):
        bench.prepare(ENTRY_A, tmp_path, run=run_no_stems)


def test_prepare_separate_code_2_shows_meaningful_error_not_info_line(tmp_path):
    """Exit code 2 (real error) shows the last meaningful line, skipping
    INFO logs and progress lines."""
    _mk_source(tmp_path, ENTRY_A)

    def run_with_error(cmd, **k):
        if len(cmd) > 3 and cmd[3] == "separate":
            stderr = "some processing\n - INFO - Processing step 1\n - INFO - Processing step 2\nreal error here"
            return subprocess.CompletedProcess(cmd, 2, "", stderr)
        return _fake_run()[0](cmd, **k)

    with pytest.raises(RuntimeError, match=r"song-a: separate failed \(real error here\)"):
        bench.prepare(ENTRY_A, tmp_path, run=run_with_error)


def test_prepare_separate_code_2_with_tqdm_shows_meaningful_error(tmp_path):
    """Exit code 2 with tqdm progress lines still shows the actual error message."""
    _mk_source(tmp_path, ENTRY_A)

    def run_with_error(cmd, **k):
        if len(cmd) > 3 and cmd[3] == "separate":
            stderr = "step 1 100%|####| 10/10 [00:01<00:00, 10it/s]\nActual failure reason: disk full"
            return subprocess.CompletedProcess(cmd, 2, "", stderr)
        return _fake_run()[0](cmd, **k)

    with pytest.raises(RuntimeError, match=r"song-a: separate failed \(Actual failure reason: disk full\)"):
        bench.prepare(ENTRY_A, tmp_path, run=run_with_error)


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
# calibrate (Task 7): fake metric values -- median logic, degenerate/flip
# guard. `scorer.part_metrics` and `scorer.PART_TYPE` are faked throughout,
# so no real audio/model is touched; `_fake_run` fakes `separate` the same
# way the `prepare` tests do.
# --------------------------------------------------------------------------

_CALIB_ENTRIES = [
    {"name": "song-a", "source": "audio/test/song-a.wav", "start": None, "end": None,
     "title": "A", "artist": "Artist"},
    {"name": "song-b", "source": "audio/test/song-b.wav", "start": None, "end": None,
     "title": "B", "artist": "Artist"},
    {"name": "song-c", "source": "audio/test/song-c.wav", "start": None, "end": None,
     "title": "C", "artist": "Artist"},
]


def _mk_calib_sources(tmp_path):
    for e in _CALIB_ENTRIES:
        _mk_source(tmp_path, e)


def _write_anchors(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_calibrate_ceiling_and_floor_are_medians_over_tier_a(tmp_path, monkeypatch):
    _mk_calib_sources(tmp_path)
    monkeypatch.setattr(bench, "TIERS", {"A": _CALIB_ENTRIES, "B": [], "C": []})
    monkeypatch.setattr(bench.scorer, "PART_TYPE", {"piano": "pitched"})
    monkeypatch.setattr(bench.scorer, "is_active", lambda path: True)
    run, calls = _fake_run()

    # ceiling values keyed by ref song (each song vs its own 2nd separation):
    # median of {0.90, 0.95, 0.99} is 0.95, not the mean (0.9467)
    ceiling_by_song = {"song-a": 0.90, "song-b": 0.95, "song-c": 0.99}
    # floor values keyed by (ref, est) song pair: median of the 6 ordered
    # pairs' values ({0.1, 0.1, 0.1, 0.3, 0.3, 0.3}) is 0.2
    floor_by_pair = {("song-a", "song-b"): 0.1, ("song-a", "song-c"): 0.1,
                     ("song-b", "song-c"): 0.1, ("song-b", "song-a"): 0.3,
                     ("song-c", "song-a"): 0.3, ("song-c", "song-b"): 0.3}

    def fake_part_metrics(ref_path, est_path, key, *, doc=None, drum_cache=None):
        ref_path, est_path = Path(ref_path), Path(est_path)
        ref_song = ref_path.parent.name
        if "calib" in est_path.parts:
            return {"note_f1": ceiling_by_song[ref_song]}
        return {"note_f1": floor_by_pair[(ref_song, est_path.parent.name)]}

    monkeypatch.setattr(bench.scorer, "part_metrics", fake_part_metrics)

    anchors_path = tmp_path / "anchors.json"
    _write_anchors(anchors_path, {
        "pitched": {"note_f1": {"floor": 0.0, "ceiling": 0.5, "weight": 1, "axis": "what"}}})

    res = bench.calibrate(tmp_path, run=run, anchors_path=anchors_path)

    assert res["ceilings"]["pitched"]["note_f1"] == pytest.approx(0.95)
    assert res["floors"]["pitched"]["note_f1"] == pytest.approx(0.2)
    new = json.loads(anchors_path.read_text())
    assert new["pitched"]["note_f1"]["floor"] == pytest.approx(0.2)
    assert new["pitched"]["note_f1"]["ceiling"] == pytest.approx(0.95)
    assert new["pitched"]["note_f1"]["weight"] == 1 and new["pitched"]["note_f1"]["axis"] == "what"
    assert not anchors_path.with_name(anchors_path.name + ".part").exists()

    # a second separation per tier-A song, into out/bench/calib/<name>/
    for e in _CALIB_ENTRIES:
        assert (tmp_path / "out" / "bench" / "calib" / e["name"] / "piano.wav").exists()
        assert (tmp_path / "out" / "stems" / e["name"] / "piano.wav").exists()


def test_calibrate_leaves_degenerate_or_flipped_anchors_and_warns(tmp_path, monkeypatch):
    _mk_calib_sources(tmp_path)
    monkeypatch.setattr(bench, "TIERS", {"A": _CALIB_ENTRIES, "B": [], "C": []})
    monkeypatch.setattr(bench.scorer, "PART_TYPE", {"piano": "pitched"})
    monkeypatch.setattr(bench.scorer, "is_active", lambda path: True)
    run, _ = _fake_run()

    def fake_part_metrics(ref_path, est_path, key, *, doc=None, drum_cache=None):
        is_ceiling = "calib" in Path(est_path).parts
        # "chroma": ceiling == floor (degenerate); "onset_f1": measured
        # direction (ceiling < floor) flips the anchor's stored direction
        # (ceiling > floor, higher is better)
        return {"chroma": 0.7, "onset_f1": 0.2 if is_ceiling else 0.8}

    monkeypatch.setattr(bench.scorer, "part_metrics", fake_part_metrics)

    anchors_path = tmp_path / "anchors.json"
    _write_anchors(anchors_path, {"pitched": {
        "chroma": {"floor": 0.5, "ceiling": 0.98, "weight": 1, "axis": "what"},
        "onset_f1": {"floor": 0.0, "ceiling": 0.9, "weight": 1, "axis": "what"},
        "mert": {"floor": 0.6, "ceiling": 0.97, "weight": 1, "axis": "sound"}}})

    res = bench.calibrate(tmp_path, run=run, anchors_path=anchors_path)

    new = json.loads(anchors_path.read_text())
    assert new["pitched"]["chroma"] == {"floor": 0.5, "ceiling": 0.98, "weight": 1, "axis": "what"}
    assert new["pitched"]["onset_f1"] == {"floor": 0.0, "ceiling": 0.9, "weight": 1, "axis": "what"}
    assert new["pitched"]["mert"] == {"floor": 0.6, "ceiling": 0.97, "weight": 1, "axis": "sound"}

    warnings = res["warnings"]
    assert any("pitched.chroma" in w and "degenerate" in w for w in warnings)
    assert any("pitched.onset_f1" in w and "flip" in w for w in warnings)
    assert any("pitched.mert" in w and "not enough data" in w for w in warnings)
    assert res["ceilings"]["pitched"]["mert"] is None and res["floors"]["pitched"]["mert"] is None


def test_calibrate_excludes_silent_stems_from_both_ceiling_and_floor(tmp_path, monkeypatch):
    """`separate` always writes every stem file, so song-b's piano is
    silent (pure bleed) in this scenario -- both its own ceiling sample
    (vs. its 2nd separation) and every floor pair it appears in (as either
    side) must be excluded from the medians, per `scorer.is_active`."""
    _mk_calib_sources(tmp_path)
    monkeypatch.setattr(bench, "TIERS", {"A": _CALIB_ENTRIES, "B": [], "C": []})
    monkeypatch.setattr(bench.scorer, "PART_TYPE", {"piano": "pitched"})
    monkeypatch.setattr(bench.scorer, "is_active", lambda path: "song-b" not in Path(path).parts)
    run, _ = _fake_run()

    # song-b's ceiling value (0.9) and every pair it's in (0.05) are
    # deliberately extreme outliers -- if the silent-stem gate leaks, the
    # medians below would be pulled toward them, not just toward song-a/c's.
    ceiling_by_song = {"song-a": 0.1, "song-b": 0.9, "song-c": 0.2}
    floor_by_pair = {("song-a", "song-c"): 0.3, ("song-c", "song-a"): 0.7,
                     ("song-a", "song-b"): 0.05, ("song-b", "song-a"): 0.05,
                     ("song-b", "song-c"): 0.05, ("song-c", "song-b"): 0.05}

    def fake_part_metrics(ref_path, est_path, key, *, doc=None, drum_cache=None):
        ref_path, est_path = Path(ref_path), Path(est_path)
        ref_song = ref_path.parent.name
        if "calib" in est_path.parts:
            return {"note_f1": ceiling_by_song[ref_song]}
        return {"note_f1": floor_by_pair[(ref_song, est_path.parent.name)]}

    monkeypatch.setattr(bench.scorer, "part_metrics", fake_part_metrics)

    anchors_path = tmp_path / "anchors.json"
    _write_anchors(anchors_path, {
        "pitched": {"note_f1": {"floor": 0.0, "ceiling": 0.5, "weight": 1, "axis": "what"}}})

    res = bench.calibrate(tmp_path, run=run, anchors_path=anchors_path)

    # ceiling: median({0.1, 0.2}) -- song-b's 0.9 excluded
    assert res["ceilings"]["pitched"]["note_f1"] == pytest.approx(0.15)
    # floor: median({0.3, 0.7}) -- every pair touching song-b excluded
    assert res["floors"]["pitched"]["note_f1"] == pytest.approx(0.5)


def test_calibrate_degenerate_guard_is_relative_to_old_spread(tmp_path, monkeypatch):
    """A measured spread of 0.0015 is far above the old flat 1e-6 epsilon,
    but still < 10% of a small existing anchor's own 0.02 spread -- must be
    caught as degenerate. The same-sized spread against a wide existing
    anchor would also be degenerate, so `wide_range` uses a measured spread
    (0.5) that's comfortably above 10% of its anchor's spread (1.0), to
    prove a large-enough spread still updates normally."""
    _mk_calib_sources(tmp_path)
    monkeypatch.setattr(bench, "TIERS", {"A": _CALIB_ENTRIES, "B": [], "C": []})
    monkeypatch.setattr(bench.scorer, "PART_TYPE", {"piano": "pitched"})
    monkeypatch.setattr(bench.scorer, "is_active", lambda path: True)
    run, _ = _fake_run()

    def fake_part_metrics(ref_path, est_path, key, *, doc=None, drum_cache=None):
        is_ceiling = "calib" in Path(est_path).parts
        return {"small_range": 0.0115 if is_ceiling else 0.01,    # spread 0.0015
                "wide_range": 0.75 if is_ceiling else 0.25}       # spread 0.5

    monkeypatch.setattr(bench.scorer, "part_metrics", fake_part_metrics)

    anchors_path = tmp_path / "anchors.json"
    _write_anchors(anchors_path, {"pitched": {
        "small_range": {"floor": 0.0, "ceiling": 0.02, "weight": 1, "axis": "what"},
        "wide_range": {"floor": 0.0, "ceiling": 1.0, "weight": 1, "axis": "what"}}})

    res = bench.calibrate(tmp_path, run=run, anchors_path=anchors_path)

    new = json.loads(anchors_path.read_text())
    assert new["pitched"]["small_range"] == {"floor": 0.0, "ceiling": 0.02, "weight": 1, "axis": "what"}
    assert any("pitched.small_range" in w and "degenerate" in w for w in res["warnings"])
    assert new["pitched"]["wide_range"]["floor"] == pytest.approx(0.25)
    assert new["pitched"]["wide_range"]["ceiling"] == pytest.approx(0.75)


def test_calibrate_never_writes_a_floor_looser_than_its_floor_cap(tmp_path, monkeypatch):
    """`f0_cents`/`lag_ms_abs`/`word_mae_s` carry a `floor_cap` (design doc
    amendment, 2026-09-28): a different-song floor for an error-size metric
    can sit far past the point where the ear calls something wrong, so
    calibration must never write a floor looser than the cap, even when
    that's what it measured -- and must keep the `floor_cap` key itself."""
    _mk_calib_sources(tmp_path)
    monkeypatch.setattr(bench, "TIERS", {"A": _CALIB_ENTRIES, "B": [], "C": []})
    monkeypatch.setattr(bench.scorer, "PART_TYPE", {"piano": "pitched"})
    monkeypatch.setattr(bench.scorer, "is_active", lambda path: True)
    run, _ = _fake_run()

    def fake_part_metrics(ref_path, est_path, key, *, doc=None, drum_cache=None):
        is_ceiling = "calib" in Path(est_path).parts
        return {"f0_cents": 5.0 if is_ceiling else 500.0}   # measured floor 500 >> cap 100

    monkeypatch.setattr(bench.scorer, "part_metrics", fake_part_metrics)

    anchors_path = tmp_path / "anchors.json"
    _write_anchors(anchors_path, {"pitched": {
        "f0_cents": {"floor": 300.0, "ceiling": 10.0, "weight": 3, "axis": "what", "floor_cap": 100}}})

    res = bench.calibrate(tmp_path, run=run, anchors_path=anchors_path)

    # reported raw (before the cap) for the caller to see what was measured
    assert res["floors"]["pitched"]["f0_cents"] == pytest.approx(500.0)
    new = json.loads(anchors_path.read_text())
    assert new["pitched"]["f0_cents"]["floor"] == pytest.approx(100.0)      # capped, not 500
    assert new["pitched"]["f0_cents"]["ceiling"] == pytest.approx(5.0)
    assert new["pitched"]["f0_cents"]["weight"] == 3
    assert new["pitched"]["f0_cents"]["floor_cap"] == 100                   # kept
    assert new["pitched"]["f0_cents"]["floor_measured"] == pytest.approx(500.0)
    assert any("looser than its perceptual cap" in w for w in res["warnings"])
    assert '"floor_cap": 100, "floor_measured": 500.0' in anchors_path.read_text()


def test_calibrate_leaves_mix_and_sung_wer_manual_with_no_warning(tmp_path, monkeypatch):
    """`mix` (no `mix.wav` stem is ever separated) and the lyrics-aligned
    vocal metrics (calibration always calls `part_metrics` with `doc=None`)
    are never measured -- left untouched, reported in `manual`, and never
    warned about as "not enough data"."""
    _mk_calib_sources(tmp_path)
    monkeypatch.setattr(bench, "TIERS", {"A": _CALIB_ENTRIES, "B": [], "C": []})
    monkeypatch.setattr(bench.scorer, "PART_TYPE", {"lead_vocals": "vocal"})
    monkeypatch.setattr(bench.scorer, "is_active", lambda path: True)
    run, _ = _fake_run()

    def fake_part_metrics(ref_path, est_path, key, *, doc=None, drum_cache=None):
        assert doc is None
        is_ceiling = "calib" in Path(est_path).parts
        return {"word_mae_s": 0.05 if is_ceiling else 0.4}

    monkeypatch.setattr(bench.scorer, "part_metrics", fake_part_metrics)

    anchors_path = tmp_path / "anchors.json"
    _write_anchors(anchors_path, {
        "vocal": {"word_mae_s": {"floor": 0.5, "ceiling": 0.05, "weight": 1, "axis": "what"},
                  "sung_wer_excess": {"floor": 0.6, "ceiling": 0.0, "weight": 1, "axis": "what"}},
        "mix": {"chroma": {"floor": 0.5, "ceiling": 0.98, "weight": 1, "axis": "what"}}})

    res = bench.calibrate(tmp_path, run=run, anchors_path=anchors_path)

    new = json.loads(anchors_path.read_text())
    # measured normally
    assert new["vocal"]["word_mae_s"]["floor"] == pytest.approx(0.4)
    assert new["vocal"]["word_mae_s"]["ceiling"] == pytest.approx(0.05)
    # left untouched, no warning
    assert new["vocal"]["sung_wer_excess"] == {"floor": 0.6, "ceiling": 0.0, "weight": 1, "axis": "what"}
    assert new["mix"]["chroma"] == {"floor": 0.5, "ceiling": 0.98, "weight": 1, "axis": "what"}

    assert res["manual"] == {"vocal": ["sung_wer_excess"], "mix": ["chroma"]}
    assert not any("sung_wer_excess" in w for w in res["warnings"])
    assert not any("mix.chroma" in w for w in res["warnings"])
    # "print only the measured part types": mix never appears in ceilings/floors
    assert "mix" not in res["ceilings"] and "mix" not in res["floors"]
    assert "sung_wer_excess" not in res["ceilings"].get("vocal", {})


def test_calibrate_writes_anchors_sorted_and_compact(tmp_path, monkeypatch):
    _mk_calib_sources(tmp_path)
    monkeypatch.setattr(bench, "TIERS", {"A": _CALIB_ENTRIES, "B": [], "C": []})
    monkeypatch.setattr(bench.scorer, "PART_TYPE", {"piano": "pitched"})
    monkeypatch.setattr(bench.scorer, "is_active", lambda path: True)
    run, _ = _fake_run()
    monkeypatch.setattr(bench.scorer, "part_metrics",
                        lambda *a, **k: {"onset_f1": 0.8, "chroma": 0.9})

    anchors_path = tmp_path / "anchors.json"
    # deliberately out of alphabetical order, both part types and metrics
    _write_anchors(anchors_path, {
        "zeta": {"onset_f1": {"floor": 0.0, "ceiling": 0.9, "weight": 1, "axis": "what"}},
        "alpha": {"z_metric": {"floor": 0.0, "ceiling": 1.0, "weight": 1, "axis": "what"},
                  "chroma": {"floor": 0.5, "ceiling": 0.98, "weight": 1, "axis": "what"}}})

    bench.calibrate(tmp_path, run=run, anchors_path=anchors_path)

    text = anchors_path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]

    def _index_containing(needle: str) -> int:
        return next(i for i, ln in enumerate(lines) if needle in ln)

    # part types in sorted order
    assert _index_containing('"alpha":') < _index_containing('"zeta":')
    # metrics within a part type in sorted order, one compact line each
    chroma_lines = [ln for ln in lines if '"chroma"' in ln]
    assert len(chroma_lines) == 1
    assert '"floor"' in chroma_lines[0] and '"axis"' in chroma_lines[0]
    assert _index_containing('"chroma":') < _index_containing('"z_metric":')
    # still valid, round-tripping JSON with the same values
    assert json.loads(text)["alpha"]["chroma"]["ceiling"] == pytest.approx(0.98)


def test_calibrate_separate_sum_check_failure_is_a_warning_not_a_crash(tmp_path, monkeypatch, capsys):
    """Calibrate's separate calls also handle exit code 1 (sum check failure)
    as a warning, not a crash, so calibration continues to measure."""
    _mk_calib_sources(tmp_path)
    monkeypatch.setattr(bench, "TIERS", {"A": _CALIB_ENTRIES, "B": [], "C": []})
    monkeypatch.setattr(bench.scorer, "PART_TYPE", {"piano": "pitched"})
    monkeypatch.setattr(bench.scorer, "is_active", lambda path: True)

    run, _ = _fake_run(separate_returncode=1,
                       separate_stdout="sum check: FAILED  (level diff +0.6 dB)")

    def fake_part_metrics(ref_path, est_path, key, *, doc=None, drum_cache=None):
        return {"note_f1": 0.8}

    monkeypatch.setattr(bench.scorer, "part_metrics", fake_part_metrics)

    anchors_path = tmp_path / "anchors.json"
    _write_anchors(anchors_path, {
        "pitched": {"note_f1": {"floor": 0.0, "ceiling": 0.5, "weight": 1, "axis": "what"}}})

    res = bench.calibrate(tmp_path, run=run, anchors_path=anchors_path)

    # Calibration succeeded despite sum check failures
    assert res["ceilings"]["pitched"]["note_f1"] == pytest.approx(0.8)
    # Warning was printed for each sum check failure (one per separate call)
    out = capsys.readouterr()
    assert "separation sum check failed" in out.err


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_bench_calibrate_calls_calibrate_and_prints_a_summary(tmp_path, monkeypatch, capsys):
    from soundcode import cli

    seen = {}

    def fake_calibrate(root, run=subprocess.run, anchors_path=None):
        seen["root"] = root
        return {"anchors": {"pitched": {"note_f1": {"floor": 0.1, "ceiling": 0.9}}},
                "warnings": ["pitched.chroma: degenerate, keeping the existing anchor"],
                "ceilings": {"pitched": {"note_f1": 0.9}},
                "floors": {"pitched": {"note_f1": 0.1}}}

    monkeypatch.setattr(bench, "calibrate", fake_calibrate)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["bench", "--calibrate"]) == 0
    assert seen["root"] == Path(".")
    out = capsys.readouterr().out
    assert "pitched.note_f1: floor=0.1 ceiling=0.9" in out
    assert "degenerate" in out
    assert str(bench.anchors.ANCHORS_PATH) in out


def test_cli_bench_calibrate_labels_manual_entries_and_skips_unmeasured_part_types(
        tmp_path, monkeypatch, capsys):
    from soundcode import cli

    def fake_calibrate(root, run=subprocess.run, anchors_path=None):
        return {"anchors": {}, "warnings": [],
                "manual": {"mix": ["chroma", "onset_f1"], "vocal": ["sung_wer_excess"]},
                "ceilings": {"vocal": {"word_mae_s": 0.05}}, "floors": {"vocal": {"word_mae_s": 0.4}}}

    monkeypatch.setattr(bench, "calibrate", fake_calibrate)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["bench", "--calibrate"]) == 0
    out = capsys.readouterr().out
    assert "vocal.word_mae_s: floor=0.4 ceiling=0.05" in out
    assert "mix.chroma: manual (not calibrated)" in out
    assert "mix.onset_f1: manual (not calibrated)" in out
    assert "vocal.sung_wer_excess: manual (not calibrated)" in out
    # "print only the measured part types": no bare "mix" floor/ceiling line
    assert "mix: floor" not in out and "mix.chroma: floor" not in out


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


# --------------------------------------------------------------------------
# final-review fixes (Amendment 3 and the minor items)
# --------------------------------------------------------------------------

def _two_verse_result():
    r = _mk_result("song-a", 60.0, {"piano": 50.0})
    r["parts"]["piano"]["sections"] = [
        {"label": "verse", "a": 0.0, "b": 12.0, "score": 40.0, "axes": {}, "metrics": {}},
        {"label": "verse", "a": 72.0, "b": 84.0, "score": 90.0, "axes": {}, "metrics": {}}]
    r["worst"] = {"part": "piano", "label": "verse", "score": 40.0, "a": 0.0, "b": 12.0}
    r["missing"] = ["guitar"]
    return r


def test_history_stores_worst_missing_and_sections_keyed_by_label_at_time():
    h = bench._song_history(_two_verse_result())
    assert h["worst"] == {"part": "piano", "label": "verse", "score": 40.0, "a": 0.0, "b": 12.0}
    assert h["missing"] == ["guitar"]
    # two "verse"s no longer collapse into one key
    assert h["parts"]["piano"]["sections"] == {"verse@0:00": 40.0, "verse@1:12": 90.0}


def test_readme_lists_missing_parts_per_song():
    line = {"date": "d1", "label": "cur", "commit": "c1", "tier": "A",
            "songs": {"song-a": bench._song_history(_two_verse_result()),
                      "song-b": bench._song_history(_mk_result("song-b", 70.0, {"piano": 70.0}))}}
    readme = bench._render_readme([line])
    rows = [ln for ln in readme.splitlines() if "**song**" in ln]
    assert "missing: guitar" in rows[0] and "song-a" in rows[0]
    assert "missing" not in rows[1]


def test_readme_header_and_bench_help_carry_the_cache_note(capsys):
    from soundcode import cli
    note = "stages are cached by input mtime; after changing encode/render code run with --force"
    assert note in bench._render_readme([])
    with pytest.raises(SystemExit):
        cli.main(["bench", "--help"])
    assert note in " ".join(capsys.readouterr().out.split())


def test_fmt_delta_takes_its_sign_from_the_rounded_delta():
    assert bench._fmt_delta(70.0, 69.7) == " (+0)"          # -0.3 rounds to 0: not "(-0)"
    assert bench._fmt_delta(70.0, 69.4) == " (-1)"
    assert bench._fmt_delta(70.0, 71.2) == " (+1)"
    assert bench._fmt_delta(None, 71.2) == ""


def test_diff_reports_state_transitions_and_the_readme_shows_them():
    prev = {"date": "d0", "label": "p", "commit": "c0", "tier": "A", "songs": {
        "a": {"score": 70.0, "worst": None, "parts": {"piano": {"score": 37.0},
                                                       "bass": {"score": None},
                                                       "drums": {"score": None}}}}}
    cur = {"date": "d1", "label": "c", "commit": "c1", "tier": "A", "songs": {
        "a": {"score": 70.0, "worst": None, "parts": {"piano": {"score": None},
                                                       "bass": {"score": 55.0},
                                                       "drums": {"score": None}}}}}
    d = bench.diff(prev, cur)
    assert ("a", "piano", 37.0, None) in d                  # scored -> silent
    assert ("a", "bass", None, 55.0) in d                   # silent -> scored
    assert not any(p == "drums" for _, p, *_ in d)          # silent both times: no change
    readme = bench._render_readme([prev, cur])
    piano = [ln for ln in readme.splitlines() if ln.startswith("| a | piano |")][0]
    assert "37 → —" in piano
    assert "(-" not in readme.split("### Regressed")[1]      # transitions aren't +/- deltas


def test_run_bench_prints_state_transitions(tmp_path, monkeypatch, capsys):
    _mk_source(tmp_path, ENTRY_A)
    monkeypatch.setattr(bench, "TIERS", {"A": [ENTRY_A]})
    results = [_mk_result("song-a", 70.0, {"piano": 37.0}), _mk_result("song-a", 70.0, {"piano": 37.0})]
    results[1]["parts"]["piano"].update(silent=True, song=None, sections=[])
    monkeypatch.setattr(bench.scorer, "score_song", lambda *a, **k: results.pop(0))
    run, _ = _fake_run()
    bench.run_bench("A", "first", tmp_path, run=run)
    bench.run_bench("A", "second", tmp_path, run=run)
    assert "song-a piano 37 → —" in capsys.readouterr().out


def test_update_anchor_stores_the_measured_floor_when_the_cap_applies():
    old = {"floor": 300.0, "ceiling": 10.0, "weight": 3, "axis": "what", "floor_cap": 100}
    w: list[str] = []
    new = bench._update_anchor("vocal", "f0_cents", old, 5.0, 747.0, w)
    assert new["floor"] == 100 and new["floor_measured"] == pytest.approx(747.0)
    # a measured floor inside the cap is written as is, and a stale
    # floor_measured from an earlier capped calibration is dropped
    new2 = bench._update_anchor("vocal", "f0_cents", new, 5.0, 80.0, w)
    assert new2["floor"] == pytest.approx(80.0) and "floor_measured" not in new2


def test_update_anchor_degenerate_guard_compares_capped_spreads():
    # the old anchor's *effective* spread is 100 - 0 = 100 (its 700 floor is
    # capped); a measured 0..50 spread is 50% of that -- a real update, even
    # though it is < 10% of the uncapped 700
    old = {"floor": 700.0, "ceiling": 0.0, "weight": 1, "axis": "what", "floor_cap": 100}
    w: list[str] = []
    new = bench._update_anchor("drums", "lag_ms_abs", old, 0.0, 50.0, w)
    assert new["floor"] == pytest.approx(50.0) and not w
    # and the measured side is capped too: 0..9 is 9% of 100 -> degenerate
    new = bench._update_anchor("drums", "lag_ms_abs", old, 0.0, 9.0, w)
    assert new == old and any("degenerate" in x for x in w)


def test_shipped_anchors_apply_their_floor_caps():
    from soundcode.score import anchors
    capped = 0
    for part_type, metrics in anchors.ANCHORS.items():
        for metric, a in metrics.items():
            cap = a.get("floor_cap")
            if cap is None:
                assert "floor_measured" not in a, f"{part_type}.{metric}"
                continue
            assert a["floor"] <= cap, f"{part_type}.{metric}"
            if "floor_measured" in a:
                capped += 1
                assert a["floor_measured"] > cap
                assert a["floor"] == min(a["floor_measured"], cap)
    assert capped >= 5
    # Check vocal f0_cents has proper capped-floor structure
    a = anchors.ANCHORS["vocal"]["f0_cents"]
    assert "floor_measured" in a
    assert a["floor_measured"] > a["floor_cap"]
    assert a["floor"] == min(a["floor_measured"], a["floor_cap"])
