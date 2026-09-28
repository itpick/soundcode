"""Anchors and the part/section/song scorer (spec 2026-09-28-benchmark-scorer, Task 4)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soundcode.score import anchors, scorer  # noqa: E402


# --- anchors.to_score ---------------------------------------------------


def test_to_score_higher_is_better_both_directions_and_clamping():
    a = {"floor": 0.5, "ceiling": 0.98}
    assert anchors.to_score(0.5, a) == pytest.approx(0.0)
    assert anchors.to_score(0.98, a) == pytest.approx(100.0)
    mid = 0.5 + (0.98 - 0.5) / 2
    assert anchors.to_score(mid, a) == pytest.approx(50.0)
    assert anchors.to_score(0.2, a) == pytest.approx(0.0)     # below floor, clamped
    assert anchors.to_score(1.0, a) == pytest.approx(100.0)   # above ceiling, clamped


def test_to_score_lower_is_better_both_directions_and_clamping():
    a = {"floor": 150, "ceiling": 5}
    assert anchors.to_score(150, a) == pytest.approx(0.0)
    assert anchors.to_score(5, a) == pytest.approx(100.0)
    assert anchors.to_score(0, a) == pytest.approx(100.0)      # better than ceiling, clamped
    assert anchors.to_score(400, a) == pytest.approx(0.0)      # worse than floor, clamped
    assert anchors.to_score(77.5, a) == pytest.approx(50.0)


def test_to_score_none_is_none():
    assert anchors.to_score(None, {"floor": 0.0, "ceiling": 1.0}) is None


def test_anchors_json_has_every_part_type_and_lower_is_better_pairs():
    for part_type in ("pitched", "bass", "drums", "vocal", "mix"):
        assert part_type in anchors.ANCHORS
    # spot-check the exact values from the design doc's table
    assert anchors.ANCHORS["pitched"]["note_f1"] == {
        "floor": 0.0, "ceiling": 0.9, "weight": 1, "axis": "what"}
    assert anchors.ANCHORS["vocal"]["sung_wer_excess"] == {
        "floor": 0.6, "ceiling": 0.0, "weight": 1, "axis": "what"}
    assert anchors.ANCHORS["drums"]["decay_ratio_err"]["axis"] == "sound"
    assert anchors.ANCHORS["mix"]["lufs_diff_abs"]["axis"] == "dyn"


# --- scorer.score_slice ---------------------------------------------------


def test_score_slice_perfect_metrics_score_100():
    metrics = {"note_f1": 0.9, "chroma": 0.98, "onset_f1": 0.9,
               "env_corr": 0.95, "level_diff_db": 0}
    # level_diff_db isn't an anchor name (level_diff_db_abs is) -> ignored
    out = scorer.score_slice(metrics, "pitched")
    assert out["axes"]["what"] == pytest.approx(100.0)
    assert out["axes"]["dyn"] == pytest.approx(100.0)
    assert out["score"] == pytest.approx(100.0)
    assert "level_diff_db" not in out["metrics"]


def test_score_slice_unknown_metric_names_are_ignored_not_a_crash():
    out = scorer.score_slice({"note_f1": 0.9, "totally_unknown_metric": 42.0}, "pitched")
    assert "totally_unknown_metric" not in out["metrics"]
    assert out["metrics"]["note_f1"]["score"] == pytest.approx(100.0)


def test_axis_with_all_none_metrics_is_excluded_not_zero():
    # every "dyn" metric present is None -> dyn axis is None, and the part
    # score is the mean of the remaining (non-None) axes only, never 0.
    metrics = {"note_f1": 0.9, "chroma": 0.98, "onset_f1": 0.9,   # what: 100
               "mert": 0.97, "logspec_db": 1.5,                    # sound: 100
               "env_corr": None, "level_diff_db_abs": None}        # dyn: all None
    out = scorer.score_slice(metrics, "pitched")
    assert out["axes"]["dyn"] is None
    assert out["axes"]["what"] == pytest.approx(100.0)
    assert out["axes"]["sound"] == pytest.approx(100.0)
    assert out["score"] == pytest.approx(100.0)   # renormalised over what+sound only


def test_score_slice_all_none_metrics_scores_none():
    out = scorer.score_slice({"note_f1": None, "chroma": None}, "pitched")
    assert out["axes"] == {"what": None, "sound": None, "dyn": None}
    assert out["score"] is None


def test_low_weight_metric_barely_moves_its_axis():
    # logspec_db (weight 1) at a perfect score, spectral_db (weight 0.1) at
    # its worst: the sound axis should sit close to logspec_db's 100, far
    # from the 50/50 average of 100 and 0.
    metrics = {"logspec_db": 1.5, "spectral_db": 12}
    out = scorer.score_slice(metrics, "pitched")
    assert out["axes"]["sound"] == pytest.approx(100.0 * (1.0 / 1.1), abs=0.01)
    assert out["axes"]["sound"] > 85.0   # nowhere near the unweighted 50.0


def test_score_slice_renormalises_axis_weights_when_an_axis_is_missing():
    # only "what" metrics given -> score equals the what axis exactly, not
    # 0.4 * what (which would be the un-renormalised formula).
    metrics = {"note_f1": 0.45}   # -> 50.0
    out = scorer.score_slice(metrics, "pitched")
    assert out["axes"]["what"] == pytest.approx(50.0)
    assert out["score"] == pytest.approx(50.0)


# --- scorer.score_part -----------------------------------------------------


def _perfect(part_type):
    if part_type == "pitched":
        return {"note_f1": 0.9, "chroma": 0.98, "onset_f1": 0.9,
                 "mert": 0.97, "logspec_db": 1.5, "env_corr": 0.95}
    raise NotImplementedError


def test_score_part_worst_section_is_picked_correctly():
    song = _perfect("pitched")
    sections = [
        ("intro", {"note_f1": 0.9}),          # 100
        ("verse", {"note_f1": 0.0}),           # 0 <- worst
        ("chorus", {"note_f1": 0.45}),         # 50
    ]
    windows = []
    out = scorer.score_part(song, sections, windows, "pitched")
    assert out["worst"] == {"label": "verse", "score": pytest.approx(0.0)}


def test_score_part_worst_ignores_none_score_sections():
    song = _perfect("pitched")
    sections = [
        ("intro", {"note_f1": 0.45}),          # 50
        ("silent-gap", {}),                     # score None -> not a candidate
    ]
    out = scorer.score_part(song, sections, [], "pitched")
    assert out["worst"] == {"label": "intro", "score": pytest.approx(50.0)}


def test_score_part_missing_scores_zero_and_flags_missing():
    out = scorer.score_part({}, [], [], "pitched", missing=True)
    assert out["missing"] is True
    assert out["silent"] is False
    assert out["song"]["score"] == pytest.approx(0.0)
    assert out["sections"] == []
    assert out["worst"] is None


def test_score_part_silent_is_not_scored_and_not_a_failure():
    out = scorer.score_part({}, [], [], "pitched", silent=True)
    assert out["silent"] is True
    assert out["missing"] is False
    assert out["song"] is None
    assert out["worst"] is None


def test_score_part_song_field_matches_score_slice():
    song = _perfect("pitched")
    out = scorer.score_part(song, [], [], "pitched")
    assert out["song"] == scorer.score_slice(song, "pitched")


# --- scorer.song_score -------------------------------------------------


def _part(score):
    return {"song": {"axes": {}, "score": score, "metrics": {}},
            "sections": [], "windows": [], "worst": None,
            "missing": False, "silent": False}


def test_song_score_missing_active_part_counts_as_zero():
    parts = {"piano": _part(100.0), "bass": scorer.score_part({}, [], [], "bass", missing=True)}
    energy = {"piano": 0.5, "bass": 0.5}
    got = scorer.song_score(parts, energy, mix=100.0)
    # bass counts as 0 but still carries its energy weight -> pulls the mean down
    parts_mean = 0.5 * 100.0 + 0.5 * 0.0
    assert got == pytest.approx(0.5 * parts_mean + 0.5 * 100.0)


def test_song_score_silent_part_is_excluded_not_zeroed():
    parts = {"piano": _part(100.0), "bass": scorer.score_part({}, [], [], "bass", silent=True)}
    energy = {"piano": 0.5, "bass": 0.5}
    got = scorer.song_score(parts, energy, mix=100.0)
    # bass is silent -> excluded entirely; piano alone (renormalised) carries the part half
    assert got == pytest.approx(0.5 * 100.0 + 0.5 * 100.0)


def test_song_score_missing_vs_silent_differ():
    energy = {"piano": 0.5, "bass": 0.5}
    missing_parts = {"piano": _part(100.0), "bass": scorer.score_part({}, [], [], "bass", missing=True)}
    silent_parts = {"piano": _part(100.0), "bass": scorer.score_part({}, [], [], "bass", silent=True)}
    missing_score = scorer.song_score(missing_parts, energy, mix=100.0)
    silent_score = scorer.song_score(silent_parts, energy, mix=100.0)
    assert missing_score < silent_score


def test_song_score_blends_50_50_with_mix():
    parts = {"piano": _part(80.0)}
    energy = {"piano": 1.0}
    assert scorer.song_score(parts, energy, mix=40.0) == pytest.approx(60.0)


def test_song_score_accepts_a_score_part_shaped_mix():
    parts = {"piano": _part(80.0)}
    energy = {"piano": 1.0}
    mix = _part(40.0)
    assert scorer.song_score(parts, energy, mix=mix) == pytest.approx(60.0)


# --- named constants -----------------------------------------------------


def test_part_type_mapping_constant():
    assert scorer.PART_TYPE["piano"] == "pitched"
    assert scorer.PART_TYPE["guitar"] == "pitched"
    assert scorer.PART_TYPE["other"] == "pitched"
    assert scorer.PART_TYPE["bass"] == "bass"
    assert scorer.PART_TYPE["drums"] == "drums"
    assert scorer.PART_TYPE["lead_vocals"] == "vocal"
    assert scorer.PART_TYPE["backing_vocals"] == "vocal"
    assert scorer.PART_TYPE["mix"] == "mix"
    assert "residual" not in scorer.PART_TYPE


def test_axis_weights_constant():
    assert scorer.AXIS_WEIGHTS == {"what": 0.4, "sound": 0.4, "dyn": 0.2}
