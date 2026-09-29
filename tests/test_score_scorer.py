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


def test_to_score_honours_floor_cap_for_lower_is_better_metrics():
    """Amendment (2026-09-28): a lower-is-better metric's effective floor is
    `min(floor, floor_cap)`, not whatever calibration measured -- a
    different-song floor can sit far past where the ear calls something
    wrong (design doc)."""
    a = {"floor": 700.0, "ceiling": 0.0, "floor_cap": 100.0}
    assert anchors.to_score(100.0, a) == pytest.approx(0.0)     # at the cap: 0, not ~86 (vs. floor=700)
    assert anchors.to_score(50.0, a) == pytest.approx(50.0)     # halfway to the cap
    assert anchors.to_score(400.0, a) == pytest.approx(0.0)     # past the cap: clamped, not ~43
    without_cap = anchors.to_score(400.0, {"floor": 700.0, "ceiling": 0.0})
    assert without_cap > 0.0                                    # confirms the cap is what changed it


def test_to_score_floor_cap_is_ignored_for_higher_is_better_metrics():
    a = {"floor": 0.0, "ceiling": 1.0, "floor_cap": 100.0}      # nonsensical here, must be a no-op
    assert anchors.to_score(0.5, a) == pytest.approx(50.0)


def test_anchors_json_has_every_part_type_and_lower_is_better_pairs():
    for part_type in ("pitched", "bass", "drums", "vocal", "mix"):
        assert part_type in anchors.ANCHORS
    # weight/axis are design choices, pinned exactly; floor/ceiling are
    # re-measured by `bench --calibrate` and deliberately not pinned here
    # (test_score_real.py covers the anchors that must always hold).
    assert anchors.ANCHORS["pitched"]["note_f1"]["axis"] == "what"
    assert anchors.ANCHORS["pitched"]["note_f1"]["weight"] == 1
    # sung_wer_excess is never calibrated (bench._MANUAL_METRICS) -- pinned exactly
    assert anchors.ANCHORS["vocal"]["sung_wer_excess"] == {
        "floor": 0.6, "ceiling": 0.0, "weight": 1, "axis": "what"}
    assert anchors.ANCHORS["drums"]["decay_ratio_err"]["axis"] == "sound"
    assert anchors.ANCHORS["mix"]["lufs_diff_abs"]["axis"] == "dyn"


def test_anchors_json_has_the_perceptual_floor_caps_and_pitch_weight():
    """Amendment (2026-09-28): pitch, lag and word-timing floors are capped,
    and pitch weighs 3x in the vocal/bass What axis -- design choices, not
    calibrated data, so pinned exactly."""
    for part_type in ("vocal", "bass"):
        f0 = anchors.ANCHORS[part_type]["f0_cents"]
        assert f0["floor_cap"] == 100
        assert f0["weight"] == 3
    for part_type in ("bass", "drums", "mix", "pitched", "vocal"):
        assert anchors.ANCHORS[part_type]["lag_ms_abs"]["floor_cap"] == 100
    assert anchors.ANCHORS["vocal"]["word_mae_s"]["floor_cap"] == 0.5


# Exact metric-name set per part type, from the spec's "Metrics per part"
# table (docs/superpowers/specs/2026-09-28-benchmark-scorer-design.md),
# plus `lag_ms_abs` (What, every part — "the lag per 20 s window" applies
# on every part) and the low-weight `spectral_db` guard (Sound, 0.1) where
# that part already carries a Sound metric it could self-optimise against.
# Pinned here so `anchors.json` can't silently drift from the spec table.
EXPECTED_METRIC_NAMES = {
    "pitched": {"note_f1", "chroma", "onset_f1", "lag_ms_abs",
                "mert", "logspec_db", "spectral_db",
                "env_corr", "level_diff_db_abs"},
    "bass": {"onset_f1", "f0_cents", "chroma", "lag_ms_abs",
             "mert", "logspec_db", "spectral_db",
             "env_corr", "level_diff_db_abs"},
    "drums": {"onset_f1", "drum_voice_f1", "lag_ms_abs",
              "mert", "decay_ratio_err", "spectral_db",
              "env_corr"},
    "vocal": {"f0_cents", "word_mae_s", "sung_wer_excess", "lag_ms_abs",
              "voice_sim", "mert", "spectral_db",
              "env_corr", "level_diff_db_abs"},
    "mix": {"chroma", "onset_f1", "lag_ms_abs",
            "mert", "width_diff", "spectral_db",
            "lufs_diff_abs", "lufs_section_diff"},
}


def test_mix_loudness_anchors_follow_amendment_3():
    """The whole-song LUFS difference is a self-optimised gain offset (low
    weight); the dynamic arc once that offset is removed carries Dyn."""
    assert anchors.ANCHORS["mix"]["lufs_diff_abs"]["weight"] == 0.1
    assert anchors.ANCHORS["mix"]["lufs_diff_abs"]["axis"] == "dyn"
    assert anchors.ANCHORS["mix"]["lufs_section_diff"] == {
        "floor": 6, "ceiling": 0.5, "weight": 1, "axis": "dyn"}


def test_anchors_json_metric_names_match_the_spec_table_exactly():
    for part_type, expected in EXPECTED_METRIC_NAMES.items():
        assert set(anchors.ANCHORS[part_type].keys()) == expected, part_type


# --- scorer.score_slice ---------------------------------------------------


def _at_ceiling(part_type: str, metric: str) -> float:
    """The raw value that scores exactly 100 for this (part_type, metric)
    right now -- so a test stays correct across `bench --calibrate`
    re-measuring `anchors.json`, instead of pinning a raw number that only
    happened to hit 100 under some earlier calibration."""
    return anchors.ANCHORS[part_type][metric]["ceiling"]


def _at_floor(part_type: str, metric: str) -> float:
    """The raw value that scores exactly 0 for this (part_type, metric)."""
    return anchors.ANCHORS[part_type][metric]["floor"]


def test_score_slice_perfect_metrics_score_100():
    metrics = {"note_f1": _at_ceiling("pitched", "note_f1"),
               "chroma": _at_ceiling("pitched", "chroma"),
               "onset_f1": _at_ceiling("pitched", "onset_f1"),
               "env_corr": _at_ceiling("pitched", "env_corr"),
               "level_diff_db": 0}
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
    metrics = {"note_f1": _at_ceiling("pitched", "note_f1"),
               "chroma": _at_ceiling("pitched", "chroma"),
               "onset_f1": _at_ceiling("pitched", "onset_f1"),      # what: 100
               "mert": _at_ceiling("pitched", "mert"),
               "logspec_db": _at_ceiling("pitched", "logspec_db"),  # sound: 100
               "env_corr": None, "level_diff_db_abs": None}         # dyn: all None
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
    metrics = {"logspec_db": _at_ceiling("pitched", "logspec_db"),
               "spectral_db": _at_floor("pitched", "spectral_db")}
    out = scorer.score_slice(metrics, "pitched")
    assert out["axes"]["sound"] == pytest.approx(100.0 * (1.0 / 1.1), abs=0.01)
    assert out["axes"]["sound"] > 85.0   # nowhere near the unweighted 50.0


def test_score_slice_renormalises_axis_weights_when_an_axis_is_missing():
    # only "what" metrics given -> score equals the what axis exactly (a
    # single axis's weighted geometric mean over itself is itself), not
    # 0.4 * what (which would be the un-renormalised formula).
    a = anchors.ANCHORS["pitched"]["note_f1"]
    metrics = {"note_f1": (a["floor"] + a["ceiling"]) / 2}   # -> 50.0
    out = scorer.score_slice(metrics, "pitched")
    assert out["axes"]["what"] == pytest.approx(50.0)
    assert out["score"] == pytest.approx(50.0)


# --- scorer.score_slice: geometric axis combination (design doc amendment,
# 2026-09-28) -------------------------------------------------------------


def _synthetic_anchors(monkeypatch):
    """One metric per axis, weight 1, already 0-100 -- isolates the axis-
    combination math from calibration data and from each axis's own
    weighted-metric-mean step (covered by the tests above)."""
    table = {"synth": {
        "what_m": {"floor": 0.0, "ceiling": 100.0, "weight": 1, "axis": "what"},
        "sound_m": {"floor": 0.0, "ceiling": 100.0, "weight": 1, "axis": "sound"},
        "dyn_m": {"floor": 0.0, "ceiling": 100.0, "weight": 1, "axis": "dyn"},
    }}
    monkeypatch.setattr(anchors, "ANCHORS", table)


def test_geometric_combination_penalises_one_bad_axis_far_more_than_arithmetic_would(monkeypatch):
    _synthetic_anchors(monkeypatch)
    # An arithmetic 0.4/0.4/0.2 mix of what=0, sound=100, dyn=100 would give
    # 60 -- "fine, mostly". The geometric mean (the failed axis floored at 1
    # before the log) lands far below that: one badly wrong axis can no
    # longer be averaged out by the other two.
    out = scorer.score_slice({"what_m": 0.0, "sound_m": 100.0, "dyn_m": 100.0}, "synth")
    assert out["axes"] == {"what": 0.0, "sound": 100.0, "dyn": 100.0}
    assert out["score"] < 40.0


def test_geometric_combination_renormalises_weights_when_an_axis_is_none(monkeypatch):
    _synthetic_anchors(monkeypatch)
    # dyn is None (no dyn_m metric given) -> what/sound renormalise from
    # 0.4/0.4 to 0.5/0.5, and the part score is their geometric mean
    # (sqrt(100 * 25) = 50), not the arithmetic (100 + 25) / 2 = 62.5.
    out = scorer.score_slice({"what_m": 100.0, "sound_m": 25.0}, "synth")
    assert out["axes"]["dyn"] is None
    assert out["score"] == pytest.approx((100.0 * 25.0) ** 0.5)


# --- scorer.score_part -----------------------------------------------------


def _perfect(part_type):
    if part_type == "pitched":
        return {"note_f1": 0.9, "chroma": 0.98, "onset_f1": 0.9,
                 "mert": 0.97, "logspec_db": 1.5, "env_corr": 0.95}
    raise NotImplementedError


def test_score_part_worst_section_is_picked_correctly():
    song = _perfect("pitched")
    a = anchors.ANCHORS["pitched"]["note_f1"]
    mid = (a["floor"] + a["ceiling"]) / 2
    sections = [
        ("intro", {"note_f1": a["ceiling"]}),   # 100
        # below floor -> axis 0, floored to 1 before the geometric mean's
        # log (design doc amendment) -> a single-axis section score of 1.0,
        # not 0 -- still the worst of the three, just never truly 0
        ("verse", {"note_f1": 0.0}),
        ("chorus", {"note_f1": mid}),            # 50, strictly between
    ]
    windows = []
    out = scorer.score_part(song, sections, windows, "pitched")
    assert out["worst"] == {"label": "verse", "score": pytest.approx(1.0)}


def test_score_part_worst_ignores_none_score_sections():
    song = _perfect("pitched")
    a = anchors.ANCHORS["pitched"]["note_f1"]
    mid = (a["floor"] + a["ceiling"]) / 2
    sections = [
        ("intro", {"note_f1": mid}),            # 50
        ("silent-gap", {}),                      # score None -> not a candidate
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
    got = scorer.song_score(parts, energy, mix=_part(100.0))
    # bass counts as 0 but still carries its energy weight -> pulls the mean down
    parts_mean = 0.5 * 100.0 + 0.5 * 0.0
    assert got == pytest.approx(0.5 * parts_mean + 0.5 * 100.0)


def test_song_score_silent_part_is_excluded_not_zeroed():
    parts = {"piano": _part(100.0), "bass": scorer.score_part({}, [], [], "bass", silent=True)}
    energy = {"piano": 0.5, "bass": 0.5}
    got = scorer.song_score(parts, energy, mix=_part(100.0))
    # bass is silent -> excluded entirely; piano alone (renormalised) carries the part half
    assert got == pytest.approx(0.5 * 100.0 + 0.5 * 100.0)


def test_song_score_missing_vs_silent_differ():
    energy = {"piano": 0.5, "bass": 0.5}
    missing_parts = {"piano": _part(100.0), "bass": scorer.score_part({}, [], [], "bass", missing=True)}
    silent_parts = {"piano": _part(100.0), "bass": scorer.score_part({}, [], [], "bass", silent=True)}
    missing_score = scorer.song_score(missing_parts, energy, mix=_part(100.0))
    silent_score = scorer.song_score(silent_parts, energy, mix=_part(100.0))
    assert missing_score < silent_score


def test_song_score_blends_50_50_with_mix():
    parts = {"piano": _part(80.0)}
    energy = {"piano": 1.0}
    assert scorer.song_score(parts, energy, mix=_part(40.0)) == pytest.approx(60.0)


def test_song_score_accepts_a_score_part_shaped_mix():
    parts = {"piano": _part(80.0)}
    energy = {"piano": 1.0}
    mix = _part(40.0)
    assert scorer.song_score(parts, energy, mix=mix) == pytest.approx(60.0)


def test_song_score_rejects_a_bare_number_as_mix():
    parts = {"piano": _part(80.0)}
    energy = {"piano": 1.0}
    with pytest.raises(ValueError):
        scorer.song_score(parts, energy, mix=40.0)


def test_song_score_rejects_an_unrecognised_mix_shape():
    parts = {"piano": _part(80.0)}
    energy = {"piano": 1.0}
    with pytest.raises(ValueError):
        scorer.song_score(parts, energy, mix={"score": 40.0})   # missing "song" key


def test_song_score_all_parts_silent_comes_from_mix_alone():
    parts = {"piano": scorer.score_part({}, [], [], "pitched", silent=True),
              "bass": scorer.score_part({}, [], [], "bass", silent=True)}
    energy = {"piano": 0.5, "bass": 0.5}
    assert scorer.song_score(parts, energy, mix=_part(73.0)) == pytest.approx(73.0)


def test_song_score_all_parts_silent_and_no_mix_is_none():
    parts = {"piano": scorer.score_part({}, [], [], "pitched", silent=True)}
    energy = {"piano": 1.0}
    assert scorer.song_score(parts, energy, mix=None) is None


def test_song_score_energy_shares_summing_to_zero_falls_back_to_mix():
    # non-silent parts, but every energy share is 0 -> no weight to average
    # over on the parts side, so the song score is the mix score alone.
    parts = {"piano": _part(10.0), "bass": _part(20.0)}
    energy = {"piano": 0.0, "bass": 0.0}
    assert scorer.song_score(parts, energy, mix=_part(66.0)) == pytest.approx(66.0)


def test_song_score_a_part_with_a_none_song_score_counts_as_zero():
    # not missing, not silent, but its own song score is None (e.g. every
    # metric came back None) -> counts as 0, same treatment as "missing",
    # and still keeps its energy weight.
    none_scored = {"song": {"axes": {}, "score": None, "metrics": {}},
                    "sections": [], "windows": [], "worst": None,
                    "missing": False, "silent": False}
    parts = {"piano": _part(100.0), "bass": none_scored}
    energy = {"piano": 0.5, "bass": 0.5}
    got = scorer.song_score(parts, energy, mix=_part(100.0))
    parts_mean = 0.5 * 100.0 + 0.5 * 0.0
    assert got == pytest.approx(0.5 * parts_mean + 0.5 * 100.0)


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
