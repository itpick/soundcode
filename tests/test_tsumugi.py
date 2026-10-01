"""tsumugi subprocess wrapper (spec Part C)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import tsumugi as ts  # noqa: E402


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    py = tmp_path / "tsumugi" / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setenv("SOUNDCODE_TSUMUGI", str(tmp_path / "tsumugi"))
    return tmp_path / "tsumugi"


def test_missing_install_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCODE_TSUMUGI", str(tmp_path / "nope"))
    with pytest.raises(ts.TsumugiError, match="install_tsumugi.sh"):
        ts.home()


def test_transcribe_builds_the_cli_call(fake_home, monkeypatch, tmp_path):
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"], seen["cwd"] = cmd, kw.get("cwd")
        Path(cmd[cmd.index("--output-midi") + 1]).write_bytes(b"MThd")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    out = ts.transcribe(tmp_path / "piano.wav", tmp_path / "piano.mid", "default")
    assert out.exists()
    cmd = seen["cmd"]
    assert cmd[0].endswith(".venv/bin/python") and cmd[1:3] == ["-m", "instrument_agnostic_amt.amt.cli.infer"]
    assert cmd[cmd.index("--type") + 1] == "default"
    assert "--device" in cmd and "--disable-tqdm" in cmd
    assert seen["cwd"] == str(fake_home)


def test_nonzero_exit_names_the_step(fake_home, monkeypatch, tmp_path):
    monkeypatch.setattr(ts.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom\nRuntimeError: bad audio"))
    with pytest.raises(ts.TsumugiError, match="amt.cli.infer: RuntimeError: bad audio"):
        ts.transcribe(tmp_path / "x.wav", tmp_path / "x.mid", "default")


def test_refine_reads_global_candidates(fake_home, monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        report = {"global_candidates": [{"class_name": "piano", "probability": 0.73},
                                        {"class_name": "electric_piano", "probability": 0.27}]}
        Path(cmd[cmd.index("--output-json") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    got = ts.refine(tmp_path / "g.wav", tmp_path / "g.mid", "piano", tmp_path / "g.json")
    assert got == [("piano", 0.73), ("electric_piano", 0.27)]


def test_refine_bad_json_is_a_tsumugi_error(fake_home, monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        Path(cmd[cmd.index("--output-json") + 1]).write_text("{not json")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    with pytest.raises(ts.TsumugiError, match="refine"):
        ts.refine(tmp_path / "g.wav", tmp_path / "g.mid", "piano", tmp_path / "g.json")


def test_stem_model_table_covers_every_separation_stem():
    from soundcode.separate import STEMS
    assert set(STEMS) <= set(ts.STEM_MODEL)


def test_paths_are_passed_absolute_because_tsumugi_runs_in_its_own_dir(fake_home, monkeypatch, tmp_path):
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        Path(cmd[cmd.index("--output-midi") + 1]).write_bytes(b"MThd")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    ts.transcribe(Path("piano.wav"), Path("piano.mid"), "default")
    cmd = seen["cmd"]
    assert Path(cmd[cmd.index("--audio") + 1]).is_absolute()
    assert Path(cmd[cmd.index("--output-midi") + 1]).is_absolute()


# --- MIDI -> .sc lines --------------------------------------------------------------------

import pretty_midi  # noqa: E402
from soundcode import tsumugi_sc as tsc  # noqa: E402

GRID = {"downbeat": 1.0, "bar_dur": 2.0}          # 120 bpm, 4/4, bar 1 at 1.0 s


def _midi(tmp_path, tracks):
    pm = pretty_midi.PrettyMIDI()
    for program, is_drum, notes in tracks:
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
        for s, e, p, v in notes:
            inst.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=e))
        pm.instruments.append(inst)
    path = tmp_path / "t.mid"
    pm.write(str(path))
    return path


def test_read_tracks_labels_by_program_and_drums(tmp_path):
    p = _midi(tmp_path, [(5, False, [(1.0, 1.5, 60, 90)]), (0, True, [(1.0, 1.1, 39, 100)])])
    tracks = tsc.read_tracks(p)
    assert [(t.klass, t.inst) for t in tracks] == [("electric_piano", "keys.ep"),
                                                   ("drums", "drums.kit.standard")]


def test_positions_keep_performed_timing_and_pickups():
    assert tsc.position(1.0, GRID) == "1:1.000"
    assert tsc.position(1.0 + 0.5 + 0.0625, GRID) == "1:2.125"     # a 32nd after beat 2
    assert tsc.position(3.0, GRID) == "2:1.000"
    assert tsc.position(0.4, GRID) == "@0.400"                     # before the first downbeat


def test_note_lines_pitched_and_drums():
    t = tsc.Track("piano", "keys.piano", [(1.5625, 1.9375, 60, 88)])
    assert tsc.note_lines(t, GRID) == ["1:2.125  C4  0.750b 88"]
    d = tsc.Track("drums", "drums.kit.standard", [(3.0, 3.1, 39, 96)])
    assert tsc.note_lines(d, GRID) == ["2:1.000 clap 96"]


# --- velocity rescaling -----------------------------------------------------------------------

def test_rescale_velocities_maps_p5_p95_to_40_120_rank_preserving():
    vels = list(range(25, 36))                                    # 25..35, 11 notes
    t = tsc.Track("piano", "keys.piano", [(float(i), float(i) + 0.5, 60, v)
                                          for i, v in enumerate(vels)])
    tsc.rescale_velocities([t])
    got = [v for _, _, _, v in t.notes]
    assert min(got) >= 40 - 5 and max(got) <= 120 + 5             # near the 40..120 spread
    assert got == sorted(got)                                     # order (rank) kept
    assert len(set(got)) > 1                                      # not flattened


def test_rescale_velocities_flattens_a_flat_or_tiny_stream_to_90():
    t = tsc.Track("piano", "keys.piano", [(float(i), float(i) + 0.5, 60, 80) for i in range(10)])
    tsc.rescale_velocities([t])
    assert all(v == 90 for _, _, _, v in t.notes)

    few = tsc.Track("piano", "keys.piano", [(0.0, 0.5, 60, 20), (1.0, 1.5, 60, 100)])
    tsc.rescale_velocities([few])
    assert all(v == 90 for _, _, _, v in few.notes)              # fewer than 5 notes


def test_rescale_velocities_clamps_outliers_to_1_127():
    vels = [1] + list(range(40, 50)) + [127]
    t = tsc.Track("drums", "drums.kit.standard", [(float(i), float(i) + 0.1, 39, v)
                                                  for i, v in enumerate(vels)])
    tsc.rescale_velocities([t])
    got = [v for _, _, _, v in t.notes]
    assert all(1 <= v <= 127 for v in got)
    assert got[0] < got[-1]                                       # order kept end to end


def test_rescale_velocities_spans_several_tracks_in_one_stream():
    """A drum stream is several Track objects (one per voice) joined by the
    encoder into one `:perc.drums` stream; the stats are over all of them."""
    a = tsc.Track("drums", "drums.kit.standard", [(0.0, 0.1, 36, v) for v in range(20, 40)])
    b = tsc.Track("drums", "drums.kit.standard", [(1.0, 1.1, 38, v) for v in range(60, 80)])
    tsc.rescale_velocities([a, b])
    all_v = [v for t in (a, b) for _, _, _, v in t.notes]
    assert min(all_v) >= 1 and max(all_v) <= 127
    assert all_v == sorted(all_v)                                 # a's low notes stay below b's


def test_bleed_tracks_are_dropped_with_a_reason():
    big = tsc.Track("piano", "keys.piano", [(i, i + 0.5, 60, 90) for i in range(200)])
    tiny = tsc.Track("strings", "strings.ensemble", [(0, 0.9, 60, 90), (1, 2, 62, 90)])
    rare = tsc.Track("organ", "organ.drawbar", [(i, i + 0.5, 60, 90) for i in range(3)])
    kept, why = tsc.drop_bleed([big, tiny, rare])
    assert [t.klass for t in kept] == ["piano"]
    assert any("strings" in w and "2 notes" in w for w in why)
    assert any("organ" in w for w in why)                          # 3 of 205 < 2 %


def test_empty_midi_gives_no_tracks(tmp_path):
    assert tsc.read_tracks(_midi(tmp_path, [])) == []


def test_stream_names_do_not_clash():
    taken: set[str] = set()
    assert tsc.stream_name("keys.piano", "piano", taken) == "notes.piano"
    assert tsc.stream_name("keys.piano", "guitar", taken) == "notes.piano.guitar"
    assert tsc.stream_name("keys.ep", "guitar", taken) == "notes.ep"


# --- inventory voting ------------------------------------------------------------------------

from soundcode import gm, inventory as inv  # noqa: E402


def _t(klass, onsets, pitch=60):
    return tsc.Track(klass, gm.TSUMUGI[klass]["inst"], [(o, o + 0.3, pitch, 90) for o in onsets])


def test_mix_votes_count_matching_notes_by_family():
    stem = [_t("distorted_guitar", [1.0, 2.0, 3.0, 4.0])]
    mix = [_t("electric_piano", [1.02, 2.01, 3.03]), _t("strings", [4.0])]
    assert inv.mix_votes(stem, mix) == {"keys": 3, "strings": 1}


def test_guitar_stem_that_the_mix_calls_keys_is_reassigned():
    d = inv.decide("guitar", "gtr", [("distorted_guitar", 0.97)], {"keys": 8, "gtr": 1, "strings": 1})
    assert d.family == "keys" and d.reassigned
    assert "reassigned" in d.warn and "0.80" in d.warn


def test_agreement_keeps_the_stem_family_with_combined_confidence():
    d = inv.decide("piano", "keys", [("piano", 0.97)], {"keys": 9, "gtr": 1})
    assert d.family == "keys" and not d.reassigned
    assert abs(d.conf - 0.97 * 0.9) < 1e-9


def test_too_few_votes_leaves_refinement_in_charge():
    d = inv.decide("bass", "bass", [("electric_bass", 0.88)], {"keys": 3})
    assert d.family == "bass" and not d.reassigned and d.conf == 0.88
    assert "few mix votes" in d.warn


# --- encoder stage (tsumugi faked) ---------------------------------------------------------

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from soundcode import encode as enc  # noqa: E402


def _stem(tmp_path, name, db=-20.0, secs=8):
    sr = 22050
    y = (np.random.default_rng(1).standard_normal(sr * secs) * 10 ** (db / 20)).astype(np.float32)
    p = tmp_path / f"{name}.wav"
    sf.write(str(p), y, sr)
    return p, y, sr


def _fake_tsumugi(monkeypatch, tmp_path, per_model, mix_tracks, refine_top):
    def _write(out_midi, tracks):
        out_midi.parent.mkdir(parents=True, exist_ok=True)
        _midi(tmp_path, tracks).replace(out_midi)
        return out_midi

    def transcribe(audio, out_midi, model):
        tracks = mix_tracks if Path(audio).name == "mix.wav" else per_model[model]
        return _write(out_midi, tracks)
    monkeypatch.setattr(ts, "transcribe", transcribe)
    monkeypatch.setattr(ts, "refine", lambda audio, midi, stem, js: refine_top[stem])
    monkeypatch.setattr(ts, "add_velocity", lambda midi, wav, out: midi)


def test_stage_tsumugi_reassigns_the_river_guitar_and_keeps_claps(tmp_path, monkeypatch):
    piano, y, sr = _stem(tmp_path, "piano")
    guitar, _, _ = _stem(tmp_path, "guitar", -25)
    drums, _, _ = _stem(tmp_path, "drums", -22)
    mix = np.concatenate([y])
    onsets = [1.0 + 0.5 * i for i in range(12)]
    per_model = {
        "default": [(0, False, [(o, o + 0.4, 60, 90) for o in onsets])],
        "guitar_v1_5": [(30, False, [(o, o + 0.4, 64, 80) for o in onsets])],
        "drums_v1_5": [(0, True, [(o, o + 0.1, 39, 100) for o in onsets])],
    }
    mix_tracks = [(4, False, [(o + 0.01, o + 0.4, 64, 80) for o in onsets] +
                            [(o + 0.01, o + 0.4, 60, 90) for o in onsets])]
    refine_top = {"piano": [("piano", 0.97)], "guitar": [("distorted_guitar", 0.97)]}
    _fake_tsumugi(monkeypatch, tmp_path, per_model, mix_tracks, refine_top)
    grid = {"downbeat": 1.0, "bar_dur": 2.0}
    stages, handled = enc.stage_tsumugi({"piano": piano, "guitar": guitar, "drums": drums},
                                        tmp_path / "mix.wav", mix, sr, grid, tmp_path / "work")
    text = "\n".join(line for st in stages if st.ok for line in enc._stage_lines(st))
    assert handled == {"piano", "guitar", "drums"}
    assert ":instruments" in text and "reassigned to keys" in text
    headers = [ln for ln in text.splitlines() if ln.startswith(":notes.")]
    assert any("inst=keys." in h for h in headers)
    assert not any("inst=gtr." in h for h in headers)
    assert " clap " in text


def test_stage_tsumugi_falls_back_per_stem_on_error(tmp_path, monkeypatch):
    piano, y, sr = _stem(tmp_path, "piano")
    bass, _, _ = _stem(tmp_path, "bass")
    def transcribe(audio, out_midi, model):
        if model == "bass_v2":
            raise ts.TsumugiError("amt.cli.infer: boom")
        out_midi.parent.mkdir(parents=True, exist_ok=True)
        _midi(tmp_path, [(0, False, [(1.0 + i * 0.5, 1.4 + i * 0.5, 60, 90) for i in range(8)])]).replace(out_midi)
        return out_midi
    monkeypatch.setattr(ts, "transcribe", transcribe)
    monkeypatch.setattr(ts, "refine", lambda *a: [("piano", 0.9)])
    monkeypatch.setattr(ts, "add_velocity", lambda midi, wav, out: midi)
    stages, handled = enc.stage_tsumugi({"piano": piano, "bass": bass}, tmp_path / "mix.wav",
                                        y, sr, {"downbeat": 1.0, "bar_dur": 2.0}, tmp_path / "w")
    assert handled == {"piano"}
    assert any("bass" in n and "boom" in n for n in enc.STEM_NOTES)


def test_silent_stem_is_gated_before_tsumugi(tmp_path, monkeypatch):
    piano, y, sr = _stem(tmp_path, "piano")
    other, _, _ = _stem(tmp_path, "other", -90)
    calls = []
    monkeypatch.setattr(ts, "transcribe", lambda a, o, m: calls.append(m) or (_ for _ in ()).throw(ts.TsumugiError("x")))
    enc.stage_tsumugi({"other": other}, tmp_path / "mix.wav", y, sr,
                      {"downbeat": 1.0, "bar_dur": 2.0}, tmp_path / "w")
    assert "other_v1_5" not in calls


# --- real-run findings (Task 5) -------------------------------------------------------------

def test_vote_prefers_the_stem_family_when_a_matching_note_of_it_exists():
    stem = [_t("melody", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
    mix = [_t("piano", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), _t("melody", [1.01, 2.0, 3.0, 4.0, 5.0, 6.0])]
    assert inv.mix_votes(stem, mix, prefer="voice") == {"voice": 6}


def test_vocal_and_drum_stems_are_never_reassigned():
    for stem, fam in (("lead_vocals", "voice"), ("vocals", "voice"), ("drums", "drums")):
        d = inv.decide(stem, fam, [("melody", 0.9)], {"keys": 20})
        assert d.family == fam and not d.reassigned


def test_tracks_with_the_same_instrument_merge():
    a = tsc.Track("piano", "keys.piano", [(1.0, 1.5, 60, 90)])
    b = tsc.Track("piano", "keys.piano", [(0.5, 0.9, 64, 80)])
    c = tsc.Track("electric_piano", "keys.ep", [(2.0, 2.5, 67, 70)])
    merged = tsc.merge_same_inst([a, b, c])
    assert [(t.inst, len(t.notes)) for t in merged] == [("keys.piano", 2), ("keys.ep", 1)]
    assert merged[0].notes[0][0] == 0.5                      # sorted by onset


def test_stream_names_stay_unique_on_repeated_clashes():
    taken: set[str] = set()
    names = [tsc.stream_name("keys.piano", "piano", taken) for _ in range(3)]
    assert len(set(names)) == 3


def test_a_stem_split_into_many_small_tracks_is_not_dropped_as_bleed(tmp_path, monkeypatch):
    """tsumugi can split one bass line over many tracks; bleed is judged
    after merging same-instrument tracks, not per fragment."""
    bass, y, sr = _stem(tmp_path, "bass", -20)
    frags = [(33, False, [(1.0 + i + 0.1 * k, 1.05 + i + 0.1 * k, 40, 90) for k in range(2)])
             for i in range(30)]                                   # 30 tracks x 2 notes
    _fake_tsumugi(monkeypatch, tmp_path, {"bass_v2": frags}, [],
                  {"bass": [("electric_bass", 0.9)]})
    stages, handled = enc.stage_tsumugi({"bass": bass}, tmp_path / "mix.wav", y, sr,
                                        {"downbeat": 1.0, "bar_dur": 2.0}, tmp_path / "w")
    notes = [st for st in stages if st.name.startswith("notes.") and st.ok]
    assert len(notes) == 1 and len(notes[0].lines) == 60


def test_a_long_drone_is_not_bleed():
    """corona_radiata's bass is one F1 note held ~25 s: few notes, all music."""
    drone = tsc.Track("electric_bass", "bass.electric", [(0.0, 24.7, 29, 90)])
    blip = tsc.Track("strings", "strings.ensemble", [(3.0, 3.2, 60, 40)])
    kept, why = tsc.drop_bleed([drone, blip])
    assert [t.klass for t in kept] == ["electric_bass"]
    assert any("strings" in w for w in why)


# --- final-review fixes ----------------------------------------------------------------------

def test_bleed_is_judged_after_relabel_so_own_notes_survive(tmp_path, monkeypatch):
    piano, y, sr = _stem(tmp_path, "piano")
    tracks = [(0, False, [(1.0 + 0.4 * i, 1.3 + 0.4 * i, 60, 90) for i in range(50)]),
              (33, False, [(3.0, 3.2, 40, 60), (5.0, 5.2, 41, 60)]),
              (26, False, [(7.0, 7.1, 64, 50)])]
    _fake_tsumugi(monkeypatch, tmp_path, {"default": tracks}, [], {"piano": [("piano", 0.97)]})
    stages, _ = enc.stage_tsumugi({"piano": piano}, tmp_path / "mix.wav", y, sr,
                                  {"downbeat": 1.0, "bar_dur": 2.0}, tmp_path / "w")
    notes = [st for st in stages if st.name.startswith("notes.") and st.ok]
    assert len(notes) == 1 and len(notes[0].lines) == 53
    assert not any("bleed" in w for w in notes[0].warns)


def test_a_stem_that_is_all_bleed_says_so(tmp_path, monkeypatch):
    piano, y, sr = _stem(tmp_path, "piano")
    _fake_tsumugi(monkeypatch, tmp_path, {"default": [(65, False, [(1.0, 1.2, 60, 40)])]}, [],
                  {"piano": [("piano", 0.9)]})
    stages, _ = enc.stage_tsumugi({"piano": piano}, tmp_path / "mix.wav", y, sr,
                                  {"downbeat": 1.0, "bar_dur": 2.0}, tmp_path / "w")
    omitted = [st for st in stages if st.name == "notes.piano" and not st.ok]
    assert omitted and any("bleed" in w for w in omitted[0].warns)
    assert any("none (bleed)" in ln for ln in stages[0].lines)


def test_pickup_notes_carry_seconds_durations():
    t = tsc.Track("piano", "keys.piano", [(0.4, 0.775, 60, 88)])
    assert tsc.note_lines(t, GRID) == ["@0.400  C4  0.375s 88"]


def test_never_reassigned_stems_keep_refinement_confidence_and_note_the_disagreement():
    d = inv.decide("lead_vocals", "voice", [("melody", 0.98)], {"keys": 20})
    assert d.conf == 0.98 and "disagrees" in d.warn


def test_timpani_is_pitched_not_a_drum_kit():
    t = gm.target_for("notes.timpani", "perc.timpani")
    assert not t.drums and t.preset == 47


def test_sparse_staccato_line_next_to_a_long_pad_is_kept():
    pad = tsc.Track("synth_pad", "synth.pad", [(0.0, 20.0, p, 60) for p in (60, 64, 67)])
    pluck = tsc.Track("plucked_keyboard", "keys.clav", [(i, i + 0.1, 72, 90) for i in range(10)])
    kept, _ = tsc.drop_bleed([pad, pluck])
    assert {t.klass for t in kept} == {"synth_pad", "plucked_keyboard"}
