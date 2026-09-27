# compare results with the tsumugi backbone

Date: 2026-09-26. Encoder: tsumugi@020edc1 per stem (`default`/`guitar_v1_5`/`bass_v2`/`other_v1_5`/`drums_v1_5`/`vocal_harmony_v1_5`) plus velocity, with refinement + mix-level vote for the instrument inventory, behind the Plan 1 separation and loudness gate. Previous: `2026-09-26-compare-after-fixes.md` (`after2`, basic-pitch).

## Summary

Mean note F1 over stems present in the original (orig stem > −60 dBFS):

| Song | basic-pitch (after2) | tsumugi | |
|---|---|---|---|
| river | 0.14 | **0.25** | better |
| discipline | 0.12 | **0.26** | better |
| lights_in_the_sky | 0.18 | **0.35** | better |
| 999999 | 0.12 | 0.13 | same (ambient, sparse) |
| corona_radiata | 0.09 | 0.05 | lower on this metric: see below |

- **Levels:** within about 1 dB on most stems, and exactly matched on river and lights. The −1 to −3 dB bias is gone after the level-matching fix.
- **Standouts:**
  - lights piano: F1 0.43, onset F1 0.88;
  - discipline bass: F1 0.40, onset 0.75;
  - discipline drums: onset 0.83;
  - river "guitar" stem (really keys): F1 0.49, onset 0.88;
  - river drums: onset 0.98.
- **corona_radiata is drones.** Its bass is one F1 note held for about 25 s, which tsumugi transcribes correctly as a single note. compare's note F1 transcribes the *original* stem with basic-pitch, which chops a drone into many short notes, so a correct drone scores near 0. Chroma (0.89) and energy (0.59) show the bass matches. Note F1 is the wrong metric for sustained material; see the follow-ups.

## Acceptance (spec Evaluation item 1)

- **Note F1 improves on every song:** met on 3 of 5. 999999 ties. corona is lower on note F1 but correct by chroma and level (the drone artefact above).
- **River inventory = keys + drums with claps, with no guitar stream:** met.

  ```
  :instruments
  meta    warn="bass stem reassigned to keys by mix-level vote 0.67"
  meta    warn="guitar stem reassigned to keys by mix-level vote 0.90"
  drums    drums.kit
  bass     keys.piano ?0.42   | electric_piano 0.37
  guitar   keys.piano ?0.67   | electric_piano 0.25
  piano    keys.piano   | electric_piano 0.02
  # other: silent (below the loudness gate)
  lead_vocals voice.lead ?0.49   | choir 0.02
  ```

  `:perc.drums` has 19 `clap` hits.
- **Listening checkpoint:** river played for the user, instruments only, then the full rebuild with the DiffSinger → Seed-VC vocal (`out/preview/river-30s.full-rebuild.wav`). Corona played before and after the drone fix. Verdict: pending the user's notes.

## Bugs found by real runs and fixed

1. Relative paths broke tsumugi, which runs in its own directory.
2. The vocal stem was reassigned to keys (a sung line doubled by piano). Voice and drums are never reassigned, and the vote prefers the stem's own family.
3. Duplicate stream names. Same-instrument tracks now merge.
4. A bass line split over many tracks was dropped as bleed. Bleed is now judged after merging.
5. A single-note drone was dropped as bleed. Bleed is now judged by sounding time too.

## Follow-ups

- compare: add a chroma/pitch-class and f0-based score for sustained parts, so drones and pads are not scored by note onsets.
- Most songs had few mix-level votes (0–2 matched notes): the mix-level `default` run finds little on dense or ambient mixes. MuScriptor (licence click needed) or tsumugi's `stem-splitter` are candidate second opinions.
- Clap precision: ADT_STR as a drum cross-check.

## Raw tables

### 999999-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -46.5   -43.7   +2.8    0.14    0.14    0.88    0.34    0.69
backing_vocals    -72.8       —      —    0.00    0.00       —       —       —
drums             -69.1       —      —       —       —       —    0.00       —
bass              -30.9   -31.8   -0.9    0.11    0.16    0.91    0.29    0.56
guitar            -74.3       —      —    0.00    0.00       —       —       —
piano             -71.5       —      —    0.00    0.00       —       —       —
other             -33.6   -29.9   +3.7    0.11    0.24    0.90    0.08   -0.05
```
### corona_radiata-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -34.0   -39.9   -6.0    0.09    0.09    0.90    0.08    0.60
backing_vocals        —       —      —       —       —       —       —       —
drums             -86.1       —      —       —       —       —       —       —
bass              -21.1       —      —    0.00    0.00       —    0.00       —
guitar            -30.4   -32.9   -2.5    0.06    0.06    0.82    0.09    0.39
piano             -56.9   -54.8   +2.1    0.05    0.14    0.95    0.41    0.82
other             -24.9   -22.1   +2.8    0.08    0.11    0.89    0.17    0.11
```
### discipline-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -21.9   -21.4   +0.5    0.34    0.34    0.87    0.63    0.83
backing_vocals    -35.7       —      —    0.00    0.00       —    0.00       —
drums             -14.5   -14.2   +0.3       —       —    0.97    0.83    0.77
bass              -16.4   -16.2   +0.2    0.40    0.44    0.91    0.75    0.62
guitar            -20.5   -20.4   +0.1    0.29    0.35    0.98    0.20    0.16
piano             -31.4   -33.3   -1.9    0.25    0.28    0.91    0.33    0.85
other             -74.9       —      —    0.00    0.00       —       —       —
```
### lights_in_the_sky-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -39.4   -39.5   -0.0    0.30    0.30    0.89    0.72    0.81
backing_vocals    -71.1       —      —    0.00    0.00       —       —       —
drums             -87.6       —      —       —       —       —       —       —
bass              -85.0       —      —    0.00    0.00       —       —       —
guitar            -74.0       —      —    0.00    0.00       —       —       —
piano             -25.7   -25.8   -0.1    0.43    0.49    0.91    0.88    0.81
other             -33.8   -33.5   +0.3    0.33    0.35    0.89    0.76    0.83
```
### river-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -31.5   -31.5   +0.0    0.38    0.41    0.80    0.58    0.52
backing_vocals    -43.6       —      —    0.00    0.00       —    0.00       —
drums             -32.0   -31.8   +0.2       —       —    0.95    0.98    0.77
bass              -58.9   -57.8   +1.2    0.14    0.17    0.90    0.15    0.24
guitar            -41.4   -41.2   +0.2    0.49    0.52    0.94    0.88    0.59
piano             -36.3   -36.1   +0.1    0.25    0.35    0.92    0.70    0.80
other             -83.6       —      —    0.00    0.00       —       —       —
```

### corona_radiata-30s (after the drone fix)

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -34.0   -39.9   -6.0    0.09    0.09    0.90    0.08    0.60
backing_vocals        —       —      —       —       —       —       —       —
drums             -86.1       —      —       —       —       —       —       —
bass              -21.1   -21.4   -0.3    0.04    0.13    0.89    0.03    0.59
guitar            -30.4   -34.5   -4.1    0.06    0.06    0.80    0.05    0.46
piano             -56.9   -56.2   +0.7    0.00    0.00    0.97    0.06    0.52
other             -24.9   -21.7   +3.1    0.03    0.08    0.89    0.17    0.08
```
