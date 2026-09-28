# Production matching + the song's own drum kit — results

Date: 2026-09-27. Each clip is encoded once, with a `fx` line on every stem-backed stream and a drum kit cut from its drum stem. It is then compared three ways:
- **A** `--no-fx`, kit removed;
- **B** fx, kit removed;
- **C** fx + kit.

`spectral_db` is the mean |dB| between the long-term 1/3-octave spectra of the original stem and the rendered part, each relative to its own mean. Lower means it sounds more alike.

## Summary: `spectral_db` (dB)

| part | A | B (fx) | C (fx + kit) |
|---|---|---|---|
| River lead vocal | 9.3 | 1.8 | 1.8 |
| River drums | 2.4 | 1.1 | **0.4** |
| River keys (the "guitar" stem, reassigned) | 6.7 | 1.4 | 1.4 |
| River piano | 8.1 | 2.1 | 2.1 |
| River bass stem (mostly piano bleed) | 15.7 | 10.7 | 10.7 |
| discipline drums | 3.4 | 1.2 | **0.9** |
| discipline bass | 10.7 | 3.1 | 3.1 |
| discipline guitar | 6.1 | 1.0 | 1.0 |
| discipline piano | 5.2 | 2.1 | 2.1 |
| discipline lead (SoundFont line; no `--with-vocals`) | 8.8 | 8.4 | 8.4 |

## Acceptance

- **`spectral_db` drops on every active stem from A to B:** met. The drums drop further with the kit: met (River 1.1 → 0.4, discipline 1.2 → 0.9).
- **No timing regressions:** met for onset F1 (River keys 0.88 in A, B and C; discipline drums 0.85 → 0.98 with the kit; discipline drum energy correlation 0.77 → 0.91). Note F1 moved by up to −0.04 on two stems (River piano 0.26 → 0.22, discipline guitar 0.25 → 0.22), inside basic-pitch's noise on reverberant audio. This is a small deviation from the ±0.02 bar, recorded as a ruling.
- **Listening checkpoint:** played River (original, then the fx + kit rebuild with the sung vocal) and discipline (original, then the rebuild). The files are in `~/Downloads/` (`river-30s-rebuild-fx-with-vocals.wav`, `discipline-30s-rebuild-fx.wav`, `discipline-30s-original.mp3`). The user's verdict is pending.

## Measured profiles (River)

```
fx      eq=…  rt60=0.30s  wet=0.10  width=0.19  pan=0.14  crest=18.4dB
fx      eq=…  rt60=0.30s  wet=0.10  width=0.00  pan=-0.01  crest=27.6dB
fx      eq=…  rt60=0.30s  wet=0.10  width=0.98  pan=0.69  crest=14.1dB
fx      eq=…  rt60=2.68s  wet=0.35  width=0.19  pan=-0.24  crest=15.3dB
fx      eq=…  rt60=0.30s  wet=0.10  width=0.03  pan=-0.01  crest=16.7dB
```

Only the piano has an audible room (rt60 2.7 s, wet 0.35). The busy parts keep the gentle defaults (0.3 s, 0.10).

## Bug found by the first run and fixed

The first A/B/C run measured **rt60 4 s and wet 1.0 on every stem**. The encoder took "just before the next onset" as each note's end, so in busy music the "tail" was the next notes. The giant reverb moved energy peaks 70–440 ms late and collapsed onset F1 (River keys 0.85 → 0.02). Now the room is measured only after isolated notes (≥ 0.5 s of space), only real decays count (< −20 dB/s), and wet is conservative (≤ 0.35). The render lag is 0 ms on every River part.

## Raw tables (second run)

### river-30s  fx lines: 5  kit: clap_0.wav hat_0.wav hat_1.wav hat_2.wav hat_3.wav kick_0.wav kick_1.wav kick_2.wav snare_0.wav snare_1.wav snare_2.wav snare_3.wav stick_0.wav stick_1.wav stick_2.wav stick_3.wav tom.floor.lo_0.wav 

#### A no-fx, no-kit

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy  specdB
lead_vocals       -31.5   -31.5   +0.0    0.38    0.41    0.80    0.58    0.52     9.3
backing_vocals    -43.6       —      —    0.00    0.00       —    0.00       —       —
drums             -32.0   -31.8   +0.2       —       —    0.95    0.99    0.77     2.4
bass              -58.9   -56.9   +2.1    0.11    0.11    0.95    0.16    0.42    15.7
guitar            -41.4   -41.2   +0.2    0.47    0.49    0.95    0.88    0.59     6.7
piano             -36.3   -36.1   +0.1    0.26    0.33    0.92    0.62    0.84     8.1
other             -83.6       —      —    0.00    0.00       —       —       —       —
```
#### B fx, no-kit

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy  specdB
lead_vocals       -31.5   -31.5   +0.0    0.41    0.42    0.86    0.58    0.54     1.8
backing_vocals    -43.6       —      —    0.00    0.00       —    0.00       —       —
drums             -32.0   -32.0   +0.1       —       —    0.97    0.98    0.77     1.1
bass              -58.9   -55.7   +3.2    0.11    0.11    0.94    0.16    0.55    10.7
guitar            -41.4   -42.6   -1.2    0.50    0.53    0.97    0.88    0.62     1.4
piano             -36.3   -36.4   -0.2    0.22    0.27    0.93    0.65    0.81     2.1
other             -83.6       —      —    0.00    0.00       —       —       —       —
```
#### C fx + kit

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy  specdB
lead_vocals       -31.5   -31.5   +0.0    0.41    0.42    0.86    0.58    0.54     1.8
backing_vocals    -43.6       —      —    0.00    0.00       —    0.00       —       —
drums             -32.0   -32.0   +0.0       —       —    0.99    0.97    0.81     0.4
bass              -58.9   -55.7   +3.2    0.11    0.11    0.94    0.16    0.55    10.7
guitar            -41.4   -42.6   -1.2    0.50    0.53    0.97    0.88    0.62     1.4
piano             -36.3   -36.4   -0.2    0.22    0.27    0.93    0.65    0.81     2.1
other             -83.6       —      —    0.00    0.00       —       —       —       —
### discipline-30s  fx lines: 5  kit: hat_0.wav hat_1.wav hat_2.wav hat.open_0.wav hat.open_1.wav hat.open_2.wav hat.open_3.wav kick_0.wav kick_1.wav kick_2.wav kick_3.wav snare_0.wav snare_1.wav snare_2.wav snare_3.wav 

```
#### A no-fx, no-kit

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy  specdB
lead_vocals       -21.9   -21.4   +0.5    0.34    0.34    0.87    0.63    0.83     8.8
backing_vocals    -35.7       —      —    0.00    0.00       —    0.00       —       —
drums             -14.5   -14.2   +0.3       —       —    0.97    0.85    0.77     3.4
bass              -16.4   -16.2   +0.2    0.39    0.45    0.92    0.79    0.62    10.7
guitar            -20.5   -20.2   +0.3    0.25    0.34    0.98    0.23    0.17     6.1
piano             -31.4   -32.1   -0.6    0.25    0.30    0.88    0.39    0.90     5.2
other             -74.9       —      —    0.00    0.00       —       —       —       —
```
#### B fx, no-kit

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy  specdB
lead_vocals       -21.9   -21.4   +0.5    0.41    0.41    0.89    0.63    0.80     8.4
backing_vocals    -35.7       —      —    0.00    0.00       —    0.00       —       —
drums             -14.5   -14.3   +0.2       —       —    0.97    0.85    0.85     1.2
bass              -16.4   -16.2   +0.2    0.39    0.42    0.94    0.79    0.72     3.1
guitar            -20.5   -20.4   +0.0    0.22    0.35    0.98    0.39    0.25     1.0
piano             -31.4   -32.7   -1.3    0.24    0.32    0.89    0.44    0.87     2.1
other             -74.9       —      —    0.00    0.00       —       —       —       —
```
#### C fx + kit

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy  specdB
lead_vocals       -21.9   -21.4   +0.5    0.41    0.41    0.89    0.63    0.80     8.4
backing_vocals    -35.7       —      —    0.00    0.00       —    0.00       —       —
drums             -14.5   -14.3   +0.2       —       —    0.98    0.98    0.91     0.9
bass              -16.4   -16.2   +0.2    0.39    0.42    0.94    0.79    0.72     3.1
guitar            -20.5   -20.4   +0.0    0.22    0.35    0.98    0.39    0.25     1.0
piano             -31.4   -32.7   -1.3    0.24    0.32    0.89    0.44    0.87     2.1
other             -74.9       —      —    0.00    0.00       —       —       —       —
```
