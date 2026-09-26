# compare baseline — before encoder fixes

Date: 2026-09-26. Encoder at commit f2aaf32 (Milestone 1 separation, no loudness gate, +33 c bias). Renderer: sf2 (GeneralUser GS), instruments by stream name. Clips: 30 s. Command: `soundcode compare <clip> out/sc/baseline/<clip>.sc`.

Columns: level of original stem / rendered stream (dBFS), difference, note F1 (onset ±50 ms, pitch ±50 c), octave-agnostic F1, mean chroma cosine per bar, onset F1 (±70 ms), RMS-envelope correlation. — = nothing to measure (a silent side).

## 999999-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -46.5   -33.5  +13.0    0.09    0.09    0.93    0.12    0.61
backing_vocals    -72.8       —      —    0.00    0.00       —       —       —
drums             -69.1   -22.0  +47.1       —       —    0.93    0.74    0.23
bass              -30.9   -23.7   +7.2    0.05    0.09    0.93    0.29    0.41
guitar            -74.3   -27.2  +47.0    0.18    0.18       —    0.29       —
piano             -71.5   -31.9  +39.6    0.13    0.17    0.79    0.26       —
other             -33.6   -28.0   +5.7    0.37    0.39    0.91    0.57    0.15
```
## corona_radiata-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals      -110.8   -27.9  +82.9    0.00    0.00       —    0.06       —
backing_vocals    -34.0       —      —    0.00    0.00       —    0.00       —
drums             -86.1   -22.5  +63.6       —       —       —    0.64       —
bass              -21.1   -24.0   -2.9    0.05    0.07    0.93    0.31    0.21
guitar            -30.4   -27.3   +3.0    0.17    0.20    0.90    0.29    0.23
piano             -56.9   -30.2  +26.7    0.08    0.10    0.95    0.15   -0.15
other             -24.9   -31.6   -6.8    0.06    0.10    0.87    0.34   -0.16
```
## discipline-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -21.9   -30.8   -8.9    0.08    0.08    0.81    0.30   -0.09
backing_vocals    -35.7       —      —    0.00    0.00       —    0.00       —
drums             -14.5   -32.6  -18.1       —       —    0.97    0.77    0.11
bass              -16.4   -18.5   -2.1    0.09    0.14    0.87    0.49    0.39
guitar            -20.5   -23.7   -3.2    0.04    0.06    0.91    0.49   -0.16
piano             -31.4   -35.5   -4.1    0.02    0.02    0.86    0.26    0.21
other             -74.9   -34.1  +40.8    0.02    0.03    0.87    0.34       —
```
## lights_in_the_sky-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -39.4   -32.9   +6.6    0.03    0.03    0.89    0.33    0.47
backing_vocals    -71.1       —      —    0.00    0.00       —       —       —
drums             -87.6   -37.2  +50.4       —       —       —    0.30       —
bass              -85.0   -22.5  +62.4    0.05    0.07       —    0.28       —
guitar            -74.0   -29.5  +44.5    0.00    0.05    0.95    0.12       —
piano             -25.7   -30.1   -4.4    0.15    0.15    0.90    0.28    0.31
other             -33.8   -30.9   +2.9    0.08    0.08    0.89    0.31    0.40
```
## river-30s

```
stem            orig dB rend dB   Δ dB  noteF1  anyOct  chroma onsetF1  energy
lead_vocals       -31.5   -28.2   +3.3    0.05    0.07    0.79    0.42    0.21
backing_vocals    -43.6       —      —    0.00    0.00       —    0.00       —
drums             -32.0   -30.6   +1.4       —       —    0.96    0.79    0.31
bass              -58.9   -21.2  +37.7    0.03    0.04    0.66    0.32   -0.15
guitar            -41.4   -22.0  +19.4    0.05    0.07    0.80    0.36    0.36
piano             -36.3   -29.6   +6.7    0.00    0.01    0.56    0.07   -0.07
other             -83.6   -32.7  +51.0    0.01    0.01       —    0.19       —
```

## Worst stem per song

| Song | Worst | Why |
|---|---|---|
| 999999 | drums +47 dB, guitar +47 dB | stems near-silent in the original (−69 / −74 dB); bleed transcribed into full parts |
| corona_radiata | lead_vocals +83 dB | pairing artefact: encoder used lead+backing (silent lead stem) but compare paired with the lead stem; drums +64 dB is real bleed |
| discipline | other +41 dB | silent stem transcribed; also drums −18 dB (real kit rendered too quiet: family gain, no level matching) |
| lights_in_the_sky | bass +62 dB, drums +50 dB | silent stems transcribed |
| river | other +51 dB, bass +38 dB | silent / late-entering stems transcribed |

## What the River images show

- `bars.png`: drums match every bar (chroma ≈ 0.95+); piano diverges in bars 4–8 (≈ 0.3–0.5) where the render smears extra overlapping notes; bass is only comparable from bar ~10, when the original bass enters — before that the original is silent and the render is not.
- Across songs, where a part really exists harmony mostly matches (chroma 0.8–0.97) and drum timing is good (onset F1 0.74–0.79); note F1 is low (0.00–0.37) because invented and extra notes swamp the real ones.

## Findings added to Task 9

1. Gate percussion as well as notes (invented drums on near-silent drum stems).
2. When the encoder falls back to lead+backing, write `meta stem=vocals`, and have compare pair it with lead+backing summed.
