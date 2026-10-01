# Benchmark results

Regenerated after every `soundcode bench` run (spec 2026-09-28-benchmark-scorer). No raw audio, no lyrics.

Note: stages are cached by input mtime; after changing encode/render code run with --force.

## Tier A
_last run: 2026-09-30-2101 — drums-velocity (commit 6dad240)_

| song | part | what | sound | dyn | score | change |
|---|---|---|---|---|---|---|
| river-30s | **song** (missing: backing_vocals) | | | | 81 | (+0) |
| river-30s | lead_vocals | 81 | 89 | 60 | 79 | (+0) |
| river-30s | backing_vocals | — | — | — | 0 | (+0) |
| river-30s | piano | 66 | 55 | 73 | 62 | (-2) |
| river-30s | guitar | 85 | 72 | 59 | 74 | (+1) |
| river-30s | bass | — | — | — | — | |
| river-30s | drums | 76 | 61 | 72 | 69 | (-1) |
| river-30s | other | — | — | — | — | |
| discipline-30s | **song** (missing: backing_vocals) | | | | 81 | (-1) |
| discipline-30s | lead_vocals | 81 | 62 | 66 | 70 | (-3) |
| discipline-30s | backing_vocals | — | — | — | 0 | (+0) |
| discipline-30s | piano | 18 | 32 | 41 | 26 | (-30) |
| discipline-30s | guitar | 65 | 66 | 32 | 57 | (+0) |
| discipline-30s | bass | 92 | 67 | 68 | 76 | (+3) |
| discipline-30s | drums | 80 | 56 | 91 | 71 | (-3) |
| discipline-30s | other | — | — | — | — | |
| lights_in_the_sky-30s | **song** | | | | 74 | (-5) |
| lights_in_the_sky-30s | lead_vocals | 83 | 61 | 56 | 68 | (+3) |
| lights_in_the_sky-30s | backing_vocals | — | — | — | — | |
| lights_in_the_sky-30s | piano | 81 | 44 | 64 | 61 | (-9) |
| lights_in_the_sky-30s | guitar | — | — | — | — | |
| lights_in_the_sky-30s | bass | — | — | — | — | |
| lights_in_the_sky-30s | drums | — | — | — | — | |
| lights_in_the_sky-30s | other | 40 | 41 | 67 | 44 | (-3) |
| 999999-30s | **song** | | | | 48 | (+5) |
| 999999-30s | lead_vocals | 80 | 24 | 58 | 46 | (+1) |
| 999999-30s | backing_vocals | — | — | — | — | |
| 999999-30s | piano | — | — | — | — | |
| 999999-30s | guitar | — | — | — | — | |
| 999999-30s | bass | 59 | 4 | 54 | 19 | (-4) |
| 999999-30s | drums | — | — | — | — | |
| 999999-30s | other | 40 | 43 | 8 | 30 | (-7) |
| corona_radiata-30s | **song** (missing: backing_vocals) | | | | 53 | (-3) |
| corona_radiata-30s | lead_vocals | — | — | — | — | |
| corona_radiata-30s | backing_vocals | — | — | — | 0 | (+0) |
| corona_radiata-30s | piano | — | — | — | — | |
| corona_radiata-30s | guitar | 20 | 6 | 8 | 10 | (-8) |
| corona_radiata-30s | bass | 66 | 41 | 60 | 54 | (+4) |
| corona_radiata-30s | drums | — | — | — | — | |
| corona_radiata-30s | other | 17 | 56 | 14 | 26 | (-7) |

### Improved (≥ +5)
_none_

### Regressed (≤ −5)
- discipline-30s piano: 56 → 26 (-30)
- lights_in_the_sky-30s piano: 69 → 61 (-9)
- corona_radiata-30s guitar: 18 → 10 (-8)
- 999999-30s other: 37 → 30 (-7)
- corona_radiata-30s other: 34 → 26 (-7)

## Tier C
_last run: 2026-09-29-0546 — final (commit 560557e)_

| song | part | what | sound | dyn | score | change |
|---|---|---|---|---|---|---|
| discipline-full | **song** (missing: backing_vocals) | | | | 84 | (+35) |
| discipline-full | lead_vocals | 91 | 68 | 61 | 75 | (+0) |
| discipline-full | backing_vocals | — | — | — | 0 | (+0) |
| discipline-full | piano | 64 | 69 | 64 | 66 | (-1) |
| discipline-full | guitar | 64 | 81 | 66 | 71 | (+9) |
| discipline-full | bass | 79 | 59 | 52 | 65 | (+9) |
| discipline-full | drums | 82 | 82 | 88 | 83 | (+13) |
| discipline-full | other | 39 | 49 | 43 | 44 | (-1) |
| lights_in_the_sky-full | **song** | | | | 80 | (+14) |
| lights_in_the_sky-full | lead_vocals | 76 | 71 | 73 | 74 | (+5) |
| lights_in_the_sky-full | backing_vocals | — | — | — | — | |
| lights_in_the_sky-full | piano | 82 | 65 | 78 | 74 | (+8) |
| lights_in_the_sky-full | guitar | — | — | — | — | |
| lights_in_the_sky-full | bass | — | — | — | — | |
| lights_in_the_sky-full | drums | — | — | — | — | |
| lights_in_the_sky-full | other | 70 | 63 | 73 | 67 | (+8) |
| corona_radiata-full | **song** (missing: backing_vocals) | | | | 50 | (+11) |
| corona_radiata-full | lead_vocals | — | — | — | — | |
| corona_radiata-full | backing_vocals | — | — | — | 0 | (+0) |
| corona_radiata-full | piano | — | — | — | — | |
| corona_radiata-full | guitar | 14 | 5 | 22 | 10 | (+2) |
| corona_radiata-full | bass | 57 | 3 | 27 | 15 | (-9) |
| corona_radiata-full | drums | 45 | 34 | 40 | 39 | (+12) |
| corona_radiata-full | other | 22 | 50 | 30 | 32 | (+0) |
| river-full | **song** (missing: backing_vocals) | | | | 75 | (+15) |
| river-full | lead_vocals | 74 | 88 | 31 | 67 | (+8) |
| river-full | backing_vocals | — | — | — | 0 | (+0) |
| river-full | piano | 62 | 69 | 67 | 66 | (-1) |
| river-full | guitar | 65 | 73 | 39 | 62 | (+9) |
| river-full | bass | 81 | 66 | 70 | 72 | (+6) |
| river-full | drums | 73 | 44 | 70 | 59 | (+13) |
| river-full | other | 47 | 67 | 68 | 58 | (+10) |

### Improved (≥ +5)
- discipline-full song: 48 → 84 (+35)
- river-full song: 61 → 75 (+15)
- lights_in_the_sky-full song: 66 → 80 (+14)
- discipline-full drums: 70 → 83 (+13)
- river-full drums: 47 → 59 (+13)
- corona_radiata-full drums: 27 → 39 (+12)
- corona_radiata-full song: 38 → 50 (+11)
- river-full other: 48 → 58 (+10)
- discipline-full guitar: 61 → 71 (+9)
- river-full guitar: 53 → 62 (+9)
- discipline-full bass: 56 → 65 (+9)
- lights_in_the_sky-full piano: 66 → 74 (+8)
- lights_in_the_sky-full other: 59 → 67 (+8)
- river-full lead_vocals: 59 → 67 (+8)
- river-full bass: 67 → 72 (+6)
- lights_in_the_sky-full lead_vocals: 68 → 74 (+5)

### Regressed (≤ −5)
- corona_radiata-full bass: 24 → 15 (-9)

