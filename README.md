# soundcode

Transcribe a song into readable code, then regenerate it.

soundcode analyses a recording and writes a `.sc` file — plain text describing
what the song *is*: grid, structure, harmony, notes, lyrics, mix character. From
that file it renders new audio back.

The bar is **a great cover version**, not a copy: same song, same words, same
groove, different performance. `.sc` is not a codec — the point is that it's
editable. Change a chord, rewrite a line, re-render.

Everything runs locally (M1 Max, 32 GB, no CUDA).

```
audio ──encode──> song.sc ──render──> mock.wav ──remix──> cover.wav
        (MIR stack)        (crude, but      (ACE-Step 1.5)
                            correctly timed
                            and pitched)
```

Encode stages fail soft and mark their own confidence — a stream with no
evidence is omitted and a warning written into the file, never guessed at. The
crude mock render is what carries note-level detail into the generator; that's
the fidelity path, and proving it works is the open question.

## Status

`check`, `render`, `separate`, `encode` and `serve` work. `decode` isn't built
yet — real `.sc` files have been encoded from real recordings, but no cover has
been generated.

**Milestone 1 (clean separation) is done.** `soundcode separate` splits a song
into lead vocals, backing vocals, drums, bass, guitar, piano and other (three
passes: BS-RoFormer → karaoke Mel-RoFormer → HTDemucs 6-stem, on MPS), plus
`vocals.wav`, `instrumental.wav` and a `residual.wav` so the stems always rebuild
the original. Results on the M1 Max:

| Input | Length | Level diff | Residual | Time |
|---|---|---|---|---|
| 999999 | 30 s | +0.09 dB | −19.2 dB | 112 s |
| corona_radiata | 30 s | −0.49 dB | −19.1 dB | 99 s |
| discipline | 30 s | −0.14 dB | −23.0 dB | 91 s |
| lights_in_the_sky | 30 s | −0.7 dB | −14.0 to −15.5 dB (borderline) | 88 s |
| The River (full song) | 196 s | −0.10 dB | −22.0 dB | 573 s |

Pass = level within ±1 dB and residual ≤ −15 dB. Times include model loading
(about 2.9× real time on a full song). Results vary by about 1 dB between runs
(Demucs uses random time shifts), so lights_in_the_sky sits on the line and
fails about half the time; its losses are in the Demucs pass.

The karaoke model sometimes files the whole vocal as backing (corona_radiata:
lead stem ≈ silent). `encode` detects this and transcribes lead + backing
together, with a `# NOTE:` in the `.sc` header. Proper lead/backing splitting is
Milestone 2.

**Sampled rendering and comparison.** `soundcode render song.sc` plays a `.sc`
through a General MIDI SoundFont (GeneralUser GS, downloaded on first use), leaving
vocal streams out unless `--with-vocals`. `soundcode compare original.wav song.sc`
scores each rendered part against its source stem (level, note F1, chroma, onsets,
energy) and writes an HTML report with players and spectrograms under
`out/compare/`, served by `soundcode serve` at `/compare/`. Before/after numbers:
`docs/results/2026-09-26-compare-baseline.md`,
`docs/results/2026-09-26-compare-after-fixes.md`.

**Transcription backbone: tsumugi.** `encode` transcribes each stem with
[tsumugi](https://github.com/anime-song/tsumugi) (MIT, pinned in `external/tsumugi`,
installed by `scripts/install_tsumugi.sh`). It writes one note stream per identified
instrument, drums with claps, and an `:instruments` inventory whose confidence
comes from tsumugi's refinement model agreeing with a mix-level vote. Note F1
roughly doubled on three of five test songs:
`docs/results/2026-09-26-compare-tsumugi.md`. Singing spike (DiffSinger → Seed-VC,
voice similarity 0.90 to the original singer): `docs/research/2026-09-26-singing-spike.md`.

**Singing.** `render --with-vocals` sings the lead vocal from the `.sc` (notes,
lyrics and the `:contour.vox` pitch curve) with DiffSinger, then converts it to the
original singer's voice with Seed-VC. Set `SOUNDCODE_SEEDVC_HOST=framepick` to run
Seed-VC on a CUDA box (`scripts/install_seedvc_remote.sh`). River: 19 cents pitch
error, 0.93 voice similarity (`docs/results/2026-09-27-singing-thin-slice.md`).

Roadmap: `docs/superpowers/plans/2026-09-26-infinity-engine-roadmap.md`.

Design and format spec: `docs/superpowers/specs/2026-07-31-soundcode-design.md`.

## Run

Python 3.11 (hard pin — basic-pitch has no 3.12 wheel).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .            # core: check, render, serve
pip install -e '.[encode]'  # + analysis stack (large, CPU-bound on macOS)

soundcode check  examples/signal-lost.v3.sc
soundcode render examples/signal-lost.v3.sc -o out/mock.wav
soundcode separate audio/test/song.mp3            # stems -> out/stems/song/
soundcode render  out/sc/song.sc                  # sampled instruments -> song.render.wav
soundcode compare audio/test/song.mp3 out/sc/song.sc   # per-stem scores + report
soundcode encode audio/test/song.wav -o out/sc/song.sc --keep-work
soundcode serve                                  # A/B listening, :8720
```

ACE-Step 1.5 is ~8 GB and not vendored here. Clone it to `external/ACE-Step-1.5`
with its own venv, then `scripts/cover_test.py` sweeps remix strength.

## Plan forward

1. **Prove the remix path** — sweep strength 0.4–0.9 from our mocks and listen.
   Go/no-go for the whole fidelity thesis; everything below assumes it passes.
2. **`soundcode decode`** — promote `cover_test.py` into a real command.
3. **Alignment loop** — re-encode the output, repaint sections that drifted.
4. **`diff` + round-trip eval** — `encode → decode → re-encode → compare` scores
   quality without a human in the loop, which is what makes iteration cheap.
5. **Full-length songs** — 4 minutes, anchors against drift, compaction.

## Licensing

Corpus audio is never committed. MAESTRO and NIN material is CC BY-NC-SA and
anything derived from it inherits that. No third-party lyrics in specs or
examples.
