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

`check`, `render`, `encode` and `serve` work. `decode` isn't built yet — real
`.sc` files have been encoded from real recordings, but no cover has been
generated.

Design and format spec: `docs/superpowers/specs/2026-07-31-soundcode-design.md`.

## Run

Python 3.11 (hard pin — basic-pitch has no 3.12 wheel).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .            # core: check, render, serve
pip install -e '.[encode]'  # + analysis stack (large, CPU-bound on macOS)

soundcode check  examples/signal-lost.v3.sc
soundcode render examples/signal-lost.v3.sc -o out/mock.wav
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
