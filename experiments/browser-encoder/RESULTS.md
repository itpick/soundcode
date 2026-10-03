# Browser encoder spike: results

**Question:** how close can an encoder that runs entirely in the browser, built
only from open JS/WASM tools and with no server, get to the native encoder?

**Short answer:** it works end to end, and it is fast once the models are
cached. All four clips produce `.sc` files that the repo parser accepts. The
benchmark scorer rates them at roughly half of what the native encoder gets:
an average song score of 32 against 64 for native. The gap comes
almost entirely from having no source separation. Everything that depends on a
clean stem (instrument parts, the vocal line, drum timbre) suffers. Things
measured on the whole mix hold up: tempo is within 2 BPM of native on three
of the four clips, and on Discipline the mix part scores 72 against 77 for a
native file scored the same way.

Run date 2026-09-30, on an Apple-silicon Mac (10 cores) with Chromium 148.

## Per-clip timings

These are warm runs: all four clips encoded in one page session, in the order
Discipline, Lights, 999999, Corona. So Discipline's row also includes the
one-time model loading and WebGL shader compilation. Times are wall-clock
seconds for each stage, from `performance.now()` in the page.

**Headless shell, no GPU** (`chrome-headless-shell` 1223). WebGPU reports no
adapter and there is no WebGL, so Whisper runs on WASM (one thread, because the
page is not cross-origin isolated) and basic-pitch on the TF.js CPU backend.

| clip | decode+resample | centre extract | essentia+JS DSP | melody | whisper load | whisper ASR | LRCLIB | basic-pitch load | basic-pitch | total |
|---|---|---|---|---|---|---|---|---|---|---|
| discipline-30s | 0.09 | 0.58 | 4.09 | 4.09 | 2.73 | 3.66 | 0.27 | 0.25 | 28.91 | **44.7** |
| lights_in_the_sky-30s | 0.06 | 0.47 | 3.43 | 2.89 | 0.00 | 4.09 | 0.17 | 0.00 | 29.85 | **41.0** |
| 999999-30s | 0.10 | 0.50 | 3.42 | 2.91 | 0.00 | 3.15 | — | 0.00 | 30.69 | **40.8** |
| corona_radiata-30s | 0.08 | 0.65 | 3.76 | 2.61 | 0.01 | 3.86 | — | 0.00 | 28.29 | **39.3** |

**Full Chromium, `--headless=new --enable-unsafe-webgpu`.** WebGPU gets the
`apple/metal-3` adapter, so Whisper runs on WebGPU and basic-pitch on WebGL.

| clip | decode+resample | centre extract | essentia+JS DSP | melody | whisper load | whisper ASR | LRCLIB | basic-pitch load | basic-pitch | total |
|---|---|---|---|---|---|---|---|---|---|---|
| discipline-30s | 0.07 | 0.48 | 3.53 | 3.73 | 1.66 | 2.63 | 0.22 | 0.08 | 15.56 | **28.0** |
| lights_in_the_sky-30s | 0.07 | 0.51 | 3.34 | 2.86 | 0.00 | 1.18 | 0.18 | 0.00 | 1.11 | **9.2** |
| 999999-30s | 0.06 | 0.47 | 3.28 | 2.69 | 0.00 | 0.44 | — | 0.00 | 1.10 | **8.0** |
| corona_radiata-30s | 0.06 | 0.55 | 3.40 | 2.61 | 0.00 | 0.38 | — | 0.00 | 0.72 | **7.7** |

Both backends produce byte-identical `.sc` output, apart from the comment line
that names the backends.

- **basic-pitch on the CPU is the bottleneck**, at about 29 s per 30 s clip.
  On WebGL it takes about 1 s. The first WebGL call takes 15 s because the
  shaders compile. `@tensorflow/tfjs-backend-wasm@3.19.0` (the TF.js version
  that basic-pitch pins) fails inside basic-pitch's graph with "Unknown dtype
  undefined", so a machine without a GPU falls back to the pure-JS CPU backend.
- Whisper-base takes 3–4 s on WASM with one thread and 0.4–1.2 s on WebGPU
  (2.6 s on the first run, for warm-up). Cross-origin isolation, which would
  enable WASM threads, is wired up (`tools/serve.py --isolate`) but was not
  needed: Whisper is not the bottleneck.
- Run-to-run variation on the CPU path was about 10%. An earlier identical run
  took 36–40 s per clip.
- The page's main thread is blocked while basic-pitch runs. A Web Worker would
  fix this; see the recommendation.

## Download sizes

| What | Size | Source |
|---|---|---|
| The page itself (HTML and JS) | ~60 KB | same origin |
| essentia.js 0.1.3 (core and WASM) | 0.75 MB transferred | jsDelivr |
| transformers.js 3.7.6 and the onnxruntime-web WASM | 4.35 MB transferred (21.6 MB uncompressed WASM) | jsDelivr |
| TF.js 3.19.0 (core, converter, layers, CPU and WebGL backends) | 0.43 MB transferred | jsDelivr |
| basic-pitch 1.0.1 (code and model) | 0.11 MB transferred (model 0.9 MB raw) | jsDelivr |
| **Libraries total** | **5.65 MB** | |
| Whisper base, WASM config: fp32 encoder and int8 merged decoder | 82.5 + 53.7 MB, plus 2.8 MB tokenizer = **~139 MB** | `onnx-community/whisper-base_timestamped` on huggingface.co |
| Whisper base, WebGPU config as coded: fp32/fp32 | 82.5 + 208.7 MB = **~294 MB** (q4f16 would cut this to ~110 MB, untested) | same |
| Whisper base, local export actually used in these runs | 82.5 + 314.8 MB (fp32) | `tools/export_whisper.py` |

The browser caches the models (transformers.js uses Cache Storage) after the
first visit. The timing tables above do not include any Hugging Face download,
because of the next point.

## What worked, and what didn't

The F1 figures below treat native's stem-based output as the reference. It is
not ground truth, so read them as agreement with native, not accuracy.

**The Hugging Face model CDN was unreachable from the spike machine.**
`huggingface.co` itself answered, but the large-file redirect targets
(`us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`, `cdn-lfs*.hf.co`) failed with
DNS or connect errors. This happened from the headless browser and from `curl`,
inside and outside the sandbox. So transformers.js could not fetch
`onnx-community/whisper-base`. As a workaround, `tools/export_whisper.py`
re-exported the cached open `openai/whisper-base` checkpoint to ONNX in the
transformers.js layout, with cross-attention outputs for word timestamps. The
page loads it from its own origin (`models/`, gitignored) when the model is
set to `local:whisper-base_timestamped`. The Hugging Face code path is the
page's default, but it is **untested end to end**. It fails cleanly here: the
stage is marked failed and the `.sc` is still written without lyrics. Export
gotcha: transformers ≥ 4.4x adds a `cache_position` decoder input that
transformers.js 3.7 does not feed. Exporting with transformers 4.40.2 and
optimum 1.19.2 works.

| Component | Status | Notes |
|---|---|---|
| Decode and resample (Web Audio) | ✅ | MP3 decodes in headless Chromium; `OfflineAudioContext` does the resampling. |
| Tempo and beats (essentia.js `RhythmExtractor2013`) | ✅ | Within 1–2 BPM of native on three clips. Corona came out at 99.9 vs native 120.2 (librosa); for a slow ambient piece the "true" tempo is debatable. |
| Key (essentia.js `KeyExtractor`) | ✅ | Written as `tonal_center` in `:harmony`. Native writes no key. |
| Chords (HPCP and triad templates) | ✅ | Same method as native (chroma templates per bar). |
| Sections (`:struct`) | ✅ (weak) | Novelty in bar-level chroma and loudness. The vocal flag comes from the ASR. |
| Drums from the mix | ⚠️ partly | Kick, snare and hat from band-wise spectral flux. An HPSS "percussive share" gate removes them correctly on the three drumless clips (native also has no drums on those). On Discipline, onset F1 against native's stem-based drums is kick 0.53, snare 0.50, hat 0.80. Tuned on that one clip, so treat it as a starting point. |
| Notes (basic-pitch on the mix) | ✅ runs / ⚠️ quality | `:notes.bass` (below MIDI 48) and `:notes.keys` (`keys.piano`). Pitch bends become cent offsets. Against native, onset+pitch F1 is 0.14 on Discipline, 0.32 on Lights, 0.05 on 999999, 0.00 on Corona. Native transcribes each separated stem; basic-pitch here has to deal with the whole mix. |
| Lyrics (Whisper-base, transformers.js) | ⚠️ partly | Word timestamps work through cross-attention. On the raw mix Whisper heard only "[Music]" on Discipline. Running it on a **centre-channel extraction** (a JS STFT mask that keeps bins where L≈R, 120 Hz–7 kHz) gave 37 words. Discipline and Lights get lyrics, and with LRCLIB 41 and 20 words. Native also has no lyrics on 999999 and Corona, where Whisper here outputs only "(dramatic music)", which is filtered out. Whisper-small (local export, 1.1 GB fp32) did *worse* on Discipline ("[Music]" again), so base was kept. transformers.js returns no per-word probabilities, so confidence is set at stream level. |
| LRCLIB | ✅ | CORS is open (`access-control-allow-origin: *`), and a lookup takes about 0.2 s. Reconciliation is ported from `lyrics.py`. The match threshold is lowered from 0.5 to 0.35 for whisper-base. Discipline's true window scores 0.38, but other windows of that repetitive song score 0.32–0.35, so the **offset** is a weak call (it matched native's 101.49 here). On Lights it picked 42.66 vs native 36.53, but the words reconcile either way. |
| Vocal melody and contour | ⚠️ partly | essentia.js `PredominantPitchMelodia` on the centre channel, kept only where the ASR heard words, then `PitchContourSegmentation`. This gives `:notes.lead` (`voice.lead`) and `:contour.vox` at 50 Hz, in the same format as native. Onset+pitch F1 against native's lead is 0.20 on Discipline and 0.06 on Lights, where the melody tracker follows the piano. |
| CREPE | ❌ skipped | The available TF.js CREPE builds are hosted outside npm, for example ml5's models on GitHub or essentia's on essentia.upf.edu. That breaks the rule of using only jsDelivr/unpkg npm packages, and without a vocal stem CREPE has nothing clean to track. Melodia covers the lite case. |
| Mix scalars (`:mix`) | ✅ | Same definitions as native: `lufs_int` is the RMS dBFS of the mono mix, and the other scalars match too. True EBU R128 from essentia.js is added as a comment. |
| WebGPU | ✅ in full Chromium / ❌ in headless shell | The headless shell reports "No available adapters". With `--enable-unsafe-swiftshader` it gets a software SwiftShader adapter, which is not useful. Full Chromium in `--headless=new` gets Metal. |
| tfjs WASM backend | ❌ | See above. The CPU fallback works but is slow. |

## How close it gets (benchmark scorer)

These are `soundcode score <clip>.mp3 out/browser-spike/<clip>.sc` results,
using the repo's renderer and scorer and the native stems in `out/stems/`. The
native column is the tier-A song score from `docs/results/benchmark/README.md`,
from the 2026-09-29 run.

| clip | browser song | native song (bench) | browser parts (part: score) |
|---|---|---|---|
| discipline-30s | **45** | 81 | lead_vocals 53, bass 28, piano 25, drums 11, mix 72; guitar and backing_vocals not rebuilt |
| lights_in_the_sky-30s | **37** | 78 | piano 13, lead_vocals 16, mix 64; other not rebuilt |
| 999999-30s | **19** | 43 | mix 38; lead_vocals, bass and other not rebuilt |
| corona_radiata-30s | **28** | 55 | bass 35, mix 34; guitar, other and backing_vocals not rebuilt |
| **mean** | **32** | **64** | |

For a like-for-like check, `site/media/discipline-30s.sc` (an older native
file) was scored with the same command. It got a song score of 66 and a mix
part of 77, against 45 and 72 for the browser file.

The scorer files parts by `meta stem`, or by instrument family when that is
absent. Without separation, the browser has to guess which part a stream
belongs to. Tagging the unseparated upper notes `stem=other` instead of
leaving them as `keys.piano` was tried, and it moved the song scores by −2 to
+1, which is a wash.

## What's missing compared with native

- **Source separation.** This is the root cause of most of the gap. Native
  splits the song into six stems with BS-RoFormer, MelBand-RoFormer karaoke
  and htdemucs_6s, and transcribes each stem. The browser transcribes the
  whole mix, so every pitched instrument and the voice land in one
  `keys.piano` stream plus a bass split. Lead and backing vocals cannot be
  told apart. The parts the scorer cannot find are reported as "not rebuilt".
- **tsumugi instrument labels.** There is no instrument classifier, so `inst`
  is fixed to `keys.piano` or `bass.electric` and there is no
  `:instruments` stream.
- **Contour from CREPE on the vocal stem.** It is replaced by Melodia on the
  centre channel, which is noisier and sometimes follows another instrument.
- **Per-stream `fx` lines and `meta level`.** Native measures EQ, reverb,
  width, pan and crest on each stem. Without stems there is nothing to measure
  them on.
- **The drum kit.** Native ships sampled one-shots per song (`meta
  kit=<clip>.kit`) from the drum stem. The browser's drums render with the
  General MIDI kit, which is why Discipline's drum "sound" axis is 0.
- **ASR quality.** The browser runs whisper-base on a centre-channel
  approximation; native runs large-v3-turbo on the separated vocal. Lyrics
  come through only where LRCLIB can correct them.

## Recommendation for the "full" version: Demucs in the page

Separation is the step that matters most. Here is what adding it would take.
These are estimates, not measurements.

1. **Model.** htdemucs (4 stems, about 80 MB fp32 and about 40 MB fp16)
   rather than the native RoFormer stack. RoFormers are larger (BS-RoFormer is
   about 250 MB or more) and depend heavily on attention. Two routes:
   - **ONNX through onnxruntime-web.** Export htdemucs with the STFT/iSTFT
     moved outside the graph (do them in JS or WASM; `js/stft.js` already has
     the FFT), because the ONNX STFT ops are poorly supported on WebGPU.
     Community exports exist, but check their licence and provenance before
     using one.
   - **A dedicated WASM build.** For example, the approach of sevagh's
     *free-music-demixer*: Demucs v4 in C++ compiled with Emscripten, which
     already runs htdemucs in the browser. It is proven, but CPU-only, and
     takes minutes per song.
2. **Runtime budget.** Expect roughly 1–3× real time on WASM with threads, and
   well under real time on WebGPU, for a 30 s clip. A full 4-minute song on a
   machine without a GPU is a multi-minute job. That needs a Web Worker with a
   progress bar, cross-origin isolation (`--isolate` headers) for threads, and
   chunked processing to keep memory under about 2 GB.
3. **Pipeline changes once stems exist.** Run Whisper and Melodia (or CREPE)
   on the vocal stem; this removes the centre-channel hack and the
   ASR-gating. Run basic-pitch per stem (bass, other); with WebGL that is
   about 1 s per stem per 30 s. Take drums from the drum stem; the band-flux
   detector is already there, and per-hit one-shots for a kit could be sliced
   in JS. Add `meta level`, `fx` and the loudness gate, which are all plain
   DSP and port directly from `encode.py`.
4. **Still missing after that.** tsumugi instrument labels would need its own
   ONNX export, and lead/backing vocal splitting would need a karaoke model.
   Both are possible later additions.
5. **Ship the models from a host the page can reach.** Plan for the Hugging
   Face CDN being unreachable: this machine could not reach it. A mirror on
   the project's own static hosting avoids that, and so does the
   `local:` model path this spike added.

Overall, a page with Demucs is feasible and would probably close most of the
part-score gap. Real separation is CPU-heavy, though, so treat WebGPU as a
requirement for full-length songs, and make "lite" (this spike) the fallback
on machines without one.

## Reproduce

```sh
python experiments/browser-encoder/tools/serve.py 18932 &
# optional, only if the HF CDN is blocked (needs optimum 1.19.2 / transformers 4.40.2):
python experiments/browser-encoder/tools/export_whisper.py openai/whisper-base experiments/browser-encoder/models/whisper-base_timestamped
PLAYWRIGHT=… CHROME=… node experiments/browser-encoder/tools/run-headless.mjs \
  --url http://127.0.0.1:18932/ --out out/browser-spike --meta site/media \
  --model local:whisper-base_timestamped audio/test/{discipline,lights_in_the_sky,999999,corona_radiata}-30s.mp3
```

The four `.sc` outputs are in `out/browser-spike/`. They are not committed
because they contain lyrics.
