# Browser encoder (spike)

A static page that turns a song into a `.sc` file entirely in the browser, with
no server-side processing. This is an experiment: the code stays out of `src/` and
`site/` until there is a decision to promote it. The results and the
recommendation are in [RESULTS.md](RESULTS.md).

## Run it

```sh
cd experiments/browser-encoder
python tools/serve.py 8000            # or: python -m http.server 8000
open http://127.0.0.1:8000/
```

Drop an audio file on the page and press **Encode**. To use LRCLIB, type the
title and artist first. Download the `.sc` when it finishes. The page fetches
its libraries from cdn.jsdelivr.net, pinned to exact versions. The first run
also downloads the Whisper model from huggingface.co, and the browser caches it
after that. The page itself is about 60 KB of JavaScript.

## Pipeline (all in the page)

| Step | Tool | Output |
|---|---|---|
| Decode and resample | Web Audio API: `decodeAudioData`, `OfflineAudioContext` | 44.1 kHz stereo, 22.05 kHz and 16 kHz mono |
| Tempo and beats | essentia.js 0.1.3 `RhythmExtractor2013` (WASM) | `:grid`: one tempo, a downbeat, an anchor every 16 bars |
| Key and chords | essentia.js `KeyExtractor`; `HPCP` with a per-bar triad template | `:harmony`, including `tonal_center` |
| Sections | Novelty in bar-level chroma and loudness (JS) | `:struct`; the vocal flag is set where ASR heard words |
| Drums | HPSS percussive share as a gate, then band-wise spectral flux onsets (JS) | `:perc.drums` with kick, snare and hat |
| Notes | `@spotify/basic-pitch` 1.0.1 (TF.js) on the full mix | `:notes.bass` (below MIDI 48) and `:notes.keys` (`keys.piano`) |
| Vocal pre-filter | Centre-channel extraction (JS STFT mask) | Input for Whisper and the melody tracker |
| Lyrics | Whisper `whisper-base_timestamped` via `@huggingface/transformers` 3.7.6, with word timestamps | `:text.vox` |
| Published lyrics | LRCLIB `/api/search` (CORS is open), reconciled with ASR (port of `lyrics.py`) | Corrected words and `@offset` |
| Vocal melody | essentia.js `PredominantPitchMelodia` and `PitchContourSegmentation`, kept only where words were heard | `:notes.lead` and `:contour.vox` |
| Mix scalars | JS STFT and essentia.js `LoudnessEBUR128` | `:mix` (same definitions as the native encoder) |

Backends: Whisper uses WebGPU when the browser exposes an adapter and falls
back to WASM otherwise. basic-pitch uses WebGL and falls back to the TF.js CPU
backend. The output conventions follow the native encoder: the header,
3-decimal `bar:beat` positions, `meta` lines, and `?conf` marks.

## Files

- `index.html`, `js/main.js`: the UI. They also expose `window.scEncodeUrl()`
  for headless runs.
- `js/pipeline.js`: runs the stages, times each one, and assembles the `.sc`.
- `js/analysis.js`: essentia.js stages, drums, `:struct`, and `:mix`.
- `js/notes.js`: basic-pitch. `js/lyrics.js`: Whisper, LRCLIB, and reconciliation.
- `js/sc.js`: `.sc` formatting helpers ported from `src/soundcode`.
- `js/stft.js`: FFT, STFT, and centre-channel extraction.
- `tools/serve.py`: the static server. `--isolate` adds COOP/COEP headers so
  WASM can use threads.
- `tools/run-headless.mjs`: a Playwright driver that encodes a list of clips
  and writes `<clip>.sc` and `<clip>.json` (timings, backends, downloads).
- `tools/export_whisper.py`: exports `openai/whisper-*` to ONNX in the
  transformers.js layout, with cross-attention outputs. It writes to `models/`,
  which is gitignored. You only need it when the Hugging Face model CDN is
  unreachable (see RESULTS.md).

## Headless

```sh
PLAYWRIGHT=/path/to/node_modules/playwright \
CHROME=~/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell \
node tools/run-headless.mjs --url http://127.0.0.1:8000/ --out ../../out/browser-spike \
  --meta ../../site/media --model local:whisper-base_timestamped \
  /abs/path/discipline-30s.mp3 ...
```

`--meta DIR` reads `@title` and `@artist` from `DIR/<clip>.sc`, which stands
in for a user typing them. Point `CHROME` at a full Chromium and add
`--headless-new --gpu` for a WebGPU (Metal) run. Generated `.sc` files contain
lyrics, so do not commit them.
