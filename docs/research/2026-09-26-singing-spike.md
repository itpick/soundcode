# Singing spike — can we re-create the River lead vocal from `.sc` symbols? (2026-09-26)

Throwaway feasibility spike, run on the M1 Max (32 GB, MPS), no CUDA, no paid
services, ~2.9 GB of downloads. Everything lives under the session scratchpad
(`scratchpad/svs/`); nothing in the repo changed except this file and two
preview wavs.

Question: from **notes + lyrics** (tsumugi vocal MIDI, 81 notes → 71
monophonic; ASR word onsets from `:text.vox` on the `:grid`), how close can we
get to the original singer, and with which open tools?

## TL;DR

- **Yes, and the chain works Mac-local today: DiffSinger (OpenUtau ONNX bank,
  headless via onnxruntime) → Seed-VC singing model (MPS) with the separated
  lead stem as the voice reference.** From symbols only it lands at **40 cents
  median pitch error** vs the original f0 and **speaker-embedding cosine
  0.90** to the original singer (same-singer ceiling 0.99; Seed-VC's own
  upper bound, converting the real stem to itself, is 0.96).
- The pitch floor is the **symbolic data, not the synth**: the quantised MIDI
  itself is already 30 cents median / 60 % of frames within 50 cents of the
  sung f0, because the real vocal has scoops, vibrato and slides the note list
  does not carry. A `:contour.vox`-derived f0 curve is the lever that would
  close that gap.
- Timing: DiffSinger 30–36 s per 30 s clip on CPU; Seed-VC 270–370 s wall per
  clip on MPS (only 31 s of it is the 50-step diffusion; the rest is model
  loading + whisper/RMVPE/CAMPPlus/BigVGAN, measured under heavy unrelated
  CPU load). Fine for offline rendering; not interactive.
- Blockers: the zero-shot SVS models (SoulX-Singer 2.8 GB, YingMusic-Singer
  ~2.8 GB) did not fit the 4 GB free-disk floor and were not tried; the
  DiffSinger English bank used (Azure Cobalt, CC BY-SA 4.0) is a feminine
  hobbyist voice, so intelligibility and naturalness are "vocaloid-like"
  before conversion.

## Previews (listening server)

- `out/preview/river-30s.vocal-diffsinger-seedvc.wav` — DiffSinger (f0 from
  MIDI, CORE mode) → Seed-VC, lead stem as reference. The symbolic result.
- `out/preview/river-30s.vocal-oohs-seedvc.wav` — GM "Voice Oohs" render of
  the MIDI → Seed-VC. No SVS at all; melody + timbre only, no words.

Compare against `out/stems/river-30s/lead_vocals.wav`.

## What ran

### 1. Seed-VC (voice conversion), singing model

Repo `Plachtaa/seed-vc` (GPL-3.0), model
`DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` (821 MB)
+ `nvidia/bigvgan_v2_44khz_128band_512x` (489 MB) + RMVPE (181 MB) +
CAMPPlus (28 MB); `openai/whisper-base` was already in the HF cache.

Install that worked (isolated venv, Python 3.11; the repo's
`requirements-mac.txt` pins torch nightly, numpy 1.26 and transformers 4.46):

```
uv venv --python 3.11 venv
uv pip install torch==2.13.0 torchaudio numpy==1.26.4 scipy==1.13.1 librosa==0.10.2 \
    munch einops "huggingface-hub>=0.28.1" transformers==4.46.3 soundfile pyyaml \
    resemblyzer descript-audio-codec==1.0.0 "setuptools<80"
git clone --depth 1 https://github.com/Plachtaa/seed-vc
ln -s ~/.cache/huggingface/hub seed-vc/checkpoints/hf_cache   # reuse cached whisper-base
```

Three patches to `inference.py` were needed on this stack:

1. `descript-audio-codec` is imported by `modules/length_regulator.py` even
   though it is unused for this model (install it).
2. MPS has no float64: cast RMVPE output with
   `torch.from_numpy(F0_ori.astype(np.float32))` (and `F0_alt`).
3. torchaudio 2.11 `save` requires torchcodec; replaced with
   `soundfile.write(path, vc_wave.cpu().numpy().T, sr)`.

Command (all runs): `--diffusion-steps 50 --length-adjust 1.0
--inference-cfg-rate 0.7 --f0-condition True --auto-f0-adjust False
--semi-tone-shift 0 --fp16 False`, `--target lead_vocals.wav` (Seed-VC uses
the first 25 s as reference).

### 2. DiffSinger, headless, from an OpenUtau ONNX voicebank

`openvpi/DiffSinger`'s `scripts/infer.py` needs PyTorch checkpoints, and no
English PyTorch bank is published (HF search: only old 24 kHz Chinese
opencpop-format checkpoints). English banks are distributed as OpenUtau ONNX
packages, so I drove the ONNX graphs directly with `onnxruntime` (CPU EP; a
CoreML EP is also available) instead of installing DiffSinger's requirements
at all. Bank: **Azure Cobalt Δ v0.4.28** (CC BY-SA 4.0, EN/JA/ZH/KO, 502 MB,
github.com/agentasteriski/azurecobaltsynth), vocoder
`pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` (53 MB `.oudep` from
openvpi/vocoders, CC BY-NC-SA 4.0).

Graph interface (introspected; this is the `.sc` → SVS contract):

| model | inputs | outputs |
|---|---|---|
| `dsvariance/linguistic.onnx` | `tokens[1,T]`, `languages[1,T]`, `ph_dur[1,T]` (frames) | `encoder_out[1,T,384]`, `x_masks` |
| `dsvariance/variance.onnx` | `encoder_out`, `ph_dur`, `pitch[1,F]` (MIDI units), `breathiness/voicing/tension[1,F]`, `retake[1,F,3]`, `spk_embed[1,F,384]`, `steps` | predicted breathiness, voicing, tension |
| `dspitch/pitch.onnx` | `encoder_out`, `ph_dur`, `note_midi[1,N]`, `note_rest`, `note_dur[1,N]`, `pitch[1,F]`, `expr[1,F]`, `retake[1,F]`, `spk_embed`, `steps` | `pitch_pred[1,F]` (MIDI) |
| `dsmain/acoustic.onnx` | `tokens`, `languages`, `durations[1,T]`, `f0[1,F]` (Hz), `breathiness`, `voicing`, `tension`, `gender`, `velocity`, `spk_embed[1,F,384]`, `depth`, `steps` | `mel[1,F,128]` |
| vocoder | `mel[1,F,128]`, `f0[1,F]` | `waveform[1,S]` |

Hop 512 @ 44.1 kHz (11.6 ms frames), 128 mel bins, 7 vocal-mode embeddings
(`embeds/*.emb` = 384 float32). The driver is
`scratchpad/svs/ds_render.py` (about 180 lines):

- words: `:text.vox` `bar:beat` → seconds via the `:grid` anchor/tempo; each
  onset snapped to the nearest MIDI note onset within 250 ms; word end = next
  word onset, capped by the end of the last note starting inside the word.
- phonemes: CMUdict (`pip install cmudict`) → the bank's lowercase ARPAbet
  (`AH0`→`ax`, `HH`→`hh`); the bank's own `dsdict-en.yaml` only has 1137
  entries and covered 1 of our 45 distinct words, so a G2P fallback is
  mandatory. Syllables split at vowels; onset consonants 70 ms placed *before*
  the note onset, vowel fills the note, codas 70 ms; `SP` in gaps, `AP`
  (breath) for the last 300 ms of gaps > 450 ms.
- f0: MIDI note pitch per frame, gaps interpolated, 30 ms Gaussian
  portamento, → Hz. Alternative `--pitch model` runs `dspitch` conditioned on
  the note list instead.
- variance model fills breathiness / voicing / tension (retake = all), then
  acoustic (`steps=20`, `depth=0.6`, `velocity=1`, `gender=0`), then vocoder.

Wall time for the 30 s clip: 30–36 s on CPU (phonemising 0.4 s, variance
5 s, acoustic 17–22 s, vocoder ~3 s).

### 3. Sources for the "VC-only" test

`scratchpad/svs/render_src.py`: tsumugi MIDI made monophonic (at overlaps keep
the note nearest the median register, drops the A#4 harmony notes) →
tinysoundfont + GeneralUser-GS (program 53 Voice Oohs, 52 Choir Aahs) and a
plain sine.

## Metrics

Pitch: `librosa.pyin` (C2–C6, 22.05 kHz, hop 256) on both files; median
absolute cents error over frames voiced in both; `within50` = share of those
frames within 50 cents; `recall` = share of the original's voiced frames that
the candidate also voices. Timbre: resemblyzer speaker-embedding cosine
against `lead_vocals.wav` (`scratchpad/svs/metrics.py`, `timbre.py`).

| candidate | pitch median cents | within 50 c | voiced recall | timbre cos |
|---|---|---|---|---|
| original lead stem (self) | 0 | 1.00 | 1.00 | 1.000 |
| `vocals.wav` (same singer, full vocal stem) | 0 | 0.96 | 0.99 | 0.991 |
| `backing_vocals.wav` | — | — | — | 0.712 |
| MIDI → sine | 30 | 0.63 | 0.85 | 0.420 |
| MIDI → GM Voice Oohs | 30 | 0.57 | 0.97 | 0.640 |
| MIDI → GM Choir Aahs | 60 | 0.45 | 0.98 | — |
| DiffSinger CORE, f0 from MIDI | 30 | 0.59 | 0.94 | 0.660 |
| DiffSinger CORE, f0 from `dspitch` | 80 | 0.37 | 0.93 | 0.657 |
| DiffSinger MASK, f0 from MIDI | 30 | 0.59 | 0.94 | 0.672 |
| **Seed-VC ← lead stem (upper bound)** | **10** | **0.85** | 0.94 | **0.959** |
| Seed-VC ← sine | 30 | 0.59 | 0.90 | 0.815 |
| Seed-VC ← Voice Oohs | 50 | 0.51 | 0.92 | 0.857 |
| **Seed-VC ← DiffSinger CORE (MIDI f0)** | **40** | 0.56 | 0.94 | **0.897** |
| Seed-VC ← DiffSinger CORE (`dspitch`) | 80 | 0.38 | 0.92 | 0.909 |
| Seed-VC ← DiffSinger MASK (MIDI f0) | 40 | 0.58 | 0.94 | 0.889 |

Timings (wall, 30 s clip): Seed-VC 272 s (lead→lead), 344 s (oohs), 347 s
(sine), 357 s (DS core), 370 s (DS dspitch), 330 s (DS mask); the
50-step diffusion is ~31 s, the rest is model load + feature extraction +
BigVGAN. Model downloads: 1.5 GB in 48 s. DiffSinger renders: 30–36 s each.

Reading the numbers:

- Seed-VC keeps the source pitch (lead→lead: 10 cents, 85 % within 50) and
  moves timbre most of the way to the reference from *any* source: 0.42→0.82
  for a sine, 0.64→0.86 for a GM choir, 0.66→0.90 for DiffSinger. A real
  vocal-like source (DiffSinger) converts measurably better than a synth.
- Pitch accuracy of every symbolic path is bounded at ~30 cents by the note
  list; Seed-VC adds ~10 cents on top (its own RMVPE → f0-conditioning
  round trip), so 40 cents median is the symbol-only floor with this data.
- `dspitch` (the bank's pitch predictor) adds expressive deviations of 1.3
  semitones on average from the notes — natural-sounding but wrong for a
  faithful rebuild. Explicit f0 is the right input.
- The 30 cents symbolic baseline comes from the 46-note basic-pitch /
  81-note tsumugi transcription being quantised to semitones with no bends;
  the original has a 205–262 Hz p10–p90 range around A#3 with continuous
  slides. Rendering `:contour.vox` (scoop/fall/vibrato) into the f0 curve
  is where the next 20 cents come from.

## Blockers and caveats

- **Disk**: SoulX-Singer (Apache 2.0, 2.8 GB weights) and YingMusic-Singer
  (MIT, ~2.8 GB needed) would both have pushed free space under 4 GB; not
  tried. Vevo2 is 11 GB. They remain the candidates for "arbitrary voice from
  lyrics + MIDI in one model" once disk allows (or on a rented GPU).
- **openvpi `scripts/infer.py` path is moot** without a PyTorch English bank;
  the ONNX route replaces it and is lighter (onnxruntime only). OpenUtau's
  own phonemizer/G2P and its duration model were not used; `dsdur` exists in
  the bank if we want predicted phoneme durations instead of my heuristic.
- **Bank fit**: Azure Cobalt is a feminine mid/deep voice; the River lead
  sits at A#3 (233 Hz median). Rendering in the MASK ("faux-masculine") mode
  only nudged timbre (0.672 vs 0.660 raw). Seed-VC does the heavy lifting on
  timbre either way.
- **Lyric timing**: `:text.vox` word onsets are quantised to half-beats; 3
  of 56 words collided (same onset) and were pushed 120 ms; "I" is placed
  before the first transcribed note. Word-level intelligibility after
  conversion was not scored (no ASR pass); listen to the preview.
- **Licences**: Seed-VC GPL-3.0; BigVGAN v2 MIT; NSF-HiFiGAN vocoder CC
  BY-NC-SA; Azure Cobalt CC BY-SA 4.0; RMVPE from the RVC project (MIT).
  Fine for research, NC for the DiffSinger vocoder path.
- Seed-VC's `inference.py` hard-codes `HF_HUB_CACHE=./checkpoints/hf_cache`;
  symlink it to the shared cache or whisper-base downloads twice.

## Recommendation for the singing pipeline

**Adopt the two-stage chain: DiffSinger (ONNX bank, headless) → Seed-VC
(singing model) with the separated lead stem as the voice reference.** It is
fully open, Mac-local, deterministic given a seed, and the `.sc` data maps
onto it one-to-one.

What each stage needs from `.sc`:

| stage | `.sc` data | notes |
|---|---|---|
| word → phoneme | `:text.vox` words (ideally `align=syllable`) | CMUdict / G2P fallback is mandatory; store the resolved phoneme string in the stream (`ph=`) so renders are reproducible. |
| phoneme durations | `:text.vox` onsets + `:notes.vox` onsets/durations | onset consonants precede the note, vowel = note; or run the bank's `dsdur` model constrained to word spans. |
| f0 curve | `:notes.vox` (cents, onset, dur) **+ `:contour.vox`** (scoop / fall / vibrato / bend) | explicit `f0` input to the acoustic model; do **not** use `dspitch`. This is the single biggest lever on the 40-cent number. |
| expression | `:mix`/velocity → `velocity`, `gender`; breathiness / tension / voicing from the variance model (or from `:contour.vox` verbs if we add them) | optional |
| voice | a 5–25 s reference clip: the separated `lead_vocals.wav` (`:audio.vox` in v0.4), or any target voice | Seed-VC `--target`, `--f0-condition True --auto-f0-adjust False`, `--semi-tone-shift` if the bank's range is off |
| tempo / grid | `:grid` for bar:beat → seconds | |

Next steps in order of expected gain:

1. Render `:contour.vox` into the f0 curve (and add per-note bend/portamento
   to the encoder), then re-measure; target < 25 cents median before VC.
2. Score intelligibility: run whisper on the converted vocal and compare to
   `:text.vox` (WER) — the encoder's ASR already exists.
3. Cut Seed-VC wall time: cache the loaded models in a long-lived process,
   try 25–30 steps, and try `--fp16 True` on MPS.
4. When disk allows, try SoulX-Singer (Apache 2.0, MIDI-or-F0 + lyrics +
   reference in one model) as the single-stage alternative; keep the
   DiffSinger→Seed-VC chain as the deterministic baseline.
5. For backing vocals, the same chain with a different reference clip
   (`backing_vocals.wav`) per `:notes.<trk>`.
