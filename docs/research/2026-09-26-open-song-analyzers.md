# Open song analyzers as an encoder backbone — 2026-09-26

Question: is there an existing open-source project that already analyses a
full song end to end (notes **with instrument labels**, instrument inventory,
drums incl. claps) well enough to become the backbone of `soundcode encode`,
instead of the hand-built audio-separator → basic-pitch → librosa chain?

Constraints: free/open only (non-commercial OK), M1 Max 32 GB, Python 3.11,
MPS/CPU; a second "Framework" machine of unknown spec for GPU-only models.
Builds on `2026-09-26-open-codifying.md` (§3 transcription table, SheetSage2,
MuScriptor, YourMT3+, ADTOF/DrumSep recipe, audio-LLMs) — facts already there
are only repeated when they changed or matter for the decision.

Method note: the web-search budget of this session was already spent, so every
claim below comes from direct fetches of the GitHub API (stars, `pushed_at`,
licence), PyPI/HF APIs (versions, file sizes, gating), the arXiv API, and
project READMEs/papers, all on 2026-09-26. Hands-on numbers are from this
machine.

## TL;DR

- **New, and the best fit: `tsumugi`** (anime-song, MIT, Sept 2026). One
  repo that transcribes pitched notes with a **36-class instrument
  taxonomy**, has **drum models that emit GM percussion pitches including
  hand clap (39)**, a per-stem **instrument-refinement classifier that
  returns probabilities**, a beat/chord/key head, a velocity model and its
  own stem splitter. Runs on **MPS**, checkpoints are ~55 MB each, install is
  `git clone` + `uv sync`. On `river-30s.wav` it found piano / electric piano
  + a vocal melody on the mix, said **piano 0.97** for our piano stem and
  **piano 0.73 / e-piano 0.27** for the stem our pipeline labels "guitar",
  and found **16 hand-clap hits** on the drum stem. 30 s takes ~5 s.
- **MuScriptor** (Kyutai/Mirelo) is the strongest *published* mix-level
  multi-instrument transcriber (multi-instrument F1 47.8 vs YourMT3+ 21.9 on
  real recordings), MIT code, MPS, `uvx muscriptor`, but the CC BY-NC weights
  are **gated**: anonymous download returns HTTP 401 and there is no HF
  token on this machine, so it could not be run. Unblock with
  `hf auth login` + accept the licence on the model page.
- **YourMT3+** (GPL-3.0, 2024, unmaintained) installs from its HF Space clone
  (562 MB ckpt) and runs on CPU (59 s for 30 s). On river it labels the
  accompaniment **piano** (38 notes) and the voice (101 notes) but invents a
  71-note string ensemble and 20-odd toms, and its GM drum vocabulary has no
  clap.
- **Instrument presence from AudioSet taggers is not usable as-is on this
  material**: PANNs Cnn14 on the river mix gives Piano 0.00, Clapping 0.00,
  Guitar 0.02, Singing 0.28; on our stems it calls the piano stem "Guitar
  0.31 / Piano 0.03". Timbre judgement is better done by transcription-side
  classifiers (tsumugi refinement, MuScriptor labels) than by taggers.
- **Claps**: only two open drum transcribers have a clap class: **tsumugi
  `drums_v1_5`** (GM pitches; 16 claps on the river drum stem, 5 s on MPS)
  and **ADT_STR** (Jan 2026, 26 GM classes incl. Hand Clap + velocities,
  ENST F1 0.73 / MDB 0.79, CC BY-SA, 289 MB; 26 claps on the same stem, 93 s
  on CPU, killed on MPS). Their kick onsets agree to ~10 ms but their clap
  onsets do not (tsumugi 6–11 s, ADT_STR 13–17 s), so clap timing needs a
  listening check. Everything else (ADTOF, Omnizart, Separate-and-detect,
  LarsNet, YourMT3+ GM vocab) is kick/snare/hat/tom/cymbal.
- **No complete "song analyzer" beats a stem-wise pipeline**, but tsumugi is
  close enough to a toolkit that we should wrap it: keep our separation,
  replace basic-pitch + band-split drums + PANNs with tsumugi models per stem,
  and keep SheetSage2 / Beat This! for the annotation layer as decided in the
  codifying doc.

---

## (a) Multi-instrument transcription from the mix, with instrument labels

| name | what it does | instruments / labels | accuracy evidence (multi-instrument, with labels) | license | Mac? | install | maintained? | link |
|---|---|---|---|---|---|---|---|---|
| **tsumugi** (`instrument-agnostic-amt`, anime-song, 2026) | Neural semi-CRF AMT (Transkun lineage) on CQT; one instrument-agnostic backbone + per-stem variants (`default`, `bass_v2`, `guitar_v1_5`, `vocal`, `vocal_harmony_v1_5`, `other_v1_5`, `drums_v1_5`), instrument-refinement classifier, velocity model, beat/chord/key head, stem splitter | 36-class taxonomy (piano, electric_piano, plucked_keyboard, acoustic/electric-clean/muted/distorted guitar, 4 basses, melody, vocal_harmony, choir, strings, brass, sax, woodwind, flute, organ, synth lead/pad/fx, chromatic perc, harp, ethnic, timpani, drums…) → GM programs in the MIDI; drums as GM pitches | No Slakh/MusicNet/URMP numbers published. Author's held-out sets: RWC-I instrument top-1 **71.3 % (AMT) → 74.5 % (refinement)**; MIR-ST500 vocal COnP 0.68; drums exact F1 0.689 (kit) / 0.404 (all percussion) on an in-house real set. Hands-on below | MIT (code + weights) | ✅ MPS native (`--device mps`), CPU; torch 2.13 pinned; Py 3.10–3.14 | `git clone https://github.com/anime-song/tsumugi && uv sync --locked`; `python -m instrument_agnostic_amt.amt.cli.infer --audio x.wav [--type drums_v1_5]`; weights auto-download from HF `anime-song/instrument_agnostic_amt` (55–57 MB each) | ✅ very active: 66 stars, pushed 2026-09-24, dated changelog every 1–2 weeks since May 2026; single author | [GitHub](https://github.com/anime-song/tsumugi) · [HF](https://huggingface.co/anime-song/instrument_agnostic_amt) |
| **MuScriptor** (Kyutai + Mirelo, Jul 2026) | Encoder–decoder seq2seq (MT3 tokens) on 5 s segments; small 103M / medium 307M / large 1.4B; optional instrument conditioning; `--format sheets` via MuseScore | 36 instrument groups (MT3_FULL_PLUS taxonomy: acoustic_piano, acoustic_guitar, acoustic_bass, …; `muscriptor list-instruments`); drums onset-only; no velocity; no overlapping same-pitch notes | Paper test set (372 real tracks): **multi-instrument F1 47.8, onset F1 60.4** vs YourMT3+ 21.9 / 32.5; also reports Bach10, ChoirSet, PHENICX, RWC-P/C/G/J/R. No Slakh/MusicNet/URMP table in the paper HTML. Drums onset F1 41.6 | code MIT; weights **CC BY-NC 4.0, gated** (auto-approve after accepting terms) | ✅ "runs on Metal (MPS) automatically"; large ≈ 3 GB fp16 | `uvx muscriptor transcribe --model medium x.wav -o x.mid` (PyPI 0.3.0, 2026-08-05); needs `hf auth login` — anonymous `config.json` fetch → HTTP 401 | ✅ 1 506 stars, pushed 2026-09-04 | [GitHub](https://github.com/muscriptor/muscriptor) · [arXiv](https://arxiv.org/abs/2607.08168) · [HF](https://huggingface.co/MuScriptor/muscriptor-large) |
| **YourMT3+** (mimbres, MLSP 2024) | Perceiver-TF + MoE + multi-channel T5 decoder (45.8 M); 13 decoding channels | MT3_FULL_PLUS: 34 program classes + singing (melody / chorus) + drums; drum vocab `gm` = 20 GM drum classes (kick, snare, x-stick, HH ×2, toms ×7, ride ×3, china, crash, splash, cowbell) — **no clap** | Slakh onset F1 84.56, **multi-F1 (onset-offset+instrument) 74.84**; URMP agnostic 81.79 / multi 67.98; MusicNet-EM strings 91.32 / winds 83.46; ENST drums 87.27; MIR-ST500 singing 72.05; MAESTRO 96.52; GuitarSet 88.87. Authors: on commercial pop, non-main instruments "< 10 %" | GPL-3.0 | 🟡 CPU fp32 (works, see hands-on); MPS untested (`device="cpu"` then `.to("cuda")` hard-coded in the Space app) | clone HF Space with `GIT_LFS_SKIP_SMUDGE=1`, download one ckpt (`…_nops/checkpoints/last.ckpt`, 562 MB); torch, lightning, transformers==4.45.1, numpy==1.26.4, einops, mido, mir_eval, **wandb** (imported by model code); torchaudio ≥ 2.9 `load` needs torchcodec (patch with soundfile) | ❌ 247 stars, last push 2024-11-29 | [GitHub](https://github.com/mimbres/YourMT3) · [arXiv](https://arxiv.org/abs/2407.04822) · [Space](https://huggingface.co/spaces/mimbres/YourMT3) |
| **MT3** (Magenta, 2021) | T5 seq2seq, JAX/T5X | MT3 full: 128 GM programs (+drums) | Slakh multi-F1 ~0.62 in original paper (superseded by all of the above) | Apache-2.0 | 🟡 JAX on CPU only; painful | clone + t5x; Colab is the supported path | 🟡 1 760 stars, repo touched 2026-09-15 (housekeeping), model unchanged | [GitHub](https://github.com/magenta/mt3) |
| **MR-MT3** (2024) | MT3 + memory retention to cut instrument leakage between segments | MT3 vocab | Slakh instrument-leakage study; numbers not in README (prior doc) | MIT | 🟡 Py 3.10 + TF 2.11 + torch; CPU possible | conda + 15 pip deps; ckpt HF `gudgud1014/MR-MT3` | ❌ 57 stars, last push 2025-06 | [GitHub](https://github.com/gudgud96/MR-MT3) |
| **Omnizart** 0.6.3 (May 2026) | TF toolkit: music (notes+instruments), drum, chord, beat, vocal, vocal-contour | music **Pop** model: 5 program groups (guitar 24, bass 32, strings 40, piano 0, brass/other 56); **Stream** model: 11 MusicNet instruments (piano, harpsichord, violin, viola, cello, contrabass, horn, oboe, bassoon, clarinet, flute); drum: 13 classes predicted, only kick/snare/hihat written | 2021 JOSS paper; no modern benchmark numbers | MIT | ❌ README still states "incompatible for ARM-based macOS"; 0.6.x (May 30–31 2026) modernised deps to Py 3.14 / NumPy 2, replaced Spleeter with demucs CLI / sherpa-onnx, but no ARM statement changed; sdist only | `pip install omnizart` (TF ≥ 2.5) — unverified on M1 | 🟡 1 981 stars, pushed 2026-05-31 (compat release after 3 years) | [GitHub](https://github.com/Music-and-Culture-Technology-Lab/omnizart) · [releases](https://github.com/Music-and-Culture-Technology-Lab/omnizart/releases) |
| **Transkun v2** | Piano-only semi-CRF AMT with velocity + pedal | piano only | MAESTRO onset F1 0.983 (prior doc) | MIT | ✅ CPU | `pip install transkun` (2.0.1, Sept 2024) | ❌ pushed 2024-11 (tsumugi is its multi-instrument descendant) | [GitHub](https://github.com/Yujia-Yan/Transkun) |
| **TUTTI** (Sept 2026) | Audio-to-score transformer trained on a synthetic multi-instrument corpus | multi-instrument (list not in abstract) | "new SOTA across A2S baselines" (abstract) | CC BY-NC-ND paper | — | **code promised, link is a placeholder** | ❌ paper only | [arXiv](https://arxiv.org/abs/2609.00640) |
| **Two-branch contrastive clustering** (TISMIR 9(1), 2026) | Timbre-agnostic transcription + note-level timbre clustering (joint AMT + separation) | dynamic instrument clusters | "competitive with heavier baselines" | — | — | no code found | ❌ | [arXiv](https://arxiv.org/abs/2509.12712) |
| **MIROS / 2025 AMT Challenge**, **Jointist**, **Timbre-Trap**, **Basic Pitch**, **NeuralNote** | see codifying doc §3; NeuralNote (2 947 stars, pushed 2026-09-24) is a JUCE plugin around Basic Pitch — instrument-agnostic | — | — | — | — | — | — | [NeuralNote](https://github.com/DamRsn/NeuralNote) |

Reading the evidence: only three systems output **labelled** notes and run
here: tsumugi, MuScriptor (once logged in) and YourMT3+. MuScriptor has the
best real-recording numbers by a wide margin; tsumugi has no cross-dataset
numbers but is the only one with an explicit, probability-emitting instrument
classifier and a clap-capable drum model; YourMT3+ is strong on Slakh but
its own authors report < 10 % on non-main instruments in pop and it is
unmaintained.

## (b) Instrument recognition / presence detection

| name | what it does | instruments / labels | accuracy evidence | license | Mac? | install | maintained? | link |
|---|---|---|---|---|---|---|---|---|
| **tsumugi instrument refinement** | Classifies notes of a separated stem into the 36-class taxonomy; `--mode single` (one class per stem, with **probabilities**) or `--mode cluster` (timbre clusters); candidate set is restricted by `--stem-name` | 36 classes (no drums) | RWC-I (554 solo files, held out): top-1 74.5 % vs 71.3 % for the AMT head; acoustic guitar / chromatic perc / harp 100 %, sax 31 %, harpsichord 0 % (absorbed by piano) | MIT | ✅ MPS/CPU, 56 MB | part of tsumugi; `python -m instrument_agnostic_amt.instrument_refinement.cli.infer --audio stem.wav --midi stem.mid --stem-name other --mode single` | ✅ | [benchmark](https://github.com/anime-song/tsumugi/blob/main/instrument_agnostic_amt/instrument_refinement/RWC_BENCHMARK.md) |
| **PANNs Cnn14** (AudioSet) | Clip-level 527-class tagger (framewise variant `Cnn14_DecisionLevelMax` exists) | AudioSet: Piano, Electric piano, Guitar (ac./el.), Bass guitar, Drum kit, Snare, Bass drum, Hi-hat, **Clapping, Hands, Applause, Finger snapping**, Singing… | AudioSet mAP 0.431 | MIT | ✅ CPU, 312 MB ckpt (already at `~/panns_data`) | `pip install panns-inference` (0.1.1, 2023) | ❌ pushed 2024-07 | [GitHub](https://github.com/qiuqiangkong/audioset_tagging_cnn) |
| **BEATs iter3+ (AS2M)** | Tokenised SSL audio transformer, AudioSet fine-tuned (90 M) | AudioSet 527 | **AudioSet-2M mAP 50.6** (best single audio-only model), ESC-50 98.1 | MIT (unilm) | ✅ CPU/MPS; checkpoints via links in README | clone `microsoft/unilm/beats`; `BEATs(cfg).extract_features()` | 🟡 unilm active, BEATs frozen since 2023 | [GitHub](https://github.com/microsoft/unilm/tree/master/beats) · [arXiv](https://arxiv.org/abs/2212.09058) |
| **EfficientAT** (mn10/mn40/dymn) | MobileNet AudioSet taggers | AudioSet 527 | mn40_as_ext 48.7, dymn20_as 49.1 mAP; no SED variant | MIT | ✅ CPU (fp16-trained, "slight degradation" in fp32) | clone + `requirements.txt` (Py 3.10) | ❌ pushed 2024-11 | [GitHub](https://github.com/fschmid56/EfficientAT) |
| **PaSST / AST / HTS-AT** | Transformer AudioSet taggers | AudioSet 527 | PaSST 0.476, AST 0.459, HTS-AT 0.471 | Apache-2.0 / BSD | ✅ CPU | `pip install hear21passt`; HF `MIT/ast-finetuned-audioset-10-10-0.4593` | 🟡 | [PaSST](https://github.com/kkoutini/PaSST) |
| **FlexSED** (JHU, WASPAA 2025) | **Open-vocabulary sound event detection with timestamps**: text prompt ("Clapping", "Piano") → frame activity | any text query (CLAP text encoder; trained on AudioSet-Strong) | paper: AudioSet-Strong PSDS; numbers not in README | MIT | 🟡 example uses `device='cuda'`; CPU unverified; ckpt 431 MB | clone + `requirements.txt`; HF `Higobeatz/FlexSED` | 🟡 57 stars, pushed 2025-12 | [GitHub](https://github.com/JHU-LCAP/FlexSED) · [arXiv](https://arxiv.org/abs/2509.18606) |
| **Essentia `mtg_jamendo_instrument`** | Discogs-EffNet embeddings → 40 instrument tags (clip-level, multi-label) | accordion, acoustic/classical/electric guitar, bass, doublebass, drums, drummachine, electricpiano, rhodes, piano, keyboard, organ, pipeorgan, strings, violin, viola, cello, brass, trumpet, trombone, horn, sax, clarinet, oboe, flute, harmonica, harp, bell, bongo, percussion, synthesizer, pad, sampler, computer, orchestra, voice, beat — **no clap** | test **ROC-AUC 0.78, PR-AUC 0.20** (weak) | CC BY-NC-SA 4.0 | ✅ `essentia-tensorflow` macOS arm64 wheels for cp3.10–3.13 (`2.1b6.dev1389`, Jul 2025); the May 2026 `dev1438` build ships **only cp314** for arm64 | `pip install essentia-tensorflow==2.1b6.dev1389` + `.pb` model files | ✅ Essentia pushed 2026-09-21 | [models](https://essentia.upf.edu/models.html) |
| **Essentia AudioSet-YAMNet** | 521-class AudioSet tagger (incl. Clapping) | AudioSet | YAMNet mAP ≈ 0.31 (weaker than PANNs/BEATs) | CC BY-NC-SA | ✅ | as above | ✅ | [models](https://essentia.upf.edu/models.html) |
| **OpenMIC-2018 baseline** (cosmir) | VGGish + per-class random forest | 20 OpenMIC classes (no clap) | 2018 baseline F1 ≈ 0.8 | MIT | ✅ | clone (Py 3.6-era) | ❌ pushed 2022 | [GitHub](https://github.com/cosmir/openmic-2018) |
| **LAION-CLAP / MS-CLAP zero-shot** | Text–audio similarity for arbitrary labels | any | ISMIR 2024 *"I can listen but cannot read"*: two-tower models are "sensitive towards specific words, favouring generic prompts over musically informed ones"; text tower is the weak link → **not reliable for instrument inventory** | CC0 / MIT | ✅ | `pip install laion-clap` (1.1.7, May 2025) / `msclap` (1.3.4) | 🟡 | [paper](https://arxiv.org/abs/2407.18058) |
| **MuQ-MuLan**, **MERT / MuQ probes**, **MOSS-Music-8B** | see codifying doc §2 (zero-shot tags; audio-LLM instrument description) | — | MARBLE-style probes; no calibrated presence scores | CC BY-NC / Apache | ✅ | — | — | — |
| **Lead Instrument Detection** (ICASSP 2025) | Frame-level "which instrument is lead" on multitracks | MedleyDB classes | paper only; 4 HF checkpoints | MIT | 🟡 Py 3.12 | clone | ❌ 3 stars | [GitHub](https://github.com/Sonata165/LeadInstrumentDetection) |
| **musedetect** (rare-instrument detection, 2025) | Hierarchical (Hornbostel–Sachs) instrument detection on MedleyDB | MedleyDB taxonomy | paper only; weights unclear | BSD-3 | 🟡 | `pip install .` | ❌ 2 stars | [GitHub](https://github.com/Seon82/musedetect) |
| **MT2 multi-class-token transformer** (2025) | SSL MIR model incl. instrument tagging, "18× fewer params than MERT" | — | beats MERT on tagging (abstract) | — | — | no code found | ❌ | [arXiv](https://arxiv.org/abs/2507.12996) |

Which is most accurate for "is there a guitar / piano / clap here?" — there
is no off-the-shelf OpenMIC-tuned checkpoint with a published mAP that we can
just download (PaSST and EfficientAT only ship fine-tuning scripts), and the
AudioSet taggers failed on this song (see hands-on). The practical answer is
two-layered: (1) run **BEATs or PANNs on each separated stem** as a cheap
prior (they were right that the "guitar" stem is *guitar-like* and that the
bass/other stems are silent), and (2) let the **transcription side decide the
label**: tsumugi refinement probabilities (piano 0.97 on the piano stem) or
MuScriptor's per-note groups. For claps specifically, use a drum transcriber
with a clap class (next section) rather than a tagger — PANNs gave Clapping
0.01 on a drum stem in which tsumugi found 16 clap hits.

## (c) Drum transcription beyond kick/snare/hat

| name | what it does | classes | accuracy evidence | license | Mac? | install | maintained? | link |
|---|---|---|---|---|---|---|---|---|
| **ADT_STR** (Jan 2026, "Towards Realistic Synthetic Data for ADT") | Encoder–decoder transformer on 2.56 s mel windows → MIDI tokens (onset, class, velocity); trained on synthesized data from a curated one-shot corpus | **26 GM-percussion classes incl. Hand Clap (39)**, side stick, 5 toms, HH open/closed/pedal, ride/bell, crash, china, splash, cowbell, tambourine… | **ENST F1 0.73 (BD .87 SD .80 TT .55 HH .77 CY .49), MDB 0.79 (BD .92 SD .85 TT .77 HH .74 CY .52)**, claimed > ADTOF and TMIDT; per-class clap F1 not stated in text. Hands-on: 26 claps + velocities on the river drum stem | CC BY-SA 4.0 | 🟡 Py 3.10–3.12, **torch 2.8.0 / torchaudio 2.8.0 + torchcodec pinned**; **CPU works (93 s / 30 s, 2.1 GB), MPS run was SIGKILLed**; 289 MB per tau checkpoint; drum-stem input only (mix → cymbal garbage) | `uv venv --python 3.12 && uv pip install -r pyproject.toml`; HF `Pierfrancesco/adt-str` (`snapshot_download(..., allow_patterns=["*.py","*.json","*.yaml","setting-tau-0.8/*"])` to avoid 4 × 289 MB) + `configs/` from GitHub; `ADTTranscriber.from_pretrained(dir, variant="setting-tau-0.8", device="cpu").transcribe(wav)` | 🟡 19 stars, pushed 2026-09-19 | [GitHub](https://github.com/pier-maker92/ADT_STR) · [arXiv](https://arxiv.org/abs/2601.09520) |
| **tsumugi `drums_v1_5`** | Semi-CRF drum model emitting GM drum pitches on channel 10 (`40→38`, `57→49` canonicalised) | GM percussion pitches: kick, snare, side stick, **hand clap**, HH closed/open, toms, crash/ride…; "experimental" | author's real-audio set: exact F1 0.689 drum kit, 0.404 all percussion (50 ms) | MIT | ✅ MPS | `--type drums_v1_5` | ✅ | [GitHub](https://github.com/anime-song/tsumugi) |
| **ADTOF** + **ADTOF-pytorch** | CRNN 5-class ADT; PyTorch port with bundled weights | kick, snare, hihat, tom, cymbal | MDBDrums++ F 88.74 (TF) / 88.51 (torch) | CC BY-NC-SA 4.0 / port unlisted | ✅ CPU | `pip install -e .` (torch, librosa, pretty_midi) | 🟡 ADTOF pushed 2025-09; port 2025-11 | [ADTOF](https://github.com/MZehren/ADTOF) · [port](https://github.com/xavriley/ADTOF-pytorch) |
| **adtof_plus_drum_transcription** (Riley & Dixon recipe) | MDX23C drum-stem separation (jarredou) → ADTOF-pytorch → MIDI with velocity | 5 classes (+ crash/ride split in paper's 7-class) | paper: 8-class MDB 0.84 / ENST 0.76 vs 0.72 / 0.65 plain ADTOF | MIT | ✅ (MSST on MPS) | clone + `pip install -e .` (depends on essentia) | ❌ 2 commits | [GitHub](https://github.com/xavriley/adtof_plus_drum_transcription) · [arXiv](https://arxiv.org/abs/2509.24853) |
| **Separate-and-detect** (ISMIR 2026) — **code now released** (Aug 2026) | Latent-diffusion 5-stem drum separation with onset/timbre auxiliaries | kick, snare, toms, hi-hats, cymbals | "consistently improves over U-Net separation baseline; beats an end-to-end ADT on kick/snare F1" | MIT | ❌ CUDA only documented; madmom patch for Py 3.10+ | conda `.yml`; HF `ddman1101/Separate-and-Detect` | 🟡 15 stars, pushed 2026-09-16 | [GitHub](https://github.com/ddman1101/Separate-and-detect) |
| **DrumSep** (inagoy) / **jarredou MDX23C DrumSep** / **LarsNet** | Drum-stem separators (Hybrid Demucs 4-stem: kick, snare, cymbals, toms / MDX23C / U-Net 5-stem kick, snare, toms, hihat, cymbals) | no clap stem | separation only | MIT / — / unlisted | ✅ (Demucs/MSST on MPS) | `drumsepInstall`; MSST; clone | 🟡 DrumSep pushed 2025-11; LarsNet 2024-09; PyPI `drumsep` 0.1.0 (Mar 2026) is an unrelated 1-line package | [DrumSep](https://github.com/inagoy/drumsep) · [LarsNet](https://github.com/polimi-ispl/larsnet) |
| **Omnizart drum** | CNN, 13 classes internally, writes kick/snare/hihat only | 3 written | 2021 | MIT | ❌ ARM | — | 🟡 | see (a) |
| **YourMT3+ / MuScriptor drums** | Part of the multi-instrument decoders | YourMT3+ `gm` vocab: 20 classes, **no clap**; MuScriptor: onset-only GM pitches (clap possible in vocab, untested) | YourMT3+ ENST 87.27; MuScriptor onset F1 41.6 | GPL / CC BY-NC | 🟡 / ✅ | — | — | — |
| **Noise-to-Notes** (Sony 2025) | Diffusion ADT with velocity | GM | E-GMD 89.7 | — | ❌ no code | — | ❌ | [arXiv](https://arxiv.org/abs/2509.21739) |
| **Separate-then-classify one-shots** (fallback) | Onsets from the drum stem (librosa / ADTOF) + an AudioSet tagger on 200 ms windows for "Clapping" vs "Snare drum" | any AudioSet class | none; PANNs missed claps here | — | ✅ | — | — | — |

## (d) Complete song-analyzer projects / toolkits

| name | what it does | outputs | accuracy evidence | license | Mac? | install | maintained? / wrappable? | link |
|---|---|---|---|---|---|---|---|---|
| **tsumugi** (as toolkit) | AMT + per-stem models + instrument refinement + velocity + **beat/chord/key head** (`beat_chord/cli/infer.py`, 87 MB `best_beat_chord_key.pth`) + own **stem-splitter** (350 MB, `--extra stem`) + Colab full workflow | multi-track MIDI with GM programs, drums ch. 10, velocities; beat grid / chords / key (not tested here) | see (a); beat/chord/key numbers unpublished | MIT | ✅ MPS | `uv sync --locked --extra stem` | ✅ active; clean CLI per stage → **easy to wrap** | [GitHub](https://github.com/anime-song/tsumugi) |
| **SheetSage2** (m-a-p, Sept 2026) | Beats, downbeats, key, chords, structure, vocal + instrumental melody from the mix → ABC/MIDI/events | annotation layer; **no per-instrument notes, no drums, no inventory** | SOTA on 12/15 annotation metrics (codifying doc) | CC BY-NC 4.0 | 🟡 CPU documented, MPS untested | `AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)` | ✅ new | [HF](https://huggingface.co/m-a-p/SheetSage2) |
| **MuScriptor** | multi-instrument MIDI + sheet music (MuseScore) + `serve` web UI | labelled MIDI, MusicXML | (a) | MIT / CC BY-NC | ✅ | `uvx muscriptor serve` | ✅ | (a) |
| **all-in-one (`allin1`)** | Beats, downbeats, tempo, segments + labels | structure only | 2023 paper | MIT | 🟡 needs NATTEN + madmom from git; **open PR (2026-09-20) "Run neighborhood attention in plain PyTorch on CPU and MPS; make NATTEN optional"** not merged | `pip install allin1` (1.1.0, Oct 2023) | ❌ last commit 2023-10; 851 stars; PRs unmerged | [GitHub](https://github.com/mir-aidj/all-in-one) |
| **madmom** | Beat/downbeat/tempo, chords, key, onsets | annotation layer | reference baselines | BSD + CC BY-NC-SA models | ✅ from git | `pip install git+https://github.com/CPJKU/madmom` (PyPI 0.16.1 is 2018) | ❌ last commit 2024-08 ("CI and NumPy compatibility") | [GitHub](https://github.com/CPJKU/madmom) |
| **Essentia** (`streaming_extractor_music` + TF models) | Low-level/tonal/rhythm descriptors, key, BPM, tags, instrument/mood/genre heads | descriptors, tags; **no notes** | model cards | AGPL-3.0 / CC BY-NC-SA models | ✅ arm64 wheels (see (b) caveat) | pip | ✅ pushed 2026-09-21 | [GitHub](https://github.com/MTG/essentia) |
| **Omnizart** | notes+instruments, drums, chords, beats, vocals | MIDI/CSV | 2021 | MIT | ❌ ARM | pip | 🟡 May 2026 compat release | (a) |
| **MIRFLEX** (AMAAI) | Wrapper over key CNN, BTC chords, BeatNet, Essentia taggers | annotation layer | — | MIT | ✅ | clone | ❌ pushed 2024-11 | [GitHub](https://github.com/AMAAI-Lab/mirflex) |
| **To-Sheet-Music-Skill** (Sept 2026) | Agent "skill": Demucs → (optional) Basic Pitch / librosa → LLM-authored 5-part band arrangement → MuseScore MIDI/PDF/MSCZ | arrangement, not transcription; instruments are assigned by design (2 guitars, bass, keys, drums) | none | MIT | ✅ CPU | Py 3.11 + FFmpeg + MuseScore Studio | 🟡 158 stars, v0.2.0 2026-09-13 | [GitHub](https://github.com/kiri603/To-Sheet-Music-Skill) |
| **NeuralNote**, **Melodfy**, **amt-apc** | Basic-Pitch plugin; piano-only converters; piano-cover generator | instrument-agnostic / piano | — | MIT | ✅ | — | 🟡 | [NeuralNote](https://github.com/DamRsn/NeuralNote) |
| Songle-like open tools | none found: Songle itself is a closed web service; the open equivalents of its layers are SheetSage2 (melody/chords/beats/structure) + tsumugi/MuScriptor (notes) | — | — | — | — | — | — | — |

Maintained and wrappable today: **tsumugi**, **MuScriptor**, **SheetSage2**,
Essentia. allin1 and madmom are frozen but installable; Omnizart is
effectively Linux/x86.

---

## Hands-on

Environment: `uv venv --python 3.11` venvs under the session scratchpad;
input `audio/test/river-30s.wav` (44.1 kHz stereo, 30 s; user: piano +
clapping + singing). Our own stems from `soundcode separate` were in
`out/stems/river-30s/`. **Disk was the binding constraint**: the data
volume started at 4.3 GiB free and ended at ~150 MiB (uv cache growth plus
the 562 MB YourMT3+ checkpoint); `uv cache prune`/`clean` reclaimed ~2 GB.
Nothing over ~1 GB was downloaded per project.

### tsumugi — worked on MPS

Install: `git clone --depth 1 https://github.com/anime-song/tsumugi &&
uv sync --locked --python 3.11` (~1 min; torch 2.13.0 macOS wheel). Every
checkpoint auto-downloaded from HF on first use (53–55 MB, ~5 s). All runs
`--device mps`; wall-clock includes model load:

| run | input | time | result |
|---|---|---|---|
| `default` (all instruments) | mix | 43 s incl. 53 MB download (≈7 s compute) | 172 notes, 8 tracks: **electric_piano 67** (F2–D#5), **melody (vocal) 62** (F4–C#5), **piano 22** (G#2–G#4), electric_bass 12 (F1–D#2), strings 6, organ/synth_bass/synth_pad 1 each |
| `drums_v1_5` | mix | 8.6 s incl. download | 32 hits: kick 28, **hand clap 4** |
| `drums_v1_5` | drums stem | 4.8 s | 197 hits: closed HH 71, snare 48, side stick 35, kick 20, **hand clap 16**, low floor tom 7 |
| `default` | piano stem | 4.8 s | 98 notes: piano 49, distorted_guitar 17, e-piano 12, strings 12, el. guitar clean 6, singletons |
| refinement `--stem-name piano --mode single` | piano stem | 6.7 s | **piano 0.971**, electric_piano 0.029, plucked_keyboard 0.00002 → all 98 notes relabelled piano |
| `default` | "guitar" stem (our separator's label) | 4.6 s | 164 notes: distorted_guitar 109, el. guitar clean 29, e-piano 14, strings 6, brass 2 … |
| refinement on "guitar" stem, `--stem-name piano` | | 3 s | **piano 0.731, electric_piano 0.269** |
| refinement on "guitar" stem, `--stem-name guitar` | | 3 s | distorted_guitar 0.968 (candidate set forced to guitars) |
| refinement on "guitar" stem, `--stem-name other` | | 3 s | brass 0.32, plucked_keyboard 0.19, strings 0.13 (nothing convincing) |
| `vocal_harmony_v1_5` | vocals stem | 7.5 s | 81 melody notes D#3–A#4, median 237 ms |
| `default` | other stem (−84 dBFS) | 4.6 s | 0 notes (silence gate) |

Observations: (1) on the mix the top labels are exactly the user's
description (keys + voice), with bass/strings as low-count bleed;
(2) the refinement probabilities are **softmax over the candidate set chosen
by `--stem-name`**, so a cross-family question ("guitar or piano?") has to be
asked by running it under both priors and comparing — done that way, the
"guitar" stem is a keyboard (piano 0.73 under the piano prior vs a diffuse
0.32 brass under the open prior); (3) claps appear as GM 39 on the drum stem,
mixed with snare/side-stick hits, which is the right vocabulary for
`:perc.drums`; (4) all notes carry velocity 100 unless the separate velocity
model is run (`instrument_agnostic_amt.velocity.cli.infer_velocity`, needs a
stems dir); (5) "Missing keys … interval_instrument_predictor" warnings on
load are benign (older checkpoints without the newer head).

### YourMT3+ (YPTF.MoE+Multi, noPS) — worked on CPU, 62 s

Install: `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/mimbres/YourMT3`,
download the 562 MB `last.ckpt` for `mc13_256_g4_all_v7_…_b36_nops`, venv
with torch 2.14 / lightning / transformers 4.45.1 / numpy 1.26.4 / einops /
mido / mir_eval + `wandb` (unlisted but imported), and `torchaudio.load`
monkey-patched with soundfile (torchaudio ≥ 2.9 delegates to torchcodec).
Model args copied from the Space's `app.py`; precision `32`; 8 CPU threads.
Load 2.5 s, **inference 59 s** for 30 s (≈ 0.5× real time on CPU; MPS not
attempted because the Space code hard-codes cuda/cpu).

Output (`model_output/river-30s-ymt3.mid`, 7 tracks):

| track | notes | range | median dur | note |
|---|---|---|---|---|
| Singing Voice | 101 | D#3–C#4 | 230 ms | plausible melody |
| **Acoustic Piano** | 38 | F1–C#5 | 1.9 s | the accompaniment, labelled piano |
| Strings (ens.) | 71 | F#2–C5 | **70 ms** | 70 ms "string" notes at piano onsets — instrument leakage / hallucination |
| Guitar (distortion) | 4 | F2–D#3 | 1.4 s | leakage |
| Bass | 2 | G#2 | 1.7 s | leakage |
| Drums | 177 hits | closed HH 39, snare 33, low floor tom 23, low tom 22, kick 17, ride 13, hi floor tom 8, crash 3, side stick 3, unknown pitch 82 ×14 | — | **no clap (not in vocab)**; claps land in snare/toms |

So YourMT3+ does say "piano" from the mix, but it pads the arrangement with
a phantom string ensemble and cannot express claps. It is usable as a
mix-level second opinion but not as a backbone (GPL, 2024, CPU-bound here).

### PANNs Cnn14 (AudioSet) — presence check, CPU, 4 s

2 s windows, 1 s hop, max over windows:

| input | Piano | Clapping | Guitar | Singing | Drum kit | top classes |
|---|---|---|---|---|---|---|
| mix | 0.00 | 0.00 | 0.06 | 0.55 | 0.04 | Music 0.75, Singing 0.28, Pop music 0.27, Music of Asia 0.26 |
| lead_vocals stem | 0.00 | 0.00 | 0.01 | 0.14 | 0.01 | Music, Speech, Singing |
| drums stem | 0.00 | 0.01 | 0.03 | 0.01 | 0.17 | Music 0.80, Electronic/House 0.1, Drum 0.06 |
| guitar stem | 0.02 | 0.00 | **0.74** | 0.02 | 0.01 | Guitar 0.56, Plucked string 0.48, Electric guitar 0.23 |
| piano stem | **0.03** | 0.00 | 0.31 | 0.01 | 0.01 | Music 0.72, Musical instrument 0.20, Brass 0.08 |
| bass / other stems (< −58 dBFS) | 0 | 0 | 0 | 0 | 0 | Silence |

AudioSet tagging does not see the piano or the claps in this recording at
all, and it agrees with the separator's wrong "guitar" label. Use taggers
only as a silence/energy prior per stem, not as the instrument inventory.

### MuScriptor — blocked (gating)

`curl https://huggingface.co/MuScriptor/muscriptor-medium/resolve/main/config.json`
→ HTTP 401; no HF token in `~/.cache/huggingface`, shell profile or
keychain. To run: create an HF account, accept the CC BY-NC terms on
`MuScriptor/muscriptor-{small,medium,large}`, `hf auth login`, then
`uvx muscriptor transcribe --model medium audio/test/river-30s.wav -o out.mid`
(medium ≈ 1.2 GB fp32 download; large ≈ 3 GB). The package itself installs
fine on Py 3.11 and selects MPS automatically.

### ADT_STR (tau-0.8) — worked on CPU (93 s); killed on MPS

A first attempt was aborted when free disk fell below 300 MB; after
reclaiming space (`uv cache prune`/`clean` freed ~2 GB, then macOS purged
several GB) it installed and ran:

```
uv venv --python 3.12 adtvenv && uv pip install --python adtvenv/bin/python \
  torch==2.8.0 torchaudio==2.8.0 "torchcodec>=0.7,<0.8" "transformers[torch]" \
  safetensors huggingface-hub pretty_midi mir_eval omegaconf soundfile pedalboard
python -c 'from huggingface_hub import snapshot_download as s; s("Pierfrancesco/adt-str", local_dir="adt-str", allow_patterns=["*.py","*.json","*.yaml","*.md","setting-tau-0.8/*"], ignore_patterns=["model.safetensors"])'
# the HF bundle lacks configs/: copy configs/ from the GitHub repo, then
python -c 'from adt_transcriber import ADTTranscriber as A; A.from_pretrained("adt-str", variant="setting-tau-0.8", device="cpu").transcribe("drums.wav", output_dir="out")'
```

Gotchas: `inference.py` expects `configs/config_default.yaml`, which is
only in the GitHub repo, not the HF snapshot; `select_inference_device()`
picks **MPS, where the process was SIGKILLed (exit 137) after ~80 s** twice
— pass `device="cpu"` (2.1 GB RSS). torch 2.8 pin → own venv (tsumugi pins
2.13). 30 s of drum stem = **93 s on CPU**; the 30 s mix = 131 s.

| input | hits | classes (GM pitch → count) | velocities |
|---|---|---|---|
| drums stem | 173 | kick 43, e-snare(40) 33, closed HH 32, **hand clap (39) 26**, snare(38) 20, crash2 10, low floor tom 5, vibraslap(58) 3 | per-note velocities (see comparison below) |
| full mix | 257 | china 57, snare 33, e-snare 30, kick 26, splash 21, tambourine 15, ride bell 10 … | nonsense — it is a drum-stem model; pitched content becomes cymbals |

On the drum stem ADT_STR and tsumugi agree on the picture (kick + snare +
closed hat + claps + a few toms). Kick onsets match to ~10 ms (2.93, 4.15,
4.87, 6.08, 6.81, 8.02 s in both). Claps do not: ADT_STR finds **26 claps
(vel 13–80) at 3.65 s and 13.3–17 s**, tsumugi **16 claps (vel 100, no
dynamics) at 6.3–11.4 s** — one of them is mislabelling snare/side-stick
hits as claps in a different section. Which one is right needs a listen,
but both put claps at GM 39, which is what `:perc.drums` needs, and ADT_STR
is the only one that also gives per-hit velocity.

Scratchpad footprint left behind (delete when done): `adtvenv` 0.9 GB,
`ymt3venv` 1.1 GB, `tsumugi` 0.9 GB, `yourmt3` 0.5 GB (562 MB ckpt),
`adt-str` 0.3 GB, plus `~/panns_data` 312 MB.

---

## RECOMMENDATION

**Adopt tsumugi as the note/drum/inventory backbone, per stem, behind our
existing separation; keep SheetSage2 (+ Beat This!) for the annotation
layer; add MuScriptor as the mix-level second opinion once the HF licence is
accepted.** Reasons: it is the only open project that (i) runs natively on
MPS with tiny checkpoints, (ii) labels notes with a 36-class taxonomy that
maps cleanly to GM, (iii) emits calibrated-looking instrument probabilities
per stem, (iv) has a drum model whose vocabulary includes hand clap, and (v)
is actively maintained under MIT. Its weakness is the absence of public
cross-dataset numbers; MuScriptor (47.8 multi-F1 on real recordings) is the
model to measure it against with `soundcode compare`, and YourMT3+ is not
worth keeping (GPL, unmaintained, < 10 % on non-main pop instruments).

Concrete pipeline (replaces basic-pitch, band-split drum onsets and PANNs;
keeps `separate`):

| `.sc` stream | source | mapping |
|---|---|---|
| `:notes.<inst>` per pitched stem | `tsumugi default` (or `bass_v2` / `guitar_v1_5` / `other_v1_5` chosen by stem) on `piano.wav`, `guitar.wav`, `bass.wav`, `other.wav`; then `instrument_refinement --mode cluster` | each MIDI track → one `:notes.<class>` stream; taxonomy class → GM program via `taxonomy/gm_instrument_classes.json`; drop tracks with < N notes or < X % of stem notes as bleed; run `velocity` model for dynamics |
| `:notes.vox` | `vocal_harmony_v1_5` on `vocals.wav` (`vocal` on lead only) | `melody` / `vocal_harmony` tracks → lead / backing voices; cross-check pitch classes with SheetSage2's vocal-melody voice as already planned |
| `:perc.drums` | `drums_v1_5` on `drums.wav`, cross-checked with **ADT_STR tau-0.8 on CPU** (agrees on kicks to ~10 ms, gives velocities; disagrees on clap timing) | GM pitch → our drum names; **39 → `clap`**, 37 → `stick`, 38/40 → `snare`, 42/44/46 → `hh`, 49/57 → `crash`, 51/53/59 → `ride`, toms → `tom`; velocity from ADT_STR or stem loudness (tsumugi drums have no velocity); emit `meta warn=` where the two disagree on clap vs snare |
| instrument inventory (with confidence) | refinement `--mode single` run under two or three `--stem-name` priors per stem + note count + stem loudness; BEATs/PANNs per stem only as a silence/energy gate | emit one line per stem: `class`, `p_within_prior`, `notes`, `dBFS`; write `meta warn=` when the top class under the *open* prior disagrees with the separator's stem label (this is exactly the river "guitar"-that-is-a-piano case) |
| `:grid`, `:harmony`, `:struct` | SheetSage2 / Beat This! as per the codifying doc; optionally compare with tsumugi's `beat_chord` head (untested here) | unchanged |
| `:text.vox`, `:contour.vox` | unchanged (Qwen3-ASR + aligner; torchcrepe/penn/SwiftF0) | unchanged |

Second opinions to wire into `compare`: MuScriptor medium on the full mix
(labelled notes without separation — the check against separation bleed
inventing parts; needs the HF licence click) and ADT_STR tau-0.8 on the drum
stem (already runs here on CPU, 93 s per 30 s; the Framework machine would
make it interactive). If either consistently beats tsumugi on the test
songs, promote it.

Things to verify before committing: tsumugi's beat/chord/key head quality;
whether its `stem-splitter` (350 MB) is better or worse than our
BS-RoFormer → Mel-RoFormer → HTDemucs chain on `compare`; clap precision (the
16 clap hits on the drum stem need listening against the 48 snare hits);
running MPS with `--amp` for speed on full songs; and the exact GM mapping in
`taxonomy/gm_instrument_classes.json`.

## Unverified / open

- tsumugi has no Slakh/MusicNet/URMP evaluation; its drum numbers are on a
  private set; single maintainer.
- MuScriptor's Slakh/URMP numbers are not in the paper HTML; its clap
  behaviour is untested; HF gating needs the user's account.
- ADT_STR per-class clap F1 and why MPS gets SIGKILLed (memory? torch 2.8
  MPS kernels?); which of ADT_STR / tsumugi has the clap timing right on
  river (they disagree by a section).
- FlexSED on CPU; BEATs checkpoint mirrors; Essentia `dev1438` arm64 wheels
  for cp311 (only cp314 published).
- Omnizart on Apple Silicon after the May 2026 release (README still says
  incompatible; not tried — sdist only, TF).
- allin1 MPS PR (2026-09-20) unmerged; not tried.
- TUTTI code link is a placeholder; MT2 and the TISMIR two-branch model have
  no code.
