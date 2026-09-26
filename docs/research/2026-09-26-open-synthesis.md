# Open synthesis stack research — 2026-09-26

Scope: open-source / open-weight tools that can turn `.sc` (notes, lyrics,
chords, structure, mix) back into audio. Hard constraints: nothing paid, no
hosted APIs, non-commercial licences acceptable (private research), primary
hardware M1 Max 32 GB / Python 3.11 / no CUDA. Where a tool is CUDA-only that
is called out as "rent an NVIDIA GPU".

Verification note: facts below were checked on 2026-09-26 against the linked
repos, model cards and papers. Items marked *(verify)* were not confirmed on a
primary page and should be re-checked before adoption. Release cadence in this
space is weeks, not months.

Column key for tables: **Mac?** = runs on Apple Silicon without CUDA
(✅ official, 🟡 community port or "should work on MPS/CPU, unverified",
❌ CUDA-only); **maturity** = 1–5 (5 = production-grade, actively maintained).

---

## 1. Singing voice synthesis (SVS) from notes + lyrics

What `.sc` can supply: `:notes.vox` (pitch in cents, onset, duration,
velocity), `:text.vox` (word- or syllable-aligned lyrics), `:contour.vox`
(vibrato / scoop / fall / bend events), `:grid` (tempo). Two families exist:

- **Voicebank SVS** (DiffSinger, NNSVS, VISinger2): deterministic, score-native
  (phoneme + note + duration, optional explicit F0 curve), one voice per
  trained model. Best fidelity to the score; needs a voicebank.
- **Zero-shot SVS** (SoulX-Singer, YingMusic-Singer, Vevo2, TCSinger2):
  timbre from a 5–30 s reference clip; melody from MIDI or an F0 contour.
  Newer, higher naturalness, mostly CUDA-first.

| name | what it does | input / conditioning | license | Mac? | install | maturity | link |
|---|---|---|---|---|---|---|---|
| **DiffSinger (openvpi fork)** | Diffusion / rectified-flow acoustic + variance models, 44.1 kHz. De-facto community SVS standard. | `.ds` JSON: phoneme seq + phoneme durations + note seq/durations + slur flags; optional explicit `f0_seq` (pitch curve), energy, breathiness, tension. Variance model predicts pitch/duration when absent. Languages per voicebank (zh, ja, en, ko, ru, …). | Code Apache 2.0. Community vocoders (NSF-HiFiGAN, PC-NSF-HiFiGAN) **CC BY-NC-SA 4.0**. Voicebanks: each has its own terms. | ✅ inference via OpenUtau arm64 or `scripts/infer.py` on CPU/MPS; training needs CUDA (Colab) | `git clone openvpi/DiffSinger`, pip; models from voicebank authors; ONNX export for OpenUtau | 5 | [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger), [vocoders releases](https://github.com/openvpi/vocoders/releases) |
| **OpenUtau** | Editor / renderer front end for DiffSinger, ENUNU (NNSVS) and classic UTAU banks. USTX project files, phonemizers for English (ARPAsing, DiffSinger EN). | Notes + lyrics in USTX; pitch curves editable; expressions (vibrato etc.) per note. | MIT | ✅ `OpenUtau-osx-arm64.dmg` (macOS 11+ for DiffSinger; .NET 8 from 0.1.549). Mac render is CPU via onnxruntime *(verify: no CoreML EP)* | dmg download | 5 | [openutau.com](https://www.openutau.com/), [install wiki](https://github.com/openutau/OpenUtau/wiki/Install) |
| **NNSVS** (+ ENUNU) | Research SVS library, PyTorch, Kaldi-style recipes; 8+ community languages. | HTS full-context labels from MusicXML (pysinsy) or UST via ENUNU plugin in OpenUtau. Explicit F0 possible via label features. | MIT | ✅ pure PyTorch, CPU inference fine | `pip install nnsvs` (v0.1.0) | 3 (stable, slower cadence than DiffSinger) | [nnsvs/nnsvs](https://github.com/nnsvs/nnsvs), [docs](https://nnsvs.github.io/) |
| **VISinger2** | End-to-end VITS + DDSP synth SVS. | Phonemes + MIDI notes + durations (Opencpop format); Mandarin checkpoint. | not stated in README *(verify)* | 🟡 PyTorch, untested | clone + Google Drive checkpoint | 2 (last update 2023) | [zhangyongmao/VISinger2](https://github.com/zhangyongmao/VISinger2) |
| **SoulX-Singer** | Zero-shot SVS, 42k h training data; zh / en / Cantonese. Released 2026-02-06; SVC model added 2026-03-16. | Lyrics + **MIDI score (note + duration)** *or* **F0 melody contour** + reference vocal for timbre. Provided MIDI editor for alignment fixes. | **Apache 2.0** code + weights | 🟡 PyTorch, Python 3.10; CUDA documented; MPS/CPU unverified | conda + `pip -r requirements.txt` + HF download | 4 | [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer), [HF](https://huggingface.co/Soul-AILab/SoulX-Singer), [paper](https://arxiv.org/abs/2602.07803) |
| **YingMusic-Singer** (GiantAILab) | Zero-shot SVS + lyric editing with annotation-free melody guidance; zh/en. V1 2026-02-09. | Lyrics + 5–7 s reference + melody from **MIDI file or audio**; no phoneme alignment needed. | MIT (code); weights on HF/ModelScope | 🟡 PyTorch 2.9 + CUDA 12.6 documented | conda + pip | 3 | [GiantAILab/YingMusic-Singer](https://github.com/GiantAILab/YingMusic-Singer), [paper](https://arxiv.org/abs/2512.04779) |
| **Vevo2** (Amphion, 2026-03-25) | Unified speech + singing: TTS, text-to-singing, SVS, SVC, humming-to-singing, instrument-to-singing, melody control via prosody tokens. 6 languages. | Text + reference audio for prosody / style / timbre; melody as audio (hum, instrument) rather than MIDI. | Code MIT; **weights CC BY-NC-ND 4.0** | 🟡 PyTorch, untested | Amphion repo + HF | 3 | [Amphion vevo2](https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevo2/README.md), [HF amphion/Vevo2](https://huggingface.co/amphion/Vevo2) |
| **TCSinger 2** (ACL 2025) | Zero-shot multilingual SVS with style transfer / style control. | Phonemes + MIDI (`ep_pitches`, `ep_notedurs`) + style prompt (audio or text). Train on GTSinger (80 h, 9 languages). | MIT; **no public checkpoints noted** | ❌ CUDA required | conda | 2 (research code) | [AaronZ345/TCSinger2](https://github.com/aaronz345/tcsinger2), [GTSinger](https://github.com/AaronZ345/GTSinger) |
| **TechSinger** (AAAI 2025) | Technique-controllable SVS (mixed voice, falsetto, breathy…). | Phonemes + MIDI + technique labels. | MIT *(verify; repo fetch returned 404 today)* | ❌ | conda | 2 | [AaronZ345](https://github.com/AaronZ345) |
| **Prompt-Singer** (NAACL 2024) | SVS with natural-language control of gender, volume, range. | Phonemes + notes + text prompt (FLAN-T5). Mandarin datasets. | not stated *(verify)* | ❌ | clone + checkpoints | 2 | [cyanbx/Prompt-Singer](https://github.com/cyanbx/Prompt-Singer) |
| Watch list (papers, code status unknown) | CoMelSinger (token-based zero-shot SVS with structured melody control), VocalRender ("score-native SVS for real-world composition", 2607.27768), CLASVS (melody-preserving lyric editing), "Synthetic Singers" survey (Jan 2026). | — | — | — | — | — | [CoMelSinger](https://arxiv.org/pdf/2509.19883), [VocalRender](https://arxiv.org/pdf/2607.27768), [CLASVS](https://arxiv.org/pdf/2608.03253), [survey](https://arxiv.org/pdf/2601.13910) |

### Voicebanks for DiffSinger (English)

Free English DiffSinger banks named by the community: Raine Rena, Hoshino Hanami,
Nishiren Gard, Ameko Kero, Hareford
([UtaForum thread](https://utaforum.net/threads/looking-for-english-diffsinger-ai-voicebanks.25396/)),
plus the **LUNAI Project** banks ("completely ethical", EN/RU/JA; ships its own
OpenUtau fork with an Apple Silicon dmg)
([lunaiproject.github.io](https://lunaiproject.github.io/),
[OpenUtau-lunai](https://github.com/keirokeer/OpenUtau-DiffSinger-Lunai)).
The [DiffSinger wiki voicebank category](https://diffsinger.miraheze.org/wiki/Category:DiffSinger_voicebanks)
is the running index. Every bank carries its own terms (usually free for
non-commercial), and all of them depend on the CC BY-NC-SA vocoder, so the
whole DiffSinger vocal path is NC — acceptable here.

### Recommendation (SVS)

**Default: DiffSinger (openvpi) driven directly from `.ds` files**, with an
English voicebank (LUNAI or a UtaForum bank), rendered on the Mac CPU. Reasons:

1. The `.ds` input is essentially `.sc` already: phoneme sequence + phoneme
   durations + note sequence + note durations + slur flags, **plus an optional
   explicit `f0_seq`** — so `:notes.vox` + `:text.vox` (syllable-aligned via
   a G2P/phonemizer) + `:contour.vox` (rendered to an F0 curve) map 1:1.
   Nothing else in this table lets you dictate the pitch curve sample by sample.
2. Apache 2.0 code, runs on Mac today, scriptable (`scripts/infer.py`), no
   GPU needed for inference.
3. Weakness: one voice per bank, English banks are hobbyist quality, and the
   natural-sounding "performance" (breath, dynamics) is only as good as the
   variance model. Use OpenUtau interactively to debug alignment.

**Alternative for realism / arbitrary voice: SoulX-Singer** (Apache 2.0,
zero-shot, takes MIDI score *or* F0 contour + lyrics). It is the strongest
open zero-shot SVS as of Q1 2026 and the licence is clean. Plan to run it on
a rented GPU first; try MPS once the pipeline is proven. YingMusic-Singer
(MIT) is the second zero-shot option, and Vevo2 the most flexible but NC-ND.

**Not recommended:** TCSinger2 / TechSinger / Prompt-Singer (no checkpoints or
Chinese-only, research code), VISinger2 (stale, Mandarin only).

---

## 2. Singing voice conversion / cloning (re-sing an existing vocal)

Use case: render `:notes.vox` through DiffSinger (or any voice), then convert
the timbre to a target voice while keeping melody and timing; or convert a
separated lead-vocal stem directly.

| name | what it does | input / conditioning | license | Mac? | install | maturity | link |
|---|---|---|---|---|---|---|---|
| **Seed-VC** | Zero-shot VC + SVC + real-time. V1 whisper-base 200M @ 44.1 kHz is the singing model; V2 (67M CFM + 90M AR) does voice + accent. | Source vocal + 1–30 s reference; F0 conditioning + `semi-tone-shift`; 30–50 diffusion steps for singing. Fine-tune: ≥1 utterance/speaker, ~2 min on T4, 100+ steps. | GPL-3.0 *(verify on repo LICENSE)* | ✅ "Mac M series support" since 2025-03 | `pip install seed-vc` or clone | 4 | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) |
| **YingMusic-SVC** | Zero-shot SVC built on Seed-VC + robust SFT + Flow-GRPO RL; singing-trained timbre shifter, F0-aware adaptor. Includes its own accompaniment separator. | Source vocal (or full mix via built-in separator) + reference clip; zh/en. | MIT (code); checkpoint on HF | 🟡 PyTorch 3.10; CUDA documented | conda + ffmpeg + sox | 3 (released 2025-11-25) | [GiantAILab/YingMusic-SVC](https://github.com/GiantAILab/YingMusic-SVC), [paper](https://arxiv.org/pdf/2512.04793) |
| **SoulX-Singer SVC model** | SVC head of SoulX-Singer (2026-03-16). | Source vocal + reference. | Apache 2.0 | 🟡 | same repo | 3 | [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) |
| **RVC** (Retrieval-based VC) | Trained-speaker VC, the 2023 standard; huge model zoo. | Needs a **trained model per voice** (typically ≥10 min clean vocals; RMVPE F0). | MIT | 🟡 inference via Mac forks (RVC-WebUI-MacOS); **training on Apple Silicon is not supported upstream** (issue #767) | community forks / Pinokio | 3 (original unmaintained) | [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), [RVC-WebUI-MacOS](https://github.com/NevilPatel01/RVC-WebUI-MacOS) |
| **Applio** | Maintained RVC distribution; realtime tab in v3.5. Now in maintenance mode (security / deps). | Same as RVC. | MIT | 🟡 Linux/macOS install documented (Python 3.10); Mac training unclear | venv + script | 4 | [IAHispano/Applio](https://github.com/iahispano/Applio), [docs](https://docs.applio.org/getting-started/installation/) |
| **DDSP-SVC 6.3** | Lightweight SVC (DDSP + reflow); much lower train/infer cost than so-vits-svc. | Train: ~1000 clips ≥2 s (recommended); ContentVec / HubertSoft; RMVPE F0. Real-time GUI. | MIT | 🟡 PyTorch; MPS not documented | clone + pretrained encoders | 3 | [yxlllc/DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) |
| **so-vits-svc / so-vits-svc-fork** | SoftVC VITS SVC. | Trained speaker model (tens of minutes). | MIT / fork MIT | 🟡 fork has `--device mps` ("probably supported") | `pip install so-vits-svc-fork` | 2 (2023-era, low activity) | [voicepaw/so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork) |
| **Vevo2 (SVC / singing style conversion)** | Zero-shot SVC + style conversion. | Source + reference audio. | Code MIT; weights CC BY-NC-ND 4.0 | 🟡 | Amphion | 3 | [Amphion](https://github.com/open-mmlab/Amphion) |
| Research (code varies) | HQ-SVC (AAAI 2026, low-resource zero-shot), SaMoye (zero-shot incl. non-human timbre), R2-SVC, InvoxSVC. | — | — | — | — | 1–2 | [HQ-SVC](https://arxiv.org/pdf/2511.08496), [SaMoye](https://arxiv.org/pdf/2407.07728), [R2-SVC](https://arxiv.org/pdf/2510.20677) |

### Recommendation (SVC)

**Default: Seed-VC (singing model, V1 whisper-base 44.1 kHz)** — zero-shot,
official Apple Silicon support, explicit F0 conditioning and semitone shift so
melody and timing survive, and fine-tuning from a single utterance in minutes
if a target voice needs more identity. The `.sc` pipeline stays
training-free: DiffSinger render → Seed-VC → target voice.

**Alternative: YingMusic-SVC** (MIT, built on Seed-VC, better robustness to
harmony bleed and F0 errors on *real* separated stems — relevant when
converting an encoded song's actual vocal stem). CUDA first; MPS to be tried.

**Trained-model path (only if a specific voice must be nailed):** RVC via
Applio, trained on a rented GPU with ≥10 min of clean vocals, inference on the
Mac. Skip so-vits-svc; DDSP-SVC only if training on the Mac itself is
required.

---

## 3. Open full-song generators (music + vocals)

The question that matters for Infinity Engine: **which of these accept
symbolic structure** (melody, chords, BPM, key, section timing, timed lyrics)
so that `.sc` → song stays faithful?

| name | what it does | input / conditioning (symbolic control in **bold**) | license | Mac? | install | maturity | link |
|---|---|---|---|---|---|---|---|
| **ACE-Step 1.5** (installed) | Qwen3 LM planner + DiT (2B; **XL 4B since 2026-04-02**). text2music, cover, repaint, lego (add tracks), extract (stems), complete, vocal→BGM, LRC output. 50+ languages; 10 s–10 min. | Lyrics with structure tags (`[verse]`…; **timestamps must be stripped**), **BPM, key/scale, time signature, duration**, `reference_audio` (global timbre/style), `src_audio` + `audio_cover_strength` 0–1 (cover: melodic/structural control via semantic codes), `repainting_start/end` (local edit), LoRA from ~8 songs (~1 h on RTX 3090). | **MIT** (code + weights) | ✅ official MLX/MPS scripts + portable macOS package | done (`external/ACE-Step-1.5`) | 5 | [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5), [paper](https://arxiv.org/html/2602.00744v1) |
| **DiffSynth-Music** (Alibaba / ModelScope, Sept 2026) | Audio-conditioned KV-cache adapters **on top of ACE-Step 1.5**. Three templates: Control (**beats** click-track → BPM alignment; **vocals** → generate accompaniment; **accompaniment** → generate vocals), Prosody (**resynthesised vocal preserving pitch + timing** → new vocals), Reference (style). Composable. | Audio controls (a click track, a vocal or instrumental stem, a pitch-and-timing-only vocal) + lyrics + text. | **Apache 2.0** | 🟡 CUDA documented (low-VRAM offload modes); PyTorch so MPS is plausible, unverified | `git clone DiffSynth-Studio && pip install -e .[audio]` | 3 (new) | [HF model card](https://huggingface.co/DiffSynth-Studio/DiffSynth-Music), [paper](https://arxiv.org/abs/2609.12774), [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio) |
| **MuLaCover** (HeartMuLa / MuLa Labs, 2026-09-16) | Controllable cover / remix: injects a **symbolic lead sheet** into a pretrained text-to-song backbone via gated adaptive cross-attention. | Path A: **melody MIDI + chord MIDI (+ optional drum MIDI)** + lyrics (`[Verse]`/`[Chorus]`) + style fields (`topic/genre/instrument/mood`). Path B: reference audio → auto-transcribed (YourMT3 + ChordNet). MIDI read in beats; tempo carried but not a separate condition. | Code Apache 2.0; **weights CC BY-NC 4.0 + MODEL_LICENSE; outputs non-commercial** | ❌ Linux + NVIDIA only (tested PyTorch 2.10 / CUDA 13 / B300) | conda, Python 3.10 | 3 (new) | [HeartMuLa/MuLaCover](https://github.com/HeartMuLa/MuLaCover), [HF](https://huggingface.co/HeartMuLa/MuLaCover) |
| **YuE2-3B** (m-a-p, v0.1.6 2026-09-09; skill 1.2.0 2026-09-25) | AR/NAR mixture-of-transformers; "symbolic planning": generates or accepts an **editable ABC score (melody + chords)** before rendering; zero-shot covers via SheetSage2 transcription; instrumental generation and instrumental covers; agentic editing. Best-of-8 tops WildSongBench/SongBench among open models. | Lyrics + style prompt; **ABC score (supplied or edited)**, `cot="melody"` cover mode from reference audio; sections. | Code Apache 2.0; **weights CC BY-NC 4.0** (+ creator permission for outputs) | 🟡 official: NVIDIA ≥24 GB, Python 3.12. **MLX ports run on 32 GB Macs**: mlx-Yue (Torch-free, ~10–11 GiB peak, ~187 s for 32-step synthesis on M3 Max, 8-step fast mode faster than realtime; includes SheetSage2+MERT2 transcription in MLX), yue2-mlx-swift, YuE Studio | official: `pip install .`; Mac: `uv sync` in mlx-Yue | 4 | [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE), [releases](https://github.com/multimodal-art-projection/YuE/releases), [mlx-Yue](https://github.com/vanch007/mlx-Yue), [yue2-mlx-swift](https://github.com/VincentGourbin/yue2-mlx-swift) |
| **HeartMuLa-oss-3B** (2026-01-14; "happy-new-year" 2026-02-13; RL variant) | Music LM over HeartCodec (12.5 Hz); strong lyric controllability; 7B planned. | Lyrics with section markers + comma tags. **No reference audio / fine control yet** (listed TODO). | **Apache 2.0** (code + weights, since 2026-01-20) | 🟡 community **heartlib-mlx**: ~11 GB for 1 min, ~15 GB for 5 min, 2× faster than PyTorch-MPS on M2 Max; 32 GB+ recommended | `pip install -e .`; Mac: heartlib-mlx `uv run` | 4 | [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib), [heartlib-mlx](https://github.com/Acelogic/heartlib-mlx) |
| **LeVo 2 / SongGeneration-v2-large** (Tencent AI Lab, 2026-03-01) | Hybrid LLM-diffusion, 4B; up to 4m30s; PER 8.55 %; zh/en/es/ja. Quality rivals closed models. | Lyrics with `[Verse]`/`[Chorus]` + description/tags + 10 s prompt audio. No symbolic control. | **Custom Tencent terms** (GitHub shows NOASSERTION) — review before use | ❌ CUDA (22–28 GB for v2-large per vllm-omni issue; a blog claims 10/16 GB with offload) | clone + HF | 4 | [tencent-ailab/SongGeneration](https://github.com/tencent-ailab/SongGeneration), [vllm-omni issue #3390](https://github.com/vllm-project/vllm-omni/issues/3390), [LeVo 2 paper](https://arxiv.org/pdf/2606.30642) |
| **DiffRhythm 2** (ASLP, 2025-10-30) | Semi-autoregressive block flow matching; full-length; "precise lyric alignment". | Lyrics + reference audio (style); espeak-ng phonemes. (DiffRhythm v1 accepted **LRC-timestamped lyrics** *(verify)*; v2 docs show plain lyrics.) | **Apache 2.0** | 🟡 Linux scripts; PyTorch | `pip -r requirements.txt` + brew espeak-ng | 3 | [ASLP-lab/DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2), [paper](https://arxiv.org/pdf/2510.22950) |
| **SongBloom** (2025) | Interleaved AR sketching + diffusion refinement; 1.3B (60 s) / 2B (150 s) @ 48 kHz. | Lyrics + 10 s prompt wav. | **Apache 2.0** | 🟡 PyTorch, CUDA docs | clone + HF | 3 | [Cypress-Yang/SongBloom](https://github.com/Cypress-Yang/SongBloom) |
| **JAM-0.5** (declare-lab, 2025-07-29) | 530M rectified-flow song generator with **word- and phoneme-level timing control**; up to 3:50. | Lyrics as JSON `{"start","end","word"}` (**timed lyrics**), reference audio for style, optional text. | **Non-commercial** (Project Jamify licence + Stability AI Community License) | ❌ "CUDA GPU 8 GB+" | script install | 3 | [declare-lab/jamify](https://github.com/declare-lab/jamify), [HF](https://huggingface.co/declare-lab/JAM-0.5) |
| Qwen-Music (2026-07-12) | Full songs with sung lyrics. | — | **weights not released** | — | — | — | [report](https://arxiv.org/pdf/2607.11699) |
| Papers to watch | SongEcho (cover generation), MPEcho (melody+phoneme-aware cover), SketchSong (sketch planning + multi-track), WanSong v1.0, "MIDI-informed singing accompaniment generation". Code not confirmed. | — | — | — | — | — | [SongEcho](https://arxiv.org/pdf/2602.19976), [MPEcho](https://www.alphaxiv.org/abs/2607.26698), [SketchSong](https://arxiv.org/pdf/2606.03169), [2602.22029](https://arxiv.org/pdf/2602.22029) |

### Ranking

**By controllability from `.sc`** (what can be dictated, not just suggested):

1. **MuLaCover** — melody MIDI + chord MIDI + drums MIDI + lyrics. The only
   model that takes the lead sheet directly. `.sc` `:notes.vox`, `:harmony`,
   `:perc.drums` → MIDI is a straight export. CUDA-only, NC weights.
2. **YuE2** — editable ABC score (melody + chords) as the planning stage, plus
   sections and style; cover mode from reference audio. `.sc` → ABC is a small
   converter. NC weights, but runs on the Mac via MLX ports.
3. **DiffSynth-Music** — beat click track + a pitch/timing-only vocal (our
   mock render) + a stem → ACE-Step-quality output that follows them. Apache
   2.0. Not symbolic, but our mock renderer already produces exactly the
   "prosody" audio it wants.
4. **ACE-Step 1.5** — BPM / key / meter / duration metadata, lyrics with
   sections, `src_audio` cover with `audio_cover_strength`, repaint windows.
   No timed lyrics, no MIDI. The current plan (mock render → cover) is sound;
   controllability is indirect.
5. **JAM** — the only one with word-level lyric timestamps as input, but small,
   NC, CUDA, and no melody control.
6. LeVo 2, HeartMuLa, DiffRhythm 2, SongBloom — lyrics + style/reference only.

**By quality (open, as of Sept 2026):** YuE2-3B (best-of-N) ≈ LeVo 2 ≈
ACE-Step 1.5 XL > HeartMuLa-oss-3B > DiffRhythm 2 > SongBloom > JAM. YuE2's
SongBench 6.96 (best-of-8) and LeVo 2's PER 8.55 % are the headline numbers;
ACE-Step 1.5 XL is the most versatile editor.

**By Mac viability:** ACE-Step 1.5 (official MLX) > YuE2 (mature MLX ports,
fits 32 GB) ≈ HeartMuLa (MLX port, ~11–15 GB) > DiffSynth-Music / DiffRhythm 2
/ SongBloom (PyTorch, untested on MPS) > MuLaCover / LeVo 2 / JAM (rent a GPU).

### Recommendation (full-song)

**Default: keep ACE-Step 1.5 as the local decoder, and add DiffSynth-Music as
the first experiment** — it is Apache 2.0, sits on the ACE-Step 1.5 weights we
already have, and its *prosody* (pitch+timing vocal), *vocals→accompaniment*
and *beats* templates are precisely the control signals the `.sc` mock
renderer emits. Try it on MPS; fall back to a rented GPU.

**Symbolic-faithful experiments (rented GPU): MuLaCover first, YuE2 second.**
MuLaCover is the direct `.sc`-to-lead-sheet test; YuE2's ABC plan is the
second and has the advantage of Mac ports for iteration once the converter
works.

**Alternative Mac-local generator: YuE2 via mlx-Yue** (NC) when a song needs
stronger melody adherence than an ACE-Step cover gives; **HeartMuLa** (Apache
2.0) when licence hygiene matters more than control.

Skip for now: LeVo 2 (custom licence, heavy, no control), JAM (NC, weak
quality; revisit only for lyric timing experiments), SongBloom / DiffRhythm 2
(no control advantage over ACE-Step).

---

## 4. Open instrument-only / accompaniment generators with symbolic conditioning

| name | what it does | input / conditioning | license | Mac? | install | maturity | link |
|---|---|---|---|---|---|---|---|
| **Stable Audio 3.0 Small / Medium** (Stability AI, 2026) | Fast latent diffusion, licensed training data. Small 433M (120 s, CPU/MLX/CoreML/TFLite), Medium ~1.4–2B (380 s). Large is API-only. | Text (BPM/genre in prose), duration, `init_audio` audio-to-audio with noise level, inpainting/continuation masks. **No MIDI/chord/melody conditioning.** LoRA training extra. | Stability AI Community License (free < $1M revenue; outputs yours) + Gemma terms for T5Gemma encoder | ✅ Small: CPU/MLX/CoreML on Apple Silicon; Medium: CUDA (Flash-Attn 2) — model card claims "a few seconds on M4 MacBook Pro" *(verify for Medium)* | `uv` install per repo | 4 | [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3), [HF medium](https://huggingface.co/stabilityai/stable-audio-3-medium), [paper](https://arxiv.org/html/2605.17991v1) |
| **Stable Audio Open 1.0** | 1B DiT, 47 s, 44.1 kHz; the fine-tuning base for the two rows below. | Text + `seconds_start/total`. | Stability AI Community License | 🟡 stable-audio-tools on MPS/CPU (slow) | `pip install stable-audio-tools` | 4 | [HF](https://huggingface.co/stabilityai/stable-audio-open-1.0) |
| **MuseControlLite** (ICML 2025) | 85M-parameter adapters on Stable Audio Open: **melody, dynamics, rhythm** and audio in/out-painting, any combination. | Time-varying attribute curves extracted from reference audio (melody = chroma-like salience; script provided) + text. Checkpoints trained on MTG-Jamendo (mostly electronic). | **MIT**; checkpoints on HF (2026-01-02) | 🟡 Python 3.11, Linux instructions; PyTorch | conda + HF login | 3 | [fundwotsai2001/MuseControlLite](https://github.com/fundwotsai2001/MuseControlLite), [paper](https://arxiv.org/abs/2506.18729) |
| **stable-audio-controlnet** | DiT ControlNet fine-tune of Stable Audio Open for audio conditioning. | Audio control signal. | see repo | 🟡 | clone | 2 | [EmilianPostolache/stable-audio-controlnet](https://github.com/EmilianPostolache/stable-audio-controlnet) |
| **JASCO** (audiocraft) | Flow-matching text-to-music with **chords + melody + drums** local controls; 400M / 1B; **10 s clips**. | Chords as `[(name, time)]`, melody as salience matrix (Deepsalience), drum stem audio, text. | Code MIT; **weights CC BY-NC 4.0** | 🟡 audiocraft has no Mac support; PyTorch, Python 3.9 | `pip install audiocraft` | 3 (10 s limit is the blocker) | [JASCO.md](https://github.com/facebookresearch/audiocraft/blob/main/docs/JASCO.md), [paper](https://arxiv.org/pdf/2406.10970) |
| **MusicGen-melody / -large** (audiocraft) | 1.5B LM; 30 s windows with continuation. | Text + **melody as chromagram** from reference audio (so a mock render of `:notes.*` works as the melody source). Community `musicgen-chord` adds chord-text conditioning. | Code MIT; weights CC BY-NC 4.0 | 🟡 runs on CPU/MPS with patches (community) | `pip install audiocraft` | 4 (dated) | [audiocraft](https://github.com/facebookresearch/audiocraft), [musicgen-chord](https://github.com/sakemin/cog-musicgen-chord) |
| **Magenta RealTime 2** (Google, 2026-06-04) | Live, chunked music model; 230M (any Apple Silicon) and 2.4B (Pro/Max chips); 48 kHz stereo; ~200 ms control latency; ~71k h mostly instrumental. | Text, audio examples, **MIDI note control** (jam example app); offline inference in Python. | Code Apache 2.0; **weights CC BY 4.0** | ✅ native MLX (`magenta-rt[mlx]`) | `uv` + `mrt models download` | 4 | [magenta/magenta-realtime](https://github.com/magenta/magenta-realtime), [HF](https://huggingface.co/google/magenta-realtime) |
| **DiffSynth-Music (vocals→accompaniment, beats)** | See §3; the *Control* template generates accompaniment from a vocal stem and locks tempo to a click track. | Vocal audio + click track + text. | Apache 2.0 | 🟡 | DiffSynth-Studio | 3 | [HF](https://huggingface.co/DiffSynth-Studio/DiffSynth-Music) |
| **ACE-Step 1.5 lego / instrumental** | `lego` adds a track to existing audio in a time window; instrumental text2music with BPM/key/meter. | Audio + metadata + caption. | MIT | ✅ | installed | 5 | [ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) |
| Research, no code found | Break-the-Beat (drum MIDI → drum audio with reference timbre, ICASSP 2026), P-MUSE (prompt+MIDI instrumental synthesis/editing; **explicitly not open-sourced**), seconds-aligned latent drum rendering, pitch-class steering for diffusion models. | — | — | — | — | — | [Break-the-Beat](https://arxiv.org/abs/2605.14555), [P-MUSE](https://arxiv.org/abs/2608.01920) ([eval only](https://github.com/FEAfeatherTHER/P-MUSE-eval)), [2605.13404](https://arxiv.org/pdf/2605.13404), [2609.04516](https://arxiv.org/pdf/2609.04516) |

### Recommendation (instrumental generation)

**Default: ACE-Step 1.5 (instrumental / lego) and DiffSynth-Music's
beats + vocals templates** — same weights family, MIT/Apache, Mac-local. For
"give me a bed that follows these chords and this melody", render the `.sc`
mock to audio and use it as `src_audio` (ACE cover) or as the *vocals*
control (DiffSynth-Music).

**Alternatives:** **MuseControlLite** (MIT) when a melody/rhythm/dynamics
curve must be followed explicitly on a ≤47 s section; **Magenta RealTime 2**
(CC BY 4.0, native MLX) for live jamming / MIDI-steered textures and for
`:texture.*` beds; **Stable Audio 3 Small** for CPU-only SFX/texture fills.
JASCO is the most symbolic (chords + melody + drums) but its 10 s window and
NC weights make it a research toy for this project.

---

## 5. Faithful instrument rendering from notes (non-generative)

Goal: play `:notes.*` and `:perc.drums` back through realistic sounds,
deterministically, on the Mac. Three tiers: SoundFont (SF2), SFZ sample
libraries, and hosted plugins (VST3/AU). Plus neural MIDI-to-audio.

### 5a. Engines and hosts

| name | what it does | input / conditioning | license | Mac? | install | maturity | link |
|---|---|---|---|---|---|---|---|
| **FluidSynth** (+ pyfluidsynth) | Reference SF2/SF3 synthesiser; offline render via `fluidsynth -F`. | MIDI file or API note events; SF2 bank. | LGPL 2.1 | ✅ `brew install fluid-synth` | `pip install pyfluidsynth` | 5 | [fluidsynth.org](https://www.fluidsynth.org/) |
| **tinysoundfont** (Python) | Self-contained SF2/SF3/SFO synth with MIDI playback and offline generation; no system deps. v0.3.7 (2025-06-03). | MIDI + SF2. | MIT | ✅ arm64 wheels, CPython 3.7–3.12 | `pip install tinysoundfont` | 4 | [PyPI](https://pypi.org/project/tinysoundfont/) |
| **sfizz** 1.2.3 (+ pysfizz) | SFZ engine (library + VST3/AU/LV2/standalone in sfizz-ui). **Repos archived 2026-06-21** (read-only, still works). Fork **sfizioso** continues (libs BSD-2, app AGPL-3). | SFZ instruments; MIDI. | BSD-2 (library); ISC for LV2 build | ✅ macOS universal builds; pysfizz 0.1.2 (2026-01-01) | brew / releases; `pip install pysfizz` *(verify wheel)* | 4 (frozen) | [sfizz releases](https://github.com/sfztools/sfizz/releases), [sfizz-ui](https://github.com/sfztools/sfizz-ui/releases), [sfizioso](https://github.com/rullopat/sfizioso-player), [pysfizz](https://libraries.io/pypi/pysfizz) |
| **DawDreamer** | Python DAW: hosts VST3 (and VST2) instruments/effects, MIDI in PPQN or seconds, audio-rate parameter automation, FAUST instruments, samplers, multi-track graphs, offline render. | MIDI events + plugin state. | GPLv3 | ✅ arm64 wheels, Python 3.11–3.14, macOS 11+ (import before JAX/LLVM libs) | `pip install dawdreamer` | 4 | [DBraun/DawDreamer](https://github.com/dbraun/DawDreamer), [plugin compatibility](https://github.com/DBraun/DawDreamer/wiki/Plugin-Compatibility) |
| **pedalboard** 0.9.25 (Spotify) | Effects/instrument host; **VST3 and Audio Unit instrument plugins accept MIDI for rendering since 0.7.4**; simpler API than DawDreamer, no automation graph. | MIDI messages + duration + sample rate → audio. | GPLv3 | ✅ arm64 wheels, Python 3.10–3.15 | `pip install pedalboard` | 5 | [docs](https://spotify.github.io/pedalboard/), [FAQ](https://spotify.github.io/pedalboard/faq.html) |

### 5b. Free sound banks

| name | what it does | format / coverage | license | Mac? | install | maturity | link |
|---|---|---|---|---|---|---|---|
| **FluidR3_GM** | Full GM bank, ~141 MB; the default in most Linux/MuseScore setups. | SF2, 128 GM + drums | MIT | ✅ | download (archive.org / distro) | 5 | [miditoolbox list](https://miditoolbox.com/posts/best-free-general-midi-soundfonts-2026) |
| **GeneralUser GS 2.0.x** | Compact, balanced GM/GS bank, ~30 MB, actively maintained by S. Christian Collins. | SF2 | GeneralUser GS License (free, incl. commercial) *(verify current text)* | ✅ | download | 5 | [schristiancollins.com](https://www.schristiancollins.com/generaluser.php) |
| **MuseScore_General / MS Basic** | MuseScore's bank (FluidR3 lineage, HQ variant available). | SF2/SF3 | MIT *(verify)* | ✅ | ships with MuseScore | 5 | [MuseScore handbook](https://handbook.musescore.org/sound-and-playback/soundfonts) |
| **Arachno, Timbres of Heaven, SGM-V2.01** | Larger GM banks (148 / 399 / 235 MB) with better rock, orchestral, RPG palettes. | SF2 | freeware (author terms; not OSI) | ✅ | archive.org | 4 | [miditoolbox list](https://miditoolbox.com/posts/best-free-general-midi-soundfonts-2026), [Zanderjaz catalog](https://www.zanderjaz.com/downloads/soundfonts/) |
| **Salamander Grand Piano v3** | 16-velocity Yamaha C5 sampled at 48 kHz/24-bit. | SFZ (FLAC); SF2 conversions exist | CC BY 3.0 | ✅ | git / sfzinstruments | 5 | [sfzinstruments/SalamanderGrandPiano](https://github.com/sfzinstruments/SalamanderGrandPiano) |
| **Virtual Playing Orchestra 3.3** | Curated full orchestra (sections + solos, sus/stac/pizz/trem), 603 MB. Built from Sonatina, VSCO 2 CE, No Budget Orchestra, U. Iowa, Philharmonia. | SFZ | mixed CC; author: no restrictions on making music | ✅ | download | 5 | [virtualplaying.com](https://virtualplaying.com/virtual-playing-orchestra/) |
| **VSCO 2 Community Edition** | 3 GB chamber orchestra, raw samples + SFZ. | SFZ / WAV | **CC0** | ✅ | download | 5 | [Versilian](https://versilian-studios.com/vsco-community/) |
| **Sonatina Symphonic Orchestra** | 440 MB orchestra. | SFZ | CC Sampling Plus 1.0 | ✅ | download | 4 | [BPB](https://bedroomproducersblog.com/2011/01/28/440mb-of-free-orchestral-samples-in-sonatina-symphonic-orchestra/) |
| **Karoryfer Samples free line** | Cello, double bass, bass guitar, drums, odd instruments. | SFZ | CC BY 4.0 *(verify per instrument)* | ✅ | download | 4 | [sfz instruments index](https://sfzlab.github.io/sfz-website/instruments/) |
| **Freepats / Musical Artifacts** | Catalogues of free SF2/SFZ (drums, guitars, basses, synths). | SF2 / SFZ | per item (GPL/CC/PD) | ✅ | download | 4 | [Musical Artifacts .sfz](https://musical-artifacts.com/?formats=sfz&order=most_downloaded), [sfzformat players](https://sfzformat.com/software/players/) |

### 5c. Free instrument plugins (host through DawDreamer / pedalboard)

| name | what it does | license | Mac? | link |
|---|---|---|---|---|
| **Surge XT** | Hybrid subtractive / wavetable / FM synth, 1000+ presets, full FX; VST3 + AU + CLAP. | GPLv3 | ✅ | [surge-synthesizer.github.io](https://surge-synthesizer.github.io/) |
| **Vital** | Spectral-warping wavetable synth. Source is GPLv3; the free binary tier needs an account. | GPLv3 (source) / free binary | ✅ | [vital.audio](https://vital.audio/) |
| **Dexed** | DX7 FM emulation, loads original SysEx patches. | GPLv3 | ✅ | [asb2m10/dexed](https://github.com/asb2m10/dexed) |
| **Odin 2**, **OB-Xd (1.x)**, **Helm**, **Vaporizer2**, **Cardinal** | Semi-modular / Oberheim / poly / wavetable / Rack modular synths. | GPLv3 (OB-Xd 2.x binaries via discoDSP) | ✅ | [BPB free synths 2026](https://bedroomproducersblog.com/free-vst-plugins/synthesizer/) |
| **Decent Sampler + Pianobook** | Free (closed) sampler with hundreds of free community libraries. | freeware; libraries per author | ✅ | [pianobook.co.uk](https://www.pianobook.co.uk/) |
| Drums: **DrumGizmo** kits, **AVL Drumkits**, **Sitala** (free, closed), Salamander Drumkit SFZ | Multi-velocity acoustic kits. | GPL / CC BY-SA / freeware *(verify per kit)* | ✅ (Sitala AU; DrumGizmo LV2/VST) | [Zanderjaz drums](https://www.zanderjaz.com/downloads/soundfonts/drums/) |

### 5d. Neural MIDI-to-audio renderers

| name | what it does | input / conditioning | license | Mac? | install | maturity | link |
|---|---|---|---|---|---|---|---|
| **MIDI-DDSP** (Magenta) | Hierarchical DDSP rendering of **monophonic** MIDI for 13 URMP orchestral instruments with expression controls (vibrato, brightness, attack…). | Monophonic MIDI. | Apache 2.0 (not officially supported) | ❌ "cannot be installed on M1 MacBook" (TensorFlow); repo **archived 2024-02-01** | pip (Colab) | 2 | [magenta/midi-ddsp](https://github.com/magenta/midi-ddsp), [C# port (no TF)](https://github.com/denmase/csharp-midi-dssp) |
| **TokenSynth** (ICASSP 2025) | Token-based neural synth: **polyphonic single-instrument** audio from MIDI + timbre embedding (clone any instrument from a reference clip, or text-to-instrument via CLAP). | MIDI + reference audio / text; DAC codec. | MIT; weights auto-download | ✅ CPU supported (slow), CUDA recommended | clone | 3 | [KyungsuKim42/tokensynth](https://github.com/kyungsukim42/tokensynth), [paper](https://arxiv.org/pdf/2502.08939) |
| **MIDI-VALLE** (ISMIR 2025) | Expressive **piano** performance synthesis via codec LM; Piano-Encodec + MIDI-VALLE checkpoints on Zenodo. | Piano MIDI (+ optional prompt pair). | Code Apache 2.0; models CC BY 4.0 | 🟡 Colab inference; 24 GB for training | clone | 3 | [nii-yamagishilab/MIDI-VALLE](https://github.com/nii-yamagishilab/MIDI-VALLE) |
| nii midi-to-audio (ICASSP 2023) | TTS-style MIDI→audio (piano). | MIDI. | see repo | 🟡 | clone | 2 | [nii-yamagishilab/midi-to-audio](https://github.com/nii-yamagishilab/midi-to-audio) |
| Papers, no code confirmed | Anysynth (zero-shot instrument cloning via in-context learning, 2607.11143); DDSP polyphonic guitar (string-wise MIDI); "Beyond Piano" differentiable SoundFont proxies (velocity estimation). | — | — | — | — | — | [Anysynth](https://arxiv.org/pdf/2607.11143), [DDSP guitar](https://arxiv.org/pdf/2309.07658), [2608.08985](https://arxiv.org/pdf/2608.08985) |

### Recommendation (faithful rendering)

**Default: a two-tier sampler renderer.**

1. **tinysoundfont (MIT, pure pip) with GeneralUser GS / FluidR3_GM** as the
   zero-dependency baseline that replaces the current sine-ish mock — every
   `inst=` class in the `.sc` vocabulary maps to a GM program, `:perc.drums`
   voices map to GM channel-10 keys, `vel` passes straight through.
2. **sfizz (pysfizz or the sfizz CLI) with SFZ libraries** for the instruments
   that matter most for recognisability: Salamander (`keys.piano`), Virtual
   Playing Orchestra / VSCO 2 CE (`strings`, `brass`, `winds`), Karoryfer
   basses. sfizz is archived but stable; keep sfizioso as the upgrade path.

**Alternative: DawDreamer hosting Surge XT / Dexed / Vital** for
`synth.lead`, `synth.pad`, `bass.synth`, `keys.ep` where a sampled GM patch
sounds wrong; the `timbre{}` and `tags{}` blocks can pick presets. pedalboard
is the lighter host if only "notes → audio through one AU/VST3" is needed.

**Neural: TokenSynth** is the one worth an experiment — it renders polyphonic
MIDI in the timbre of a reference clip, which is exactly "play `:notes.gtr`
with the sound of the original guitar stem". MIDI-DDSP is dead on Apple
Silicon; MIDI-VALLE is piano-only and heavy.

---

## Recommended open synthesis stack for Infinity Engine

Two render paths, as in the roadmap (Milestone 4). Everything below is
Mac-local unless marked **GPU**.

### Path (a): faithful rebuild (deterministic, from `.sc` + stems)

| `.sc` stream(s) | renderer | how it is fed | notes |
|---|---|---|---|
| `:notes.vox` + `:text.vox` + `:contour.vox` | **DiffSinger (openvpi) + English voicebank** (LUNAI / community), CPU | `.sc` → `.ds`: notes → `note_seq`/`note_dur` (cents → note names, microtones via `f0_seq`), `:text.vox` syllables → phonemes via the bank's dictionary (`align=syllable` needed; word-aligned lyrics get syllabified and spread across notes), `:contour.vox` verbs rendered to an explicit `f0_seq` (vibrato / scoop / fall / bend) at `f0_timestep` | Only tool that accepts an explicit pitch curve. Vocoder NC (fine for research). |
| same, target voice | **Seed-VC singing model** (Mac) after DiffSinger; **SoulX-Singer** (**GPU**, Apache) as the zero-shot alternative that takes MIDI/F0 + lyrics directly | DiffSinger render → Seed-VC with a reference clip from the separated vocal stem (Milestone 1) | Keeps melody/timing via F0 conditioning; semitone shift if the bank's range is off. |
| `:notes.vox` (backing / harmony voices) | same DiffSinger pipeline per voice, separate banks or Seed-VC to differentiate | one `.ds` per `:notes.<trk>` with `inst=voice.backing` | |
| `:notes.bass`, `:notes.gtr*`, `:notes.keys*`, `:notes.strings/brass/winds` | **sfizz + SFZ** (Salamander, VPO/VSCO 2 CE, Karoryfer) with **tinysoundfont + GM SF2** as fallback per track | `inst=` → SFZ instrument or GM program; `vel` → velocity; `dev±ms` micro-timing applied to onsets; `env{}` ignored by samplers (use ADSR opcodes in SFZ when present) | Deterministic; same seed = same audio. |
| `:notes.synth.*`, `bass.synth`, `keys.ep` | **DawDreamer hosting Surge XT / Dexed / Vital** | preset chosen from `tags{}` / `timbre{}` (centroid, harmonicity, dist); MIDI in PPQN from `:grid` | Optional; sampler fallback if no host. |
| `:perc.drums` | **tinysoundfont GM drum kit** (baseline) → **SFZ kit** (Salamander Drumkit / DrumGizmo / Karoryfer) | `kick/snare/hat…` atoms → GM note numbers; `vel` → layers; `dev±ms` groove | |
| `:texture.*` | **Magenta RealTime 2 small** (CC BY 4.0, MLX) or **Stable Audio 3 Small** (CPU) | text from `~"gloss"` + `tags{}`; band-energy rows as loudness automation | Generative, but the only sane way to get drones/ambience; keep NC-free options. |
| `:harmony` | not rendered directly — resolves `.` atoms in `:notes.*`; optional pad from `:harmony` chords through the sampler for scaffolding | | |
| `:struct` + `:grid` | timeline / tempo map for every renderer above (PPQN in DawDreamer, seconds elsewhere); section `energy` → track gain | | |
| `:mix` | pedalboard effects chain (GPLv3): per-track reverb from `rt60`, compression from `comp_est`, stereo width, `!auto` ramps as parameter automation; LUFS normalisation to `lufs_int` / `by_section` | | |
| `:audio.<trk>` stems (v0.4) | pass-through, time-aligned by `@offset`; replaced part rendered by the row above and summed | | |

### Path (b): generative full song

| `.sc` stream(s) | Mac-local default | GPU experiment (rented NVIDIA) | how it is fed |
|---|---|---|---|
| everything (bridge bundle) | **ACE-Step 1.5** cover: path-(a) mock render → `src_audio`, `audio_cover_strength` sweep 0.4–0.9; `reference_audio` = original mix (or stem) for timbre; BPM / key / meter / duration from `:grid` + `:harmony`; lyrics with `[verse]`/`[chorus]` from `:struct` (timestamps stripped) | **MuLaCover**: `:notes.vox` → melody MIDI, `:harmony` → chord MIDI, `:perc.drums` → drum MIDI, `:text.vox` → sectioned lyrics, `@style` → style fields. The direct lead-sheet test. | |
| vocal line with exact pitch + timing | **DiffSynth-Music *prosody* template** on ACE-Step 1.5 (Apache 2.0): feed the DiffSinger/mock vocal as the prosody control, lyrics as text → new vocals that keep pitch and timing; *vocals* template then generates the accompaniment | — (try MPS first; offload modes exist) | best candidate for "faithful but generated" vocals |
| tempo lock | DiffSynth-Music *beats* template: click track synthesised from `:grid` | — | |
| melody + chords as a score | **YuE2 via mlx-Yue**: `.sc` → ABC (melody from `:notes.vox`, chords from `:harmony`, sections from `:struct`), "Full + Supplied ABC Score" mode; cover mode from the original recording as a cross-check | YuE2 official CUDA build (24 GB) | NC weights |
| lyrics-only / licence-clean regeneration | **HeartMuLa-oss-3B via heartlib-mlx** (Apache 2.0) with `:text.vox` lyrics + tags | LeVo 2 (custom licence) only for a quality ceiling comparison | no symbolic control |
| word-level lyric timing test | — | **JAM-0.5** with `:text.vox` word times as its JSON input | NC, small; only to measure timing adherence |
| accompaniment for an existing vocal | DiffSynth-Music *vocals* template, or ACE-Step `lego` to add tracks in `repainting_start/end` windows from `:struct` | MuseControlLite for melody/rhythm-curve-locked ≤47 s sections | |
| section repaint after the alignment loop | ACE-Step `repaint` with `repainting_start/end` from `:struct` boundaries | | |
| evaluation | re-encode outputs (Milestone 4 done-criteria); compare `:notes.vox` and `:text.vox` timing to the input `.sc` | | |

**Order of experiments:** (1) ACE-Step cover sweep from the sampler-based
mock (already planned); (2) DiffSynth-Music prosody/vocals/beats on the same
mock; (3) DiffSinger vocal render feeding both; (4) MuLaCover on a rented GPU
with straight MIDI export; (5) YuE2 ABC path on the Mac. Adopt whichever
keeps melody and lyric timing on re-analysis.
