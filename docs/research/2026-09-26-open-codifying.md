# Open codifying stack research — 2026-09-26

Scope: open-source / open-weight tools that turn audio into `.sc` (the
"codify" direction) and that write, repair and edit `.sc` from a text brief.
Hard constraints: nothing paid, no hosted APIs (Claude / GPT / Gemini are out
for pipeline stages), non-commercial licences acceptable (private research),
primary hardware M1 Max 32 GB / Python 3.11 / no CUDA. Where a tool is
CUDA-only that is called out as "rent an NVIDIA GPU".

Companion document: `2026-09-26-open-synthesis.md` covers the decode
direction (`.sc` → audio).

Verification note: facts below were checked on 2026-09-26 against the linked
model cards, repos and papers (five parallel research passes plus direct
re-checks of the load-bearing claims). Items marked *(verify)* were not
confirmed on a primary page and should be re-checked before adoption. The
session's web-search budget was exhausted partway through, so the long tail
was checked by direct fetch only; a list of unverified items closes the file.

Column key for tables: **Mac? (RAM needed)** = runs on Apple Silicon without
CUDA (✅ official or confirmed, 🟡 community port / "should work on MPS or CPU,
unverified", ❌ CUDA-only) plus the memory or disk it needs; **maturity** =
1–5 (5 = production-grade, actively maintained).

---

## TL;DR

- **`.sc` author**: Qwen3.8-27B (Apache-2.0, Aug 2026) at 4-bit on MLX
  (16 GB), thinking off for emission, on for repair. Qwen3.6-35B-A3B for fast
  loops, gpt-oss-20b as the cheap checker-loop model.
- **Constrain it** with one Lark grammar for `.sc` run through Outlines'
  `mlxlm` backend (llguidance engine); the same grammar works in llama.cpp
  (`-DLLAMA_LLGUIDANCE=ON`) and vLLM on a rented GPU. Keep the grammar
  syntactic; the strict checker owns semantics. Few-shot with 15–30 curated
  checker-clean files first; QLoRA/DoRA on `mlx_lm.lora` second.
- **Listener**: MOSS-Music-8B (Apache-2.0, May 2026) on MLX for captions,
  timestamped lyrics, section descriptions and targeted questions. Never let
  an audio LLM set key/chords/tempo — dedicated MIR models beat every audio
  LLM by 60+ points.
- **Transcription backbone**: SheetSage2 (CC BY-NC 4.0, Sept 2026) on the full
  mix for beats, key, chords, structure and lead melody; per-stem specialists
  for drums (DrumSep → ADTOF), piano (Transkun), bass/other (Basic Pitch or
  MuScriptor), lyrics (Qwen3-ASR + ForcedAligner), contour (torchcrepe/penn
  or SwiftF0). No end-to-end system beats separate-then-transcribe on drums,
  piano detail or note timing.
- **Interop**: symusic (MIDI/ABC), music21 (MusicXML), JAMS + mir_eval
  (annotations, Harte chords, metrics), MidiTok (tokens), abcMIDI + abcjs
  (ABC), MuseScore 4.6 CLI + LilyPond 2.26 (engraving). `.sc` stays custom;
  borrow ABC's `|` and mini-notation ideas, allow optional ABC islands.
- **Eval data**: RWC-Popular (now an open CC BY-NC download, Feb 2026) for
  full-song round-trip; MUSDB18-HQ + its lyrics extension; Slakh2100 for
  per-instrument notes; JamendoLyrics for word alignment; Harmonix +
  SongFormBench for structure; Lakh/MidiCaps rendered via FluidSynth as the
  `.sc` fine-tuning corpus.

---

## 1. Open LLMs for writing and editing music as code

The job: (a) write `.sc` from a brief (structure, harmony, melody, lyrics),
(b) repair it when `check --strict` returns line-numbered errors, (c) edit an
existing file. "Fits" below means weights ≤ ~20 GB so that 8–12 GB remain for
KV cache at 16–64K context plus the OS.

### 1.1 General open LLMs that fit in 32 GB

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **Qwen3.8-27B** (dense, hybrid Gated DeltaNet + gated attention, 262K ctx, thinking on by default with `reasoning_effort`) | Strongest ≤32B coder / instruction follower as of Sept 2026; released 2026-08-14 | Apache-2.0 | ✅ 4-bit MLX 16.1 GB on disk, ~17–19 GB at 64K ctx; GGUF UD-Q4_K_M 16.5 GB, Q5_K_M 19.8 GB, Q6_K 22 GB; 8-bit (29 GB) does not fit | mlx-community quant (made with mlx-vlm 0.6.8; text-only load through `mlx_lm` *(verify)*), or `unsloth/Qwen3.8-27B-GGUF` via llama.cpp | 4 (six weeks old, ~85K downloads/month) | [Qwen3.8 repo](https://github.com/QwenLM/Qwen3.8) · [HF](https://huggingface.co/Qwen/Qwen3.8-27B) · [mlx 4-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) · [GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| **Qwen3.6-27B** (dense, Apr 2026) | Previous 27B; LiveCodeBench v6 83.9, SWE-bench Verified 77.2 | Apache-2.0 | ✅ 4-bit ≈16–17 GB | mlx-community 4-bit / OptiQ-4bit / MTP-4bit; GGUF | 4 | [HF](https://huggingface.co/Qwen/Qwen3.6-27B) · [mlx](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) |
| **Qwen3.6-35B-A3B** (MoE, 3B active, Apr 2026) | Fast agentic coder; LCB v6 80.4, SWE-bench V 73.4; ~3× the tok/s of the dense 27B | Apache-2.0 | ✅ 4-bit ≈20 GB (leaves ~10 GB for KV) | mlx-community `Qwen3.6-35B-A3B-OptiQ-4bit`; GGUF | 4 | [HF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [mlx](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit) |
| **Qwen3.5-27B / 35B-A3B** (Feb 2026) | Superseded but explicitly supported by mlx-lm 0.30.7+ (incl. LoRA); IFEval 95.0 / IFBench 76.5 (27B) | Apache-2.0 | ✅ 27B 4-bit ≈16.8 GB; 35B-A3B 4-bit 20.4 GB | `pip install mlx-lm`; mlx-community quants | 5 | [HF 27B](https://huggingface.co/Qwen/Qwen3.5-27B) · [mlx 27B](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) · [mlx 35B-A3B](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) |
| **Qwen3-Coder-30B-A3B-Instruct** (MoE, non-thinking, May 2025) | Still the default "MLX coding model"; no Qwen3.5/3.6 Coder at this size has shipped (Qwen3-Coder-Next is 80B-A3B, ~45 GB at 4-bit) | Apache-2.0 | ✅ 4-bit ≈17–18 GB | mlx-lm / llama.cpp | 5 | [HF](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| **Gemma 4 31B-it** (dense, 256K ctx, `<\|think\|>` toggle; family launched Apr 2026) | Second lineage for cross-checks; LCB v6 80.0, MMLU-Pro 85.2, AIME 2026 89.2 | Apache-2.0 (licence changed from Gemma ToU) | ✅ 4-bit MLX 18.4 GB (tight; ~48K ctx practical) | mlx-vlm quant; mlx-lm 0.31.2 added Gemma 4 text | 4 | [model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [HF](https://huggingface.co/google/gemma-4-31B-it) · [mlx](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit) |
| **Gemma 4 26B-A4B** (MoE, 3.8B active) | Fast Gemma; LCB v6 77.1 | Apache-2.0 | ✅ 4-bit 15.3 GB | same | 4 | [mlx](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) |
| **gpt-oss-20b** (21B / 3.6B active, MXFP4, Aug 2025) | Cheap fast reasoner for checker loops; SWE-bench V 53.2 | Apache-2.0 | ✅ ~12–13 GB; **62–72 tok/s measured on an M1 Max 32 GB** | mlx-lm, llama.cpp, ollama | 5 | [HF](https://huggingface.co/openai/gpt-oss-20b) · [M1 Max figures](https://intuitionlabs.ai/articles/hardware-requirements-gpt-oss-20b) |
| **GLM-4.7-Flash** (30B-A3B, Jan 2026) | Local coding/agents; SWE-bench V 59.2, LCB v6 64.0 | MIT | ✅ 4-bit ≈17–18 GB | llama.cpp / ollama; community MLX | 4 | [HF](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| **Nemotron-3-Nano-30B-A3B** (Mamba-2/MoE hybrid, 1M ctx, Dec 2025) | Reasoning + coding; LCB v6 68.3, IFBench 71.5 | NVIDIA Nemotron Open Model License | 🟡 4-bit ≈17 GB; GGUF yes, MLX *(verify)* | llama.cpp / ollama | 4 | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |
| **Devstral-Small-2-24B** (Dec 2025) | Agentic coding; SWE-bench V 68.0; Mistral itself lists "Mac with 32 GB RAM" | Apache-2.0 | ✅ 4-bit ≈14 GB | mlx-lm / llama.cpp | 4 | [HF](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) |
| **Magistral-Small-2509 / Mistral-Small-3.2-24B** | 24B reasoner (LCB v5 70.9) / plain instruct baseline | Apache-2.0 | ✅ 4-bit ≈14 GB | mlx-lm / llama.cpp | 4 | [Magistral](https://huggingface.co/mistralai/Magistral-Small-2509) · [Small 3.2](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| **Seed-OSS-36B-Instruct** (Aug 2025, 512K ctx, thinking budget) | IFEval 85.8, LCB v6 67.4 | Apache-2.0 | 🟡 4-bit ≈20–21 GB (borderline; 3-bit or short ctx) | llama.cpp / mlx-lm | 4 | [HF](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) |
| **Granite 4.1 30B / 8B** (Apr 2026, 512K ctx) | Native JSON / tool-output emphasis | Apache-2.0 | 🟡 30B 4-bit ≈17 GB; MLX *(verify)* | llama.cpp | 3 | [IBM](https://research.ibm.com/blog/granite-4-1-ai-foundation-models) |
| **Phi-4-reasoning-plus 14B** | Small dense reasoner; no Phi-5 found | MIT | ✅ 4-bit ≈8 GB | mlx-lm / llama.cpp | 4 | [announcement](https://venturebeat.com/ai/microsoft-launches-phi-4-reasoning-plus-a-small-powerful-open-weights-reasoning-model) |
| **DeepSeek-R1-Distill-Qwen-32B** | Distilled reasoner; no V4 distil exists | MIT | ✅ 4-bit ≈18 GB | mlx-lm / llama.cpp | 3 (dated) | [HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| *Do not fit 32 GB*: Mistral Small 4 (119B-A6B), Qwen3-Coder-Next (80B-A3B), Llama 4 Scout (109B, 45–55 GB Q4), GLM-4.5-Air / GLM-5.3-Flash, gpt-oss-120b, Kimi K2.x, MiniMax M2.7, Qwen3.5-122B-A10B, DeepSeek V3.x/V4 | — | — | ❌ rent an NVIDIA GPU (or a 64–128 GB Mac) | — | — | [Mistral Small 4](https://mistral.ai/news/mistral-small-4/) · [Scout on 32 GB](https://llmcheck.net/blog/llama-4-scout-maverick-mac/) |

Speed on the M1 Max: only gpt-oss-20b has a published M1 Max figure
(62–72 tok/s). For 27B dense 4-bit expect roughly 10–18 tok/s and for
35B-A3B 4-bit roughly 35–45 tok/s (extrapolated from M4-series numbers; not
measured). Multi-token-prediction quants (`-MTP-4bit`) exist for speculative
speed-ups.

MLX caveat: the Qwen3.6/3.8 and Gemma 4 community quants were produced with
`mlx-vlm`, not `mlx-lm`. `mlx-lm` 0.30.7–0.31.3 explicitly lists Qwen 3.5,
Gemma 4, GLM5 and gpt-oss; text-only loading of the Qwen3.6/3.8 quants
through `mlx_lm` should be tested before committing. `mlx-lm` has no native
structured output; it exposes `logits_processors`, which is the hook Outlines
uses. Sources: [mlx-lm releases](https://github.com/ml-explore/mlx-lm/releases),
[mlx-lm Qwen3.5 issue #1136](https://github.com/ml-explore/mlx-lm/issues/1136).

**Recommended default: Qwen3.8-27B, 4-bit, MLX.** Thinking off for `.sc`
emission (constrained), thinking on for repair reasoning, context ≤64K to
stay inside 32 GB. It is the strongest ≤32 GB instruction follower on record
(IFBench 79.5, LiveCodeBench v6 90.3, Terminal-Bench 2.1 73.0 per its model
card), Apache-2.0, and available in every Mac runtime.

**Alternatives.** (1) Qwen3.6-35B-A3B 4-bit when iteration speed matters
(repair loops, many-shot ICL), ~3× faster, ~4 points lower on coding.
(2) gpt-oss-20b as the fast checker-loop model (measured on this exact
machine). (3) Gemma 4 31B for a second opinion from a different lineage;
26B-A4B as its cheap sibling. (4) Devstral-Small-2 / Mistral-Small-3.2 as
the safest LoRA bases (mlx-lm's LoRA path lists Mistral/Llama first, and
`mlx_lm.fuse` → GGUF export is supported for them).

### 1.2 Music-specialised open LLMs

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **ChatMusician** (LLaMA-2-7B continued-pretrain + SFT on ABC) | Text ↔ ABC chat, harmonisation; MusicTheoryBench author | MIT | ✅ 4-bit GGUF ≈4 GB | transformers or GGUF (llama.cpp / Ollama) | 2 (frozen 2024, Llama-2 quality) | [HF](https://huggingface.co/m-a-p/ChatMusician) · [arXiv](https://arxiv.org/abs/2402.16153) |
| **NotaGen / NotaGen-X** (110M–516M, interleaved ABC, CLaMP-DPO) | Classical score generation conditioned on period / composer / instrumentation | MIT | 🟡 CUDA instructions; small model fits CPU; MPS untested | `git clone`, PyTorch 2.3 | 3 (IJCAI 2025, maintained) | [GitHub](https://github.com/ElectricAlexis/NotaGen) · [HF](https://huggingface.co/ElectricAlexis/NotaGen) |
| **MuPT-v1** (190M–1.97B, SMT-ABC, Llama-2 arch) | ABC pretraining base | licence not stated *(verify)* | ✅ ≤2B, MPS/MLX after conversion | transformers | 2 (ICLR 2025, unmaintained) | [HF](https://huggingface.co/m-a-p/MuPT-v1-8192-1.97B) · [arXiv](https://arxiv.org/abs/2404.06393) |
| **MIDI-LLM** (Llama-3.2-1B + 55K MIDI tokens, Nov 2025 / ISMIR 2026) | Text → multitrack MIDI; beats Text2midi (FAD 0.173, CLAP 22.1) | Llama 3.2 Community License | ✅ 1.5B, transformers, MPS fine | transformers / vLLM | 3 | [HF](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B) · [GitHub](https://github.com/slSeanWU/MIDI-LLM) |
| **Text2midi / t2m-InferAlign** (T5 encoder + 18-layer decoder) | Text → MIDI with chord / tempo attributes | Apache-2.0 | ✅ card lists CUDA, MPS, CPU | GitHub | 3 (AMAAI Lab) | [HF](https://huggingface.co/amaai-lab/text2midi) · [InferAlign](https://github.com/AMAAI-Lab/t2m-inferalign) |
| **SongComposer** (InternLM2-7B) | Lyric ↔ melody, text → song in word-level `(lyric, pitch, dur, rest)` tuples | Apache-2.0 | 🟡 7B, quantise | transformers | 3 (ACL 2025) | [HF](https://huggingface.co/Mar2Ding/songcomposer_sft) · [arXiv](https://arxiv.org/abs/2402.17645) |
| **Anticipatory Music Transformer** (128M–780M, arrival-time MIDI tokens) | Infilling / accompaniment on Lakh MIDI | Apache-2.0 | ✅ small, CPU/MPS | `anticipation` package | 3 (2023, stable) | [GitHub](https://github.com/jthickstun/anticipation) |
| **BACH-1B** ("Via Score to Performance", ICASSP 2026) | Bar-level ABC song generation from lyrics, "CPU friendly" | not stated | weights not released | — | 1 | [arXiv](https://arxiv.org/abs/2508.01394) |
| **YuE2-3B** | Lyrics + tags → 48 kHz audio; also emits an ABC score plan | code Apache-2.0; weights CC BY-NC 4.0 | ❌ 24 GB NVIDIA BF16 (rent a GPU) | GitHub | 4 | [GitHub](https://github.com/multimodal-art-projection/YuE) |
| **Libretto** (Jun 2026) | *Not a model*: MIDI → bar-block text grammar → 39-axis fingerprint environment for LLM agents with a repair loop; generator pluggable | open source (type unstated) | ✅ pure Python | `pip install -e .` | 2 (single author) | [arXiv](https://arxiv.org/pdf/2606.22708) · [GitHub](https://github.com/Xyc-arch/Libretto) |
| **Agogic** (Aug 2026) | *Paper + tokenisation*: performance-timed tokens (10 ms shifts, 32 velocities) trained on Qwen3.5 0.8B–35B; shows representation beats scale | CC-BY paper | n/a | — | 2 | [arXiv](https://arxiv.org/html/2608.03999) |
| **Text2Score** (May 2026) / **SongSage** (Jan 2026) / **MIDI-LLaMA** (ICASSP 2026) / **MelodyT5** | Two-stage text → ABC planner+executor / lyric-centric CPT / MIDI understanding / ABC score-to-score | papers CC-BY; weights unclear *(verify)* | — | — | 1–2 | [Text2Score](https://arxiv.org/abs/2605.13431) · [SongSage](https://arxiv.org/abs/2601.01153) · [MIDI-LLaMA](https://arxiv.org/html/2601.21740) · [MelodyT5](https://arxiv.org/abs/2407.02277) |

Not found: "ScoreLLM", "MusicInfuser"; NotaGen-X is a checkpoint inside the
NotaGen repo, not a separate model. Kimi / DeepSeek have no music variants.

**Recommended default: none of these as the primary `.sc` writer.** Every
symbolic model emits ABC or MIDI tokens, not `.sc`; they are ≤7B and 2024-era
except MIDI-LLM (1B). Use **MIDI-LLM** and **Text2midi** as *data
generators* (brief → MIDI → render to `.sc` for synthetic training pairs) and
as melody/bass suggestion tools inside the compose loop; use ChatMusician's
MusicTheoryBench to check the general model's theory knowledge.

**Alternatives.** NotaGen-X as a source of polished classical multi-voice
material (interleaved ABC converts cleanly to `:notes.*` events). YuE2 is the
only open lyrics → full-song system and needs a rented GPU (covered in the
synthesis doc). Libretto and Agogic are worth reading as design validation:
both independently arrive at explicit-onset, performance-timed, bar-scoped
text with a checker/repair loop — the `.sc` design.

### 1.3 Structured-output tooling on Mac

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **Outlines 1.3.x** (`outlines[mlxlm]`) | JSON schema / regex / **Lark CFG** / choice on MLX via `from_mlxlm`; streaming; CFG backend is llguidance by default, XGrammar optional; no batching with constraints | Apache-2.0 | ✅ requires Metal; negligible RAM | `pip install "outlines[mlxlm]"` | 4 (MLXLM is a first-class backend) | [mlxlm docs](https://dottxt-ai.github.io/outlines/latest/features/models/mlxlm/) · [backends](https://dottxt-ai.github.io/outlines/latest/features/advanced/backends/) |
| **llguidance 1.x** (Rust Earley parser; Lark-variant CFG + JSON schema + regex; inline `%json{}`) | ~50 µs/token masks; the engine under Outlines, llama.cpp, vLLM (≥0.8.2), SGLang (≥0.4.4), OpenAI structured outputs | MIT | ✅ arm64 wheels | `pip install llguidance`; llama.cpp via `-DLLAMA_LLGUIDANCE=ON`, grammars prefixed `%llguidance` | 5 | [GitHub](https://github.com/guidance-ai/llguidance) · [syntax](https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md) · [llama.cpp doc](https://github.com/ggml-org/llama.cpp/blob/master/docs/llguidance.md) |
| **llama.cpp GBNF** (`grammar` / `json_schema` on llama-server; `--grammar-file` on llama-cli) | Native CFG sampling: ranges, `{m,n}`, token refs; no left recursion or lookahead; `x? x? x?` patterns are slow | MIT | ✅ CPU-side masks | `brew install llama.cpp` or cmake with Metal | 5 | [grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) |
| **XGrammar 0.2.x / XGrammar-2 (May 2026)** | JSON / regex / EBNF CFG masks; `pip install "xgrammar[metal]"` | Apache-2.0 | ✅ Metal wheels; usable as Outlines backend on MLX | pip | 5 (default in vLLM / SGLang / MLC) | [GitHub](https://github.com/mlc-ai/xgrammar) |
| **mlx-lm 0.31.x native** | `logits_processors` hook only; no built-in grammar / JSON schema | MIT | ✅ | `pip install mlx-lm` | 5 | [GitHub](https://github.com/ml-explore/mlx-lm) |
| **vllm-mlx / vllm-metal** | OpenAI-compatible server on MLX with `JSONSchemaLogitsProcessor` and a thinking-aware processor | Apache-2.0 | ✅ | pip | 2 (v0.2.0 Apr 2026) | [vllm-mlx](https://vllm-mlx.is-a.dev/reference/api/vllm_mlx/constrained/) · [vllm-metal](https://github.com/vllm-project/vllm-metal) |
| **LM Studio mlx-engine** | JSON-schema structured output for MLX models (Outlines inside) | app proprietary / engine OSS | ✅ | app | 4 | [docs](https://lmstudio.ai/docs/developer/core/structured-output) |
| **ollama** | `format` = JSON schema only (compiled to GBNF); no raw grammar | MIT | ✅ | brew | 4 (MLX engine had a non-terminating structured-output bug, #18567) | [blog](https://ollama.com/blog/structured-outputs) · [issue](https://github.com/ollama/ollama/issues/18567) |
| **llama-cpp-python 0.3.x** | `response_format` JSON schema; `LlamaGrammar` GBNF | MIT | ✅ Metal wheels py3.10–3.12 | `pip install llama-cpp-python --extra-index-url …/whl/metal` | 4 (lags llama.cpp) | [docs](https://llama-cpp-python.readthedocs.io/en/latest/) |
| **guidance 0.3.x** | Programmatic constrained generation on llguidance (llama.cpp / transformers backends) | MIT | ✅ | pip | 4 | [PyPI](https://pypi.org/project/guidance/) |
| **SGLang on Metal** | MLX backend exists (macOS 14+); structured output on Metal undocumented | Apache-2.0 | 🟡 experimental | pip | 2 | [docs](https://lmsysorg.mintlify.app/docs/hardware-platforms/apple_metal) |
| **lm-format-enforcer 0.11.x** | JSON schema / regex only, pure Python, slow | MIT | ✅ | pip | 2 | [PyPI](https://pypi.org/project/lm-format-enforcer/) |

**Recommended default: Outlines (`outlines[mlxlm]`) with a Lark grammar for
`.sc`, backed by llguidance, on mlx-lm.** It is the only path that gives CFG
(not just JSON-schema) constraints on MLX today, and the same Lark file is
reusable verbatim in llama.cpp and vLLM if the compose loop later moves to a
rented GPU.

**Alternatives.** llama.cpp GBNF via llama-server for a zero-Python server
(write GBNF, or convert with llguidance's `gbnf_to_lark.py` in the other
direction); ollama or LM Studio only for JSON-first sub-tasks (schema only).
Avoid lm-format-enforcer (no CFG) and SGLang-Metal (unverified).

### 1.4 Fine-tuning tooling on Mac

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **mlx-lm `lora`** (LoRA / QLoRA on quantised base / DoRA `--fine-tune-type dora` / full) | Reference Mac trainer; jsonl `chat` / `completions` / `text`; `--grad-checkpoint`, `--num-layers`; `mlx_lm.fuse` → GGUF for Mistral/Llama | MIT | ✅ doc: Mistral-7B LoRA on **M1 Max 32 GB ≈250 tok/s** (batch 1, 4 layers); QLoRA of a 27B 4-bit base ≈16 GB weights + activations, feasible at seq ≤2K, batch 1, few layers | `pip install mlx-lm` | 5 | [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) |
| **mlx-tune 0.6.0** (ex "unsloth-mlx", Jun 2026) | Unsloth-compatible API on MLX: SFT / DPO / ORPO / KTO / GRPO, QLoRA, 39+ archs incl. Qwen3.5 and Gemma 4 | Apache-2.0 | ✅ 16 GB+ | `pip install mlx-tune` | 3 (active community project) | [GitHub](https://github.com/ARahim3/mlx-tune) |
| **Unsloth Studio / Desktop (beta)** | No-code UI; "training, MLX and GGUF inference all work" on macOS; the core Unsloth library is still CUDA-first | Apache-2.0 | ✅ macOS 12+ | installer | 2 (beta) | [requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) · [Studio](https://unsloth.ai/docs/new/studio) |
| **Axolotl** | YAML trainer; M-series "partial", no bitsandbytes → no 4-bit QLoRA on Mac | Apache-2.0 | 🟡 fp16/32 LoRA on MPS | pip | 4 (CUDA) | [GitHub](https://github.com/axolotl-ai-cloud/axolotl) |
| **LLaMA-Factory** | Broad CUDA trainer; README has no macOS/MPS statement; QLoRA needs bitsandbytes | Apache-2.0 | ❌ not recommended on Mac | pip | 5 (CUDA) | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |

**Recommended default: `mlx_lm.lora`, QLoRA on a 4-bit base, DoRA on,
`--grad-checkpoint`, `--num-layers 8–16`, batch 1–2, sequences ≤2K tokens.**
Start on Qwen3.8/3.6-27B-4bit if mlx-lm loads it; fall back to
Qwen3.5-27B-4bit (confirmed in mlx-lm 0.30.7+) or Devstral-Small-2 (GGUF fuse
confirmed).

**Alternatives.** mlx-tune when you want DPO/GRPO against the checker
(reward = `check --strict` pass rate) with scripts that port to Unsloth-CUDA
unchanged; Unsloth Studio for a GUI. Full fine-tunes, anything >27B in bf16,
or DPO at scale → rent an NVIDIA GPU.

### 1.5 Benchmark evidence

- **Qwen3.8-27B** vs 3.6-27B: IFBench 79.5 (+10.4), LiveCodeBench v6 90.3
  (+6.4), Terminal-Bench 2.1 73.0 (+9.6), SWE-bench Pro 61.7
  ([HF card](https://huggingface.co/Qwen/Qwen3.8-27B)). Qwen3.5-27B: IFEval
  95.0, IFBench 76.5 ([HF](https://huggingface.co/Qwen/Qwen3.5-27B)).
- **Gemma 4 31B**: LCB v6 80.0, MMLU-Pro 85.2, Codeforces 2150; 26B-A4B
  LCB 77.1 ([model card](https://ai.google.dev/gemma/docs/core/model_card_4)).
- **Aider-polyglot**: the top open model is DeepSeek-V3.2-Exp at 74.5 % (needs
  a GPU); none of the ≤32 GB models appears on it
  ([llm-stats](https://llm-stats.com/benchmarks/aider-polyglot)).
- **ABC-Eval** (Sep 2025; 1,086 items, 10 sub-tasks; frontier closed models
  only): GPT-5 55.0 %, DeepSeek-reasoner 52.0 %, Gemini-2.5-pro 51.3 %;
  syntax tasks >90 % but error-detection macro-F1 only 6–27 % and
  next-bar prediction 26–50 % ([arXiv](https://arxiv.org/abs/2509.23350)).
  Even frontier models fail at bar-level structural reasoning — precisely
  the class of check the `.sc` checker must own.
- **LilyBench** (Jun 2026; open 14–22B models): zero-shot LilyPond compile
  rates Codestral-22B 79.3 %, Phi-4 71.1 %, Qwen2.5-Coder-14B 69.0 %; bar
  counting 0–3 % exact; **curated few-shot examples raised validity to
  97–99.9 %, poorly chosen ones lowered it to 20–45 %**
  ([arXiv](https://arxiv.org/html/2606.08722)). Strongest evidence that
  curated many-shot ICL is the highest-leverage lever for a niche format.
- **"How far can pretrained LLMs go in symbolic music"** (Jan 2026):
  LLaMA-3.1-8B-Instruct has perplexity ~694 on short ABC — unadapted open
  models parse ABC but are not fluent in it; SFT fixes short sequences, long
  ones remain hard ([arXiv](https://arxiv.org/html/2601.22764)).
- **MusicTheoryBench** (372 college-level MCQs): ChatMusician beat LLaMA-2 and
  GPT-3.5 zero-shot ([dataset](https://huggingface.co/datasets/m-a-p/MusicTheoryBench)).
  **ZIQI-Eval** (14K items, 16 LLMs): "all LLMs perform poorly"
  ([arXiv](https://arxiv.org/abs/2406.15885)). A 2026 study finds Qwen2.5
  7B/14B/32B all near 36 % on music-theory understanding — size barely helps
  ([arXiv](https://arxiv.org/pdf/2606.05522)).
- **Agogic** (Aug 2026): performance-timed tokens reach FMD 159 vs 272–286
  for beat-grid MIDI-Like and 406 for ABC; "representation Pareto-dominates
  model scale" ([arXiv](https://arxiv.org/html/2608.03999)). **Libretto**
  (Jun 2026): explicit-onset bar-block grammar + retrieval + repair loop
  reaches 94 % full-piece pass rate ([arXiv](https://arxiv.org/html/2606.22708)).
- **LIMA**: 1,000 curated examples suffice for alignment; response formats
  are learned "from only a handful of examples"
  ([arXiv](https://arxiv.org/abs/2305.11206)).
- **llguidance** mask cost ~50 µs/token p50, 0.5 ms p99; GBNF's backtracking
  parser is "significantly slower" ([GitHub](https://github.com/guidance-ai/llguidance)).

### 1.6 Verdict: grammar and LoRA for `.sc`

**Write one Lark grammar for `.sc` (not GBNF, not JSON-first).** Lark is what
llguidance consumes, and llguidance is the common engine under
Outlines-on-MLX, llama.cpp, vLLM and SGLang, so a single grammar follows the
project from the M1 Max to a rented GPU. llguidance has a real lexer, so a
line-oriented format with tokens like `13:1.0`, `F1-6c`, `1.5b`, `?0.85`,
`%pat` and `Am | F | C | G` is expressed naturally with terminals and stays
fast, whereas GBNF's backtracking parser degrades on optional-heavy
productions. The existing EBNF sketch in the design spec (§4.6) is already
two-level by intent; the Lark grammar should mirror only its lexical layer.

Keep the grammar **syntactic**: stream headers, line shapes, lexemes,
bar-range syntax, note-name/octave alphabets. Do **not** encode semantics
(bar numbers monotonic, events inside the declared `:grid`, chords consistent
with key, `%pat` defined before use) — CFGs cannot express them, and trying
inflates the grammar and the mask cost. The checker owns semantics.

**Hybrid loop**: grammar-constrained generation → `check --strict` →
line-numbered errors fed back with the offending lines quoted → regenerate
only the affected stream (grammar re-applied). Two warnings: constrained
decoding forces *some* legal token, so a weak model produces syntactically
valid nonsense — let the model emit `?0.NN` when unsure and treat that as a
signal; and thinking models must have their `<think>` section excluded from
the constraint (vllm-mlx's thinking-aware processor does this; with raw
mlx-lm apply the processor only after the think-close token, or run with
`enable_thinking: False`).

**JSON-first** (schema → deterministic emitter) is right for two sub-tasks
only: the *brief → structure/harmony plan* stage (small, nested, Pydantic
validated, works in any runtime) and programmatic edit operations
("transpose bars 9–16"). For note and lyric streams JSON inflates tokens
3–5× and loses the line locality that makes checker feedback cheap, so
generate `.sc` directly there.

**Many-shot ICL first.** With 262K native context and ~64K practical on
32 GB, put the spec plus 15–30 complete, checker-clean examples (~1–2K
tokens each) in the prompt; LilyBench shows example quality dominates. Use
mlx-lm prompt caching so the shared prefix is prefilled once.

**LoRA verdict: feasible and worth doing, as step two.** Format teaching is
the easiest LoRA target. Budget 300–1,000 checker-validated pairs for a
syntax LoRA and 2–5K if brief-following and repair behaviour are included.
Build them synthetically: render MIDI corpora (Lakh via MIDI-LLM/Text2midi,
NotaGen-X classical ABC, GigaMIDI, POP909) through the project's own
`emit.py`, generate briefs with the base model, and add repair pairs by
injecting checker-detectable faults into clean files. QLoRA of a 27B 4-bit
base fits the M1 Max but is slow — extrapolating from the documented
250 tok/s for a 7B LoRA, expect ~40–80 tok/s for 27B, i.e. roughly 10–20
hours per epoch over 1M training tokens (estimate). A 24B Mistral/Devstral
base is ~30 % faster with confirmed GGUF export. Anything beyond rank-16
QLoRA on 27B, DPO/GRPO at scale, or bf16 full fine-tunes → rent an NVIDIA
GPU (mlx-tune scripts port to Unsloth unchanged). Every recommended base is
Apache-2.0 or MIT.

---

## 2. Open audio-language models that listen and describe

The job: captioning, instrumentation/timbre description, structure,
key/chords/tempo, lyrics, and targeted questions ("what plays the riff at
0:32?") to feed `@style`, `@mix`, `desc=`, `~"gloss"` and the lyrics
reconciliation step.

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **MOSS-Music-8B-Instruct / -Thinking** (~9.1B: Qwen3-8B + MOSS-Audio-Encoder, May 2026) | Music captioning, **timestamped** lyrics ASR, section/structure analysis, chord/key/tempo/beat reasoning, instrument & voice ID, long-form QA | Apache-2.0 | ✅ MLX: `mlx-community/MOSS-Music-8B-Thinking-{4,6,8}bit`, 8-bit ≈10 GB disk/RAM, ~23 tok/s on an M4, 75 s track in ~34 s; PyTorch-MPS ≈18 GB bf16 and reported to stall; GGUFs exist but are not llama.cpp-loadable | `pip install -U mlx-audio` (`moss_music` model dir) — the mlx-community card names a `moss_music_mlx` backend; confirm which package is current | 3 (four months old, actively updated; official path is SGLang on CUDA 12.8) | [GitHub](https://github.com/OpenMOSS/MOSS-Music) · [HF](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) · [mlx 8-bit](https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-8bit) |
| **MOSS-Audio-4B/8B** (Apr 2026) | General audio (speech / sound / music) captioning, time-aware QA; MMAU-Pro 64.9, MMAR 66.5 (8B-Thinking) | Apache-2.0 | 🟡 community MLX / GGUF only; not in mlx-audio's official list | transformers / SGLang (CUDA) | 3 | [GitHub](https://github.com/OpenMOSS/MOSS-Audio) · [report](https://arxiv.org/pdf/2606.01802) |
| **Qwen3-Omni-30B-A3B Instruct / Thinking** (Sept 2025) | Best general open audio LLM; music tagging, genre, QA; ≤40 min audio; GTZAN 93.0, RUL-MuChoMusic 52.0 | Apache-2.0 | ✅ llama.cpp: official `ggml-org/…-GGUF` with audio (mtmd); Q4 ≈17–22 GB, must run alone. MLX quants are text-only; mlx-vlm audio path unverified (issue #2319) | `llama-mtmd-cli -hf ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF --audio x.wav` | 4 | [GitHub](https://github.com/QwenLM/Qwen3-Omni) · [HF](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) · [report](https://arxiv.org/html/2509.17765) |
| **Qwen3-Omni-30B-A3B-Captioner** | Prompt-free dense captions; card recommends ≤30 s clips | Apache-2.0 | 🟡 community GGUF only | as above | 2 | [HF](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) |
| **Qwen2.5-Omni-7B / 3B** (Mar 2025) | Omni; MMAU-music 69.2; best open model on HumMusQA (64.3 %) | Apache-2.0 | ✅ official `ggml-org/Qwen2.5-Omni-{3B,7B}-GGUF` with audio; 7B Q4 ≈6 GB | `llama-mtmd-cli … --audio` | 4 | [HF](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) |
| **Qwen2-Audio-7B-Instruct** (2024) | Audio QA / captioning | Apache-2.0 | ✅ mlx-audio (`qwen2_audio`); llama.cpp ships no GGUF "due to poor performance" | mlx-audio | 2 (hallucination-prone; CMI-Bench key 8.3, beat F1 7.5) | [HF](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |
| **Qwen3-ASR-1.7B / 0.6B + Qwen3-ForcedAligner-0.6B** (Jan 2026) | Lyrics/song transcription in 52 languages; word timestamps (aligner, ≤5 min chunks, 11 languages) | Apache-2.0 | ✅ mlx-audio (`qwen3_asr`, `qwen3_forced_aligner`) and official `ggml-org/Qwen3-ASR-GGUF`; <4 GB | `mlx_audio.stt.generate --model Qwen/Qwen3-ASR-1.7B …` | 5 | [GitHub](https://github.com/QwenLM/Qwen3-ASR) · [HF](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [report](https://arxiv.org/pdf/2601.21337) |
| **Music Flamingo** (8B: Qwen2.5-7B + AF-Whisper; `-hf`, `-2601-hf`, `-think-2601-hf`; Nov 2025) | Theory-aware captions / QA (harmony, structure, timbre, lyrics); ≤10–20 min; MMAU-music 76.8, MuChoMusic 74.6, MUSDB18 lyrics WER 19.6 | NVIDIA OneWay **Noncommercial** (+ Qwen Research License) | ❌ targets A100/H100; no MLX; GGUFs carry no working audio tower — rent a GPU | transformers (CUDA) | 4 | [HF](https://huggingface.co/nvidia/music-flamingo-2601-hf) · [arXiv](https://arxiv.org/abs/2511.10289) |
| **Audio Flamingo Next** Instruct / Think / Captioner (8B, Apr 2026); **AF3** (7B); **AF2** | General audio incl. music captions with BPM/key prompts, ≤30 min, timestamped CoT; MMAU 75.8, MMAR 63.0 | NVIDIA OneWay Noncommercial | ❌ CUDA | transformers | 4 | [AF-Next](https://huggingface.co/nvidia/audio-flamingo-next-hf) · [arXiv](https://arxiv.org/abs/2604.10905) · [AF3](https://huggingface.co/nvidia/audio-flamingo-3) |
| **Kimi-Audio-7B-Instruct** (Apr 2025) | Speech-centric audio foundation model | code Apache-2.0/MIT; weights tagged MIT *(verify)* | 🟡 `mlx-community/kimi-audio-7b` bf16 22.6 GB via mlx-audio — tight | mlx-audio | 3 (weak on music: MUSDB18 lyrics WER 97.5) | [GitHub](https://github.com/MoonshotAI/Kimi-Audio) |
| **MiMo-Audio-7B-Instruct** (Xiaomi, 2025) | Speech + audio understanding; MMAU-music 66.4 | MIT | ✅ mlx-audio lists it | mlx-audio | 3 | [GitHub](https://github.com/XiaomiMiMo/MiMo-Audio) |
| **Step-Audio 2 mini** (8B) / **Step-Audio-R1** (33B) | Speech dialogue + audio reasoning; MMAU-music 71.6 (mini) | Apache-2.0 | ❌ no MLX/GGUF found | transformers / vLLM | 3 | [GitHub](https://github.com/stepfun-ai/Step-Audio2) |
| **MiniCPM-o 4.5** (9B, Feb 2026) | Omni, speech-first; general audio captioning; MMAU 59.2 | Apache-2.0 | ✅ llama.cpp-omni GGUF "for Macs" | llama.cpp-omni | 4 | [GitHub](https://github.com/OpenBMB/MiniCPM-o) |
| **Gemma 4 E2B/E4B/12B** (audio input; 31B has none) · **Gemma 3n** | Speech ASR/AST; Gemma 4 audio capped at 30 s | Apache-2.0 (Gemma 4) / Gemma ToU (3n) | ✅ mlx-vlm `--audio`; `ggml-org/gemma-4-E4B-it-GGUF` | mlx-vlm / llama.cpp | 3 (not music-oriented) | [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) |
| **Voxtral Mini 3B / Small 24B**, **Ultravox 0.5**, **LFM2-Audio-1.5B**, **Phi-4-multimodal**, **Baichuan-Audio**, **DeSTA2.5** | Speech-first audio LLMs; no music claims | Apache-2.0 / MIT / LFM Open | ✅ Voxtral, Ultravox, LFM2 in llama.cpp; Voxtral in mlx-audio | llama.cpp / mlx-audio | 3–4 (speech only) | [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) · [Ultravox](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_1-8b) |
| **SALMONN · MU-LLaMA · M2UGen · MusiLingo · LLark** (2023–24) | Early music/audio LLMs; MuChoMusic 41.8 / 32.4 / 42.9 / 21.1; LLark has no released weights | mostly NC | ❌ custom PyTorch, no Mac ports | — | 1 (superseded) | [MuChoMusic](https://arxiv.org/abs/2408.01337) · [LLark](https://github.com/spotify-research/llark) |
| **Qwen3.5-Omni Plus/Flash** (Apr 2026) | Omni, >10 h audio | API-only; no open weights found | n/a | n/a | excluded | [arXiv](https://arxiv.org/abs/2604.15804) |
| **MuQ** (300M) / **MuQ-MuLan** (700M) | SSL music embeddings; MuLan = music–text contrastive (zero-shot tags, EN/ZH); MARBLE SOTA | MIT code, CC BY-NC 4.0 weights | ✅ PyTorch CPU/MPS, <3 GB | `pip install muq` | 4 | [GitHub](https://github.com/tencent-ailab/MuQ) · [arXiv](https://arxiv.org/html/2501.01108) |
| **MERT-v1-330M** | SSL music embeddings for probes (key, instrument, genre, chords) | CC BY-NC 4.0 | ✅ transformers CPU/MPS, ~1.3 GB | `AutoModel.from_pretrained("m-a-p/MERT-v1-330M")` | 4 | [HF](https://huggingface.co/m-a-p/MERT-v1-330M) |
| **LAION-CLAP** (music ckpt) / **msclap** | Zero-shot tag / instrument / genre scoring, audio–text retrieval; GTZAN zero-shot 71 % | CC0 (LAION) / MIT (MS) | ✅ CPU | `pip install laion-clap` / `msclap` | 4 | [GitHub](https://github.com/LAION-AI/CLAP) |
| **Essentia TensorFlow models** | Instrument, mood, genre (Discogs-EffNet / MAEST), TempoCNN, tonal/atonal | CC BY-NC-SA 4.0 | ✅ CPU (`essentia-tensorflow`) | pip | 4 | [models](https://essentia.upf.edu/models.html) |
| **madmom** | Beat/downbeat/tempo, majmin chords, key recognition | BSD code, CC BY-NC-SA models | ✅ CPU (install from git on 3.11) | `pip install git+https://github.com/CPJKU/madmom` | 3 (old but still the reference) | [GitHub](https://github.com/CPJKU/madmom) |

Mac runtime status (verified Sept 2026): **mlx-audio 0.5.6** (Sept 24 2026)
ships `moss_music`, `moss_transcribe_diarize`, `qwen3_asr`,
`qwen3_forced_aligner`, `qwen2_audio`, `voxtral`, `whisper`, `parakeet`,
`vibevoice_asr`, plus MiMo-Audio and Kimi-Audio
([repo](https://github.com/Blaizzy/mlx-audio)). **llama.cpp mtmd** has
official audio GGUFs for Ultravox 0.5, Voxtral-Mini, Qwen3-ASR,
Qwen2.5-Omni 3B/7B, Qwen3-Omni 30B-A3B, Gemma 4 E2B/E4B; no Music Flamingo,
AF-Next or MOSS ([docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)).
**Ollama** has no audio capability as of today
([capabilities](https://docs.ollama.com/capabilities)). RAM budget: MOSS-Music
8-bit (~10 GB) + Qwen3-ASR (~4 GB) + MuQ/MERT/Beat This (<3 GB) coexist;
Qwen3-Omni Q4 must run alone.

### Recommendations by task

1. **Captioning / timbre description — default MOSS-Music-8B (Thinking or
   Instruct) on MLX.** MusicCaps (GPT-5.4 judge) 4.53/5 vs Gemini-3.1-Pro
   4.42, Music Flamingo 4.21, AF-Next 4.10, Qwen3-Omni 3.96; Song Describer
   4.58 vs 4.48 / 4.26 / 4.02. Strongest on *structure* (4.86) and
   melody/harmony; Gemini and Music Flamingo edge it on *instrumentation*
   (4.68/4.64 vs 4.40). Thinking wins MusicCaps, Instruct wins SDD and QA —
   try both. Ground the timbre field with MuQ-MuLan / CLAP zero-shot scores
   over the `inst=` vocabulary rather than the LLM's prose alone.
   *Alternatives*: Music Flamingo-2601 / AF-Next-Captioner on a rented A100;
   Qwen3-Omni-Instruct via llama.cpp when album art or video also matters.
2. **Instrumentation tags — default a dedicated classifier, LLM second.**
   NSynth-instrument: MOSS-Music 86.6 %, AF-Next 86.4 %, Music Flamingo
   80.8 %, but Qwen3-Omni 30.9 %, Gemini-3.1-Pro 13.4 %, Kimi 6.0 %;
   CMI-Bench: Qwen2-Audio 37.6 % vs supervised 78.2 %. Stack: Essentia
   MTG-Jamendo-instrument + MuQ-MuLan zero-shot over the `.sc` vocabulary
   (MuQ probe NSynth 79.2 %, MuQ-MuLan zero-shot MTT 79.3 ROC-AUC vs
   LAION-CLAP 73.9), then ask MOSS-Music timestamped questions — it is the
   only Mac-runnable model trained with time markers.
3. **Structure — default SongFormer (boundaries + labels) → MOSS-Music to
   describe each section for `desc=`.** SongFormer beats All-In-One 0.703 vs
   0.596 HR.5F on Harmonix; among LLMs only MOSS-Music was trained for
   section boundaries and its structure caption sub-score (4.86–4.92) is >1
   point above everyone else. SheetSage2 (§3) is a strong second source.
4. **Key / chords / tempo — honest answer: dedicated MIR models win by a
   wide margin.** CMI-Bench: key score 7.7–8.3 (Qwen2-Audio, SALMONN) vs
   74.3 supervised; GTZAN beat F1 7.5–23.7 vs 88.3; models "copy input
   examples rather than generating predictions". MOSS-Music, Music Flamingo
   and AF-Next all *claim* key/BPM/chords in captions but none publishes
   accuracy (MOSS: "benchmark results will be added soon"); NSynth-pitch is
   the tell — MOSS-Music 86.9 % but Music Flamingo 0.0 %, AF-Next 0.05 %.
   MusICA-MetaBench (Jul 2026): Qwen3-Omni 46.3 % vs Gemini 3.1 Pro 59.0 %
   (random 20 %) on pitch/interval/rhythm/harmony, audio the hardest
   modality. Use SheetSage2 / Beat This! / ChordMini / madmom (§3) and let
   the LLM only sanity-check ("does the chorus modulate?").
5. **Lyrics — default Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B**
   (mlx-audio or llama.cpp): M4Singer WER 5.98 vs Whisper-large-v3 13.58,
   GPT-4o-transcribe 16.77; full songs with backing 14.6 % EN. *Alternative*:
   MOSS-Music-8B-Thinking — best of the audio LLMs (MUSDB18 WER 29.2,
   MIR-1K CER 15.8 vs Qwen3-Omni 62.7 / 20.5, Kimi 97.5, AF-Next 94.9) and
   it emits timestamps natively, so it is a good second opinion in the voting
   step of the lyrics reconciliation plan. Note the separation caveat in §3.

### Benchmark evidence and caveats

- **Text-prior problem**: on MuChoMusic 8 of 11 *text-only* LLMs exceed 50 %
  ([RUListening](https://arxiv.org/abs/2504.00369)); 8 audio LLMs retain
  60–72 % accuracy with *no audio* ([All That Glitters](https://arxiv.org/html/2604.24401)).
  Prefer RUL-MuChoMusic, MMAU-Pro-music, NSynth-pitch and CMI-Bench when
  ranking.
- **MOSS-Music table** (11 music benchmarks): Instruct 80.38 > Gemini-3.1-Pro
  75.17 > Thinking 74.26 > Music Flamingo 73.87 > AF-Next 69.89 > Qwen3-Omni
  66.75 > Kimi 56.90; Gemini still wins MMAR-music (71.6 vs 59.7) and
  MMAU-Pro-music (73.1 vs 71.0) ([HF](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct)).
- **MMAU-Pro** (5,305 items): Gemini 2.5 Flash 59.2, Qwen2.5-Omni-7B 52.2,
  AF3 51.7, human 77.9; MOSS-Audio-8B-Thinking 64.9, Qwen3-Omni 61.2
  ([arXiv](https://arxiv.org/html/2508.13992)). **MMAR**: Qwen3-Omni-Thinking
  66.4 (music 44.7), Qwen2-Audio 30.4 ([leaderboard](https://github.com/ddlBoJack/MMAR)).
- **CMI-Bench** (14 MIR tasks): key 6.5–8.3 vs 74.3; beat F1 7.5–23.7 vs
  88.3; DSing lyrics WER 116–816 vs 13.0 ([arXiv](https://arxiv.org/html/2506.12285v2)).
- **Embeddings** (MARBLE): NSynth-instr MuQ 79.2 / MERT 72.6; GiantSteps key
  MuQ 63.5 / MERT 65.6 — still below dedicated key models
  ([MuQ](https://arxiv.org/html/2501.01108)).

---

## 3. End-to-end transcription models (audio → score/code)

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **SheetSage2** (m-a-p, Sept 2026; MERT-v2-FullSong 632M backbone + 57M adapters + 6-layer AR decoder; 28.4K h training audio) | One model, full mix: beats/downbeats, key, chords, structure, vocal + instrumental melody → ABC, MIDI, events.json, `.lab`; long songs via overlapping windows | weights CC BY-NC 4.0; code licence unresolved (HF discussion #2) | 🟡 CPU documented (fp32; est. 6–10 GB RAM, minutes/song, unmeasured); MPS untested; BF16 on CUDA for benchmarks; GGUF exists for the `audio.cpp` runtime (Metal support unverified) | `huggingface-cli download m-a-p/SheetSage2`; Py 3.10/3.11, torch 2.8, FFmpeg 6.1; `AutoModel.from_pretrained(..., trust_remote_code=True).transcribe("song.mp3")` | 3 (weeks old, 17.7K downloads/month, tech report "coming soon") | [HF](https://huggingface.co/m-a-p/SheetSage2) · [YuE2 page](https://map-yue2.github.io/) · [GGUF](https://huggingface.co/audio-cpp/SheetSage2-GGUF) |
| **MuScriptor** (Kyutai + Mirelo, Jul 2026; 103M / 307M / 1.4B) | Multi-instrument audio → MIDI (36 instrument groups, MT3-style tokens), MusicXML via MuseScore; no velocity; no separation at train or test | code MIT; weights CC BY-NC 4.0 (gated) | ✅ explicit MPS support; large ≈3 GB fp16 | `pip install muscriptor`; `muscriptor transcribe --model large in.wav -o out.mid` | 4 (pip package, 1.5K stars) | [GitHub](https://github.com/muscriptor/muscriptor) · [arXiv](https://arxiv.org/abs/2607.08168) · [HF](https://huggingface.co/MuScriptor/muscriptor-large) |
| **YourMT3+** (2024, 45.8M) | Multi-instrument + vocals + drums seq2seq; Slakh onset F1 84.56, MAESTRO 96.98, GuitarSet 91.65, MIR-ST500 vocals-in-mix 72.05 | GPL-3.0 | 🟡 CPU fp32 (patch hard-coded `cuda`); 2–4 GB | clone; Lightning stack; checkpoints from HF Space | 3 (research code, last updates 2024) | [GitHub](https://github.com/mimbres/YourMT3) · [arXiv](https://arxiv.org/abs/2407.04822) |
| **MT3** (2021) / **mt3-pytorch** / **MR-MT3** (2024) | Multi-instrument seq2seq baselines | Apache-2.0 / unlisted | 🟡 JAX painful on Mac; PyTorch port ships weights | clone | 2 (superseded) | [mt3-pytorch](https://github.com/kunato/mt3-pytorch) · [MR-MT3](https://github.com/gudgud96/MR-MT3) |
| **MIROS** (2025 AMT Challenge winner) · **Harmonica** (Sept 2026, 26K–15M params, runs on iPhone) · **PerceiverTF** | Multi-instrument / instrument-agnostic transcription; Harmonica frame F1 MAESTRO 0.922, GuitarSet 0.898 | none public | ❌ no code released | — | 1 (papers only) | [MIROS](https://arxiv.org/abs/2603.27528) · [Harmonica](https://arxiv.org/abs/2609.04640) |
| **Timbre-Trap** (Sony, 2024) | Instrument-agnostic pitch salience / notes | MIT | ✅ CPU, small | clone + HF weights | 2 | [GitHub](https://github.com/sony/timbre-trap) |
| **Basic Pitch** (Spotify, 16.8K params) | Instrument-agnostic notes + pitch bends; GuitarSet F_no 79.1, Slakh 57.4 | Apache-2.0 | ✅ CPU / CoreML, <1 GB (already in the encoder) | `pip install basic-pitch` | 4 | [GitHub](https://github.com/spotify/basic-pitch) |
| **Transkun v2** | Piano → MIDI with velocity + pedal (semi-CRF); MAESTRO onset F1 0.983, onset+offset 0.935, +velocity 0.930 | MIT | ✅ CPU default, 1–2 GB | `pip install transkun`; `transkun in.mp3 out.mid` | 4 | [GitHub](https://github.com/Yujia-Yan/Transkun) |
| **Aria-AMT** (EleutherAI, 2024–25) | Whisper-style piano transcription, robust on real recordings; MAESTRO note F1 97.63, MAPS 90.58 | Apache-2.0 | 🟡 GPU-oriented (int8 needs BF16 GPU); CPU untested | clone, Py 3.11, checkpoint `piano-medium-double-1.0` | 3 | [GitHub](https://github.com/EleutherAI/aria-amt) · [arXiv](https://arxiv.org/abs/2504.15071) |
| **ByteDance piano_transcription_inference** (Kong 2021) | Piano notes + pedal; MAESTRO 96.82 | Apache-2.0 | ✅ CPU | `pip install piano_transcription_inference` | 3 (archived read-only Dec 2025) | [GitHub](https://github.com/bytedance/piano_transcription) |
| **ADTOF** (Zehren) | 5-class drum CRNN (kick/snare/HH/toms/cymbals); tested on macOS | CC BY-NC-SA 4.0 | ✅ TF/Keras (PyTorch port −0.2 F) | `pip install .` from repo | 3 | [GitHub](https://github.com/MZehren/ADTOF) |
| **Drum-stem separation → ADTOF** (Riley & Dixon, Sept 2025) | 7/8-class drums + MIDI velocity via jarredou MDX23C DrumSep; 8-class F 0.84 (MDB) / 0.76 (ENST) vs 0.72 / 0.65 for plain ADTOF | paper CC BY-NC-SA; DrumSep weights public | ✅ ZFTurbo MSST on MPS | MSST + jarredou model + ADTOF (recipe, not a package) | 3 | [arXiv](https://arxiv.org/abs/2509.24853) |
| **Noise-to-Notes** (Sony, Sept 2025) · **Separate-and-Detect** (ISMIR 2026) | Diffusion ADT with velocity (E-GMD F1 89.68, ENST 94.90) / latent-diffusion 5-stem drum separation + onsets | no code | ❌ papers only | — | 1 | [N2N](https://arxiv.org/abs/2509.21739) · [S&D](https://arxiv.org/abs/2608.01093) |
| **ROSVOT** (ACL 2024, 12M) | Note-level singing transcription (needs word boundaries; extractor included); M4Singer COnPOff 77.4, MIR-ST500 OOD 47.4 | MIT | 🟡 CPU plausible (PyTorch 2.1); Mandarin-trained | clone + Google-Drive checkpoints | 3 | [GitHub](https://github.com/RickyL-2000/ROSVOT) |
| **VocalParse-1.7B** (May 2026; Qwen3-ASR-1.7B + ~400 pitch/duration/tempo tokens) | Audio-LLM → interleaved lyrics + pitch + duration + BPM; Opencpop pitch MAE 0.35 vs ROSVOT 0.38; lyric WER 3.79 | Apache-2.0 | 🟡 1.7B (~4 GB fp16) on CPU/MPS undocumented; trained on Mandarin only (Opencpop, GTSinger, M4Singer) | `uv pip install -e .`; `snapshot_download('pymaster/VocalParse')` | 2 (new, single-org) | [arXiv](https://arxiv.org/abs/2605.04613) · [GitHub](https://github.com/pymaster17/VocalParse) |
| **Mel-RoFormer** (2024) | Vocal separation model fine-tuned for vocal note transcription; MIR-ST500 COnPOff 0.625 / COn 0.819 (SOTA) | configs via MSST; transcription weights unconfirmed | 🟡 would run (BS-RoFormer family) | — | 2 | [arXiv](https://arxiv.org/abs/2409.04702) |
| **SwiftF0** (Aug 2025, 95.8K params) | Monophonic F0, 42× faster than CREPE on CPU | CC BY 4.0 | ✅ trivial | `github.com/lars76/swift-f0` | 2 | [arXiv](https://arxiv.org/abs/2508.18440) |
| **Beat This!** (CPJKU, 2024, ~20M) | Beats/downbeats without DBN; GTZAN 89.1 / 78.3, Harmonix 95.8 / 90.7, RWC-Pop 96.1 / 93.7 | MIT | ✅ CPU fallback (already in the roadmap) | `pip install beat-this` | 4 | [GitHub](https://github.com/CPJKU/beat_this) |
| **SongFormer** (Oct 2025, 0.7B; MuQ + MusicFM fusion) | Section boundaries + 8 functional labels; SongFormBench-Harmonix HR.5F 0.703, ACC 0.807 | CC BY 4.0 | 🟡 CUDA-tested; CPU/MPS unverified (~1 GB encoders) | clone with submodules, `fetch_pretrained.py`, `infer.sh` | 3 | [GitHub](https://github.com/ASLP-lab/SongFormer) · [arXiv](https://arxiv.org/abs/2510.02797) |
| **all-in-one** (`allin1`, 2023) | Tempo, beats, downbeats, segments, labels | MIT | ✅ NATTEN auto-installs on macOS; needs madmom from git | `pip install allin1` | 3 | [GitHub](https://github.com/mir-aidj/all-in-one) |
| **ChordMini** (Feb 2026) | BTC / 2E1D chord models, 170-class vocabulary, pseudo-label + KD; MajMin 80.24, MIREX 80.16, Sevenths 69.38 | MIT (code + shipped weights) | ✅ CPU | clone + requirements | 3 | [GitHub](https://github.com/ptnghia-j/ChordMini) · [arXiv](https://arxiv.org/abs/2602.19778) |
| **ChordFormer** (Feb 2025) | Conformer large-vocab chords (301 classes); WCSR MajMin 84.09 | no code | ❌ paper only | — | 1 | [arXiv](https://arxiv.org/abs/2502.11840) |
| **SongPrep-7B** (Tencent, Sept 2025; MuCodec + Qwen2-7B) | Full-song structure + timestamped lyrics, no separation; SSLD-200 WER 23.5 % / DER 18.2 % (Gemini-2.5 29.2 / 94.6) | academic-only, no commercial use | ❌ CUDA ≥11.8 required; ~16 GB VRAM — rent a GPU | clone + HF weights | 3 | [GitHub](https://github.com/tencent-ailab/SongPrep) · [HF](https://huggingface.co/tencent/SongPrep-7B) |
| **MIRFLEX** | Wrapper: key CNN, BTC chords, BeatNet, Essentia taggers | MIT | ✅ | clone | 3 | [GitHub](https://github.com/AMAAI-Lab/mirflex) |
| **TART** / **Noise2Fret** (ISMIR 2026) | Guitar audio → tab with techniques / playability | papers; no code | ❌ | — | 1 | [TART](https://arxiv.org/abs/2609.11904) · [Noise2Fret](https://arxiv.org/abs/2608.30854) |
| **Jointist** (2022) · **Omnizart** | Joint recognition+transcription on Slakh / legacy TF toolkit (documented ARM-macOS incompatibility) | unlisted / MIT | ❌ stale, Omnizart no ARM | — | 1 | [Jointist](https://github.com/KinWaiCheuk/Jointist) · [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) |

### Recommendations per sub-task

| sub-task | recommended default | alternatives |
|---|---|---|
| Piano / keys | **Transkun v2** on the piano/other stem (MIT, CPU, F1 0.98 / 0.93 / 0.93 with velocity) | Aria-AMT for difficult real recordings (rent a GPU or test CPU); ByteDance model for pedal |
| Bass | **Basic Pitch** on the bass stem with a pitch-range prior (no dedicated 2025–26 bass model surfaced) | MuScriptor medium/large on the bass stem (MPS-native); YourMT3+ bass program from the mix |
| Drums | **DrumSep (MSST, MPS) → ADTOF** (+10–12 points over plain ADTOF; velocity from per-stem loudness) | N2N / Separate-and-Detect when code appears; MuScriptor drums (onset F1 41.6) is far behind |
| Guitar / other pitched | **MuScriptor** on the other stem (instrument labels, MusicXML, MPS) | Basic Pitch (tiny), YourMT3+ (GuitarSet 91.65 but trained on it) |
| Vocal melody (notes) | **SheetSage2 vocal-melody voice** from the mix (RWC-Pop pitch-class F1 82.51 vs SheetSage1 62.71), cross-checked with Basic Pitch / SwiftF0 note segmentation on the vocal stem for onset precision | ROSVOT (MIT; Mandarin-trained); VocalParse (joint lyrics+notes+BPM, Mandarin — test on English, low expectations); YourMT3+ singing channel |
| Vocal contour (F0) | keep **torchcrepe / penn** on the vocal stem | **SwiftF0** for a 42× CPU speed-up |
| Chords / key | **SheetSage2** (osu2017 maj/min 90.08 vs ChordFormer 86.55; GiantSteps key 77.73 vs madmom 74.62) | **ChordMini BTC** (MIT, 170-class quality labels); madmom DBN baseline |
| Structure | **SongFormer** for strict boundaries (HR.5F 0.703 vs SheetSage2 67.96) and **SheetSage2** for labels (ACC 80.51, HR3F 82.86 vs 80.03 / 79.50); since `.sc` sections are bar-quantised, SheetSage2 alone may do | all-in-one |
| Beats | **Beat This!** (already chosen) — SheetSage2 reports beating it (GTZAN 89.01 vs 86.27) but its baseline number is below the Beat This! paper's own 89.1; protocol-dependent, they agree closely | madmom |
| Lyrics | **Qwen3-ASR + ForcedAligner** (strongest open lyric ASR in VocalParse's own table) — Jam-ALT finds Whisper-family does *worse* on Demucs-isolated vocals than on the mix, while Qwen3-ASR's table shows separated vocals roughly halve WER; **run both inputs and vote** | SongPrep-7B (structure + timestamped lyrics in one pass; rented GPU, academic-only); MOSS-Music as a second opinion |
| Full-song end-to-end | **SheetSage2** as the lead-sheet backbone (one consistent event timeline → ABC/MIDI) | MuScriptor as the multi-instrument note backbone |

### Accuracy evidence

- **SheetSage2 vs specialists** (SOTA on 12/15 metrics per the YuE2 page):
  GTZAN beat F1 89.01, osu2017 downbeat 92.90, GiantSteps key 77.73, GTZAN
  key 75.77, chord maj/min osu2017 90.08, JAAH 64.50 (ChordFormer 59.45),
  Harmonix ACC 80.51 / HR3F 82.86 (SongFormer 80.03 / 79.50; SongFormer wins
  HR.5F 70.63 vs 67.96), RWC-Pop vocal melody 82.51. Melody metric is
  pitch-class F1, not COnPOff. ([HF](https://huggingface.co/m-a-p/SheetSage2),
  [YuE2](https://map-yue2.github.io/))
- **MuScriptor vs YourMT3+** on 372 real recordings: onset F1 60.4 vs 32.5,
  multi-instrument F1 47.8 vs 21.9; drums onset F1 41.6
  ([arXiv](https://arxiv.org/abs/2607.08168)).
- **2025 AMT Challenge**: MIROS 0.60 / YourMT3 0.59 / MT3 0.39; F1 falls
  0.72 → 0.44 going from one to three instruments; instrument leakage and
  hallucinated instruments are the main failures ([arXiv](https://arxiv.org/abs/2603.27528)).
- **Drums**: Riley & Dixon 8-class MDB 0.84 vs 0.72, ENST 0.76 vs 0.65 with
  drum-stem separation first ([arXiv](https://arxiv.org/abs/2509.24853)).
- **Singing** (MIR-ST500): Mel-RoFormer COnPOff 0.625 / COn 0.819; ROSVOT
  OOD 47.4 / 72.1; YourMT3+ from the mix 72.05 onset
  ([Mel-RoF](https://arxiv.org/abs/2409.04702), [ROSVOT](https://arxiv.org/html/2405.09940v1)).
- **Lyrics** (Jam-ALT, English pop): Whisper-v2 39.7 % EN WER, LyricWhiz
  24.6 %, AudioShake v3 (paid) 16.1 %; Demucs-isolated vocals
  "substantially degrade" Whisper ([arXiv](https://arxiv.org/html/2408.06370)).
  SongPrep SSLD-200: WER 23.5 %, DER 18.2 %; Qwen-Audio WER 232.7 %
  ([GitHub](https://github.com/tencent-ailab/SongPrep)).

### Verdict: end-to-end vs separate-then-transcribe

No single end-to-end system beats a well-built separate-then-transcribe
pipeline across all `.sc` fields on mixed pop, but end-to-end has won the
*annotation layer* and is the right backbone there.

Where end-to-end wins (use it): beats, key, chords, structure and lead
melody — SheetSage2 outperforms the specialists from the full mix and gives
one mutually consistent timeline, which is what `.sc` needs (full-song
context is the reason: MERT-v2-FullSong sees 30–360 s). Vocals-in-the-mix
without a separation front-end is now feasible for lyrics (YourMT3+,
VocalParse, SongPrep), and Jam-ALT shows isolated vocals can hurt ASR.

Where separation + specialists still win (keep them): drums
(separation-informed ADT is directly measured to help, +10–12 points;
MuScriptor's drum F1 41.6 is not close to ADTOF/N2N 83–95); piano and
note-level polyphony (specialists on a clean stem reach 0.93–0.98 F1, the
best general models sit at 0.48–0.60 in the wild and degrade sharply with
polyphony); note-level singing timing (the MIR-ST500 SOTA is a separation
model fine-tuned for transcription, and COnPOff from any end-to-end model
stays ≤0.63).

Practical consequence for the encoder: run SheetSage2 on the full mix and
let it own `:grid`, `:harmony` (key + chords), `:struct` and the melody
voices; run DrumSep → ADTOF for `:perc.drums`; Transkun on piano/other;
Basic Pitch or MuScriptor on bass; Qwen3-ASR + aligner on both mix and vocal
stem for `:text.vox`; torchcrepe/penn or SwiftF0 on the vocal stem for
`:contour.vox`. Reconcile everything onto SheetSage2's beat grid and record
disagreements as `meta warn=` per the spec's cross-check contract. Note that
SheetSage2 replaces the roadmap's "SheetSage2 (NC; one model covers …)" entry
with confirmed facts: CC BY-NC 4.0 weights, CPU path documented, MPS
unverified, no tech report yet.

---

## 4. Standard symbolic formats and Python libraries

All are CPU-only, arm64-native, negligible RAM unless stated. Python 3.11
caveats: **abjad and converter21 require 3.12+**; chord-extractor caps at
<3.12.

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **symusic 0.6.0** (Apr 2026) | C++20/nanobind MIDI + ABC parser; Score/Track/Note with `to("second")`, tempo map, pitch bends, lyrics, markers, pianoroll, `get_beats()/get_downbeats()`; bundles abc2midi/midi2abc | MIT | ✅ arm64 cp311 wheels | `pip install symusic` | 4 (fast-moving; ABC "experimental") | [GitHub](https://github.com/Yikai-Liao/symusic) |
| **pretty_midi 0.2.11** (Jul 2026) | Seconds-based MIDI model, tempo map, pitch bends, `fluidsynth()` render | MIT | ✅ | `pip install pretty-midi` | 4 (slow but alive) | [GitHub](https://github.com/craffel/pretty-midi) |
| **mido 1.3.3** | Low-level MIDI messages / files / ports (SysEx MTS, RPN for MPE) | MIT | ✅ | `pip install mido[ports-rtmidi]` | 4 | [GitHub](https://github.com/mido/mido) |
| **miditoolkit 1.0.1** (2023) | Tick-based MIDI model (old MidiTok backend) | MIT | ✅ | pip | 2 (superseded by symusic) | [PyPI](https://pypi.org/project/miditoolkit/) |
| **music21 10.5.0** (Jun 2026) | Score model; MusicXML / MIDI / ABC (import only) / Humdrum / LilyPond export; theory; ChordSymbol, `<sound tempo>`, rehearsal marks, decimal `<alter>` | BSD-3 | ✅ ~200 MB import | `pip install music21` (py ≥3.11) | 5 | [GitHub](https://github.com/cuthbertLab/music21) |
| **partitura 1.9.0** (May 2026) | Score / performance model; MusicXML, MIDI, **kern, MEI; score–performance alignment (derive `dev±ms` from a performance vs score) | Apache-2.0 | ✅ | `pip install partitura` | 4 | [GitHub](https://github.com/CPJKU/partitura) |
| **muspy 0.5.0** (2022) / **pypianoroll 1.0.4** | MIDI/MusicXML/ABC I/O, event & pianoroll representations, metrics (used by LilyBench) | MIT | ✅ | pip | 2 (dormant) | [muspy](https://github.com/salu133445/muspy) |
| **MidiTok 3.0.6** (release late 2024; repo commits Sept 2026) | REMI / REMI+ / TSD / MIDI-Like / Structured / CPWord / Octuple / MuMIDI / MMM / PerTok tokenizers; BPE / Unigram; ABC via symusic | MIT | ✅ | `pip install miditok` (pin git SHA — no PyPI release in ~2 years) | 4 | [GitHub](https://github.com/Natooz/MidiTok) |
| **abcMIDI** (Homebrew 2026.06.09) | `abc2midi` / `midi2abc` / `abc2abc` / `yaps` | GPL-2.0+ | ✅ arm64 bottle | `brew install abcmidi` | 5 | [GitHub](https://github.com/sshlien/abcmidi) |
| **abcjs 6.7.1** (Sept 2026) | JS render ABC → SVG, synth playback, editor, chord grid, microtone accidentals — for the A/B listening server | MIT | ✅ browser | `npm i abcjs` | 5 | [GitHub](https://github.com/paulrosen/abcjs) |
| **xml2abc / abc2xml** (W. Vree) | MusicXML ↔ ABC CLI incl. percussion maps, jazz chords, microtonal accidentals | permissive, not stated on page *(verify)* | ✅ pure Python | download `.py` | 4 | [site](https://wim.vree.org/svgParse/) |
| **EasyABC** | wxPython ABC editor bundling abcm2ps, abc2midi, xml2abc | GPL | ✅ | git clone | 3 | [GitHub](https://github.com/jwdj/EasyABC) |
| **LilyPond 2.26.0** (Apr 2026) + **python-ly 0.9.10** | Engraving; MIDI export; `\relative` octaves / parse-manipulate `.ly` text | GPL-3.0+ | ✅ arm64 bottle (~300 MB) | `brew install lilypond`; `pip install python-ly` | 5 / 3 | [lilypond.org](https://lilypond.org/) · [python-ly](https://pypi.org/project/python-ly/) |
| **abjad 3.31** | Python API generating LilyPond | MIT | ✅ but **py ≥3.12** | pip | 4 | [PyPI](https://pypi.org/project/abjad/) |
| **MuseScore Studio 4.6** (Sept 2025) | `mscore -o out.{pdf,png,svg,mid,mp3,wav,musicxml,mxl}`, `-j` batch JSON, `-r` dpi | GPL-3 | ✅ arm64 app (~1 GB) | `.dmg`; `/Applications/MuseScore 4.app/Contents/MacOS/mscore` | 5 (headless MP3 with Muse Sounds historically flaky — use `--sound-profile "MuseScore Basic"`) | [CLI handbook](https://handbook.musescore.org/appendix/command-line-usage) |
| **Verovio 6.3.0** (Aug 2026) | MEI engraving → SVG in-process; imports MusicXML, Humdrum, ABC, PAE | LGPL-3.0 | ✅ arm64 cp311 wheel | `pip install verovio` | 5 | [verovio.org](https://www.verovio.org/) |
| **MEI 5.1** / **MusicXML 4.0** / **MNX** (draft) | XML score encodings; MusicXML `<alter>` decimals, `<sound tempo>`, rehearsal marks, per-note `<lyric>`; MNX still unstable | ECL-2.0 / W3C CG | n/a | — | 5 / 5 / 1 | [MEI](https://music-encoding.org/resources/schemas.html) · [MusicXML 4.0](https://www.w3.org/2021/06/musicxml40/) · [MNX](https://mnx.formats.music/docs/) |
| **converter21 4.0.2** (Sept 2026) | Humdrum / MEI / ABC readers + writers for music21 | MIT | ✅ but **py ≥3.12** | pip | 4 | [PyPI](https://pypi.org/project/converter21/) |
| **pychord 1.4.0** (Apr 2026) | Parse / transpose jazz chord symbols (`Am7/C`) | MIT | ✅ | pip | 3 | [PyPI](https://pypi.org/project/pychord/) |
| **JAMS 0.3.5** (Jun 2025) | JSON annotation container; namespaces `chord` / `chord_harte`, `beat`, `beat_position`, `tempo`, `key_mode`, `segment_*`, `multi_segment`, `lyrics`, `pitch_contour`, `note_midi`; **per-observation `confidence` field** | ISC | ✅ | `pip install jams` | 4 | [GitHub](https://github.com/marl/jams) · [namespaces](https://jams.readthedocs.io/en/stable/namespace.html) |
| **mir_eval 0.8.2** | Metrics; `mir_eval.chord` implements the Harte grammar (parse / encode / compare at root / thirds / triads / tetrads / sevenths / mirex levels) | MIT | ✅ | `pip install mir_eval` | 5 | [docs](https://mir-eval.readthedocs.io/latest/api/chord.html) |
| **mirdata 1.0.0** (Sept 2025) | Dataset loaders → JAMS / mir_eval structures | BSD-3 | ✅ | pip | 5 | [PyPI](https://pypi.org/project/mirdata/) |
| **FluidSynth 2.6.1** + **pyfluidsynth 1.4.0** | SoundFont MIDI rendering; MTS tuning | LGPL-2.1+ | ✅ `brew install fluid-synth` | pip | 5 | [Homebrew](https://formulae.brew.sh/formula/fluid-synth) |
| **pedalboard 0.9.25** | Audio I/O + effects + VST3/AU hosting | **GPL-3** | ✅ arm64 wheel | pip | 5 | [PyPI](https://pypi.org/project/pedalboard/) |
| **PyGuitarPro 0.11** | Read / write GP3–GP5 | LGPL-3.0 | ✅ | pip | 3 | [PyPI](https://pypi.org/project/PyGuitarPro/) |
| **MIDI 2.0 / UMP**, **MPE**, **MTS** | Per-note pitch (2.0), channel-rotation per-note bend (MPE), SysEx tuning tables (MTS); no mainstream Python file library for MIDI 2.0 yet | MMA specs | — | — | 2 / 4 / 4 | [MIDI 2.0](https://midi.org/midi-2-0) · [MPE](https://midi.org/midi-polyphonic-expression-mpe) |
| **ABC 2.1**, **Humdrum **kern**, **LRC (enhanced A2)** | Text notation standards; LRC `[mm:ss.xx]` line + `<mm:ss.xx>` word | open | — | — | 5 | [ABC 2.1](https://abcnotation.com/wiki/abc:standard:v2.1) · [kern](https://www.humdrum.org/rep/kern/) |
| **Strudel / TidalCycles**, **Alda**, **Sonic Pi**, **musicpy 7.16**, **Tonal.js** | Prior-art text music languages: mini-notation (`[bd sn]*2`, `~`, `<a b>`, `@`), `V1:` voices and `@marker`, Ruby DSL, `C('CM7',3,1/4)`, JS theory | AGPL / EPL / MIT / LGPL / MIT | ✅ | — | 4–5 | [Strudel](https://strudel.cc/learn/mini-notation/) · [Alda](https://github.com/alda-lang/alda) · [musicpy](https://pypi.org/project/musicpy/) · [Tonal](https://github.com/tonaljs/tonal) |

Not usable: `pyharmony` on PyPI is a Logitech remote-control library;
`midi2audio` is abandoned (shell out to `fluidsynth` instead); `mingus` and
`chordparser` are dormant.

### Recommended `.sc` import/export targets

| target | default | alternatives / notes |
|---|---|---|
| **MIDI (in + out)** | **symusic** — seconds/ticks/quarter units, tempo map, bends, lyrics, markers, beats; same object feeds MidiTok | pretty_midi for its seconds-first API and one-line FluidSynth render; mido only for raw SysEx MTS / RPN setup |
| **MusicXML / MEI** | **music21** (py 3.11 OK) for MusicXML round-trips incl. ChordSymbol, `<sound tempo>`, rehearsal marks, per-note lyrics, decimal `<alter>` | partitura for score↔performance alignment and **kern/MEI in one small package; Verovio for MEI → SVG; converter21 once on py 3.12; MNX watch-only |
| **ABC** | **custom emitter (~200 lines; bars are already the `:harmony` unit) validated by `abc2midi` / `symusic.Score(path, fmt="abc")`** | import via symusic (loses notation) or music21 (ABC 1.6 full, 2.1 partial, no export); abc2xml/xml2abc for lossless notation transfer; abcjs to render/play ABC islands in `soundcode serve` |
| **LilyPond** | **LilyPond 2.26 via Homebrew** driven by a small `.ly` template writer (absolute pitch, not `\relative`) | abjad on py 3.12; python-ly for parsing existing `.ly` |
| **Score / audio render** | **MuseScore 4.6 CLI** (PDF/PNG/SVG/MIDI/MP3/WAV/MusicXML from one binary) | Verovio for in-process SVG; FluidSynth + GM SoundFont for cheap MIDI → WAV |
| **Annotations** | **JAMS** as the sidecar / exchange format: `:struct` → `segment_open` / `multi_segment`, `:harmony` → `chord`, `:grid` → `beat` / `beat_position` / `tempo`, `:text.<trk>` → `lyrics` (word intervals), `:contour` → `pitch_contour`, `?0.NN` → `confidence`; mirdata gives training data in the same shape and mir_eval scores repairs | plain CSV/TSV via `mir_eval.io` for quick diffs |
| **Tokens** | **MidiTok** REMI or TSD, BPE-trained, on symusic Scores (git-pinned) | Anticipatory Music Transformer arrival-time triples for performance-time tokens (Agogic argues these beat beat-grids and ABC) |
| **Chord syntax** | **keep jazz/ChordPro surface syntax (`Am7/C`, `F#m7b5`) in `.sc`; normalise internally to Harte (`A:min7/b3`) via `pychord` → `mir_eval.chord`** — LLM training data overwhelmingly uses jazz syntax, every MIR dataset and metric uses Harte, and Harte's colon collides with `bar:beat` | store Harte in JAMS exports; `?0.58 alt=Dm` maps to JAMS confidence + a second annotation; `mir_eval.chord.compare` levels give a principled "how wrong" score |
| **Lyrics timing** | **Enhanced LRC (A2)** for import/export (LLMs have seen a lot of it; `.sc` `:text` is richer, so export is lossy one way only) + **JAMS `lyrics`** for word intervals with confidence | SRT / TTML only for karaoke/video tools |
| **Pitch contour** | internal cents-vs-seconds; export to **JAMS `pitch_contour`** or a CREPE/PENN-style `time,frequency,confidence` CSV; for MIDI render `:contour` verbs to pitch bend on a dedicated channel per voice (MPE-style, RPN 0 bend range ±48 st) | MIDI 2.0 per-note pitch once a Python file library exists |
| **Drums** | static **General MIDI percussion map** on channel 10 (`kick` 36, `snare` 38, `sidestick` 37, `clap` 39, `hat` 42, `hh_pedal` 44, `hh_open` 46, `tom1` 50, `tom2` 47, `tom_floor` 41/43, `crash` 49, `ride` 51, `ride_bell` 53, `china` 52, `splash` 55); abc2xml `%%percmap` / MusicXML `<unpitched>` for score paths | — |
| **Microtonal** | cents are canonical already; MIDI export via per-note bend on rotating channels (works in FluidSynth and every DAW); MusicXML `<alter>` decimals; ABC microtonal accidentals only as an abc2xml/abcjs extension | MTS SysEx via mido for hardware / FluidSynth tuning tables |
| **Tempo curves / sections** | MIDI tempo meta events (piecewise-constant sampling of the `:grid` curve) + MIDI markers; MusicXML `<sound tempo>` + `<rehearsal>`; JAMS `tempo` + `segment_open` — `:grid` anchors remain ground truth | — |

### Which formats open LLMs handle best

- **ABC dominates music-LLM work and the compactness is quantified**:
  ChatMusician reports ~38 % fewer tokens than MIDI-as-text and 99.6 % ABC
  parse rate vs GPT-4 94.6 % / GPT-3.5 65.4 %
  ([arXiv](https://arxiv.org/abs/2402.16153)). MuPT had to invent SMT-ABC
  (bar-synchronised multi-track ABC) because plain ABC misaligns bars across
  voices — the exact problem `.sc`'s per-stream `bars 1-4` lines avoid by
  construction ([arXiv](https://arxiv.org/abs/2404.06393)). NotaGen pretrains
  on 1.6M ABC pieces, again bar-aligned ([arXiv](https://arxiv.org/abs/2502.18008)).
- **Syntax is easy, semantics is not, regardless of format**: ABC-Eval
  frontier models exceed 90 % on ABC syntax (GPT-5 96.5 %) but drop to ~44 %
  on segment/sequence reasoning ([arXiv](https://arxiv.org/abs/2509.23350));
  ZIQI-Eval "all LLMs perform poorly" ([arXiv](https://arxiv.org/abs/2406.15885)).
- **LilyPond is viable, not just ABC**: LilyBench shows executable LilyPond
  zero-shot from open models ([arXiv](https://arxiv.org/abs/2606.08722)).
- **Unadapted open models are not fluent in ABC**: LLaMA-3.1-8B-Instruct
  perplexity ~694 on short ABC; SFT helps short sequences
  ([arXiv](https://arxiv.org/html/2601.22764)).
- **For understanding, MIDI-derived embeddings beat ABC text** (MIDI-LLaMA:
  BERTScore 0.914 vs 0.854, human preference 58 vs 22 of 100
  ([arXiv](https://arxiv.org/html/2601.21740))); **for generation, custom
  performance-timed / grid-constrained schemes beat REMI and ABC**: Agogic
  (FMD 159 vs 272–286 MIDI-Like vs 406 ABC, 2.84 tok/note for ABC in Qwen
  tokenisation ([arXiv](https://arxiv.org/html/2608.03999))), Libretto
  (explicit integer onset slots; ABC forces ~70 running additions to recover
  onsets ([arXiv](https://arxiv.org/html/2606.22708))), BEAT
  ([arXiv](https://arxiv.org/abs/2604.19532)), SymPAC FSM-constrained decoding
  ([arXiv](https://arxiv.org/abs/2409.03055)).
- **Lyrics + melody**: SongComposer's word-level `(lyric, pitch, duration,
  rest)` tuples beat GPT-4 on lyric↔melody tasks; SongGLM uses 2-D
  word/phrase alignment — both custom tuple formats, not ABC `w:` lines,
  which supports `.sc`'s word-timed `:text.<trk>` design
  ([SongComposer](https://arxiv.org/abs/2402.17645), [SongGLM](https://arxiv.org/abs/2412.18107)).
- **Chord symbols**: every LLM paper uses jazz/ChordPro syntax as the
  LLM-facing surface ([GPT-as-judge](https://arxiv.org/abs/2501.13261)); no
  paper evaluates Harte as an LLM output format.
- **Humdrum and MusicXML**: no LLM+kern papers; no MusicXML *generation*
  benchmarks — neither is a sensible LLM-facing format.
- **Novel DSLs**: Grammar Prompting shows LLMs handle unfamiliar DSLs
  competitively given a BNF grammar + few-shot ([arXiv](https://arxiv.org/abs/2305.19234)).

### Verdict: keep `.sc` custom, borrow ABC's lexical conventions

1. What `.sc` encodes has no home in ABC: seconds-canonical timing,
   per-event `dev±ms`, cents pitch, confidence markers, lossy `~0.NN`
   pattern similarity, chord doubt with alternatives, contour verbs,
   texture/mix streams. Every paper that needed these (SongComposer,
   Libretto, Agogic, SymPAC, BEAT) invented a custom scheme rather than
   extending ABC, and Agogic's result removes the main "ABC is what LLMs
   like" argument for performance-level data.
2. The "LLMs know ABC" evidence is weaker than it looks for *open* models
   (perplexity ~700 at 8B; semantics collapse to chance anyway). The project
   will few-shot or fine-tune either way; Grammar Prompting and Libretto show
   grammar + examples + repair loop closes most of the gap for a novel DSL.
3. Token efficiency is fine if lines stay terse: `13:1.0 F1-6c 1.5b 112` is
   ~10–14 tokens, similar to MIDI-Like; `%pat` / `like` bindings amortise
   dense passages.
4. Borrow, don't adopt: `|` bar separators (already done); scientific pitch
   + cents (`F#4+12c`) rather than ABC's case/comma octaves or LilyPond's
   `\relative` — the context-dependent rules ABC-Eval identifies as
   error-prone; Strudel/Tidal mini-notation (`[bd sn]*2`, `~`, `<a b>`, `@`)
   as an optional rhythmic sub-syntax inside `%pat`; ChordPro/jazz chord
   surface with Harte internally; enhanced-LRC `<mm:ss.xx>` accepted as an
   input alias for `:text`; Alda's `V1:` voices and `@marker` as precedent
   for named anchors in `:grid`.
5. **Optional ABC islands**: allow a fenced `:abc.<trk>` block (with
   `K:`/`L:`/`M:` headers) that the parser converts via abc2midi/symusic into
   `:notes.<trk>` events at load time, so a model can paste ABC it already
   knows for a monophonic melody. Do not make ABC the canonical melodic form.

---

## 5. Open datasets for evaluation and training

"Mac? (RAM needed)" here = disk size and tooling; "install" = access method
(mirdata loader, Zenodo record, request form, or audio-from-YouTube).

### 5.1 Stems / separation

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **MUSDB18-HQ** | 150 pop/rock songs, 4 stems + mix, 44.1k WAV; the separation benchmark | non-commercial (MedleyDB CC BY-NC-SA + Mixing Secrets) | 22.7 GB | Zenodo 3338373 (open); `musdb` package | 5 | [Zenodo](https://zenodo.org/records/3338373) |
| **MUSDB18 lyrics extension** (Schulze-Forster) | Line-level timestamped lyrics + vocal-type labels for 141 MUSDB songs | CC BY-NC-SA 4.0 | 215 KB | Zenodo 3989267 | 3 | [Zenodo](https://zenodo.org/records/3989267) |
| **MoisesDB** | 240 unreleased songs, 12 genres, 14.4 h; hierarchical stems (guitar/keys/percussion subtypes) | CC BY-NC-SA 4.0 | ~100 GB WAV *(verify)* | music.ai/research registration + `moises-db` package | 4 | [GitHub](https://github.com/moises-ai/moises-db) |
| **SDXDB23 LabelNoise / Bleeding** | MUSDB-like sets with label errors / bleed for robustness | research, NC | MUSDB-sized | music.ai/research | 3 | [Music.AI](https://music.ai/research/) |
| **Slakh2100-redux** | 2,100 tracks synthesised from Lakh MIDI: mix + per-instrument stems + **aligned MIDI**, 145 h | CC BY 4.0 | 104 GB FLAC — take the test split (~20 GB) | Zenodo 4599666; mirdata `slakh` | 5 | [Zenodo](https://zenodo.org/records/4599666) |
| **MedleyDB 1.0 / 2.0** (+ pitch-tracking subset) | 196 multitracks, melody F0, instrument activations | CC BY-NC-SA 4.0 | tens of GB (pitch subset 275 GB) | Zenodo request; mirdata `medleydb_melody`, `medleydb_pitch` | 5 | [site](https://medleydb.weebly.com/) |
| **MDB Drums** | 23 MedleyDB tracks: drum-only + mix + 7,994 onsets in 6/21 classes | CC BY-NC-SA 4.0 | small | GitHub | 3 | [GitHub](https://github.com/CarlSouthall/MDBDrums) |
| **URMP** | 44 classical chamber pieces, per-instrument audio, MIDI scores, F0 | not stated | 12.5 GB | Rochester site | 4 | [site](https://labsites.rochester.edu/air/projects/URMP.html) |
| **CocoChorales** | 240K synthetic 4-voice chorales, stems + MIDI + F0 | CC BY 4.0 | 2.9 TB — tiny subset only | Magenta GCS | 4 | [Magenta](https://magenta.tensorflow.org/datasets/cocochorales) |
| **Filosax** | 48 jazz multitracks + beats/chords/sections + note-level sax | NC, signed agreement | ~35 h | Zenodo request; mirdata `filosax` | 3 | [site](https://dave-foster.github.io/filosax/) |
| **StemGMD** | 9-class isolated drum stems from Groove MIDI, 10 kits | CC BY 4.0 | 1.13 TB — skip | Zenodo 7860223 | 3 | [Zenodo](https://zenodo.org/records/7860223) |
| **Cambridge-MT "Mixing Secrets"** | Hundreds of raw multitracks, no annotations (source of 100 MUSDB tracks) | educational, per-song terms | 100s of GB | site (403 at fetch) *(verify)* | 4 | [site](https://www.cambridge-mt.com/ms/mtk/) |
| **CrowdioSet + PaRIRset** (Jul 2026) / **Spheres** (Nov 2025) / **Sanidha** (2025) | Live-MSS crowd/RIR augmentation; orchestral multitrack; Carnatic multitrack | CC BY | unknown; download locations partly unverified | HF DOIs / arXiv | 2 (new) | [CrowdioSet](https://enricguso.github.io/crowdioset_parirset) · [Spheres](https://arxiv.org/abs/2511.21247) |

No **MoisesDB v2**, **SDX 2025** or **MUSDB-XL** exists as of today.

### 5.2 Notes / instrument transcription

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **MAESTRO v3.0.0** | 1,276 piano performances, 199 h, Disklavier MIDI aligned ±3 ms | CC BY-NC-SA 4.0 | 101 GB (MIDI-only 56 MB) | GCS; mirdata `maestro` | 5 | [Magenta](https://magenta.tensorflow.org/datasets/maestro) |
| **MAPS** / **SMD** | Piano WAV + MIDI (isolated notes, chords, pieces) / Disklavier pairs | CC BY-NC-SA 2.0 FR / research | 31 GB / small | Télécom Paris / AudioLabs | 4 (legacy) | [MAPS](https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/) · [SMD](https://www.audiolabs-erlangen.de/resources/MIR/SMD) |
| **MusicNet** | 330 classical recordings, 1M+ aligned note labels | CC BY 4.0 | 11.1 GB | Zenodo 5120004 | 4 (~4 % label noise) | [Zenodo](https://zenodo.org/records/5120004) |
| **GiantMIDI-Piano** / **ATEPP v1.2** / **PiJAMA** / **Aria-MIDI** / **PianoCoRe** (May 2026) | Transcribed piano MIDI at scale; PianoCoRe unifies 250K performances / 21.7K h with score alignments | CC BY 4.0 / CC BY 4.0 / see repo / CC BY-NC-SA 4.0 / CC BY-NC-SA 4.0 | MB–tens of GB (Aria-MIDI 16.6 GB) | GitHub / HF `loubb/aria-midi` / Zenodo | 4 | [GiantMIDI](https://github.com/bytedance/GiantMIDI-Piano) · [ATEPP](https://github.com/BetsyTang/ATEPP) · [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) · [PianoCoRe](https://github.com/ilya16/PianoCoRe) |
| **ASAP** | 222 scores ↔ 1,068 piano performances; beats, downbeats, key, time-sig (audio from MAESTRO) | CC BY-NC-SA 4.0 | small | GitHub | 4 | [GitHub](https://github.com/fosfrancesco/asap-dataset) |
| **GuitarSet** | 360 × 30 s acoustic guitar, hex pickup + mic; per-string notes/F0, beats, tempo, chords, key | MIT | few GB | Zenodo 3371780; mirdata `guitarset` | 5 | [site](https://guitarset.weebly.com/) |
| **EGDB** / **SynthTab** | Electric guitar DI + amp renders + MIDI / 15K DadaGP tabs rendered to 6,700 h | not stated / see page | few GB / 1 GB dev set | project pages | 3 | [EGDB](https://ss12f32v.github.io/Guitar-Transcription/) · [SynthTab](https://synthtab.dev/) |
| **IDMT-SMT Bass / Guitar / Drums / Chords** | Short annotated clips per instrument | mostly CC BY-NC-ND | small | Fraunhofer IDMT pages | 4 (legacy) | [IDMT](https://www.idmt.fraunhofer.de/en/publications/datasets.html) |
| **Groove MIDI (GMD)** / **E-GMD** | 13.6 h human drumming MIDI + audio / 444 h, 43 kits, velocity | CC BY 4.0 | 4.8 GB / 90 GB (MIDI 103 MB) | Magenta; mirdata `groove_midi` | 5 | [GMD](https://magenta.tensorflow.org/datasets/groove) · [E-GMD](https://magenta.tensorflow.org/datasets/e-gmd) |
| **ENST-Drums** / **ADTOF** | 3 drummers, 8-channel multitrack / 359 h drum onsets from rhythm-game charts (audio not included) | NC click-through / CC BY-NC-SA 4.0 | few GB / mel-specs on request | Télécom Paris / GitHub + Zenodo | 4 / 3 | [ENST](https://perso.telecom-paristech.fr/grichard/ENST-drums/) · [ADTOF](https://github.com/MZehren/ADTOF) |
| **NSynth** | 306K 4-s single notes, 1,006 instruments | CC BY 4.0 | ~30 GB | Magenta | 5 | [Magenta](https://magenta.tensorflow.org/datasets/nsynth) |
| **TuttiCorpus / TUTTI** (Sept 2026) | Synthetic multi-instrument audio–score pairs | CC BY-NC 4.0 | not stated | HF `pzzzzz/TuttiCorpus` (partially uploaded) | 2 (new) | [GitHub](https://github.com/a-musiclover/TUTTI) |

### 5.3 Vocals / lyrics / captions

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **DALI v1/v2** | ~5–8K songs with notes / words / lines / paragraphs aligned; audio via YouTube IDs | CC BY-NC-SA 4.0 | annotations small; audio DIY | GitHub + `pip install dali-dataset`; mirdata `dali` | 4 (auto-aligned, noisy; YouTube rot) | [GitHub](https://github.com/gabolsgabs/DALI) |
| **JamendoLyrics MultiLang** | 80 CC songs, 4 languages, word- and line-level timestamps, MP3 included — the word-level benchmark | per-song CC | small | HF `jamendolyrics/jamendolyrics` (GitHub deprecated Apr 2025) | 5 | [GitHub](https://github.com/f90/jamendolyrics) |
| **Jam-ALT v1.4.0** | Same songs, readability-aware transcripts + line-level timings; `alt-eval` metrics | per-song CC | small | HF `jamendolyrics/jam-alt` | 4 | [HF](https://huggingface.co/datasets/jamendolyrics/jam-alt) |
| **MIR-ST500** | 500 pop songs, vocal note events; audio via YouTube (script updated 2026) | not stated | audio DIY | GitHub | 4 | [GitHub](https://github.com/york135/singing_transcription_ICASSP2021) |
| **vocadito** / **VocalSet** / **Dagstuhl ChoirSet** / **Cantoría** | Solo-voice excerpts with F0 + notes + lyrics / techniques / choir multitrack / SATB | CC BY 4.0 | 58 MB / 8.1 GB / 5.1 GB / 863 MB | Zenodo; mirdata `vocadito`, `dagstuhl_choirset` | 3–4 | [vocadito](https://zenodo.org/records/5578807) · [VocalSet](https://zenodo.org/records/1442513) · [Dagstuhl](https://zenodo.org/records/4608395) |
| **GTSinger** (2024) / **M4Singer** / **Opencpop** / **PopCS** | Studio singing with MusicXML scores + phoneme alignments (GTSinger 80.6 h, 9 languages) / Mandarin corpora with scores | CC BY-NC-SA 4.0 / custom research | tens of GB / ~10 GB / few GB | HF or Google Drive with licence acceptance | 4 | [GTSinger](https://github.com/GTSinger/GTSinger) · [M4Singer](https://github.com/M4Singer/M4Singer) |
| **VietLyrics** (Oct 2025) | 647 h Vietnamese, line-level aligned | CC BY 4.0 | large; not located on HF *(verify)* | — | 2 | [arXiv](https://arxiv.org/abs/2510.22295) |
| **Song Describer** / **MusicCaps** / **MidiCaps** / **MuChin** | 706 CC tracks + captions / 5.5K YouTube clips + captions / 168K Lakh MIDIs + captions + key/tempo/chords / Chinese songs with section + rhyme structure | CC BY-SA 4.0 / CC BY-SA 4.0 / CC BY-SA 4.0 / MIT + NC audio | 3.3 GB / audio DIY / few GB / tens of GB | Zenodo 10072001 / Kaggle / HF `amaai-lab/MidiCaps` / GitHub | 4 | [SDD](https://zenodo.org/records/10072001) · [MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps) · [MuChin](https://github.com/CarlWangChina/MuChin) |
| **Kara1k** | 2017 cover-song ID set (features; audio on request) — not a karaoke/lyrics-timing dataset | MIT (code) | n/a | GitHub | 2 | [GitHub](https://github.com/ybayle/ISM2017) |

### 5.4 Chords / beats / structure / key

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **RWC Music Database — 2026 re-release** (Popular / Classical / Jazz / Genre / Royalty-free, 315 pieces) | Audio now an open download; AIST annotations: beats (all), chords (RWC-P), beat-aligned MIDI (P, C), melody F0 (P, R), chorus/structure, lyrics, vocal activity — the best full-song ground truth for the round-trip target | CC BY-NC 4.0, no pledge | 13.4 GB WAV (Popular 4.1 GB) | Zenodo 18656623 + GitHub `rwc-music/rwc-annotations`; mirdata `rwc_popular` etc. | 5 | [Zenodo](https://zenodo.org/records/18656623) · [annotations](https://github.com/rwc-music/rwc-annotations) |
| **Harmonix Set** | 912 Western pop tracks: beats, downbeats, functional sections, BPM/meter; mel-spectrograms shipped, audio via YouTube IDs | MIT (annotations) | 1.2 GB | GitHub | 5 | [GitHub](https://github.com/urinieto/harmonixset) |
| **SongFormDB** / **SongFormBench** (Oct 2025) | 15,346-song multilingual structure set (mel-specs, BigVGAN-reconstructable) / 300-song expert 7-class benchmark (200 Harmonix + 100 Chinese) | CC BY 4.0 | 262 GB / 5.3 GB | HF `ASLP-lab/SongFormDB`, `ASLP-lab/SongFormBench` | 3 (new) | [DB](https://huggingface.co/datasets/ASLP-lab/SongFormDB) · [Bench](https://huggingface.co/datasets/ASLP-lab/SongFormBench) |
| **Isophonics** (Beatles, Queen, Carole King, MJ, Zweieck) | Chords, keys, beats, structure; no audio | not stated | KB | isophonics.net; mirdata `beatles`, `queen` | 5 | [site](http://isophonics.net/datasets) |
| **McGill Billboard** | ~890 chord + section transcriptions; no audio | CC0 (per mirdata) | KB | mirdata `billboard` | 5 | (mirdata) |
| **ChoCo** | 20,080 JAMS from 18 sources (Isophonics, Billboard, RealBook, Wikifonia, iRealPro…): chords + key + some beats/structure | CC BY 4.0 (parts BY-NC-SA) | small | GitHub | 4 | [GitHub](https://github.com/smashub/choco) |
| **CASD** / **JAAH** / **Weimar Jazz DB** | Chord annotator subjectivity (50 songs × 4) / 113 jazz tracks beat-aligned chords / 443 solo transcriptions | CC BY-NC-SA 4.0 / see repo / research | KB–small | GitHub / jazzomat | 3–4 | [CASD](https://github.com/chordify/CASD) · [JAAH](https://mtg.github.io/JAAH/) · [WJD](https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html) |
| **SALAMI v2** | ~1,400 structure annotations, 3 layers; audio via YouTube / RWC / Internet Archive | CC0 | KB | GitHub; mirdata `salami` | 5 | [GitHub](https://github.com/DDMAL/salami-data-public) |
| **GiantSteps Key** / **FMA Keys** / **Beatport EDM key** | Key labels with downloadable audio | not stated / CC BY-SA 4.0 | 850 MB / small | GitHub scripts; mirdata `giantsteps_key`, `fma_keys`, `beatport_edm_key` | 4 | [GiantSteps](https://github.com/GiantSteps/giantsteps-key-dataset) |
| **Ballroom** / **GTZAN** | Beats / tempo (+ genre) | CC0 / n.a. | ~2 GB | mirdata `ballroom`, `gtzan_genre` | 4 (legacy) | (mirdata) |
| **MTG-Jamendo** / **FMA** | 55K / 106K CC full tracks with tags and genres | CC BY-NC-SA meta, per-track CC audio / CC BY 4.0 meta | 508 GB (156 GB low-q) / 7.2 GB small – 879 GB full | GitHub downloaders | 5 | [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) · [FMA](https://github.com/mdeff/fma) |
| **EDM-98** (Mar 2026) | 98 pro-annotated EDM tracks (drops/build-ups) | CC BY-NC-ND | download not found *(verify)* | — | 2 | [arXiv](https://arxiv.org/abs/2603.08759) |

### 5.5 Symbolic

| name | purpose | license | Mac? (RAM needed) | install | maturity | link |
|---|---|---|---|---|---|---|
| **POP909** (+ **POP909-CL**, Oct 2025) | 909 pop piano arrangements (melody / bridge / piano MIDI) + beats, chords, key; CL = human-corrected chords/beats/keys/time-sigs | MIT / CC BY | small | GitHub (CL not located on HF *(verify)*) | 4 | [GitHub](https://github.com/music-x-lab/POP909-Dataset) · [CL](https://arxiv.org/abs/2510.06528) |
| **Lakh MIDI** (full / matched / aligned) | 176,581 MIDI; 45,129 matched to MSD; aligned to 30-s previews | CC BY 4.0 | ~10 GB | colinraffel.com | 5 | [site](https://colinraffel.com/projects/lmd/) |
| **MetaMIDI** / **GigaMIDI v2** | 436K MIDI (237K Spotify-matched) / 2.1M MIDI with expressive-loop tags | restricted Zenodo agreement / CC BY-NC 4.0 | ~20 GB / 29.5 GB | Zenodo request / HF `Metacreation/GigaMIDI` | 4 | [MetaMIDI](https://github.com/jeffreyjohnens/MetaMIDIDataset) · [GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) |
| **PDMX** | 250K+ public-domain MuseScore scores: MusicXML / MXL / MIDI / PDF / JSON (lyrics present where scored) | MIT code; PD content (`no_license_conflict` subset) | tens of GB | Zenodo 15571083 | 4 | [GitHub](https://github.com/pnlong/PDMX) |
| **HookTheory / TheoryTab** (Sheet Sage) | 50 h melody + chords aligned to YouTube audio; JSON 20 MB; MARBLE `HookTheory*` tasks | CC BY-NC-SA 3.0 | audio DIY | GitHub | 4 | [GitHub](https://github.com/chrisdonahue/sheetsage) |
| **IrishMAN** / **Nottingham** | 216K ABC tunes (34K with chords) / 1,200 folk tunes in ABC with chord symbols | MIT + PD / none stated | 579 MB / KB | HF `sander-wood/irishman` / ifdo.ca | 4 | [IrishMAN](https://huggingface.co/datasets/sander-wood/irishman) · [Nottingham](https://ifdo.ca/~seymour/nottingham/nottingham.html) |
| **EMOPIA** / **XMIDI** / **Los Angeles MIDI v4** / **SymphonyNet** / **NES-MDB** | Piano clips with audio + MIDI + emotion / 108K emotion+genre MIDI / 405K deduped MIDI + chord metadata / 46K symphonic MIDI / 5,278 NES songs | see record / not stated / Apache-2.0 / not stated / MIT | few GB each | Zenodo / Google Drive / HF / GitHub | 3–4 | [EMOPIA](https://annahung31.github.io/EMOPIA/) · [XMIDI](https://github.com/xmusic-project/XMIDI_Dataset) · [LA-MIDI](https://github.com/asigalov61/Los-Angeles-MIDI-Dataset) · [NES-MDB](https://github.com/chrisdonahue/nesmdb) |
| **MARBLE v2** / **MuChoMusic** / **SongEval** / **Music4All** | Benchmark harness (GTZAN beat, GiantSteps key, Chords1217, HookTheory, MTG) / music QA benchmark / human aesthetic ratings / 109K songs with lyrics + Spotify key/tempo (email request) | per-dataset / CC BY-SA 4.0 / CC BY-NC-SA 4.0 / request | varies | GitHub / email | 4 | [MARBLE](https://github.com/a43992899/MARBLE) · [MuChoMusic](https://github.com/mulab-mir/muchomusic) · [SongEval](https://github.com/ASLP-lab/SongEval) · [Music4All](https://sites.google.com/view/contact4music4all) |

### Recommended default + alternatives per evaluation target

| target | recommended default | alternatives / notes |
|---|---|---|
| Separation (Milestone 1) | **MUSDB18-HQ** test split (50 songs) — every published SDR is on it | MoisesDB for the >4-stem taxonomy (guitar/keys/percussion sub-stems match the per-instrument model); Slakh subset for synthetic per-instrument stems + MIDI; SDXDB23 Bleeding for robustness |
| Notes per instrument | **Slakh2100-redux** test split (perfectly aligned MIDI per stem) | MusicNet (real classical), URMP (real chamber), GuitarSet/EGDB, MAESTRO/ASAP (piano with beats/key), RWC-P aligned MIDI (100 real pop songs, beat-level sync) |
| Drums | **E-GMD** or Groove MIDI (real hits with velocity) | MDB Drums (in-mix real songs), ENST-Drums, ADTOF for scale |
| Vocal melody / F0 | **MedleyDB melody** (full-mix vocal F0; request access) | MIR-ST500 (note events, YouTube), RWC-P melody F0 (now open), vocadito for isolated-voice sanity checks, DALI notes (noisy) |
| Lyrics alignment (Milestone 2, ≤80 ms) | **JamendoLyrics MultiLang** (word-level, audio included) | Jam-ALT (line-level + readable), MUSDB18 lyrics extension (line-level on the same songs as separation — joint eval), DALI (scale), RWC-P lyrics |
| Chords | **RWC-Popular** (audio open, chords + beats + structure + MIDI in one place) | Isophonics Beatles (own CDs), Billboard (no audio), ChoCo (unified JAMS), CASD (annotator-disagreement tolerance for `?0.NN`), JAAH |
| Key | **GiantSteps Key** + Isophonics keys | FMA Keys / Beatport EDM; POP909(-CL); ASAP (classical) |
| Beats / downbeats | **Harmonix Set** (912 pop; mel-specs shipped, YouTube for audio) | RWC (315 pieces, real audio open), Ballroom/GTZAN, GuitarSet, ASAP |
| Structure | **Harmonix Set** sections cross-checked with **SongFormBench** (300 expert songs) | SALAMI (CC0, multi-layer), SongFormDB (training), RWC chorus sections |
| Full-song round-trip (Milestone 3 "re-analysing gives back the same `.sc`") | **RWC-Popular** — 100 real pop songs with audio + beats + chords + melody F0 + aligned MIDI + structure + lyrics, CC BY-NC, 4.1 GB | MUSDB18-HQ ∪ lyrics extension ∪ own chord/beat annotations (same 150 songs → stems + lyrics); Slakh (audio + MIDI, no lyrics); HookTheory test set (melody + chords vs YouTube audio) |
| `.sc` fine-tuning corpus (Milestone 5) | Render **Lakh-matched / MidiCaps** (captions + key/tempo/chords already extracted) via FluidSynth → `.sc` pairs; add **Slakh** for real stem renders; **POP909** for melody/chord/beat structure; **PDMX** for lyric-bearing scores | GigaMIDI (2.1M, expressive-loop tags), Aria-MIDI / PianoCoRe (expressive piano timing → `dev±ms`), HookTheory JSON (melody + chords, 20 MB), IrishMAN (chords in ABC) |

Synthetic-data notes: MIDI → audio via FluidSynth + SF2/SFZ gives
sample-accurate timing, velocity, per-instrument stems, tempo map and key
(Slakh's recipe; Slakh's `metadata.yaml` patch lists are a template for
`inst=` / `timbre{}`). For vocals, ACE-Step 1.5 produces full songs from
lyrics but no ground-truth word timings (forced alignment still needed);
DiffSinger-family SVS (see the synthesis doc) yields note- and phoneme-exact
vocals from GTSinger/M4Singer scores. Disk budget on 1–2 TB: MUSDB18-HQ 23 GB
+ RWC 13 GB + MoisesDB ~100 GB + Slakh test ~20 GB + MAESTRO MIDI-only +
E-GMD 90 GB + Harmonix 1.2 GB + SongFormBench 5 GB ≈ 250 GB covers every
target; skip CocoChorales, StemGMD, SynthTab-full, FMA-full, MTG-Jamendo-full
and SongFormDB.

---

## Recommended open codifying stack for Infinity Engine

Everything below is open-weight, free, and runs on the M1 Max unless marked
"rented GPU". Licences: Apache-2.0 / MIT unless noted; NC items are fine for
this project but everything derived from them inherits the NC term.

### `.sc`-authoring LLM and how to constrain it (Milestone 5, Phase 5)

| role | choice | notes |
|---|---|---|
| Primary author / repairer | **Qwen3.8-27B, 4-bit, MLX** (`mlx-community/Qwen3.8-27B-4bit`, 16 GB) | Thinking off for constrained emission, on for repair reasoning; ≤64K context. Fallback to Qwen3.5-27B-4bit if `mlx_lm` will not load the mlx-vlm quant. |
| Fast loop model | **Qwen3.6-35B-A3B 4-bit** (repair iterations, many-shot) and **gpt-oss-20b** (checker-loop triage; 62–72 tok/s measured on M1 Max) | |
| Second opinion | **Gemma 4 31B-it 4-bit** | different lineage; also useful as an LLM-judge for `desc=` / `@style` prose |
| Constraint | **One Lark grammar for `.sc` (lexical layer of spec §4.6) via Outlines `outlines[mlxlm]` → llguidance** | Same grammar runs in llama.cpp (`-DLLAMA_LLGUIDANCE=ON`) and vLLM on a rented GPU. Syntax only; `check --strict` owns semantics; feed line-numbered errors back with quoted lines; regenerate per stream. Exclude `<think>` from the constraint. |
| JSON-first sub-tasks | brief → structure/harmony plan (Pydantic schema), programmatic edits | any runtime (ollama / LM Studio / Outlines JSON) |
| Prompting | spec + 15–30 checker-clean `.sc` examples in a cached prefix | LilyBench: curated examples 97–99.9 % validity vs 20–45 % for careless ones |
| Fine-tuning (step two) | **`mlx_lm.lora` QLoRA + DoRA** on the 4-bit base; 300–1K pairs for syntax, 2–5K with brief-following and repair; synthetic pairs from Lakh/MidiCaps/POP909/NotaGen-X rendered through `emit.py` + injected-fault repair pairs | ~10–20 h/epoch per 1M tokens on M1 Max (estimate); mlx-tune for DPO/GRPO against the checker; bf16 full fine-tunes → rented GPU |
| Suggestion tools | MIDI-LLM (1B) / Text2midi for melody and bass proposals; ChatMusician's MusicTheoryBench for regression checks of theory knowledge | never as the primary writer |

### Audio-LLM helper (encode-time describe pass; lyrics second opinion)

| role | choice | notes |
|---|---|---|
| Captions, `@style` / `@mix` / `desc=` prose, timestamped questions | **MOSS-Music-8B-Thinking (MusicCaps) or -Instruct (SDD/QA), 8-bit on MLX (~10 GB)** via mlx-audio `moss_music` | Apache-2.0; only Mac-runnable music LLM trained with time markers; chord/key/tempo accuracy unpublished — never let it set those |
| Instrument tags (`tags{}`) | **MuQ-MuLan zero-shot** (CC BY-NC weights) + **Essentia MTG-Jamendo-instrument** + existing msclap; MOSS-Music for targeted "what plays at 0:32" questions | audio LLMs collapse on NSynth-instrument (Qwen3-Omni 30.9 %) |
| Lyrics second opinion | **MOSS-Music** (native timestamps; MUSDB18 WER 29.2, best of the audio LLMs) in the voting step of the lyrics reconciliation plan | primary ASR stays Qwen3-ASR |
| Generalist fallback | **Qwen3-Omni-30B-A3B** via llama.cpp `ggml-org` GGUF (Q4 ≈17–22 GB, runs alone) | when album art / video context matters |
| Rented GPU only | Music Flamingo-2601 / AF-Next-Captioner (NVIDIA NC) | best instrumentation prose; not needed on Mac |

### Transcription models (Milestones 2–3, Phases 1–2)

| `.sc` stream | model | input |
|---|---|---|
| `:grid` | **SheetSage2** beats/downbeats reconciled with **Beat This!** (already chosen); disagreements → `meta warn=` | full mix |
| `:harmony` key + chords | **SheetSage2** (osu2017 maj/min 90.08; GiantSteps key 77.73); **ChordMini BTC** (MIT, 170-class) for extensions/inversions; cross-check vs `:notes.bass` roots per spec | full mix |
| `:struct` | **SongFormer** boundaries (HR.5F 0.703) + **SheetSage2** labels; energy/vocal flags from stems; `desc=` from MOSS-Music | full mix |
| `:notes.vox` | **SheetSage2 vocal-melody voice** for pitch classes and octaves; onset/offset precision from Basic Pitch / SwiftF0 note segmentation on the vocal stem | mix + vocal stem |
| `:contour.vox` | **torchcrepe / penn** (or **SwiftF0** for speed) | vocal stem |
| `:text.vox` | **Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B** on both the mix and the vocal stem, voted with LRCLIB/web lyrics + MOSS-Music per the reconciliation plan | mix + vocal stem |
| `:perc.drums` | **DrumSep (MSST, MPS) → ADTOF** (CC BY-NC-SA) with velocity from per-stem loudness | drum stem |
| `:notes.bass` | **Basic Pitch** with pitch-range prior; MuScriptor bass program as second opinion | bass stem |
| `:notes.<keys/other>` | **Transkun v2** (piano, with velocity + pedal) and **MuScriptor large** (MPS; instrument labels) on the other stem | other stem |
| Rented GPU experiments | Aria-AMT (robust piano), SongPrep-7B (structure + lyrics, academic-only), VocalParse (Mandarin-trained) | |

Design rule confirmed by the evidence: end-to-end owns the annotation layer
(grid, key, chords, structure, lead melody); per-stem specialists own notes,
drums and timing. Everything is reconciled onto one beat grid.

### Interop formats and libraries (Phase 0 tooling, `inspect` / JSON export)

| target | library |
|---|---|
| MIDI in/out, beats, tokens | **symusic** (+ **MidiTok** REMI/TSD, git-pinned) |
| MusicXML in/out | **music21 10.x** (py 3.11); **partitura** for score↔performance alignment (`dev±ms`) |
| ABC | custom emitter + **abcMIDI** validation; optional `:abc.<trk>` islands parsed via abc2midi/symusic; **abcjs** in `soundcode serve` |
| Engraving / render | **MuseScore 4.6 CLI** (PDF/PNG/MIDI/MP3), **LilyPond 2.26**, **Verovio** for SVG, **FluidSynth** for GM audio |
| Annotations + metrics | **JAMS** sidecar (confidence field maps to `?0.NN`) + **mir_eval** (Harte chord compare, beat/segment/melody/transcription metrics) + **mirdata** loaders |
| Chord syntax | jazz/ChordPro surface (`Am7/C`) in `.sc`; Harte (`A:min7/b3`) internally via `pychord` → `mir_eval.chord` |
| Lyrics timing | enhanced LRC `<mm:ss.xx>` import/export alias; JAMS `lyrics` |
| Contour / microtonal / drums | JAMS `pitch_contour` or CREPE-style CSV; MPE-style per-channel bend (RPN ±48 st) for MIDI export; General MIDI percussion map on channel 10 |
| `.sc` itself | **stays custom**; borrows `\|` bars, scientific pitch + cents, optional Strudel-style mini-notation in `%pat`, Alda-style named anchors |

### Evaluation datasets (Phase 6)

| target | dataset |
|---|---|
| Separation | MUSDB18-HQ test (MoisesDB for sub-stems) |
| Full-song round-trip | **RWC-Popular** (open since Feb 2026: audio + beats + chords + melody + aligned MIDI + structure + lyrics) |
| Per-instrument notes | Slakh2100-redux test split; MAESTRO/ASAP (piano); GuitarSet |
| Drums | E-GMD / Groove MIDI; MDB Drums in-mix |
| Vocal melody | MedleyDB melody; MIR-ST500; RWC-P F0 |
| Lyrics alignment | JamendoLyrics MultiLang (word); Jam-ALT; MUSDB18 lyrics extension |
| Chords / key | RWC-P, Isophonics, ChoCo, CASD (tolerance); GiantSteps Key |
| Beats / structure | Harmonix Set; SongFormBench; SALAMI |
| LLM corpus | Lakh-matched + MidiCaps + POP909 + PDMX rendered via FluidSynth → `.sc`; GigaMIDI / Aria-MIDI for expressive timing |

Total disk for the full evaluation set ≈ 250 GB.

---

## Unverified items and open questions

- M1 Max tokens/s for 27B-dense and 35B-A3B MLX quants (estimates only);
  whether `mlx_lm` (vs `mlx_vlm`) loads the Qwen3.6/3.8 and Gemma 4
  community quants text-only; Nemotron / Granite MLX conversions; MuPT
  licence.
- mlx-lm release dates as fetched read "2025" for releases that contain
  Qwen3.5 / Gemma 4 / GLM5 support (which shipped in 2026); versions
  0.30.7–0.31.3 are reliable, the dates are not. Gemma 4 model card shows a
  July 30 2026 revision date against an April 2026 launch.
- MOSS-Music: maximum audio duration, VRAM table and chord/key/beat accuracy
  unpublished; whether `moss_music_mlx` (named on the mlx-community card) is
  the same as mlx-audio's `moss_music`.
- Qwen3-Omni audio input through mlx-vlm; Qwen2.5-Omni MLX conversions
  likely text-only; Music Flamingo GGUFs have no evidence of a working audio
  tower; Kimi-Audio weights licence.
- SheetSage2: no tech report; code licence unresolved; CPU/MPS speed and
  memory unmeasured; its Beat This! baseline (86.27) disagrees with the Beat
  This! paper (89.1); `audio.cpp` GGUF runtime Metal support.
- SongFormer, Step-Audio 2 mini, Aria-AMT on CPU/MPS; Mel-RoFormer
  transcription weights; Harmonica, N2N, Separate-and-Detect, TART,
  Noise2Fret, MIROS, ChordFormer have no public code.
- SongPrep and VocalParse are trained mainly on Chinese songs; English-pop
  transfer untested.
- MidiTok exact release date (late 2024); xml2abc/abc2xml licence text;
  basic-pitch's "M1 needs py 3.10" README note may be stale; MPE bend-range
  and CREPE CSV column order from memory; MusicXML 4.1 / ABC 2.2 draft
  status.
- Dataset download locations for Spheres, Sanidha, VietLyrics, EDM-98,
  POP909-CL; MoisesDB on-disk size and DALI song counts from memory;
  Billboard, Cambridge-MT, NUS-48E, Opencpop pages unreachable at fetch.
- Names that returned nothing and should be treated as non-existent or
  renamed: "Agogic" as a model (it is a tokenisation paper), "ScoreLLM",
  "MusicInfuser", "NoteTrans", "OmniAMT", "NoteFormer", "MoisesDB v2",
  "SDX 2025", "MUSDB-XL", a 2025 "Kara1k".
