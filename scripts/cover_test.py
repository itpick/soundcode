"""Path B smoke test: drive ACE-Step 1.5 `cover` from our mock render.

The decoder has no melody/note conditioning, so the only way symbolic detail
reaches it is by rendering .sc to crude-but-correct audio and using that as
`src_audio` for the cover task. This script sweeps `audio_cover_strength` so we
can hear where structure adherence starts trading against production quality.

Run from the ACE-Step checkout with its own venv:

    cd external/ACE-Step-1.5
    .venv/bin/python ../../scripts/cover_test.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ACE = Path(__file__).resolve().parents[1] / "external" / "ACE-Step-1.5"
SOUNDCODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACE))

from acestep.handler import AceStepHandler          # noqa: E402
from acestep.inference import (                     # noqa: E402
    GenerationConfig,
    GenerationParams,
    generate_music,
)
from acestep.llm_inference import LLMHandler        # noqa: E402

SRC = SOUNDCODE / "out" / "signal-lost.mock.wav"
OUT = SOUNDCODE / "out" / "cover"

# From the .sc header @style. The decode bridge will author this later.
CAPTION = (
    "mid-tempo industrial pop, dry compressed drums, distorted saw bass, "
    "clean electric piano pad, close-mic male vocal with doubled female "
    "backing, moderate plate reverb"
)

# From :struct + :text.vox. Section tags are what the decoder actually reads.
# (Original lyrics, written for this control track.)
LYRICS = """[Verse]
I was a signal in the static
you were the hand that turned the dial
now every word I ever said
comes back to me as numbers

[Chorus]
Hold the line, hold the line
nothing here is lost
"""

STRENGTHS = [0.4, 0.6, 0.8]


def main() -> int:
    if not SRC.exists():
        print(f"missing mock render: {SRC}", file=sys.stderr)
        print("run: soundcode render examples/signal-lost.v3.sc -o out/signal-lost.mock.wav")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    dit = AceStepHandler()
    llm = LLMHandler()          # cover is in skip_lm_tasks; never initialized

    t0 = time.time()
    print("initializing DiT (turbo, MLX)...")
    dit.initialize_service(
        project_root=str(ACE),
        config_path="acestep-v15-turbo",
        device="auto",
        use_mlx_dit=True,
    )
    print(f"  ready in {time.time() - t0:.1f}s  device={dit.device}")

    for strength in STRENGTHS:
        params = GenerationParams(
            task_type="cover",
            src_audio=str(SRC),
            audio_cover_strength=strength,
            caption=CAPTION,
            lyrics=LYRICS,
            vocal_language="en",
            bpm=128,
            keyscale="A minor",
            timesignature="4/4",
            duration=30.0,
            inference_steps=8,          # turbo
            seed=12345,                 # fixed, so strength is the only variable
            thinking=False,             # cover skips the LM anyway
            use_cot_metas=False,
            use_cot_caption=False,
        )
        config = GenerationConfig(batch_size=1, audio_format="wav")

        dest = OUT / f"strength_{strength:.1f}"
        dest.mkdir(exist_ok=True)
        print(f"\ncover @ strength {strength} ...")
        t = time.time()
        try:
            result = generate_music(dit, llm, params, config, save_dir=str(dest))
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.time() - t
        files = getattr(result, "audio_files", None) or list(dest.glob("*.wav"))
        print(f"  {elapsed:.1f}s  ({elapsed / 30.0:.2f}x realtime)  -> {files}")

    print(f"\ntotal {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
