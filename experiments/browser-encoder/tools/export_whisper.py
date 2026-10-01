"""Export an openai/whisper-* checkpoint to ONNX in the transformers.js layout,
with decoder cross-attention outputs (needed for word timestamps).

Only needed when the Hugging Face model CDN is unreachable (it was, from the
spike machine): the page then loads `models/<id>/` from its own origin instead
of `onnx-community/whisper-*_timestamped` from huggingface.co. Same recipe as
transformers.js's scripts/convert.py + scripts/extra/whisper.py.

    python tools/export_whisper.py openai/whisper-base models/whisper-base_timestamped

Needs: optimum[exporters]==1.19.2 transformers==4.40.2 torch onnx "numpy<2".
(Newer transformers adds a `cache_position` decoder input that transformers.js 3.7
does not feed.)
"""

import shutil
import sys
from pathlib import Path

from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.base import ConfigBehavior
from optimum.exporters.onnx.model_configs import WhisperOnnxConfig
from transformers import AutoConfig


class CrossAttnWhisperOnnxConfig(WhisperOnnxConfig):
    @property
    def outputs(self):
        out = super().outputs
        if self._behavior is ConfigBehavior.DECODER:
            for i in range(self._config.decoder_layers):
                out[f"cross_attentions.{i}"] = {0: "batch_size", 2: "decoder_sequence_length",
                                                3: "encoder_sequence_length_out"}
        return out


def main(model_id: str, out: str) -> None:
    out_dir = Path(out)
    tmp = out_dir / "_export"
    task = "automatic-speech-recognition-with-past"
    cfg = AutoConfig.from_pretrained(model_id)
    c = CrossAttnWhisperOnnxConfig(config=cfg, task="automatic-speech-recognition")
    main_export(
        model_id, output=tmp, task=task, opset=14,
        model_kwargs={"output_attentions": True},
        custom_onnx_configs=dict(
            encoder_model=c.with_behavior("encoder"),
            decoder_model=c.with_behavior("decoder", use_past=True, use_past_in_inputs=False),
            decoder_with_past_model=c.with_behavior("decoder", use_past=True, use_past_in_inputs=True),
        ),
    )
    (out_dir / "onnx").mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        if f.suffix == ".onnx" or f.name.endswith(".onnx_data"):
            if f.stem in ("encoder_model", "decoder_model_merged"):
                shutil.move(str(f), out_dir / "onnx" / f.name)
        elif f.suffix == ".json" or f.suffix == ".txt":
            shutil.move(str(f), out_dir / f.name)
    shutil.rmtree(tmp)

    # fp32 only: onnxruntime's dynamic quantisation leaves the merged
    # decoder's If-subgraphs (i.e. all its weights) unquantised.
    for f in sorted((out_dir / "onnx").iterdir()):
        print(f"{f.name}\t{f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
