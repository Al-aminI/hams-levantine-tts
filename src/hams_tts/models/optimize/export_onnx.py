"""Export HamsVITS to ONNX (the bridge from PyTorch to the deployable runtime).

We export the *full* graph (phoneme/lang/speaker embeddings → text encoder → stochastic
duration predictor → flow → HiFi-GAN decoder → waveform) with a dynamic phoneme-length
axis.  ONNX is the portable IR that ONNX Runtime (CUDA EP) and TensorRT both consume; it
is also where graph-level fusions happen.

Why this hits the KPIs:
  * VITS is non-autoregressive, so the whole graph is a single forward — no per-token
    loop — which is what keeps RTF well under 0.3 and TTFA low once combined with the
    front-end's short first chunk.
  * Exporting once lets us build an FP16 TensorRT engine (≈2× faster, ≈half the VRAM)
    in :mod:`build_tensorrt`, and an INT8 weight-quantised ORT model in :mod:`quantize`
    for the ≤3 GB budget.

For *frame-level* streaming (even lower TTFA) the decoder can be exported separately and
run over latent sub-windows; that split path is documented in build_tensorrt.py.  The
default full-graph export is the robust baseline and is what the benchmark uses.

Run on the GPU host:  python -m hams_tts.models.optimize.export_onnx --ckpt <dir> --out model.onnx
"""

from __future__ import annotations

import argparse


def export(checkpoint: str, out_path: str, opset: int = 17, verify: bool = True) -> str:
    import numpy as np
    import torch

    from ..hams_vits import HamsVITS

    model = HamsVITS.from_checkpoint(checkpoint).eval()

    class ExportWrapper(torch.nn.Module):
        """Flatten HamsVITS.infer into a single traceable forward."""

        def __init__(self, m: HamsVITS):
            super().__init__()
            self.m = m

        def forward(self, phoneme_ids, phoneme_lengths, language_ids, speaker_id):
            self.m.embed.cur_lang_ids = language_ids
            attention_mask = (
                torch.arange(phoneme_ids.shape[1], device=phoneme_ids.device)[None, :]
                < phoneme_lengths[:, None]
            ).long()
            kwargs = {"input_ids": phoneme_ids, "attention_mask": attention_mask}
            if self.m.config.num_speakers > 1:
                kwargs["speaker_id"] = speaker_id
            return self.m.backbone(**kwargs).waveform

    wrapper = ExportWrapper(model).eval()

    # example inputs
    L = 24
    phoneme_ids = torch.randint(1, model.config.vocab_size, (1, L), dtype=torch.long)
    phoneme_lengths = torch.tensor([L], dtype=torch.long)
    language_ids = torch.ones((1, L), dtype=torch.long)
    speaker_id = torch.zeros((1,), dtype=torch.long)

    input_names = ["phoneme_ids", "phoneme_lengths", "language_ids", "speaker_id"]
    dynamic_axes = {
        "phoneme_ids": {0: "batch", 1: "phonemes"},
        "language_ids": {0: "batch", 1: "phonemes"},
        "phoneme_lengths": {0: "batch"},
        "speaker_id": {0: "batch"},
        "waveform": {0: "batch", 1: "samples"},
    }

    torch.onnx.export(
        wrapper,
        (phoneme_ids, phoneme_lengths, language_ids, speaker_id),
        out_path,
        input_names=input_names,
        output_names=["waveform"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # use the stable TorchScript exporter (torch>=2.6 defaults to dynamo,
                       # which over-specializes our dynamic phoneme axis)
    )
    print(f"[export] wrote {out_path} (opset {opset})")

    if verify:
        import onnxruntime as ort

        with torch.inference_mode():
            ref = wrapper(phoneme_ids, phoneme_lengths, language_ids, speaker_id).cpu().numpy()
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        feeds = {
            "phoneme_ids": phoneme_ids.numpy(),
            "phoneme_lengths": phoneme_lengths.numpy(),
            "language_ids": language_ids.numpy(),
            "speaker_id": speaker_id.numpy(),
        }
        # a single-speaker graph prunes the unused speaker_id input — feed only what exists
        inames = {i.name for i in sess.get_inputs()}
        got = sess.run(None, {k: v for k, v in feeds.items() if k in inames})[0]
        # VITS has stochastic duration/prior; compare shapes + that audio is finite.
        ok = np.isfinite(got).all() and got.ndim == 2
        print(f"[verify] torch={ref.shape} onnx={got.shape} finite={bool(np.isfinite(got).all())} ok={ok}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HamsVITS checkpoint dir")
    ap.add_argument("--out", default="hams_vits.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    export(args.ckpt, args.out, args.opset, verify=not args.no_verify)


if __name__ == "__main__":
    main()
