"""Build a TensorRT engine from the exported ONNX — the production inference path on L4.

We use ``trtexec`` (ships with TensorRT) for a reproducible, scriptable build rather than
hand-rolling the Python builder: it is the same tool NVIDIA benchmarks with, and the
flags below are the ones that matter for our KPIs.

KPI mapping
-----------
* ``--fp16``  : Ada (L4) has fast FP16/BF16 tensor cores; FP16 roughly halves both
                latency and weight VRAM vs FP32 with no audible quality loss for VITS.
                → directly serves RTF<0.3 and the ≤3 GB VRAM budget.
* dynamic shape profile (min/opt/max phoneme length): lets one engine serve short
                first-chunks (low TTFA) *and* longer chunks without rebuild.
* ``--int8``  (optional, with a calibration cache): weight+activation INT8 for the
                heaviest convs; we keep it opt-in because it needs a calibration set and
                can nick quality — FP16 already meets the targets, INT8 is headroom.

Frame-level streaming (advanced): export the HiFi-GAN decoder as a second ONNX and build
a separate engine; the runtime then decodes latent windows with a few-frame receptive
overlap so first audio is emitted after the first window instead of the whole chunk.
The default single-engine path already meets <300 ms TTFA via short text chunks.

Usage (GPU host):
  python -m hams_tts.models.optimize.build_tensorrt --onnx hams_vits.onnx --engine hams_vits.plan \
      --fp16 --min 4 --opt 32 --max 256
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def build(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    min_len: int = 4,
    opt_len: int = 32,
    max_len: int = 256,
    calib_cache: str | None = None,
    workspace_mb: int = 2048,
) -> str:
    trtexec = shutil.which("trtexec")
    if not trtexec:
        sys.exit(
            "trtexec not found. Install TensorRT (it ships with the NGC TensorRT/PyTorch "
            "containers, e.g. nvcr.io/nvidia/pytorch:24.05-py3) and ensure trtexec is on PATH."
        )

    def shapes(name: str) -> str:
        return (
            f"{name}:1x{min_len},{name}:1x{opt_len},{name}:1x{max_len}"
            if name != "phoneme_lengths" and name != "speaker_id"
            else f"{name}:1,{name}:1,{name}:1"
        )

    # min/opt/max profiles for the dynamic phoneme axis
    min_shapes = f"phoneme_ids:1x{min_len},language_ids:1x{min_len},phoneme_lengths:1,speaker_id:1"
    opt_shapes = f"phoneme_ids:1x{opt_len},language_ids:1x{opt_len},phoneme_lengths:1,speaker_id:1"
    max_shapes = f"phoneme_ids:1x{max_len},language_ids:1x{max_len},phoneme_lengths:1,speaker_id:1"

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--builderOptimizationLevel=5",
    ]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")
        if calib_cache:
            cmd.append(f"--calib={calib_cache}")
    print("[trt] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[trt] saved engine -> {engine_path}")
    return engine_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", default="hams_vits.plan")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--min", type=int, default=4)
    ap.add_argument("--opt", type=int, default=32)
    ap.add_argument("--max", type=int, default=256)
    ap.add_argument("--calib", default=None)
    args = ap.parse_args()
    build(args.onnx, args.engine, fp16=args.fp16, int8=args.int8,
          min_len=args.min, opt_len=args.opt, max_len=args.max, calib_cache=args.calib)


if __name__ == "__main__":
    main()
