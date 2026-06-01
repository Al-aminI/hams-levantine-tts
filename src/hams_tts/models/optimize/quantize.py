"""ONNX Runtime quantisation — INT8 weights for the ≤3 GB VRAM budget / CPU edge.

Two modes:
  * **dynamic** (weight-only INT8): no calibration data needed, safest quality, good for
    shrinking the model footprint and speeding up matmul-heavy parts.
  * **static** (INT8 weights + activations): needs a small calibration set (we reuse the
    eval utterances); higher speed-up, slightly more quality risk — validate with the
    ASR round-trip + UTMOS in the eval harness before shipping.

On the L4 we prefer the TensorRT FP16 engine (see build_tensorrt.py); these ORT-quantised
graphs are the portable fallback (CPU/edge, or GPUs without a TRT build) and a useful
lever if FP16 alone leaves us near the VRAM ceiling under concurrency.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List


def quantize_dynamic(onnx_in: str, onnx_out: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic as _qd

    _qd(onnx_in, onnx_out, weight_type=QuantType.QInt8)
    print(f"[quant] dynamic INT8 -> {onnx_out}")
    return onnx_out


def quantize_static(onnx_in: str, onnx_out: str, calibration_texts: Iterable[str]) -> str:
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader, QuantType
    from onnxruntime.quantization import quantize_static as _qs

    from ...text.frontend import TextFrontend

    fe = TextFrontend()

    class _Reader(CalibrationDataReader):
        def __init__(self, texts: List[str]):
            self._data = []
            for t in texts:
                u = fe.process(t)
                self._data.append(
                    {
                        "phoneme_ids": np.asarray([u.phoneme_ids], dtype=np.int64),
                        "phoneme_lengths": np.asarray([len(u.phoneme_ids)], dtype=np.int64),
                        "language_ids": np.asarray([u.language_ids], dtype=np.int64),
                        "speaker_id": np.asarray([0], dtype=np.int64),
                    }
                )
            self._it = iter(self._data)

        def get_next(self):
            return next(self._it, None)

    _qs(onnx_in, onnx_out, _Reader(list(calibration_texts)), weight_type=QuantType.QInt8)
    print(f"[quant] static INT8 -> {onnx_out}")
    return onnx_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default="hams_vits.int8.onnx")
    ap.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--calib", default=None, help="text file, one utterance per line (static)")
    args = ap.parse_args()
    if args.mode == "dynamic":
        quantize_dynamic(args.onnx, args.out)
    else:
        texts = open(args.calib, encoding="utf-8").read().splitlines() if args.calib else [
            "مرحبا كيف حالك", "hello how are you", "بدي أحجز flight بكرا"
        ]
        quantize_static(args.onnx, args.out, texts)


if __name__ == "__main__":
    main()
