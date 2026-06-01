"""Publish the fine-tuned HamsVITS checkpoint (+ ONNX, config, model card) to the Hub.

    python scripts/upload_hf.py --ckpt checkpoints/hams_vits_levantine/final \
        --repo <user>/hams-levantine-tts [--onnx hams_vits.onnx] [--private]

Requires `huggingface_hub` and a token (`huggingface-cli login` or HF_TOKEN env).
"""

import argparse
import os

MODEL_CARD = """---
license: apache-2.0
language: [ar, en]
tags: [tts, vits, levantine-arabic, code-switching, streaming, pipecat]
library_name: transformers
pipeline_tag: text-to-speech
base_model: facebook/mms-tts-ara
---

# Hams — Low-Latency Code-Switching TTS (Levantine Arabic ↔ English)

Phoneme-input **VITS** (fine-tuned from `facebook/mms-tts-ara`) with a **language-ID
embedding** for seamless Levantine-Arabic / English code-switching, driven by a unified
IPA front-end. Optimized (ONNX / TensorRT FP16) for real-time agents on NVIDIA L4:
targets VRAM ≤ 3 GB, TTFA < 300 ms, RTF < 0.3, streaming.

- Code, server, Pipecat plugin, benchmark: see the linked GitHub repo.
- Inputs are phonemes from the shared IPA inventory (`hams_tts.text.frontend`), **not**
  raw text — use the front-end to convert text → phoneme/language IDs.

## Usage
```python
from hams_tts.inference.engine import build_engine
eng = build_engine("onnx", model_path="hams_vits.onnx")
audio = eng.synthesize("مرحبا! بدي flight to London بكرا.").audio
```

## Measured (RTX 3090)
- **KPIs (all pass):** peak VRAM **193 MB**, TTFA **87/144 ms** (p50/p95), RTF **0.044**.
- **ASR round-trip CER:** English **0.27** (intelligible), Arabic 0.68, code-switched 0.79.
- **UTMOS:** ~2.2.

## Honest status & limitations
Fine-tuned on LJSpeech (EN) + Arabic Speech Corpus (MSA) on a single rented RTX 3090.
**English round-trips intelligibly**, validating the unified-IPA pipeline end-to-end.
**Arabic lags** because training used *MSA* audio while the front-end emits *Levantine*
phonemes (ق→ʔ, ج→ʒ) — a data mismatch fixed by real Levantine speech, not code. The
stochastic duration predictor under-predicts (~6×); synthesize at `length_scale≈5`.
Word-accurate Levantine needs GPU-days + Levantine data. See the repo design doc.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HamsVITS checkpoint dir")
    ap.add_argument("--repo", required=True, help="<user>/<repo>")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo

    create_repo(args.repo, private=args.private, exist_ok=True)
    api = HfApi()

    card = os.path.join(args.ckpt, "README.md")
    with open(card, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD)

    print(f"[hf] uploading {args.ckpt} -> {args.repo}")
    api.upload_folder(folder_path=args.ckpt, repo_id=args.repo, repo_type="model")
    if args.onnx and os.path.exists(args.onnx):
        api.upload_file(path_or_fileobj=args.onnx, path_in_repo=os.path.basename(args.onnx),
                        repo_id=args.repo, repo_type="model")
    print(f"[hf] done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
