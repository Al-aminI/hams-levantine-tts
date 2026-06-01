"""Generate the three required sample WAVs (and optionally render the eval set).

  samples/sample_levantine_arabic.wav   – pure Levantine Arabic
  samples/sample_english.wav            – pure English
  samples/sample_codeswitched.wav       – Arabic↔English code-switch

Each WAV gets a ``.phonemes.json`` sidecar showing the front-end output (normalised
text, IPA, per-token language IDs) so the phonemisation is fully inspectable.

Backend defaults to ``espeak`` (real audio anywhere — dev/CI), and to the neural model
on the GPU host via ``--backend onnx --model-path hams_vits.onnx``.  The headline
samples in the submission are regenerated with the fine-tuned model after training.

    python scripts/make_samples.py                       # 3 samples (espeak)
    python scripts/make_samples.py --eval-set            # + render data/eval_set
    python scripts/make_samples.py --backend onnx --model-path hams_vits.onnx
"""

import argparse
import json
import os

from hams_tts.inference.engine import build_engine
from hams_tts.text.frontend import TextFrontend
from hams_tts.utils import audio as A

HEADLINE = {
    "sample_levantine_arabic": "مَرحَبا! أنا Hams. اليَوم الجَوّ كْتير حِلو، وِ بَدّي روح عَ السّوق. كيفَك إنتَ؟",
    "sample_english": "Hi, I'm Hams. I can stream speech with very low latency on a single L4 GPU.",
    "sample_codeswitched": "مَرحَبا! بَدّي إحجِز flight من بيروت to London بُكرا الساعة 9، and please confirm by email.",
}


def _write(engine, fe, text, stem, out_dir, sr):
    res = engine.synthesize(text)
    wav_path = os.path.join(out_dir, stem + ".wav")
    A.save_wav(wav_path, res.audio, res.sample_rate)
    utt = fe.process(text)
    side = {
        "text": text,
        "normalized": utt.normalized,
        "ipa": utt.ipa,
        "tokens": utt.symbols,
        "language_ids": utt.language_ids,
        "ttfa_ms": round(res.ttfa_s * 1000, 1),
        "rtf": round(res.rtf, 4),
        "audio_seconds": round(res.audio_s, 3),
        "backend": engine.backend.name,
        "sample_rate": res.sample_rate,
    }
    with open(os.path.join(out_dir, stem + ".phonemes.json"), "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False, indent=2)
    print(f"  {stem}.wav  ({res.audio_s:4.2f}s, ttfa {side['ttfa_ms']:.0f}ms, rtf {res.rtf:.3f})")
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="espeak", choices=["espeak", "onnx", "tensorrt", "torch"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--eval-set", action="store_true", help="also render data/eval_set/*")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kw = {}
    if args.backend in ("onnx", "tensorrt", "torch"):
        key = {"onnx": "model_path", "tensorrt": "engine_path", "torch": "checkpoint"}[args.backend]
        kw[key] = args.model_path
    engine = build_engine(args.backend, output_sample_rate=args.sample_rate, **kw)
    engine.warmup()
    fe = TextFrontend()

    print(f"[samples] backend={args.backend} sr={args.sample_rate}")
    manifest = {}
    for stem, text in HEADLINE.items():
        manifest[stem] = _write(engine, fe, text, stem, args.out, args.sample_rate)

    if args.eval_set:
        eval_dir = os.path.join(args.out, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        with open("data/eval_set/eval_utterances.json", encoding="utf-8") as f:
            data = json.load(f)
        print("[samples] rendering eval set ->", eval_dir)
        for utt in data["utterances"]:
            manifest[utt["id"]] = _write(engine, fe, utt["text"], utt["id"], eval_dir, args.sample_rate)

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[samples] wrote {len(manifest)} clips + manifest.json")


if __name__ == "__main__":
    main()
