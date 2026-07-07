"""Evaluate a HamsVITS checkpoint on the held-out set (precomputed phonemes, so no
front-end needed). Synthesizes each utterance, calibrates length_scale against reference
durations, and optionally runs ASR round-trip CER/WER per category (the paper's metric).

    # synth only (fast) + duration calibration
    python scripts/eval_checkpoint.py --ckpt checkpoints/flagship_spk01_gan/step_5000 \
        --manifest data/manifests/eval.phon.filtered.jsonl --out eval_out/step_5000

    # + ASR round-trip CER (Whisper). --asr-device cpu keeps the GPU free for training.
    python scripts/eval_checkpoint.py --ckpt ... --asr --asr-model large-v3 --asr-device cpu

    # compare two checkpoints (baseline vs trained)
    python scripts/eval_checkpoint.py --ckpt A --ckpt-b B --manifest ... --asr
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
import torch

_DIAC = re.compile(r"[ً-ْٰ]")
_PUNCT = re.compile(r"[^\w\s؀-ۿ]")


def normalize_ar(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = _DIAC.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    t = _PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def synth_all(ckpt: str, rows: list, out_dir: Path, length_scale: float):
    import transformers

    transformers.logging.set_verbosity_error()  # silence VitsModel from_pretrained warnings
    from hams_tts.models.hams_vits import HamsVITS

    model = HamsVITS.from_checkpoint(ckpt).cuda().eval()
    sr = model.sample_rate
    out_dir.mkdir(parents=True, exist_ok=True)
    synth = []
    ratios = []
    for i, r in enumerate(rows):
        pid = torch.tensor([r["phoneme_ids"]], device="cuda")
        lid = torch.tensor([r["language_ids"]], device="cuda")
        with torch.no_grad():
            wav = model.infer(pid, lid, length_scale=length_scale).squeeze().cpu().numpy()
        p = out_dir / f"{r.get('sentence_type','x')}_{i:03d}.wav"
        sf.write(str(p), wav, sr)
        dur = len(wav) / sr
        synth.append((p, dur))
        if r.get("duration_s"):
            ratios.append(dur / r["duration_s"])
    del model
    torch.cuda.empty_cache()
    return synth, sr, ratios


def run_asr(wavs, rows, model_name, device):
    from faster_whisper import WhisperModel
    import jiwer

    compute = "int8" if device == "cpu" else "float16"
    asr = WhisperModel(model_name, device=device, compute_type=compute)
    by_cat = defaultdict(lambda: {"cer": [], "wer": []})
    details = []
    for (p, _dur), r in zip(wavs, rows):
        segs, _ = asr.transcribe(str(p), language="ar", beam_size=5)
        hyp = normalize_ar(" ".join(s.text for s in segs))
        ref = normalize_ar(r["text"])
        if not ref:
            continue
        cer = jiwer.cer(ref, hyp)
        wer = jiwer.wer(ref, hyp) if ref.split() else 1.0
        cat = r.get("sentence_type", "x")
        by_cat[cat]["cer"].append(cer)
        by_cat[cat]["wer"].append(wer)
        details.append({"cat": cat, "ref": ref, "hyp": hyp, "cer": round(cer, 3)})
    return by_cat, details


def summarize(tag, by_cat):
    print(f"\n=== ASR round-trip [{tag}] ===")
    allc, allw = [], []
    for cat, d in by_cat.items():
        c = float(np.mean(d["cer"])) if d["cer"] else float("nan")
        w = float(np.mean(d["wer"])) if d["wer"] else float("nan")
        allc += d["cer"]; allw += d["wer"]
        print(f"  {cat:16} n={len(d['cer']):3}  CER {c:.3f}  WER {w:.3f}")
    if allc:
        print(f"  {'OVERALL':16} n={len(allc):3}  CER {np.mean(allc):.3f}  WER {np.mean(allw):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-b", default=None, help="optional second checkpoint to compare")
    ap.add_argument("--manifest", default=str(REPO / "data/manifests/eval.phon.filtered.jsonl"))
    ap.add_argument("--out", default=str(REPO / "eval_out"))
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--auto-ls", action="store_true", help="calibrate length_scale per ckpt to match reference durations")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--asr-model", default="large-v3")
    ap.add_argument("--asr-device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    print(f"eval set: {len(rows)} utterances | length_scale={args.length_scale}")

    for tag, ck in [("A", args.ckpt)] + ([("B", args.ckpt_b)] if args.ckpt_b else []):
        ls = args.length_scale
        if args.auto_ls:
            # calibrate: synth at ls=1, target reference durations (ratio -> 1)
            _s, _sr, ratios0 = synth_all(ck, rows, Path(args.out) / "_calib", 1.0)
            r0 = float(np.mean(ratios0)) if ratios0 else 1.0
            ls = round(min(max(1.0 / max(r0, 1e-3), 1.0), 8.0), 2)
            print(f"\n[{tag}] {ck}\n  auto length_scale = {ls} (ratio@1.0 = {r0:.2f})")
        out_dir = Path(args.out) / (Path(ck).name + f"_ls{ls}")
        synth, sr, ratios = synth_all(ck, rows, out_dir, ls)
        rmean = float(np.mean(ratios)) if ratios else float("nan")
        print(f"  synthesized {len(synth)} -> {out_dir} @ {sr} Hz | synth/ref ratio {rmean:.2f}")
        if args.asr:
            by_cat, details = run_asr(synth, rows, args.asr_model, args.asr_device)
            summarize(ck, by_cat)
            (out_dir / "asr_details.json").write_text(
                json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
