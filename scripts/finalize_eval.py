"""End-to-end neural eval: (optionally wait for training to finish) generate the 3
headline samples + the eval set from a checkpoint, then run the Whisper ASR round-trip
(CER/WER) and UTMOS, writing a combined results JSON.

    python scripts/finalize_eval.py --ckpt /workspace/ckpt_gan/final --out /workspace/samples_neural \
        --length-scale 5.0 --whisper large-v3 [--wait /workspace/ckpt_gan/final]
"""

import argparse
import json
import os
import time

HEADLINE = {
    "sample_levantine_arabic": "مَرحَبا! أنا Hams. اليَوم الجَوّ كْتير حِلو، وِ بَدّي روح عَ السّوق. كيفَك إنتَ؟",
    "sample_english": "Hi, I'm Hams. I can stream speech with very low latency on a single L4 GPU.",
    "sample_codeswitched": "مَرحَبا! بَدّي إحجِز flight من بيروت to London بُكرا الساعة 9، and please confirm by email.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt_gan/final")
    ap.add_argument("--out", default="/workspace/samples_neural")
    ap.add_argument("--length-scale", type=float, default=5.0)
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--wait", default=None, help="path to wait for before starting (training's final ckpt)")
    ap.add_argument("--eval-set", default="data/eval_set/eval_utterances.json")
    args = ap.parse_args()

    if args.wait:
        print(f"[finalize] waiting for {args.wait} ...", flush=True)
        while not os.path.exists(os.path.join(args.wait, "hams_vits.pt")):
            time.sleep(30)
        time.sleep(10)

    import torch
    import soundfile as sf
    from hams_tts.models.hams_vits import HamsVITS
    from hams_tts.text.frontend import TextFrontend

    m = HamsVITS.from_checkpoint(args.ckpt).cuda().eval()
    fe = TextFrontend()
    sr = m.sample_rate
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
    ls = args.length_scale

    def gen(text, path):
        u = fe.process(text)
        ids = torch.tensor([u.phoneme_ids], device="cuda")
        lg = torch.tensor([u.language_ids], device="cuda")
        with torch.inference_mode():
            w = m.infer(ids, lg, length_scale=ls).squeeze().float().cpu().numpy()
        sf.write(path, w, sr)
        with open(path.replace(".wav", ".phonemes.json"), "w", encoding="utf-8") as f:
            json.dump({"text": text, "ipa": u.ipa, "language_ids": u.language_ids,
                       "length_scale": ls, "duration_s": round(len(w) / sr, 2), "sample_rate": sr},
                      f, ensure_ascii=False, indent=2)

    print("[finalize] generating samples ...", flush=True)
    for stem, t in HEADLINE.items():
        gen(t, os.path.join(args.out, stem + ".wav"))
    with open(args.eval_set, encoding="utf-8") as f:
        data = json.load(f)
    for u in data["utterances"]:
        gen(u["text"], os.path.join(args.out, "eval", u["id"] + ".wav"))
    print(f"[finalize] wrote {3 + len(data['utterances'])} neural WAVs -> {args.out}", flush=True)

    results = {"checkpoint": args.ckpt, "length_scale": ls}
    print(f"[finalize] ASR round-trip (Whisper {args.whisper}) ...", flush=True)
    try:
        from hams_tts.eval.asr_roundtrip import run as asr_run
        results["asr"] = asr_run(os.path.join(args.out, "eval"), args.eval_set, model_size=args.whisper)["summary"]
    except Exception as e:
        results["asr"] = {"error": str(e)}
    print("[finalize] UTMOS ...", flush=True)
    try:
        from hams_tts.eval.quality import run as q_run
        results["quality"] = q_run(os.path.join(args.out, "eval"), args.eval_set)["summary"]
    except Exception as e:
        results["quality"] = {"error": str(e)}

    with open(os.path.join(args.out, "neural_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[finalize] DONE", flush=True)
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
