"""Intelligibility via ASR round-trip: synthesise → transcribe (Whisper large-v3) →
compare to the reference text with CER/WER.

A strong ASR transcribing our audio back to text is a reference-free proxy for
intelligibility: low CER/WER means the words (and the code-switches) came through.
We use faster-whisper (large-v3) and report CER+WER per category (pure AR / pure EN /
code-switched), with script-appropriate text normalisation so we score *content*, not
orthographic noise (diacritics, hamza/alef variants, casing, punctuation).

GPU host:  python -m hams_tts.eval.asr_roundtrip --audio-dir samples/eval --eval-set data/eval_set/eval_utterances.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List

_AR_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_PUNCT = re.compile(r"[^\w\s؀-ۿ]")


def normalize_ar(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _AR_DIAC.sub("", s)
    s = s.translate({ord(a): "ا" for a in "أإآ"})
    s = s.replace("ة", "ه").replace("ى", "ي")
    return " ".join(_PUNCT.sub(" ", s).split())


def normalize_en(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return " ".join(_PUNCT.sub(" ", s).split())


def normalize_mixed(s: str) -> str:
    # code-switched: normalise Arabic runs the Arabic way, keep Latin lower-cased
    out = []
    for tok in s.split():
        if re.search(r"[؀-ۿ]", tok):
            out.append(normalize_ar(tok))
        else:
            out.append(normalize_en(tok))
    return " ".join(t for t in out if t)


_NORM = {"pure_arabic": normalize_ar, "pure_english": normalize_en, "code_switched": normalize_mixed}


def _cer_wer(ref: str, hyp: str) -> tuple[float, float]:
    import jiwer

    if not ref:
        return 0.0, 0.0
    wer = jiwer.wer(ref, hyp) if ref.split() else 1.0
    cer = jiwer.cer(ref, hyp)
    return cer, wer


def run(audio_dir: str, eval_set: str, model_size: str = "large-v3", device: str = "auto") -> dict:
    from faster_whisper import WhisperModel

    if device == "auto":
        # ctranslate2's CUDA build needs CUDA-12 libs and fails at *encode* time on a
        # CUDA-13 host; CPU int8 is robust everywhere. Pass device="cuda" on a CUDA-12 box.
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    asr = WhisperModel(model_size, device=device, compute_type=compute)

    with open(eval_set, encoding="utf-8") as f:
        data = json.load(f)

    per_item: List[dict] = []
    by_cat: Dict[str, List[tuple]] = defaultdict(list)
    for utt in data["utterances"]:
        wav = os.path.join(audio_dir, f"{utt['id']}.wav")
        if not os.path.exists(wav):
            continue
        # let Whisper auto-detect language; for code-switch it still transcribes both scripts
        segments, _ = asr.transcribe(wav, beam_size=5, task="transcribe")
        hyp_raw = " ".join(seg.text for seg in segments).strip()
        norm = _NORM[utt["category"]]
        ref, hyp = norm(utt["text"]), norm(hyp_raw)
        cer, wer = _cer_wer(ref, hyp)
        per_item.append({"id": utt["id"], "category": utt["category"], "cer": round(cer, 4),
                         "wer": round(wer, 4), "ref": ref, "hyp": hyp})
        by_cat[utt["category"]].append((cer, wer))

    summary = {}
    for cat, vals in by_cat.items():
        cers = [v[0] for v in vals]
        wers = [v[1] for v in vals]
        summary[cat] = {"n": len(vals), "cer": round(sum(cers) / len(cers), 4),
                        "wer": round(sum(wers) / len(wers), 4)}
    allcer = [i["cer"] for i in per_item] or [0]
    allwer = [i["wer"] for i in per_item] or [0]
    summary["overall"] = {"n": len(per_item), "cer": round(sum(allcer) / len(allcer), 4),
                          "wer": round(sum(allwer) / len(allwer), 4)}
    return {"model": model_size, "device": device, "summary": summary, "items": per_item}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default="samples/eval")
    ap.add_argument("--eval-set", default="data/eval_set/eval_utterances.json")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--output", default="samples/asr_roundtrip.json")
    args = ap.parse_args()
    r = run(args.audio_dir, args.eval_set, args.model)
    print(json.dumps(r["summary"], ensure_ascii=False, indent=2))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
