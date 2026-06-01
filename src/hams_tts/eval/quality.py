"""Reference-free speech-quality metric (UTMOS primary, DNSMOS fallback).

A reference-free MOS predictor scores naturalness without needing ground-truth audio —
ideal here because we have no Levantine reference recordings for most prompts.  UTMOS
(``tarepan/SpeechMOS``) is the default; DNSMOS (via ``speechmos``) is the fallback.

GPU/CPU host:  python -m hams_tts.eval.quality --audio-dir samples/eval --eval-set data/eval_set/eval_utterances.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional


class UTMOS:
    """Thin wrapper over the UTMOS22 strong predictor from torch.hub."""

    def __init__(self):
        import torch

        self.torch = torch
        self.model = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
        self.model.eval()

    def score(self, wav, sr: int) -> float:
        import torch

        if wav.ndim == 1:
            wav = wav[None, :]
        with torch.no_grad():
            return float(self.model(torch.as_tensor(wav).float(), sr).mean().item())


class DNSMOS:
    def __init__(self):
        from speechmos import dnsmos  # type: ignore

        self._dnsmos = dnsmos

    def score(self, wav, sr: int) -> float:
        r = self._dnsmos.run(wav, sr)
        # use overall MOS (OVRL) as the headline number
        return float(r.get("ovrl_mos", r.get("OVRL", 0.0)))


def _load_scorer(prefer: str = "utmos"):
    if prefer == "utmos":
        try:
            return UTMOS(), "utmos22_strong"
        except Exception:
            pass
    try:
        return DNSMOS(), "dnsmos"
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"no quality metric available (install eval extra): {e}")


def run(audio_dir: str, eval_set: Optional[str], prefer: str = "utmos") -> dict:
    import soundfile as sf

    scorer, name = _load_scorer(prefer)
    cats: Dict[str, str] = {}
    if eval_set and os.path.exists(eval_set):
        with open(eval_set, encoding="utf-8") as f:
            for u in json.load(f)["utterances"]:
                cats[u["id"]] = u["category"]

    items: List[dict] = []
    by_cat: Dict[str, List[float]] = defaultdict(list)
    for fn in sorted(os.listdir(audio_dir)):
        if not fn.endswith(".wav"):
            continue
        wav, sr = sf.read(os.path.join(audio_dir, fn))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        s = scorer.score(wav, sr)
        cat = cats.get(fn[:-4], "unknown")
        items.append({"file": fn, "category": cat, "mos": round(s, 3)})
        by_cat[cat].append(s)

    summary = {c: {"n": len(v), "mos": round(sum(v) / len(v), 3)} for c, v in by_cat.items()}
    allv = [i["mos"] for i in items] or [0]
    summary["overall"] = {"n": len(items), "mos": round(sum(allv) / len(allv), 3)}
    return {"metric": name, "summary": summary, "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default="samples/eval")
    ap.add_argument("--eval-set", default="data/eval_set/eval_utterances.json")
    ap.add_argument("--metric", default="utmos", choices=["utmos", "dnsmos"])
    ap.add_argument("--output", default="samples/quality.json")
    args = ap.parse_args()
    r = run(args.audio_dir, args.eval_set, args.metric)
    print(json.dumps(r["summary"], ensure_ascii=False, indent=2))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
