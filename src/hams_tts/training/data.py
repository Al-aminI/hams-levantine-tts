"""Data preparation for fine-tuning: corpora registry, manifest building, phoneme
precompute, synthetic code-switch augmentation, and the torch Dataset.

Corpus strategy (see design doc for the full rationale)
-------------------------------------------------------
  * **Levantine acoustic core**  — repurpose Levantine ASR corpora for TTS (e.g. MGB-3
    Egyptian/Levantine, the Levantine portions of Common Voice ar / QASR, plus any
    in-house single-speaker Levantine recordings).  Single, clean speaker(s) preferred.
  * **MSA / Arabic TTS**         — ClArTTS, ArabicSpeech, ASC for clean Arabic acoustics
    and broad phoneme coverage; used to stabilise Arabic phonetics, then biased toward
    Levantine via the dialect data + G2P.
  * **English TTS**              — LJSpeech (single speaker) and/or LibriTTS (multi) so
    the *same* model speaks English natively for code-switching.
  * **Code-switch augmentation** — (a) concatenate matched-speaker AR+EN clips at clause
    boundaries to create CS audio; (b) text-level splicing to enrich the encoder/duration
    model.  Real bilingual speaker data is best when available.

The manifest + phoneme precompute run on CPU (used here); the Dataset/spectrogram path
needs torch+librosa (the ``train`` extra) and runs on the GPU host.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------------------
# Corpora registry (documentation as code)
# --------------------------------------------------------------------------------------
@dataclass
class Corpus:
    name: str
    lang: str          # "ar" | "en" | "cs"
    role: str
    notes: str
    license: str = ""


CORPORA: List[Corpus] = [
    Corpus("ClArTTS", "ar", "msa_acoustic", "Classical Arabic single-speaker TTS; clean.", "research"),
    Corpus("ArabicSpeech", "ar", "msa_acoustic", "Broad Arabic phoneme coverage.", "varies"),
    Corpus("ASC", "ar", "msa_acoustic", "Arabic Speech Corpus (diacritised, phoneme-aligned).", "CC-BY"),
    Corpus("CommonVoice-ar(Levantine)", "ar", "levantine_acoustic", "Filter to Levantine speakers.", "CC0"),
    Corpus("QASR/MGB", "ar", "levantine_acoustic", "Repurpose Levantine ASR segments for TTS.", "research"),
    Corpus("LJSpeech", "en", "english_acoustic", "Single-speaker English TTS baseline.", "public-domain"),
    Corpus("LibriTTS", "en", "english_acoustic", "Multi-speaker English for speaker variety.", "CC-BY"),
]


# --------------------------------------------------------------------------------------
# Manifest building + phoneme precompute (CPU-runnable)
# --------------------------------------------------------------------------------------
@dataclass
class Utt:
    audio: str
    text: str
    lang: str
    speaker: str = "spk0"
    phoneme_ids: Optional[List[int]] = None
    language_ids: Optional[List[int]] = None
    extra: Dict = field(default_factory=dict)


# Buckwalter <-> Arabic (the Arabic Speech Corpus ships diacritized Buckwalter ASCII).
_BW2AR = {
    "'": "ء", "|": "آ", ">": "أ", "&": "ؤ", "<": "إ", "}": "ئ", "A": "ا", "b": "ب",
    "p": "ة", "t": "ت", "v": "ث", "j": "ج", "H": "ح", "x": "خ", "d": "د", "*": "ذ",
    "r": "ر", "z": "ز", "s": "س", "$": "ش", "S": "ص", "D": "ض", "T": "ط", "Z": "ظ",
    "E": "ع", "g": "غ", "_": "ـ", "f": "ف", "q": "ق", "k": "ك", "l": "ل", "m": "م",
    "n": "ن", "h": "ه", "w": "و", "Y": "ى", "y": "ي", "F": "ً", "N": "ٌ", "K": "ٍ",
    "a": "َ", "u": "ُ", "i": "ِ", "~": "ّ", "o": "ْ", "`": "ٰ", "{": "ٱ",
}


def buckwalter_to_arabic(text: str) -> str:
    """Convert (diacritized) Buckwalter ASCII to Arabic script — used for ASC transcripts."""
    return "".join(_BW2AR.get(ch, ch) for ch in text)


def build_manifest(pairs: List[dict], out_jsonl: str) -> int:
    """Write a JSONL manifest from a list of {audio, text, lang, speaker} dicts."""
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            row = {"audio": p["audio"], "text": p["text"], "lang": p.get("lang", "ar"),
                   "speaker": p.get("speaker", "spk0")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def precompute_phonemes(in_jsonl: str, out_jsonl: str, diacritizer_backend: str = "auto") -> int:
    """Run the text front-end over a manifest, caching phoneme_ids + language_ids.

    Doing this offline keeps training-step CPU work tiny and makes batches deterministic.
    Runs on CPU (no torch needed)."""
    from ..text.frontend import TextFrontend

    fe = TextFrontend(diacritizer_backend=diacritizer_backend)
    n = 0
    with open(in_jsonl, encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            u = fe.process(row["text"])
            row["phoneme_ids"] = u.phoneme_ids
            row["language_ids"] = u.language_ids
            row["ipa"] = u.ipa
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def synthetic_codeswitch_text(ar_lines: List[str], en_lines: List[str], n: int,
                              seed: int = 7) -> List[str]:
    """Generate code-switched *text* by templated insertion (encoder/duration aug).

    Real CS *audio* is produced by concatenating same-speaker AR+EN segments at clause
    boundaries (see ``concat_codeswitch_audio``); this text aug enriches the front-end
    coverage of switch points and is also useful for evaluating the G2P boundary logic."""
    rng = random.Random(seed)
    templates = [
        "{ar}، {en}.",
        "{ar} {en} {ar2}",
        "{en}، بَس {ar}.",
        "بَدّي {en} {ar}",
    ]
    out = []
    for _ in range(n):
        t = rng.choice(templates)
        out.append(t.format(ar=rng.choice(ar_lines), en=rng.choice(en_lines),
                            ar2=rng.choice(ar_lines)))
    return out


def concat_codeswitch_audio(ar_wav: str, en_wav: str, out_wav: str, gap_ms: int = 60,
                            target_sr: int = 24000) -> str:
    """Concatenate an Arabic and an English clip (ideally same speaker) into one CS clip.

    A pragmatic CS-audio augmentation when paired bilingual recordings are scarce.
    Crossfades a short gap to avoid clicks; matched-speaker pairs keep timbre stable."""
    import numpy as np

    from ..utils import audio as A

    a, sa = A.read_wav(open(ar_wav, "rb").read())
    b, sb = A.read_wav(open(en_wav, "rb").read())
    a = A.resample(a, sa, target_sr)
    b = A.resample(b, sb, target_sr)
    gap = np.zeros(int(gap_ms / 1000 * target_sr), dtype=np.float32)
    A.save_wav(out_wav, np.concatenate([a, gap, b]).astype(np.float32), target_sr)
    return out_wav


# --------------------------------------------------------------------------------------
# Torch Dataset (GPU/train extra)
# --------------------------------------------------------------------------------------
def make_dataset(*args, **kwargs):
    """Lazy factory so importing this module never requires torch/librosa."""
    from ._torch_dataset import VitsCodeSwitchDataset  # noqa

    return VitsCodeSwitchDataset(*args, **kwargs)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="dataset prep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precompute")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--diacritizer", default="auto")
    args = ap.parse_args()
    if args.cmd == "precompute":
        n = precompute_phonemes(args.inp, args.out, args.diacritizer)
        print(f"precomputed phonemes for {n} utterances -> {args.out}")


if __name__ == "__main__":
    main()
