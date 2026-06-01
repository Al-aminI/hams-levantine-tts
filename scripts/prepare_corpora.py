"""Build a combined LJSpeech (EN) + Arabic Speech Corpus (AR) manifest and precompute
phoneme/language IDs via the front-end.  ASC transcripts are diacritized Buckwalter ASCII
-> converted to Arabic script so the Levantine G2P applies.

    python scripts/prepare_corpora.py --lj-root /workspace/data/LJSpeech-1.1 \
        --asc-root /workspace/data/arabic-speech-corpus --n-en 3000 --n-ar 1800 \
        --out /workspace/data/manifests
"""

import argparse
import os
import random
import shlex

from hams_tts.training.data import buckwalter_to_arabic, build_manifest, precompute_phonemes


def lj_rows(root):
    rows = []
    meta = os.path.join(root, "metadata.csv")
    for line in open(meta, encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) < 2:
            continue
        text = p[2] if len(p) > 2 and p[2] else p[1]
        wav = os.path.join(root, "wavs", p[0] + ".wav")
        if os.path.exists(wav):
            rows.append({"audio": wav, "text": text, "lang": "en", "speaker": "ljspeech"})
    return rows


def asc_rows(root):
    rows = []
    tr = os.path.join(root, "orthographic-transcript.txt")
    for line in open(tr, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            fn, bw = shlex.split(line)
        except ValueError:
            continue
        wav = os.path.join(root, "wav", fn)
        if os.path.exists(wav):
            rows.append({"audio": wav, "text": buckwalter_to_arabic(bw), "lang": "ar", "speaker": "asc"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lj-root", required=True)
    ap.add_argument("--asc-root", required=True)
    ap.add_argument("--n-en", type=int, default=3000)
    ap.add_argument("--n-ar", type=int, default=1800)
    ap.add_argument("--out", default="/workspace/data/manifests")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    en = lj_rows(args.lj_root); rng.shuffle(en); en = en[: args.n_en]
    ar = asc_rows(args.asc_root); rng.shuffle(ar); ar = ar[: args.n_ar]
    print(f"[prep] LJSpeech(en)={len(en)}  ASC(ar)={len(ar)}")

    rows = en + ar
    rng.shuffle(rows)
    n_eval = 16
    train, eval = rows[n_eval:], rows[:n_eval]
    raw_train = os.path.join(args.out, "train.jsonl")
    raw_eval = os.path.join(args.out, "eval.jsonl")
    build_manifest(train, raw_train)
    build_manifest(eval, raw_eval)

    print("[prep] precomputing phonemes (front-end) ...")
    nt = precompute_phonemes(raw_train, os.path.join(args.out, "train.phon.jsonl"))
    ne = precompute_phonemes(raw_eval, os.path.join(args.out, "eval.phon.jsonl"))
    print(f"[prep] wrote train.phon.jsonl ({nt}) + eval.phon.jsonl ({ne}) -> {args.out}")


if __name__ == "__main__":
    main()
