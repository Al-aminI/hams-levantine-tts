"""Prepare mohammedaly22/lahgtna-levantine-tts for HamsVITS fine-tuning.

Mirrors scripts/prepare_corpora.py but for a HF parquet audio dataset (50k clips,
66.8h, 10 speakers, 24kHz, Levantine + ~12% code-switch). Three steps:

  extract    parquet shards -> 16kHz mono WAVs on disk + full manifest.jsonl
             (keeps speaker_id per clip so speakers are NEVER blurred together)
  phonemize  run the text front-end over the manifest -> phoneme_ids + language_ids
             (the code-switch segmenter tags AR vs EN per token)
  stats      per-speaker / per-type breakdown to choose the training scope

    python scripts/prepare_lahgtna.py extract  --parquet-dir data/corpora/lahgtna/data --out data/lahgtna
    python scripts/prepare_lahgtna.py phonemize --manifest data/lahgtna/manifest.jsonl --out data/manifests
    python scripts/prepare_lahgtna.py stats     --manifest data/lahgtna/manifest.jsonl

Design choice: audio is resampled to 16 kHz (the mms-tts-ara backbone rate) at extract
time so training does zero per-step resampling. WAVs are laid out wavs/<speaker_id>/*.wav.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET_SR = 16000  # mms-tts-ara backbone rate (override with extract --target-sr 24000)


def _iter_parquet_rows(parquet_dir: Path):
    import pyarrow.parquet as pq

    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards in {parquet_dir}")
    for shard in shards:
        pf = pq.ParquetFile(str(shard))
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            cols = tbl.column_names
            for i in range(tbl.num_rows):
                yield {k: tbl[k][i].as_py() for k in cols}, shard.name


def cmd_extract(args):
    import soundfile as sf
    import soxr

    target_sr = args.target_sr
    only_spk = set(args.speaker.split(",")) if args.speaker else None
    out = Path(args.out)
    wav_root = out / "wavs"
    wav_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    parquet_dir = Path(args.parquet_dir)
    n = 0
    skipped = 0
    per_spk = {}
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for row, shard in _iter_parquet_rows(parquet_dir):
            audio = row.get("audio") or {}
            b = audio.get("bytes")
            if not b:
                skipped += 1
                continue
            spk = row.get("speaker_id", "spk0")
            if only_spk is not None and spk not in only_spk:
                continue
            # deterministic filename from the embedded path (fallback to running index)
            stem = Path(audio.get("path") or f"{spk}_{n:07d}.wav").stem
            spk_dir = wav_root / spk
            spk_dir.mkdir(exist_ok=True)
            wav_path = spk_dir / f"{stem}.wav"

            if not (args.resume and wav_path.exists()):
                wav, sr = sf.read(io.BytesIO(b), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != target_sr:
                    wav = soxr.resample(wav, sr, target_sr)
                sf.write(str(wav_path), wav, target_sr, subtype="PCM_16")

            st = row.get("sentence_type", "")
            lang = "cs" if st == "code_switching" else "ar"
            mf.write(json.dumps({
                "audio": str(wav_path),
                "text": row.get("text", ""),
                "lang": lang,
                "speaker": spk,
                "speaker_name": row.get("speaker_name", ""),
                "gender": row.get("gender", ""),
                "sentence_type": st,
            }, ensure_ascii=False) + "\n")
            per_spk[spk] = per_spk.get(spk, 0) + 1
            n += 1
            if n % 2000 == 0:
                print(f"  extracted {n} clips ...", flush=True)
    print(f"[extract] wrote {n} clips (skipped {skipped}) -> {manifest_path}")
    print(f"[extract] speakers: {per_spk}")


def cmd_phonemize(args):
    """Front-end over the manifest. Parallel across processes (espeak is a subprocess)."""
    from concurrent.futures import ProcessPoolExecutor

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out, exist_ok=True)

    # hold out a balanced eval set: first N per (speaker, sentence_type) cell
    eval_rows, train_rows = _split_eval(rows, per_cell=args.eval_per_cell)
    print(f"[phonemize] {len(train_rows)} train / {len(eval_rows)} eval "
          f"| workers={args.workers}")

    for name, subset in (("train", train_rows), ("eval", eval_rows)):
        texts = [r["text"] for r in subset]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_fe_init) as ex:
            results = list(ex.map(_fe_process, texts, chunksize=64))
        out_path = Path(args.out) / f"{name}.phon.jsonl"
        bad = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r, res in zip(subset, results):
                if res is None:
                    bad += 1
                    continue
                r = dict(r)
                r["phoneme_ids"], r["language_ids"], r["ipa"] = res
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[phonemize] {name}: wrote {len(subset)-bad} rows (skipped {bad}) -> {out_path}")


def _split_eval(rows, per_cell=3):
    seen = {}
    eval_rows, train_rows = [], []
    for r in rows:
        key = (r.get("speaker", ""), r.get("sentence_type", ""))
        if seen.get(key, 0) < per_cell:
            seen[key] = seen.get(key, 0) + 1
            eval_rows.append(r)
        else:
            train_rows.append(r)
    return eval_rows, train_rows


_FE = None


def _fe_init():
    global _FE
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from hams_tts.text.frontend import TextFrontend
    _FE = TextFrontend(diacritizer_backend="auto")


def _fe_process(text):
    try:
        u = _FE.process(text)
        return (u.phoneme_ids, u.language_ids, u.ipa)
    except Exception:
        return None


def cmd_stats(args):
    import collections

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    spk = collections.Counter(r["speaker"] for r in rows)
    typ = collections.Counter(r["sentence_type"] for r in rows)
    gen = collections.Counter(r.get("gender", "") for r in rows)
    print(f"total clips: {len(rows)}")
    print(f"sentence types: {dict(typ)}")
    print(f"gender: {dict(gen)}")
    print("per speaker:")
    for s, c in spk.most_common():
        name = next((r.get("speaker_name", "") for r in rows if r["speaker"] == s), "")
        print(f"   {s:16} {name:10} {c:6} clips")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--parquet-dir", default="data/corpora/lahgtna/data")
    e.add_argument("--out", default="data/lahgtna")
    e.add_argument("--target-sr", type=int, default=TARGET_SR, help="output sample rate (e.g. 24000)")
    e.add_argument("--speaker", default=None, help="only this speaker_id (comma-separated for several)")
    e.add_argument("--resume", action="store_true", help="skip WAVs already written")
    e.set_defaults(func=cmd_extract)

    p = sub.add_parser("phonemize")
    p.add_argument("--manifest", default="data/lahgtna/manifest.jsonl")
    p.add_argument("--out", default="data/manifests")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-per-cell", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_phonemize)

    s = sub.add_parser("stats")
    s.add_argument("--manifest", default="data/lahgtna/manifest.jsonl")
    s.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
