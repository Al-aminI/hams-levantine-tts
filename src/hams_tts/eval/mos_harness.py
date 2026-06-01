"""Subjective MOS-style listening-test harness.

Two subcommands:
  * ``form``      — build a randomised rating sheet (CSV + standalone HTML player) for
                    n≥5 listeners.  Clip order is shuffled and the category is hidden so
                    raters are blind.  Each clip is rated for **naturalness** (1–5) and,
                    for code-switched clips, **code-switching smoothness** (1–5).
  * ``aggregate`` — read the returned rating CSVs and report mean opinion scores with
                    95% confidence intervals, per category and overall.

This is the protocol + tooling; the human ratings are collected by you (the harness
cannot listen).  Keeping it blind + randomised is what makes the small-n MOS credible.

    python -m hams_tts.eval.mos_harness form --audio-dir samples/eval --out mos/
    python -m hams_tts.eval.mos_harness aggregate --ratings mos/returned/*.csv --key mos/key.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List


def build_form(audio_dir: str, out_dir: str, eval_set: str, seed: int = 13) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cats: Dict[str, str] = {}
    if os.path.exists(eval_set):
        with open(eval_set, encoding="utf-8") as f:
            for u in json.load(f)["utterances"]:
                cats[u["id"]] = u["category"]

    clips = [fn for fn in sorted(os.listdir(audio_dir)) if fn.endswith(".wav")]
    order = clips[:]
    random.Random(seed).shuffle(order)
    # blind ids
    key = {f"clip_{i:02d}": {"file": fn, "category": cats.get(fn[:-4], "unknown")}
           for i, fn in enumerate(order)}

    with open(os.path.join(out_dir, "key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)

    # CSV template the rater fills in
    with open(os.path.join(out_dir, "rating_template.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "naturalness_1to5", "codeswitch_smoothness_1to5", "notes"])
        for cid in key:
            w.writerow([cid, "", "", ""])

    # standalone HTML player
    rel = os.path.relpath(audio_dir, out_dir)
    rows = "\n".join(
        f'<tr><td>{cid}</td><td><audio controls src="{rel}/{meta["file"]}"></audio></td>'
        f'<td><input type=number min=1 max=5 step=1></td>'
        f'<td><input type=number min=1 max=5 step=1></td>'
        f'<td><input type=text></td></tr>'
        for cid, meta in key.items()
    )
    html = f"""<!doctype html><meta charset=utf-8><title>Hams TTS — MOS listening test</title>
<style>body{{font-family:sans-serif;max-width:880px;margin:2rem auto}}td{{padding:.4rem}}</style>
<h2>Hams TTS — blind listening test</h2>
<p>Rate each clip 1 (bad) – 5 (excellent) for <b>naturalness</b>, and for clips that mix
Arabic & English also rate <b>code-switching smoothness</b>. Save your numbers into
<code>rating_template.csv</code> and send it back.</p>
<table border=1 cellspacing=0><tr><th>clip</th><th>audio</th><th>naturalness</th>
<th>CS smoothness</th><th>notes</th></tr>
{rows}</table>"""
    with open(os.path.join(out_dir, "listen.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[mos] wrote {len(key)} clips -> {out_dir}/listen.html + rating_template.csv + key.json")


def _ci95(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return 1.96 * sd / math.sqrt(len(vals))


def aggregate(rating_globs: List[str], key_path: str) -> dict:
    with open(key_path, encoding="utf-8") as f:
        key = json.load(f)
    nat: Dict[str, List[float]] = defaultdict(list)
    smooth: Dict[str, List[float]] = defaultdict(list)
    n_raters = 0
    for pattern in rating_globs:
        for path in glob.glob(pattern):
            n_raters += 1
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cid = row["clip_id"]
                    cat = key.get(cid, {}).get("category", "unknown")
                    if row.get("naturalness_1to5", "").strip():
                        nat[cat].append(float(row["naturalness_1to5"]))
                    if row.get("codeswitch_smoothness_1to5", "").strip():
                        smooth[cat].append(float(row["codeswitch_smoothness_1to5"]))

    def summ(d: Dict[str, List[float]]) -> dict:
        out = {}
        allv: List[float] = []
        for cat, vals in d.items():
            allv += vals
            out[cat] = {"n": len(vals), "mos": round(sum(vals) / len(vals), 3), "ci95": round(_ci95(vals), 3)}
        if allv:
            out["overall"] = {"n": len(allv), "mos": round(sum(allv) / len(allv), 3), "ci95": round(_ci95(allv), 3)}
        return out

    return {"n_raters": n_raters, "naturalness": summ(nat), "codeswitch_smoothness": summ(smooth)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("form")
    f.add_argument("--audio-dir", default="samples/eval")
    f.add_argument("--out", default="mos")
    f.add_argument("--eval-set", default="data/eval_set/eval_utterances.json")
    a = sub.add_parser("aggregate")
    a.add_argument("--ratings", nargs="+", required=True)
    a.add_argument("--key", default="mos/key.json")
    a.add_argument("--output", default="mos/mos_results.json")
    args = ap.parse_args()

    if args.cmd == "form":
        build_form(args.audio_dir, args.out, args.eval_set)
    else:
        r = aggregate(args.ratings, args.key)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print("wrote", args.output)


if __name__ == "__main__":
    main()
