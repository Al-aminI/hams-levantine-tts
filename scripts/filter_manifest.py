"""Filter a phon manifest for VITS training stability:
  * duration in [min_s, max_s]  (bounds VRAM; drops silence/outliers)
  * phoneme_count < spec_frames  (MAS alignment needs t_x <= t_y)
Writes <manifest>.filtered.jsonl and prints duration stats.
"""
import argparse, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--min-s", type=float, default=0.8)
ap.add_argument("--max-s", type=float, default=12.0)
ap.add_argument("--hop", type=int, default=256)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
kept, drop_dur, drop_align = [], 0, 0
durs = []
for r in rows:
    info = sf.info(r["audio"])
    dur = info.frames / info.samplerate
    spec_frames = info.frames // args.hop + 1
    n_phon = len(r["phoneme_ids"])
    if not (args.min_s <= dur <= args.max_s):
        drop_dur += 1
        continue
    if n_phon >= spec_frames:
        drop_align += 1
        continue
    r["duration_s"] = round(dur, 2)
    kept.append(r)
    durs.append(dur)

out = str(Path(args.manifest).with_suffix(".filtered.jsonl"))
with open(out, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

durs.sort()
n = len(durs)
tot = sum(durs)
print(f"kept {n}/{len(rows)}  (dropped {drop_dur} by duration, {drop_align} by alignment)")
if n:
    print(f"duration: total {tot/3600:.2f} h | min {durs[0]:.1f}s | "
          f"p50 {durs[n//2]:.1f}s | p95 {durs[int(n*0.95)]:.1f}s | max {durs[-1]:.1f}s")
print(f"-> {out}")
