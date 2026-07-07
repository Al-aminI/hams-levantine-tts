"""Texture experiment: for a few sentences, render the REFERENCE (source data) and our
24 kHz model at several inference noise_scale values, into a listening page. Answers:
(1) which noise_scale sounds smoothest, (2) whether the reference is smoother than ours
(decoder gap = trainable) or similar (data ceiling).
"""
from __future__ import annotations
import os, sys, json, shutil
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, soundfile as sf
import transformers; transformers.logging.set_verbosity_error()
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.utils.trim import trim_trailing_artifact

CKPT = REPO / "checkpoints/flagship_BEST_v3_24k"
MANIFEST = REPO / "data/manifests_24k/eval.phon.filtered.jsonl"
OUT = REPO / "samples_texture"
NOISE = [0.667, 0.5, 0.4, 0.3, 0.2, 0.0]
PICKS = [0, 1, 5, 10]  # a few sentences

rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
if (OUT / "audio").exists():
    shutil.rmtree(OUT / "audio")
(OUT / "audio").mkdir(parents=True)

model = HamsVITS.from_checkpoint(str(CKPT)).cuda().eval()
sr = model.sample_rate
print(f"model @ {sr} Hz")

cards = []
for idx in PICKS:
    r = rows[idx]
    pid = torch.tensor([r["phoneme_ids"]], device="cuda")
    lid = torch.tensor([r["language_ids"]], device="cuda")
    variants = []
    # reference (source training clip)
    ref, rsr = sf.read(r["audio"])
    if ref.ndim > 1: ref = ref.mean(1)
    rp = f"audio/{idx}_ref.wav"; sf.write(str(OUT / rp), ref, rsr)
    variants.append(("REFERENCE (source data)", rp, "ref"))
    for ns in NOISE:
        with torch.no_grad():
            wav = model.infer(pid, lid, length_scale=1.0, noise_scale=ns).squeeze().cpu().numpy()
        wav = trim_trailing_artifact(wav, sr)
        p = f"audio/{idx}_ns{ns}.wav"; sf.write(str(OUT / p), wav, sr)
        variants.append((f"ours · noise_scale {ns}", p, "ours"))
    cards.append({"text": r["text"], "variants": variants})
    print(f"clip {idx}: {r['text'][:50]} -> {len(variants)} variants")


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
blocks = []
for c in cards:
    players = "".join(
        f'<div class="p"><span class="lbl {cls}">{lbl}</span>'
        f'<audio controls preload="none" src="{path}"></audio></div>'
        for lbl, path, cls in c["variants"])
    blocks.append(f'<div class="card"><div class="txt" dir="auto">{esc(c["text"])}</div>'
                  f'<div class="players">{players}</div></div>')
html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>texture sweep</title>
<style>
 :root{{color-scheme:light dark}} body{{font:15px/1.6 "Segoe UI",system-ui,sans-serif;max-width:1100px;
  margin:0 auto;padding:22px;background:#0f1319;color:#e6ebf2}}
 @media(prefers-color-scheme:light){{body{{background:#fff;color:#14181f}}}}
 .card{{border:1px solid #2a3342;border-radius:12px;padding:14px 16px;margin:10px 0;background:#161b23}}
 @media(prefers-color-scheme:light){{.card{{background:#f6f8fa;border-color:#d5dae2}}}}
 .txt{{font-size:19px;margin-bottom:10px}} .players{{display:flex;gap:12px;flex-wrap:wrap}}
 .p{{flex:1;min-width:200px}} .lbl{{display:block;font-size:12px;margin-bottom:4px;opacity:.85}}
 .lbl.ref{{color:#e8b23e}} .lbl.ours{{color:#34c98e}} audio{{width:100%}}
 .note{{opacity:.7;font-size:13.5px;margin-bottom:16px}}
</style></head><body>
<h1>Texture sweep — reference vs noise_scale</h1>
<div class="note">Amber = the source recording (our quality ceiling for this data). Green = our
24 kHz model at different inference noise_scale (0.667 default → 0.0 = no prior noise). Find the
smoothest green, and compare the best green to amber: if amber is clearly smoother, more decoder
training closes the gap; if similar, we're near the data ceiling.</div>
{''.join(blocks)}</body></html>"""
(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"\nwrote {OUT/'index.html'} ({len(cards)} sentences x {len(NOISE)+1} variants)")
