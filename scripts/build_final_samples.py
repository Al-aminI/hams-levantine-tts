"""The full journey, per sentence: reference / published baseline / our best HiFi-GAN /
FINAL (that + BigVGAN). Uses the bigvgan_vocoder integration."""
from __future__ import annotations
import os, sys, json, shutil
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, soundfile as sf, numpy as np
import transformers; transformers.logging.set_verbosity_error()
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.utils.trim import trim_trailing_artifact
from hams_tts.inference.bigvgan_vocoder import revocode

rows = [json.loads(l) for l in open(REPO / "data/manifests_24k/eval.phon.filtered.jsonl", encoding="utf-8")]
picks = [0, 1, 2, 5, 10, 15]
OUT = REPO / "samples_final"
if (OUT / "audio").exists(): shutil.rmtree(OUT / "audio")
(OUT / "audio").mkdir(parents=True)

base = HamsVITS.from_checkpoint(str(REPO / "checkpoints/resume_from_hf")).cuda().eval()   # published, 16 kHz stochastic
best = HamsVITS.from_checkpoint(str(REPO / "checkpoints/flagship_BEST_v4_tex")).cuda().eval()  # our best VITS, 24 kHz


def synth(model, r, ls):
    pid = torch.tensor([r["phoneme_ids"]], device="cuda"); lid = torch.tensor([r["language_ids"]], device="cuda")
    with torch.no_grad():
        return model.infer(pid, lid, length_scale=ls).squeeze().cpu().numpy(), model.sample_rate


cards = []
for idx in picks:
    r = rows[idx]
    ref, rsr = sf.read(r["audio"]); ref = ref.mean(1) if ref.ndim > 1 else ref
    bwav, bsr = synth(base, r, 3.24)                       # published baseline (needs ~3x stretch)
    hwav, hsr = synth(best, r, 1.0); hwav = trim_trailing_artifact(hwav, hsr)   # our best HiFi-GAN
    fwav = trim_trailing_artifact(revocode(hwav), hsr)     # FINAL: + BigVGAN
    variants = [("REFERENCE (source)", "ref", ref, rsr),
                ("published baseline (start)", "base", bwav, bsr),
                ("our best HiFi-GAN", "hifigan", hwav, hsr),
                ("FINAL — + BigVGAN", "final", fwav, hsr)]
    v = []
    for lbl, key, sig, ssr in variants:
        rel = f"audio/{idx}_{key}.wav"; sf.write(str(OUT / rel), sig, ssr); v.append((lbl, key, rel))
    cards.append({"text": r["text"], "variants": v})
    print(f"clip {idx} done")


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
blocks = []
for c in cards:
    players = "".join(f'<div class="p"><span class="lbl {k}">{l}</span>'
                      f'<audio controls preload="none" src="{p}"></audio></div>' for l, k, p in c["variants"])
    blocks.append(f'<div class="card"><div class="txt" dir="auto">{esc(c["text"])}</div>'
                  f'<div class="players">{players}</div></div>')
html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hams — final</title><style>
 :root{{color-scheme:light dark}} body{{font:15px/1.6 "Segoe UI",system-ui,sans-serif;max-width:1150px;margin:0 auto;padding:22px;background:#0f1319;color:#e6ebf2}}
 @media(prefers-color-scheme:light){{body{{background:#fff;color:#14181f}}}}
 .card{{border:1px solid #2a3342;border-radius:12px;padding:14px 16px;margin:10px 0;background:#161b23}}
 @media(prefers-color-scheme:light){{.card{{background:#f6f8fa;border-color:#d5dae2}}}}
 .txt{{font-size:19px;margin-bottom:10px}} .players{{display:flex;gap:12px;flex-wrap:wrap}}
 .p{{flex:1;min-width:180px}} .lbl{{display:block;font-size:12px;margin-bottom:4px;opacity:.85}}
 .lbl.ref{{color:#e8b23e}} .lbl.base{{color:#8b96a8}} .lbl.hifigan{{color:#ef6363}} .lbl.final{{color:#34c98e}} audio{{width:100%}}
 .note{{opacity:.75;font-size:13.5px;margin-bottom:16px}}
</style></head><body><h1>Hams Levantine TTS — the journey</h1>
<div class="note">Left→right: amber source · grey published baseline (robotic) · red our best VITS/HiFi-GAN
(natural but slight air) · <b style="color:#34c98e">green FINAL = that + BigVGAN vocoder</b>. Single flagship
speaker "Badr", 24 kHz. Levantine CER 0.69→0.11 across the project; the last step (green) closes the vocoder 'air'.</div>
{''.join(blocks)}</body></html>"""
(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"\nwrote {OUT/'index.html'}")
