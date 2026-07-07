"""A/B: old 24 kHz (buzzy) vs new 24 kHz texture model (smooth). Synthesizes both on a
set of eval sentences (trailing-trim applied) into samples_texture_ab/index.html."""
from __future__ import annotations
import os, sys, json, shutil
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, soundfile as sf
import transformers; transformers.logging.set_verbosity_error()
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.utils.trim import trim_trailing_artifact

MODELS = [
    ("flagship_BEST_v4_tex", "old", "before STFT loss (feature-matching only)"),
    ("flagship_spk01_stft/step_5000", "new", "after multi-res STFT loss"),
]
PICKS = [0, 1, 2, 5, 10, 15]
OUT = REPO / "samples_texture_ab"
rows = [json.loads(l) for l in open(REPO / "data/manifests_24k/eval.phon.filtered.jsonl", encoding="utf-8")]
if (OUT / "audio").exists(): shutil.rmtree(OUT / "audio")
(OUT / "audio").mkdir(parents=True)

synth = {}
for ck, key, _lbl in MODELS:
    m = HamsVITS.from_checkpoint(str(REPO / "checkpoints" / ck)).cuda().eval()
    sr = m.sample_rate
    for idx in PICKS:
        r = rows[idx]
        pid = torch.tensor([r["phoneme_ids"]], device="cuda"); lid = torch.tensor([r["language_ids"]], device="cuda")
        with torch.no_grad():
            w = m.infer(pid, lid, length_scale=1.0).squeeze().cpu().numpy()
        w = trim_trailing_artifact(w, sr)
        rel = f"audio/{idx}_{key}.wav"; sf.write(str(OUT / rel), w, sr)
        synth[(idx, key)] = rel
    del m; torch.cuda.empty_cache()

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
blocks = []
for idx in PICKS:
    r = rows[idx]
    players = "".join(
        f'<div class="p"><span class="lbl {key}">{lbl}</span>'
        f'<audio controls preload="none" src="{synth[(idx,key)]}"></audio></div>'
        for _ck, key, lbl in MODELS)
    blocks.append(f'<div class="card"><div class="txt" dir="auto">{esc(r["text"])}</div>'
                  f'<div class="players">{players}</div></div>')
html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>texture A/B</title><style>
 :root{{color-scheme:light dark}} body{{font:15px/1.6 "Segoe UI",system-ui,sans-serif;max-width:900px;margin:0 auto;padding:22px;background:#0f1319;color:#e6ebf2}}
 @media(prefers-color-scheme:light){{body{{background:#fff;color:#14181f}}}}
 .card{{border:1px solid #2a3342;border-radius:12px;padding:14px 16px;margin:10px 0;background:#161b23}}
 @media(prefers-color-scheme:light){{.card{{background:#f6f8fa;border-color:#d5dae2}}}}
 .txt{{font-size:19px;margin-bottom:10px}} .players{{display:flex;gap:14px;flex-wrap:wrap}}
 .p{{flex:1;min-width:240px}} .lbl{{display:block;font-size:12px;margin-bottom:4px;opacity:.85}}
 .lbl.old{{color:#e8b23e}} .lbl.new{{color:#34c98e}} audio{{width:100%}} .note{{opacity:.7;font-size:13.5px;margin-bottom:16px}}
</style></head><body><h1>Texture A/B — buzz reduction</h1>
<div class="note">Amber = the 24 kHz you heard (slight robotic buzz). Green = after the texture pass
(stronger feature-matching + a discriminator that now keeps training). Same words, natural pacing,
end-trim applied. Objective: our texture went from +0.020 buzzier-than-source to +0.002 (at source),
CER held (Levantine 0.12). Your ears: is the green cleaner/less robotic?</div>
{''.join(blocks)}</body></html>"""
(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"wrote {OUT/'index.html'} ({len(PICKS)} sentences x {len(MODELS)})")
