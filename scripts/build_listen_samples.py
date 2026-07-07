"""Package a multi-model listening comparison into a self-contained folder + HTML player.

Copies the already-synthesized eval WAVs for each model into samples_listen/ with clear
names and generates index.html with side-by-side audio players + the Arabic/code-switch
text, grouped by category. Open the HTML in a browser.
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANIFEST = REPO / "data/manifests/eval.phon.filtered.jsonl"  # text/order (same across runs)
OUT = REPO / "samples_listen"

# (eval_out subdir, short key, display label, css color class). Same 40 sentences/order.
MODELS = [
    ("resume_from_hf_ls3.24", "baseline", "Published baseline (16 kHz, robotic 3× stretch)", "base"),
    ("step_4000_ls1.03", "v16", "16 kHz — deterministic duration (natural rhythm)", "v16"),
    ("step_10000_ls1.07", "v24", "24 kHz — higher fidelity (＋ everything above)", "v24"),
]

rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
OUT.mkdir(exist_ok=True)
audio_dir = OUT / "audio"
if audio_dir.exists():
    shutil.rmtree(audio_dir)
audio_dir.mkdir()

cards = []
missing = 0
for i, r in enumerate(rows):
    cat = r.get("sentence_type", "x")
    src_name = f"{cat}_{i:03d}.wav"
    srcs = {key: (REPO / "eval_out" / d / src_name) for d, key, _lbl, _c in MODELS}
    if not all(p.exists() for p in srcs.values()):
        missing += 1
        continue
    dsts = {}
    for key, p in srcs.items():
        rel = f"audio/{i:02d}_{cat}_{key}.wav"
        shutil.copyfile(p, OUT / rel)
        dsts[key] = rel
    cards.append({"i": i, "cat": cat, "text": r["text"], "audio": dsts})

order = {"pure_levantine": 0, "code_switching": 1}
cards.sort(key=lambda c: (order.get(c["cat"], 9), c["i"]))


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


rows_html = []
cur_cat = None
for c in cards:
    if c["cat"] != cur_cat:
        cur_cat = c["cat"]
        title = "Pure Levantine Arabic" if cur_cat == "pure_levantine" else "Code-switching (Arabic + English)"
        rows_html.append(f"<h2>{title}</h2>")
    players = "".join(
        f'<div class="p"><span class="lbl {cls}">{lbl}</span>'
        f'<audio controls preload="none" src="{c["audio"][key]}"></audio></div>'
        for _d, key, lbl, cls in MODELS
    )
    rows_html.append(
        f'<div class="card"><div class="txt" dir="auto">{esc(c["text"])}</div>'
        f'<div class="players">{players}</div></div>'
    )

html = f"""<!doctype html><html lang="ar"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hams Levantine TTS — model comparison</title>
<style>
 :root{{color-scheme:light dark}}
 body{{font:16px/1.6 "Segoe UI",system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:24px;
   background:#0f1319;color:#e6ebf2}}
 @media(prefers-color-scheme:light){{body{{background:#fff;color:#14181f}}}}
 h1{{font-size:22px}} h1 small{{font-weight:400;opacity:.6;font-size:14px}}
 h2{{margin:28px 0 8px;font-size:15px;text-transform:uppercase;letter-spacing:.08em;opacity:.7}}
 .card{{border:1px solid #2a3342;border-radius:12px;padding:14px 16px;margin:10px 0;background:#161b23}}
 @media(prefers-color-scheme:light){{.card{{background:#f6f8fa;border-color:#d5dae2}}}}
 .txt{{font-size:20px;margin-bottom:10px}}
 .players{{display:flex;gap:14px;flex-wrap:wrap}}
 .p{{flex:1;min-width:220px}}
 .lbl{{display:block;font-size:12px;margin-bottom:4px;opacity:.85}}
 .lbl.base{{color:#e8b23e}} .lbl.v16{{color:#4f8cff}} .lbl.v24{{color:#34c98e}}
 audio{{width:100%}}
 .note{{opacity:.7;font-size:13.5px;margin:6px 0 18px}}
 .note b{{color:#34c98e}}
</style></head><body>
<h1>Hams Levantine TTS <small>— model progression</small></h1>
<div class="note">Same sentence, three models left→right: amber = published baseline (robotic),
blue = 16 kHz with the deterministic duration head (natural rhythm), <b>green = 24 kHz (higher
fidelity)</b>. Rhythm/intelligibility jump is baseline→blue; the fidelity/"real recording" jump is
blue→green. Held-out Levantine CER: 0.69 (baseline) → 0.11 (blue) → 0.13 (green — CER flat because
24 kHz adds fidelity, not intelligibility; judge green by ear).</div>
{''.join(rows_html)}
</body></html>"""

(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"wrote {len(cards)} sentences x {len(MODELS)} models ({missing} missing) -> {OUT}")
print(f"open: {OUT / 'index.html'}")
