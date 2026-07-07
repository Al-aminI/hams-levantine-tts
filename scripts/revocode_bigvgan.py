"""Validation: does BigVGAN remove the 'air'? Re-vocode our best model's output through
the pretrained BigVGAN (our wav -> mel -> BigVGAN -> wav). Since our magnitude spectrum
already matches the reference, the mel is good; BigVGAN regenerates clean phase/excitation.
Builds a 3-way A/B: reference / ours (HiFi-GAN) / ours->BigVGAN.
"""
from __future__ import annotations
import os, sys, json, shutil
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "vendor" / "BigVGAN"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch, soundfile as sf, numpy as np
import transformers; transformers.logging.set_verbosity_error()
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.utils.trim import trim_trailing_artifact
import bigvgan
from meldataset import get_mel_spectrogram

device = "cuda"
print("loading BigVGAN nvidia/bigvgan_v2_24khz_100band_256x ...")
bv = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
bv.remove_weight_norm(); bv = bv.eval().to(device)
print(f"BigVGAN loaded | expects {bv.h.sampling_rate} Hz, {bv.h.num_mels} mels, hop {bv.h.hop_size}")

model = HamsVITS.from_checkpoint(str(REPO / "checkpoints/flagship_BEST_v4_tex")).cuda().eval()
sr = model.sample_rate
assert sr == bv.h.sampling_rate, f"SR mismatch {sr} vs {bv.h.sampling_rate}"

rows = [json.loads(l) for l in open(REPO / "data/manifests_24k/eval.phon.filtered.jsonl", encoding="utf-8")]
picks = [0, 1, 2, 5, 10]
OUT = REPO / "samples_bigvgan"
if (OUT / "audio").exists(): shutil.rmtree(OUT / "audio")
(OUT / "audio").mkdir(parents=True)


def revocode(wav):
    w = torch.FloatTensor(wav).unsqueeze(0)
    mel = get_mel_spectrogram(w, bv.h).to(device)
    with torch.inference_mode():
        out = bv(mel).squeeze().cpu().numpy()
    return out.astype(np.float32)


cards = []
for idx in picks:
    r = rows[idx]
    pid = torch.tensor([r["phoneme_ids"]], device=device); lid = torch.tensor([r["language_ids"]], device=device)
    with torch.no_grad():
        ours = model.infer(pid, lid, length_scale=1.0).squeeze().cpu().numpy()
    ours = trim_trailing_artifact(ours, sr)
    ours_bv = trim_trailing_artifact(revocode(ours), sr)
    ref, rsr = sf.read(r["audio"]); ref = ref.mean(1) if ref.ndim > 1 else ref
    variants = [("REFERENCE (source)", "ref", ref, rsr),
                ("ours — HiFi-GAN (v4_tex)", "hifigan", ours, sr),
                ("ours → BigVGAN", "bigvgan", ours_bv, sr)]
    v = []
    for lbl, key, sig, ssr in variants:
        rel = f"audio/{idx}_{key}.wav"; sf.write(str(OUT / rel), sig, ssr)
        v.append((lbl, key, rel))
    cards.append({"text": r["text"], "variants": v})
    print(f"clip {idx}: done")


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
blocks = []
for c in cards:
    players = "".join(f'<div class="p"><span class="lbl {k}">{l}</span>'
                      f'<audio controls preload="none" src="{p}"></audio></div>' for l, k, p in c["variants"])
    blocks.append(f'<div class="card"><div class="txt" dir="auto">{esc(c["text"])}</div>'
                  f'<div class="players">{players}</div></div>')
html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BigVGAN test</title><style>
 :root{{color-scheme:light dark}} body{{font:15px/1.6 "Segoe UI",system-ui,sans-serif;max-width:1050px;margin:0 auto;padding:22px;background:#0f1319;color:#e6ebf2}}
 @media(prefers-color-scheme:light){{body{{background:#fff;color:#14181f}}}}
 .card{{border:1px solid #2a3342;border-radius:12px;padding:14px 16px;margin:10px 0;background:#161b23}}
 @media(prefers-color-scheme:light){{.card{{background:#f6f8fa;border-color:#d5dae2}}}}
 .txt{{font-size:19px;margin-bottom:10px}} .players{{display:flex;gap:12px;flex-wrap:wrap}}
 .p{{flex:1;min-width:210px}} .lbl{{display:block;font-size:12px;margin-bottom:4px;opacity:.85}}
 .lbl.ref{{color:#e8b23e}} .lbl.hifigan{{color:#ef6363}} .lbl.bigvgan{{color:#34c98e}} audio{{width:100%}}
 .note{{opacity:.7;font-size:13.5px;margin-bottom:16px}}
</style></head><body><h1>BigVGAN re-vocode test</h1>
<div class="note">Amber = source. Red = our current HiFi-GAN (with the 'air'). Green = our same output
re-vocoded through pretrained BigVGAN. If green is cleaner/closer to amber than red, BigVGAN is the
fix and we integrate it properly (train the acoustic model to feed BigVGAN directly).</div>
{''.join(blocks)}</body></html>"""
(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"\nwrote {OUT/'index.html'}")
