"""Robustness on NOVEL text (never in train/eval): long sentences, heavy code-switch,
numbers/dates/currency, rare words. Full pipeline: front-end -> VITS(Badr) -> HiFi-GAN
and -> BigVGAN. Builds a listen page + flags any failures (empty/too-short/NaN audio)."""
from __future__ import annotations
import os, sys, json, shutil
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, soundfile as sf
import transformers; transformers.logging.set_verbosity_error()
from hams_tts.text.frontend import TextFrontend
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.utils.trim import trim_trailing_artifact
from hams_tts.inference.bigvgan_vocoder import revocode

# Novel, deliberately hard sentences (NOT in the 40-clip eval or training text).
SENTENCES = [
    # long pure Levantine
    "بَعرِف إنّو الوَقِت تأخّر كْتير، بَس صَدقيني ما قَصّرِت، ضلّيت عَم حاوِل طول النّهار لَحتّى خلّصِت كِل الشّغِل اللي طَلَبتيه مِنّي.",
    # heavy code-switch (AR + lots of EN)
    "الـ meeting بُكرا الساعة تسعة، وبَدّي نراجِع الـ dashboard والـ analytics قَبِل ما نعمِل الـ presentation قُدّام الـ client.",
    # numbers, currency, date
    "دَفعِت مية وخمسة وعشرين دولار عَ الأوتيل، والرِّحلة بتكلّف تلاتة آلاف وخمسمية ليرة، والموعد يوم اتنين بشهر تلاتة.",
    # question + exclamation + rare words
    "شو صار مَعَك؟! ليش مِتضايِق هَلقَد؟ يَلّا احكيلي كِل شي بالتّفصيل، لا تِخبّي عَنّي وَلا كِلمة!",
    # tech / brand names in code-switch
    "نزّلِت آخِر version مِن الـ app بَس الـ update كَسَر الـ login، فبعتِتلهُن ticket عَ الـ support وقاللي رَح يفيكسوها.",
    # short punchy
    "تَمام، خَلَص، مْنِتّفِق.",
    # English-heavy technical
    "The neural vocoder reduces spectral artifacts, بَس لازِم نـ fine-tune الـ model عَ بيانات أنضَف لَحتّى نوصَل لَ real-time performance.",
    # emphatics + gutturals dense
    "الطّبيب قال إنّو الصّحّة بخير، والضّغِط طبيعي، والقَلِب عَم يِشتِغِل مْنيح، الحَمدُ لله عَ كِل شي.",
]

OUT = REPO / "samples_robustness"
if (OUT / "audio").exists(): shutil.rmtree(OUT / "audio")
(OUT / "audio").mkdir(parents=True)

os.environ.setdefault("ESPEAK_DATA_PATH", r"C:\Users\BOSS\Desktop\utils\espeak-ng\eSpeak NG")
fe = TextFrontend(diacritizer_backend="auto")
model = HamsVITS.from_checkpoint(str(REPO / "checkpoints/flagship_BEST_v4_tex")).cuda().eval()
sr = model.sample_rate

cards, failures = [], []
for i, text in enumerate(SENTENCES):
    u = fe.process(text)
    pid = torch.tensor([u.phoneme_ids], device="cuda"); lid = torch.tensor([u.language_ids], device="cuda")
    with torch.no_grad():
        hwav = model.infer(pid, lid, length_scale=1.0).squeeze().cpu().numpy()
    hwav = trim_trailing_artifact(hwav, sr)
    bwav = trim_trailing_artifact(revocode(hwav), sr)
    # failure checks
    dur = len(bwav) / sr
    ok = np.isfinite(bwav).all() and dur > 0.3 and np.abs(bwav).max() > 0.01
    if not ok:
        failures.append({"i": i, "text": text[:40], "dur": round(dur, 2)})
    for key, w in [("hifigan", hwav), ("bigvgan", bwav)]:
        sf.write(str(OUT / f"audio/{i}_{key}.wav"), w, sr)
    cards.append({"text": text, "ipa": u.ipa, "n_tokens": len(u), "dur": round(dur, 2),
                  "n_spans": len(u.spans)})
    print(f"[{i}] {dur:5.2f}s  {len(u):3d} tok  {len(u.spans)} spans  {'OK' if ok else 'FAIL'}  {text[:45]}")

print(f"\n{len(SENTENCES)-len(failures)}/{len(SENTENCES)} passed | failures: {failures}")


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
blocks = []
for i, c in enumerate(cards):
    blocks.append(
        f'<div class="card"><div class="txt" dir="auto">{esc(c["text"])}</div>'
        f'<div class="ipa">{esc(c["ipa"][:130])}</div>'
        f'<div class="players">'
        f'<div class="p"><span class="lbl h">HiFi-GAN</span><audio controls preload="none" src="audio/{i}_hifigan.wav"></audio></div>'
        f'<div class="p"><span class="lbl b">BigVGAN</span><audio controls preload="none" src="audio/{i}_bigvgan.wav"></audio></div>'
        f'</div></div>')
html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>robustness — novel text</title><style>
 :root{{color-scheme:light dark}} body{{font:15px/1.6 "Segoe UI",system-ui,sans-serif;max-width:920px;margin:0 auto;padding:22px;background:#0f1319;color:#e6ebf2}}
 @media(prefers-color-scheme:light){{body{{background:#fff;color:#14181f}}}}
 .card{{border:1px solid #2a3342;border-radius:12px;padding:14px 16px;margin:10px 0;background:#161b23}}
 @media(prefers-color-scheme:light){{.card{{background:#f6f8fa;border-color:#d5dae2}}}}
 .txt{{font-size:19px;margin-bottom:4px}} .ipa{{font-size:12px;opacity:.55;margin-bottom:10px;font-family:monospace}}
 .players{{display:flex;gap:14px;flex-wrap:wrap}} .p{{flex:1;min-width:260px}}
 .lbl{{display:block;font-size:12px;margin-bottom:4px;opacity:.85}} .lbl.h{{color:#ef6363}} .lbl.b{{color:#34c98e}} audio{{width:100%}}
 .note{{opacity:.75;font-size:13.5px;margin-bottom:16px}}
</style></head><body><h1>Robustness — novel unseen text</h1>
<div class="note">Sentences the model NEVER saw (long, heavy code-switch, numbers/dates/currency, rare words).
Red = HiFi-GAN, green = BigVGAN. Tests whether Badp generalises beyond the 40-clip eval set.</div>
{''.join(blocks)}</body></html>"""
(OUT / "index.html").write_text(html, encoding="utf-8")
(OUT / "report.json").write_text(json.dumps({"cards": cards, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT/'index.html'}")
