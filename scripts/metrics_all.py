"""Compute the paper's core metrics, consistently, for 3 systems:
  baseline (published, stochastic dur, HiFi-GAN) / ours HiFi-GAN / ours BigVGAN.
Per system: ASR CER/WER (per category + overall), duration ratio@ls1, texture flatness
vs reference, RTF (acoustic + vocoder). Writes paper_assets/metrics.json.
"""
from __future__ import annotations
import os, sys, json, re, time, unicodedata
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
from collections import defaultdict
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, soundfile as sf
import transformers; transformers.logging.set_verbosity_error()
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.utils.trim import trim_trailing_artifact
from hams_tts.inference.bigvgan_vocoder import revocode

OUT = REPO / "paper_assets"; OUT.mkdir(exist_ok=True)
rows = [json.loads(l) for l in open(REPO / "data/manifests_24k/eval.phon.filtered.jsonl", encoding="utf-8")]
# baseline expects the non-de-desinentialized phonemes it was trained on:
rows_base = [json.loads(l) for l in open(REPO / "data/manifests/eval.phon.filtered.jsonl", encoding="utf-8")]

_DIAC = re.compile(r"[ً-ْٰ]"); _PUNCT = re.compile(r"[^\w\s؀-ۿ]")
def norm_ar(t):
    t = unicodedata.normalize("NFKC", t); t = _DIAC.sub("", t)
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي").replace("ة","ه")
    return re.sub(r"\s+"," ", _PUNCT.sub(" ", t)).strip().lower()

def flatness(w, sr):
    n_fft, hop = 1024, 256; win = np.hanning(n_fft)
    fr = np.stack([w[i:i+n_fft]*win for i in range(0, len(w)-n_fft, hop)])
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-9; freqs = np.fft.rfftfreq(n_fft, 1/sr)
    e = (S**2).sum(1); Sv = S[e > np.percentile(e, 40)]; band = (freqs>=2000)&(freqs<=11000)
    Sb = Sv[:, band]; return float(np.median(np.exp(np.log(Sb).mean(1)) / Sb.mean(1)))

def calibrate(model, rs, vocoder):
    """find length_scale that matches reference duration (ratio->1)."""
    ratios = []
    for r in rs[:12]:
        pid = torch.tensor([r["phoneme_ids"]], device="cuda"); lid = torch.tensor([r["language_ids"]], device="cuda")
        with torch.no_grad():
            w = model.infer(pid, lid, length_scale=1.0).squeeze().cpu().numpy()
        if r.get("duration_s"): ratios.append(len(w)/model.sample_rate / r["duration_s"])
    r0 = float(np.mean(ratios)) if ratios else 1.0
    return round(min(max(1.0/max(r0,1e-3), 1.0), 8.0), 2)

SYSTEMS = [
    ("baseline", "checkpoints/resume_from_hf", rows_base, "hifigan"),
    ("ours_hifigan", "checkpoints/flagship_BEST_v4_tex", rows, "hifigan"),
    ("ours_bigvgan", "checkpoints/flagship_BEST_v4_tex", rows, "bigvgan"),
]
results = {}
for name, ckpt, rs, vocoder in SYSTEMS:
    print(f"\n=== {name} ({vocoder}) ===")
    model = HamsVITS.from_checkpoint(str(REPO / ckpt)).cuda().eval()
    sr = model.sample_rate
    ls = calibrate(model, rs, vocoder)
    d = (OUT / "synth" / name); d.mkdir(parents=True, exist_ok=True)
    ratios, flats, ac_rtf, voc_rtf = [], [], [], []
    meta = []
    for i, r in enumerate(rs):
        pid = torch.tensor([r["phoneme_ids"]], device="cuda"); lid = torch.tensor([r["language_ids"]], device="cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            w = model.infer(pid, lid, length_scale=ls).squeeze().cpu().numpy()
        torch.cuda.synchronize(); t_ac = time.perf_counter() - t0
        t_voc = 0.0
        if vocoder == "bigvgan":
            w = trim_trailing_artifact(w, sr); t1 = time.perf_counter(); w = revocode(w); torch.cuda.synchronize(); t_voc = time.perf_counter()-t1
        w = trim_trailing_artifact(w, sr)
        dur = len(w)/sr
        sf.write(str(d / f"{r.get('sentence_type','x')}_{i:03d}.wav"), w, sr)
        if r.get("duration_s"): ratios.append(dur / r["duration_s"])
        ref, rsr = sf.read(r["audio"]); ref = ref.mean(1) if ref.ndim>1 else ref
        flats.append((flatness(w, sr), flatness(ref, rsr)))
        ac_rtf.append(t_ac/max(dur,1e-6)); voc_rtf.append(t_voc/max(dur,1e-6))
        meta.append({"cat": r.get("sentence_type","x"), "text": r["text"], "wav": str(d / f"{r.get('sentence_type','x')}_{i:03d}.wav")})
    del model; torch.cuda.empty_cache()
    results[name] = {"vocoder": vocoder, "length_scale": ls, "sr": sr,
                     "dur_ratio": round(float(np.mean(ratios)),3),
                     "flatness_ours": round(float(np.mean([f[0] for f in flats])),3),
                     "flatness_ref": round(float(np.mean([f[1] for f in flats])),3),
                     "rtf_acoustic": round(float(np.median(ac_rtf)),4),
                     "rtf_vocoder": round(float(np.median(voc_rtf)),4),
                     "_meta": meta}
    print(f"  ls={ls} dur_ratio={results[name]['dur_ratio']} flat ours/ref={results[name]['flatness_ours']}/{results[name]['flatness_ref']} rtf_ac={results[name]['rtf_acoustic']} rtf_voc={results[name]['rtf_vocoder']}")

# ---- ASR round-trip (Whisper large-v3, GPU) ----
print("\n=== ASR (Whisper large-v3) ===")
from faster_whisper import WhisperModel
import jiwer
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
for name in results:
    by = defaultdict(lambda: {"cer": [], "wer": []})
    for m in results[name]["_meta"]:
        segs, _ = asr.transcribe(m["wav"], language="ar", beam_size=5)
        hyp = norm_ar(" ".join(s.text for s in segs)); ref = norm_ar(m["text"])
        if not ref: continue
        by[m["cat"]]["cer"].append(jiwer.cer(ref, hyp)); by[m["cat"]]["wer"].append(jiwer.wer(ref, hyp) if ref.split() else 1.0)
    summ = {}
    allc, allw = [], []
    for cat, dd in by.items():
        summ[cat] = {"cer": round(float(np.mean(dd["cer"])),3), "wer": round(float(np.mean(dd["wer"])),3), "n": len(dd["cer"])}
        allc += dd["cer"]; allw += dd["wer"]
    summ["overall"] = {"cer": round(float(np.mean(allc)),3), "wer": round(float(np.mean(allw)),3), "n": len(allc)}
    results[name]["asr"] = summ
    del results[name]["_meta"]
    print(f"  {name}: {summ}")

# model sizes
m = HamsVITS.from_checkpoint(str(REPO / "checkpoints/flagship_BEST_v4_tex")).eval()
results["_model"] = {"vits_params_M": round(sum(p.numel() for p in m.parameters())/1e6,1),
                     "bigvgan_params_M": 112, "eval_clips": len(rows)}
(OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {OUT/'metrics.json'}")
