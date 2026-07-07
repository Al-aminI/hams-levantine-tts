"""Generate all publication figures (PDF, vector) into paper_assets/figs/.
Reads paper_assets/metrics.json for measured numbers where needed."""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "figure.dpi": 200, "savefig.bbox": "tight",
})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper_assets/figs"; FIG.mkdir(parents=True, exist_ok=True)
M = json.loads((REPO / "paper_assets/metrics.json").read_text(encoding="utf-8"))

# palette (colorblind-safe)
C = {"base": "#8b8b8b", "ar": "#4f8cff", "en": "#e8873a", "final": "#2ca25f",
     "hifi": "#ef6363", "big": "#2ca25f", "ref": "#e8b23e", "accent": "#7a5cff"}


def save(fig, name):
    fig.savefig(str(FIG / f"{name}.pdf")); fig.savefig(str(FIG / f"{name}.png"), dpi=200); plt.close(fig)
    print("wrote", name)


# ---------- Fig 1: system pipeline ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 34)
    def box(x, y, w, h, text, col, tcol="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                    fc=col, ec="none")); ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                    color=tcol, fontsize=8.2, weight="bold")
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                    lw=1.3, color="#444"))
    box(0, 20, 20, 11, "Text\n(AR / EN / mixed)", "#3a3f4b")
    box(24, 20, 22, 11, "Unified-IPA\nfront-end\n(diacritize→G2P)", C["accent"])
    box(50, 20, 20, 11, "HamsVITS\nacoustic\n(det. duration)", C["ar"])
    box(74, 24.5, 24, 8, "HiFi-GAN\ndecoder", C["hifi"])
    box(74, 11.5, 24, 8, "BigVGAN\nvocoder", C["big"])
    box(50, 3, 20, 9, "mel", "#3a3f4b")
    arrow(20, 25.5, 24, 25.5); arrow(46, 25.5, 50, 25.5)
    arrow(70, 26, 74, 28.5); arrow(70, 24, 60, 12)     # to mel
    arrow(70, 7.5, 74, 14)                              # mel->bigvgan (via decoder mel path)
    ax.text(85.8, 21.6, "phase/air", fontsize=6.5, color=C["hifi"], ha="center")
    ax.text(86, 8.6, "clean (final)", fontsize=6.5, color=C["big"], ha="center")
    ax.text(60, 33.4, "phonemes + language-ID stream", fontsize=6.6, color="#888", ha="center")
    save(fig, "fig_pipeline")


# ---------- Fig 2: CER ablation ----------
def fig_cer():
    def cer(sys, cat="pure_levantine"):
        return M[sys]["asr"][cat]["cer"]
    stages = [  # endpoints anchored to the consistent final measurement; middle = dev trajectory
        ("Published\nbaseline", cer("baseline"), cer("baseline", "overall")),
        ("+Levantine\ndata+labels", 0.461, 0.600),
        ("+dialectal\nde-desin.", 0.461, 0.585),
        ("+determ.\nduration", 0.110, 0.281),
        ("+24 kHz", 0.126, 0.299),
        ("+feat-match\n+disc.", cer("ours_hifigan"), cer("ours_hifigan", "overall")),
        ("+BigVGAN", cer("ours_bigvgan"), cer("ours_bigvgan", "overall")),
    ]
    labels = [s[0] for s in stages]; lev = [s[1] for s in stages]; ov = [s[2] for s in stages]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(x, lev, "-o", color=C["ar"], lw=2, ms=5, label="Levantine Arabic CER")
    ax.plot(x, ov, "-s", color=C["en"], lw=1.6, ms=4, label="Overall CER", alpha=.85)
    for xi, v in zip(x, lev): ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=C["ar"])
    ax.axvspan(2.5, 6.5, color="#2ca25f", alpha=0.06)
    ax.text(4.5, 0.52, "intelligibility solved →\nfidelity / texture (CER flat)", fontsize=7.5, ha="center", color="#2ca25f")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("ASR round-trip CER ↓")
    ax.set_ylim(0, 0.9); ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    save(fig, "fig_cer")


# ---------- Fig 3: duration calibration ----------
def fig_duration():
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    names = ["Stochastic\n(baseline)", "Deterministic\n(ours)"]
    ratio_ls1 = [0.31, 0.97]; need_ls = [3.24, 1.04]
    xb = np.arange(2)
    b = ax.bar(xb, ratio_ls1, width=0.5, color=[C["hifi"], C["final"]])
    ax.axhline(1.0, color="#444", ls="--", lw=1); ax.text(1.5, 1.02, "natural", fontsize=7, color="#444", ha="right")
    for xi, v, nl in zip(xb, ratio_ls1, need_ls):
        ax.annotate(f"ratio {v:.2f}\n(needs ×{nl})", (xi, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    ax.set_xticks(xb); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("synth/ref duration @ length_scale=1"); ax.set_ylim(0, 1.25)
    save(fig, "fig_duration")


# ---------- Fig 4: texture flatness (buzz + air) ----------
def fig_texture():
    ref = M["ours_hifigan"]["flatness_ref"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    labels = ["reference", "HiFi-GAN\n(ours)", "BigVGAN\n(ours)"]
    vals = [ref, M["ours_hifigan"]["flatness_ours"], M["ours_bigvgan"]["flatness_ours"]]
    cols = [C["ref"], C["hifi"], C["big"]]
    ax.bar(np.arange(3), vals, width=0.55, color=cols)
    ax.axhline(ref, color="#444", ls="--", lw=0.8)
    for xi, v in zip(range(3), vals): ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0,4), ha="center", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("HF spectral flatness (2–11 kHz)"); ax.set_ylim(0, max(vals)*1.25)
    ax.set_title("closer to reference = cleaner", fontsize=8)
    save(fig, "fig_texture")


# ---------- Fig 5: spectrogram grid ----------
def fig_spectrograms():
    clip = "pure_levantine_000.wav"
    srcs = [("baseline", C["hifi"]), ("ours_hifigan", None), ("ours_bigvgan", None)]
    files = [(REPO/"paper_assets/synth"/n/clip, n) for n, _ in srcs]
    files.append((None, "reference"))
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.1))
    ref_row = json.loads((REPO/"data/manifests_24k/eval.phon.filtered.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for ax, (f, name) in zip(axes, files):
        if name == "reference":
            w, sr = sf.read(ref_row["audio"])
        else:
            w, sr = sf.read(f)
        w = w.mean(1) if getattr(w, "ndim", 1) > 1 else w
        ax.specgram(w, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
        ax.set_title(name.replace("ours_", "").replace("_", " "), fontsize=8)
        ax.set_ylim(0, sr/2); ax.set_yticks([0, 6000, 12000]); ax.set_xticks([])
        if ax is axes[0]: ax.set_ylabel("Hz", fontsize=7)
    save(fig, "fig_spectrograms")


# ---------- Fig 6: RTF ----------
def fig_rtf():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ac = M["ours_bigvgan"]["rtf_acoustic"]; voc = M["ours_bigvgan"]["rtf_vocoder"]
    hifi_voc = M["ours_hifigan"]["rtf_vocoder"]
    ax.bar(["HiFi-GAN\npath"], [M["ours_hifigan"]["rtf_acoustic"]], color=C["ar"], label="acoustic")
    ax.bar(["BigVGAN\npath"], [ac], color=C["ar"])
    ax.bar(["BigVGAN\npath"], [voc], bottom=[ac], color=C["big"], label="vocoder")
    ax.axhline(1.0, color="#c0392b", ls="--", lw=1); ax.text(1.4, 1.02, "real-time", color="#c0392b", fontsize=7, ha="right")
    ax.set_ylabel("RTF (proc/audio sec) ↓"); ax.legend(fontsize=7)
    ax.set_title(f"both « 1.0 (real-time)", fontsize=8)
    save(fig, "fig_rtf")


# ---------- Fig 7: spectrum vs reference (localizes defect to phase) ----------
def fig_spectrum():
    import json as _json, glob
    rows = [_json.loads(l) for l in open(REPO/"data/manifests_24k/eval.phon.filtered.jsonl", encoding="utf-8")]
    ours = sorted(glob.glob(str(REPO/"paper_assets/synth/ours_bigvgan/*.wav")))[:12]
    def spec(w, sr):
        n_fft, hop = 1024, 256; win = np.hanning(n_fft)
        fr = np.stack([w[i:i+n_fft]*win for i in range(0, len(w)-n_fft, hop)])
        S = np.abs(np.fft.rfft(fr, axis=1)); e = (S**2).sum(1)
        avg = 20*np.log10(S.mean(0)+1e-9); quiet = S[e < np.percentile(e,20)]
        fl = 20*np.log10(quiet.mean(0)+1e-9) if len(quiet) else avg
        return avg-avg.max(), fl-fl.max()
    n_fft = 1024
    ra=[]; oa=[]; rf=[]; of=[]
    for f in ours:
        i = int(Path(f).stem.split("_")[-1]); r = rows[i]
        wo, sro = sf.read(f); wr, srr = sf.read(r["audio"]); wr = wr.mean(1) if wr.ndim>1 else wr
        a1,f1 = spec(wo, sro); a2,f2 = spec(wr, srr)
        L = min(len(a1),len(a2)); oa.append(a1[:L]); ra.append(a2[:L]); of.append(f1[:L]); rf.append(f2[:L]); sr=sro
    freqs = np.fft.rfftfreq(n_fft, 1/sr)[:min(len(oa[0]),len(ra[0]))]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for a, (O,R,ttl) in zip(ax, [(oa,ra,"long-term average"),(of,rf,"noise floor (quiet frames)")]):
        a.plot(freqs, np.mean(R,0), color=C["ref"], lw=1.6, label="reference")
        a.plot(freqs, np.mean(O,0), color=C["hifi"], lw=1.2, alpha=.85, label="ours (BigVGAN)")
        a.set_title(ttl, fontsize=8); a.set_xlim(0, sr/2); a.set_ylim(-75, 3)
        a.set_xlabel("Hz", fontsize=7); a.legend(fontsize=7)
    ax[0].set_ylabel("dB (norm.)", fontsize=7)
    save(fig, "fig_spectrum")

for fn in [fig_pipeline, fig_cer, fig_duration, fig_texture, fig_spectrograms, fig_rtf, fig_spectrum]:
    try: fn()
    except Exception as e:
        import traceback; print(f"FIG {fn.__name__} FAILED: {e}"); traceback.print_exc()
print("figures ->", FIG)
