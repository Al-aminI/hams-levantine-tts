"""Where does our output diverge from the reference in the spectrum? Overlays the
long-term average spectrum and the noise-floor spectrum (quietest frames) of REF vs
ours, 0-12 kHz, to locate the 'air through a mic' (HF broadband noise)."""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np, torch, soundfile as sf, matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import transformers; transformers.logging.set_verbosity_error()
from hams_tts.models.hams_vits import HamsVITS

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "checkpoints/flagship_BEST_v4_tex"
rows = [json.loads(l) for l in open(REPO / "data/manifests_24k/eval.phon.filtered.jsonl", encoding="utf-8")]
picks = [0, 1, 5, 10]
model = HamsVITS.from_checkpoint(str(CKPT)).cuda().eval()
sr = model.sample_rate


def spectra(w, sr):
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    fr = np.stack([w[i:i+n_fft]*win for i in range(0, len(w)-n_fft, hop)])
    S = np.abs(np.fft.rfft(fr, axis=1))                       # (T,F)
    e = (S**2).sum(1)
    avg = 20*np.log10(S.mean(0) + 1e-9)                        # long-term avg
    quiet = S[e < np.percentile(e, 20)]                        # noise-floor frames
    floor = 20*np.log10(quiet.mean(0) + 1e-9) if len(quiet) else avg
    return avg, floor


freqs = np.fft.rfftfreq(1024, 1/sr)
fig, axes = plt.subplots(len(picks), 2, figsize=(14, 3*len(picks)))
hf = freqs >= 6000
for i, idx in enumerate(picks):
    r = rows[idx]
    ref, rsr = sf.read(r["audio"]); ref = ref.mean(1) if ref.ndim > 1 else ref
    pid = torch.tensor([r["phoneme_ids"]], device="cuda"); lid = torch.tensor([r["language_ids"]], device="cuda")
    with torch.no_grad():
        ours = model.infer(pid, lid, length_scale=1.0).squeeze().cpu().numpy()
    ra, rfl = spectra(ref, rsr); oa, ofl = spectra(ours, sr)
    # normalize both to 0 dB peak for shape comparison
    ra -= ra.max(); oa -= oa.max(); rfl -= rfl.max(); ofl -= ofl.max()
    axes[i, 0].plot(freqs, ra, label="reference", color="#34c98e")
    axes[i, 0].plot(freqs, oa, label="ours", color="#ef6363", alpha=.8)
    axes[i, 0].set_title(f"clip{idx} long-term avg spectrum"); axes[i, 0].set_xlim(0, sr/2)
    axes[i, 0].set_ylim(-80, 2); axes[i, 0].legend(fontsize=8); axes[i, 0].axvline(6000, color="gray", ls=":", lw=.7)
    axes[i, 1].plot(freqs, rfl, label="reference", color="#34c98e")
    axes[i, 1].plot(freqs, ofl, label="ours", color="#ef6363", alpha=.8)
    axes[i, 1].set_title(f"clip{idx} noise-floor (quiet frames)"); axes[i, 1].set_xlim(0, sr/2)
    axes[i, 1].set_ylim(-80, 2); axes[i, 1].legend(fontsize=8)
    # HF metrics
    ref_hf = 10*np.log10((10**(ra/10))[hf].mean()); our_hf = 10*np.log10((10**(oa/10))[hf].mean())
    print(f"clip{idx}: avg HF(>6k) rel-energy  ref {ref_hf:6.1f} dB  ours {our_hf:6.1f} dB  (ours-ref {our_hf-ref_hf:+.1f} dB)")
plt.tight_layout(); fig.savefig(str(REPO/"eval_out/spec_compare.png"), dpi=90)
print("wrote", REPO/"eval_out/spec_compare.png")
