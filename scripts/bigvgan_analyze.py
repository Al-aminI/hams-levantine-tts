"""Objective texture comparison of the 3 variants in samples_bigvgan/audio:
reference vs ours-HiFiGAN vs ours-BigVGAN. Spectral flatness (2-11 kHz) = buzz proxy."""
import sys, glob
from pathlib import Path
import numpy as np, soundfile as sf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
A = Path(__file__).resolve().parents[1] / "samples_bigvgan/audio"


def flatness(w, sr):
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    fr = np.stack([w[i:i+n_fft]*win for i in range(0, len(w)-n_fft, hop)])
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    e = (S**2).sum(1); Sv = S[e > np.percentile(e, 40)]
    band = (freqs >= 2000) & (freqs <= 11000); Sb = Sv[:, band]
    return float(np.median(np.exp(np.log(Sb).mean(1)) / Sb.mean(1)))


idxs = sorted({Path(f).stem.split("_")[0] for f in glob.glob(str(A / "*.wav"))}, key=int)
vals = {"ref": [], "hifigan": [], "bigvgan": []}
for idx in idxs:
    for key in vals:
        f = A / f"{idx}_{key}.wav"
        if f.exists():
            w, sr = sf.read(f); w = w.mean(1) if w.ndim > 1 else w
            vals[key].append(flatness(w, sr))
print(f"{'variant':12} {'flatness':>9}  vs ref")
ref_m = np.mean(vals["ref"])
for key in ["ref", "hifigan", "bigvgan"]:
    m = np.mean(vals[key])
    print(f"{key:12} {m:9.3f}  {m-ref_m:+.3f}")
print(f"\nHiFi-GAN gap {np.mean(vals['hifigan'])-ref_m:+.3f} | BigVGAN gap {np.mean(vals['bigvgan'])-ref_m:+.3f}"
      f"  ({'BigVGAN closer to ref' if abs(np.mean(vals['bigvgan'])-ref_m) < abs(np.mean(vals['hifigan'])-ref_m) else 'HiFi-GAN closer'})")
