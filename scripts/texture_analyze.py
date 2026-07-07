"""Objective texture metrics: compare REFERENCE vs our model across noise_scale.
Buzz/roughness proxies: high-band (>3 kHz) energy ratio, and spectral flatness in
2-11 kHz over voiced frames (noise-like=high -> buzzy; tonal=low -> clean)."""
import sys, glob
from pathlib import Path
import numpy as np, soundfile as sf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
A = REPO / "samples_texture/audio"


def metrics(w, sr):
    n_fft, hop = 1024, 256
    if len(w) < n_fft:
        return None
    win = np.hanning(n_fft)
    frames = [w[i:i+n_fft]*win for i in range(0, len(w)-n_fft, hop)]
    S = np.abs(np.fft.rfft(np.stack(frames), axis=1)) + 1e-9   # (T, F)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    energy = (S**2).sum(1)
    voiced = energy > np.percentile(energy, 40)                # drop silence frames
    Sv = S[voiced]
    hi = (freqs >= 3000)
    hf_ratio = (Sv[:, hi]**2).sum() / (Sv**2).sum()
    band = (freqs >= 2000) & (freqs <= 11000)
    Sb = Sv[:, band]
    flat = np.exp(np.log(Sb).mean(1)) / Sb.mean(1)             # spectral flatness per frame
    return hf_ratio, float(np.median(flat))


# group files by clip index
clips = {}
for f in sorted(glob.glob(str(A / "*.wav"))):
    idx = Path(f).stem.split("_")[0]
    clips.setdefault(idx, []).append(f)

print(f"{'variant':28} {'HF>3k ratio':>12} {'flatness(2-11k)':>16}")
for idx, files in clips.items():
    print(f"--- clip {idx} ---")
    # order: ref first, then ns descending
    def key(f):
        s = Path(f).stem
        return (0, 0) if s.endswith("ref") else (1, -float(s.split("ns")[1]))
    for f in sorted(files, key=key):
        w, sr = sf.read(f)
        if w.ndim > 1: w = w.mean(1)
        m = metrics(w, sr)
        if m:
            tag = Path(f).stem.split("_", 1)[1]
            print(f"  {tag:26} {m[0]*100:11.1f}% {m[1]:16.3f}")
