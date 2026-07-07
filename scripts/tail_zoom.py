"""Zoom the last 0.8s of each synth clip (spectrogram + fine RMS envelope) to pin down
the EOS artifact and where to trim."""
import os, sys, glob
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np, soundfile as sf, matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
probe = REPO / "eval_out/tail_probe"
clips = sorted(glob.glob(str(probe / "clip*_synth.wav")))

fig, axes = plt.subplots(len(clips), 1, figsize=(12, 3 * len(clips)))
for i, c in enumerate(clips):
    w, sr = sf.read(c)
    zoom_s = 0.8
    z = w[-int(zoom_s * sr):]
    t0 = len(w) / sr - zoom_s
    ax = axes[i]
    ax.specgram(z, NFFT=512, Fs=sr, noverlap=448, cmap="magma")
    ax.set_title(f"{Path(c).name} — last {zoom_s}s (file {len(w)/sr:.2f}s)")
    ax.set_ylim(0, sr / 2)
    # fine envelope (10ms) over the zoom, printed
    win = int(sr * 0.01)
    env = np.array([np.sqrt((z[k*win:(k+1)*win]**2).mean() + 1e-12) for k in range(len(z)//win)])
    peak = np.sqrt((w**2).mean())
    # find last speech frame (>10% of overall rms) then describe what's after
    thr = 0.10 * w.__abs__().max()
    line = "".join("#" if e > thr else ("." if e > thr*0.25 else " ") for e in env)
    print(f"{Path(c).name}: 10ms env over last {zoom_s}s ( # loud  . faint  ' ' silent ), t0={t0:.2f}s")
    print(f"  |{line}|")
fig.tight_layout()
fig.savefig(str(probe / "tail_zoom.png"), dpi=95)
print("wrote", probe / "tail_zoom.png")
