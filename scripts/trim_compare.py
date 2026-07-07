"""Before/after tail spectrograms: synth (with EOS click) vs trimmed."""
import os, sys, glob
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import soundfile as sf, matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[1]
probe = REPO / "eval_out/tail_probe"
clips = sorted(glob.glob(str(probe / "clip*_synth.wav")))
fig, axes = plt.subplots(len(clips), 2, figsize=(13, 3*len(clips)))
for i, c in enumerate(clips):
    for col, tag in enumerate(["synth", "trimmed"]):
        w, sr = sf.read(c.replace("_synth.wav", f"_{tag}.wav"))
        z = w[-int(0.7*sr):]
        ax = axes[i, col]
        ax.specgram(z, NFFT=512, Fs=sr, noverlap=448, cmap="magma")
        ax.set_title(f"clip{i} {tag} (last 0.7s of {len(w)/sr:.2f}s)")
        ax.set_ylim(0, sr/2)
fig.tight_layout(); fig.savefig(str(probe/"trim_compare.png"), dpi=95)
print("wrote", probe/"trim_compare.png")
