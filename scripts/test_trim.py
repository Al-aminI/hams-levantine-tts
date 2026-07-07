"""Prototype + validate the trailing-artifact trim on the probe clips."""
import sys, glob
from pathlib import Path
import numpy as np, soundfile as sf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hams_tts.utils.trim import trim_trailing_artifact  # the new util

probe = REPO / "eval_out/tail_probe"
for c in sorted(glob.glob(str(probe / "clip*_synth.wav"))):
    w, sr = sf.read(c)
    t = trim_trailing_artifact(w, sr)
    sf.write(c.replace("_synth.wav", "_trimmed.wav"), t, sr)
    # tail envelope of trimmed, last 0.4s, to confirm no isolated burst remains
    z = t[-int(0.4*sr):]
    win = int(sr*0.01)
    env = np.array([np.sqrt((z[k*win:(k+1)*win]**2).mean()+1e-9) for k in range(len(z)//win)])
    thr = 0.10*np.abs(t).max()
    line = "".join("#" if e>thr else ("." if e>thr*0.25 else " ") for e in env)
    print(f"{Path(c).name}: {len(w)/sr:.2f}s -> {len(t)/sr:.2f}s (cut {(len(w)-len(t))/sr*1000:.0f}ms) "
          f"| trimmed tail |{line}|")
