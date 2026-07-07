"""Quick texture metric for a checkpoint vs the reference: spectral flatness (2-11 kHz,
voiced frames) — lower/closer-to-ref = less buzz. Run on a training checkpoint mid-run."""
import sys, json
from pathlib import Path
import numpy as np, torch, soundfile as sf
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import transformers; transformers.logging.set_verbosity_error()
from hams_tts.models.hams_vits import HamsVITS

ckpt = sys.argv[1]
picks = [0, 1, 5, 10]
rows = [json.loads(l) for l in open(REPO / "data/manifests_24k/eval.phon.filtered.jsonl", encoding="utf-8")]
model = HamsVITS.from_checkpoint(ckpt).cuda().eval()
sr = model.sample_rate


def flatness(w, sr):
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    fr = np.stack([w[i:i+n_fft]*win for i in range(0, len(w)-n_fft, hop)])
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    e = (S**2).sum(1); Sv = S[e > np.percentile(e, 40)]
    band = (freqs >= 2000) & (freqs <= 11000)
    Sb = Sv[:, band]
    return float(np.median(np.exp(np.log(Sb).mean(1)) / Sb.mean(1)))


print(f"ckpt: {ckpt}")
print(f"{'clip':6} {'REF':>7} {'ours':>7}  (lower ours, closer to REF = less buzz)")
dref, dours = [], []
for idx in picks:
    r = rows[idx]
    ref, rsr = sf.read(r["audio"]);  ref = ref.mean(1) if ref.ndim > 1 else ref
    pid = torch.tensor([r["phoneme_ids"]], device="cuda"); lid = torch.tensor([r["language_ids"]], device="cuda")
    with torch.no_grad():
        w = model.infer(pid, lid, length_scale=1.0).squeeze().cpu().numpy()
    fr, fo = flatness(ref, rsr), flatness(w, sr)
    dref.append(fr); dours.append(fo)
    print(f"{idx:6} {fr:7.3f} {fo:7.3f}   {'BUZZIER' if fo > fr+0.01 else 'ok/clean'}")
print(f"{'MEAN':6} {np.mean(dref):7.3f} {np.mean(dours):7.3f}  gap {np.mean(dours)-np.mean(dref):+.3f}")
