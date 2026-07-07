"""Diagnose the end-of-utterance 'robotic break' artifact on the 24 kHz model.

For a few eval sentences it (1) prints the predicted per-token durations for the TAIL
tokens (to catch a mis-sized final phoneme / punctuation / EOS), (2) measures the
trailing energy envelope (where speech ends vs where the file ends, and whether the tail
sustains then cuts = robotic, or decays = natural), and (3) renders spectrograms
(synth vs reference) to a PNG for visual inspection.
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import transformers
transformers.logging.set_verbosity_error()
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.text.phoneme_inventory import ID_TO_SYMBOL

CKPT = REPO / "checkpoints/flagship_BEST_v3_24k"
MANIFEST = REPO / "data/manifests_24k/eval.phon.filtered.jsonl"
OUT = REPO / "eval_out/tail_probe"
OUT.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
picks = [rows[i] for i in (0, 1, 5)]  # a few pure-levantine

model = HamsVITS.from_checkpoint(str(CKPT)).cuda().eval()
sr = model.sample_rate
bb = model.backbone
print(f"model @ {sr} Hz | deterministic_duration={model.config.deterministic_duration}")


def token_durations(phoneme_ids, language_ids):
    """Replicate VitsModel's deterministic duration path to get per-token frame counts."""
    pid = torch.tensor([phoneme_ids], device="cuda")
    lid = torch.tensor([language_ids], device="cuda")
    model._set_lang(lid)
    mask = torch.ones_like(pid).unsqueeze(-1).float()
    enc = bb.text_encoder(input_ids=pid, padding_mask=mask, attention_mask=torch.ones_like(pid))
    hidden = enc.last_hidden_state.transpose(1, 2)
    pmask = mask.transpose(1, 2)
    log_dur = bb.duration_predictor(hidden, pmask, None)
    dur = torch.ceil(torch.exp(log_dur) * pmask).squeeze().detach().cpu().numpy()
    return dur  # frames per token (hop=256)


def envelope(w, sr, win_ms=20):
    win = int(sr * win_ms / 1000)
    n = len(w) // win
    return np.array([np.sqrt((w[i*win:(i+1)*win]**2).mean() + 1e-12) for i in range(n)]), win_ms


fig, axes = plt.subplots(len(picks), 2, figsize=(14, 3 * len(picks)))
for r_i, r in enumerate(picks):
    dur = token_durations(r["phoneme_ids"], r["language_ids"])
    syms = [ID_TO_SYMBOL.get(i, "?") for i in r["phoneme_ids"]]
    hop = 256
    print(f"\n=== clip {r_i}: {r['text'][:55]}")
    print("  TAIL tokens (symbol : frames : seconds):")
    for s, d in list(zip(syms, dur))[-8:]:
        print(f"    {s!r:8} {int(d):4d} fr  {d*hop/sr:.3f}s")

    # synth
    pid = torch.tensor([r["phoneme_ids"]], device="cuda")
    lid = torch.tensor([r["language_ids"]], device="cuda")
    with torch.no_grad():
        wav = model.infer(pid, lid, length_scale=1.0).squeeze().cpu().numpy()
    sf.write(str(OUT / f"clip{r_i}_synth.wav"), wav, sr)
    ref, rsr = sf.read(r["audio"])
    if ref.ndim > 1: ref = ref.mean(1)

    env, wm = envelope(wav, sr)
    # speech end = last frame above 8% of peak
    thr = 0.08 * env.max()
    above = np.where(env > thr)[0]
    speech_end = (above[-1] + 1) * wm / 1000 if len(above) else 0
    total = len(wav) / sr
    tail = total - speech_end
    # is the tail sustained-then-cut (robotic) or decaying? ratio of last-frame energy to peak-tail
    tail_frames = env[above[-1]:] if len(above) else env
    print(f"  synth {total:.2f}s | speech ends ~{speech_end:.2f}s | trailing {tail:.2f}s "
          f"| last-frame rms {env[-1]:.4f} (peak {env.max():.3f})")

    # spectrograms
    for c, (sig, ssr, ttl) in enumerate([(wav, sr, "SYNTH"), (ref, rsr, "REF")]):
        ax = axes[r_i, c]
        ax.specgram(sig, NFFT=1024, Fs=ssr, noverlap=768, cmap="magma")
        ax.set_title(f"clip{r_i} {ttl} ({len(sig)/ssr:.2f}s)")
        ax.set_ylim(0, ssr/2)
        if c == 0 and speech_end:
            ax.axvline(speech_end, color="cyan", lw=1, ls="--")
plt.tight_layout()
fig.savefig(str(OUT / "tail_spectrograms.png"), dpi=90)
print(f"\nwrote {OUT/'tail_spectrograms.png'}")
