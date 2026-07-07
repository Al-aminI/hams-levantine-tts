"""BigVGAN vocoder integration.

The VITS/HiFi-GAN decoder gets the magnitude spectrum right but leaves a phase/aliasing
'air' that magnitude losses can't fix. BigVGAN (anti-aliased, Snake activations) fixes it.
Since our magnitude already matches the reference, we re-vocode: VITS wav -> mel -> BigVGAN
-> clean wav (the mel keeps our good magnitude; BigVGAN regenerates clean phase/excitation).

Pretrained `nvidia/bigvgan_v2_24khz_100band_256x` is an exact match for our pipeline
(24 kHz, hop 256, fmax 12 kHz). use_cuda_kernel=False -> pure-torch anti-aliasing (Windows-OK).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "nvidia/bigvgan_v2_24khz_100band_256x"


def _bigvgan_repo() -> str:
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "vendor" / "BigVGAN"
        if (cand / "bigvgan.py").exists():
            return str(cand)
    raise RuntimeError("vendor/BigVGAN not found — clone https://github.com/NVIDIA/BigVGAN")


@lru_cache(maxsize=1)
def load_bigvgan(device: str = "cuda"):
    repo = _bigvgan_repo()
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import bigvgan  # from vendor/BigVGAN

    m = bigvgan.BigVGAN.from_pretrained(MODEL_ID, use_cuda_kernel=False)
    m.remove_weight_norm()
    return m.eval().to(device)


def revocode(wav, device: str = "cuda") -> np.ndarray:
    """Re-vocode a waveform through BigVGAN. wav: 1-D array/tensor at 24 kHz. Returns 1-D float32."""
    bv = load_bigvgan(device)
    if _bigvgan_repo() not in sys.path:
        sys.path.insert(0, _bigvgan_repo())
    from meldataset import get_mel_spectrogram

    w = torch.as_tensor(np.asarray(wav, dtype=np.float32)).reshape(1, -1)
    mel = get_mel_spectrogram(w, bv.h).to(device)
    with torch.inference_mode():
        out = bv(mel).squeeze().detach().cpu().numpy()
    return out.astype(np.float32)


def synthesize(model, phoneme_ids, language_ids, length_scale: float = 1.0,
               device: str = "cuda", trim: bool = True) -> np.ndarray:
    """Full path: HamsVITS acoustic -> BigVGAN vocoder -> (optional) trailing-artifact trim."""
    pid = torch.as_tensor([phoneme_ids], device=device)
    lid = torch.as_tensor([language_ids], device=device)
    with torch.no_grad():
        wav = model.infer(pid, lid, length_scale=length_scale).squeeze().cpu().numpy()
    wav = revocode(wav, device)
    if trim:
        from ..utils.trim import trim_trailing_artifact
        wav = trim_trailing_artifact(wav, model.sample_rate)
    return wav
