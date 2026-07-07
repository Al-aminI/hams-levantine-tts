"""Trailing-artifact trim for TTS output.

The VITS decoder emits a short broadband click for the end-of-sequence token (the
duration predictor allocates it ~0.1 s because training clips ended in silence/breath).
That transient — sometimes separated from the speech by a silence gap — is the audible
"robotic break" at the end of an utterance. This removes it by scanning back from the
end and cutting after the last *sustained* speech run (skipping short isolated bursts and
trailing silence), with a natural release pad and a short fade to avoid a new click.

Pure-numpy, no torch — safe to call in any output path (inference, eval, server).
"""
from __future__ import annotations

import numpy as np


def trim_trailing_artifact(
    wav: np.ndarray,
    sr: int,
    thr_rel: float = 0.06,   # active threshold as a fraction of peak envelope
    min_run_ms: int = 45,    # a run this long counts as real speech (EOS burst is shorter)
    pad_ms: int = 40,        # natural release kept after the last speech run
    fade_ms: int = 12,       # fade-out to avoid introducing a click at the new end
) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    hop = max(1, int(sr * 0.01))  # 10 ms frames
    n = len(wav) // hop
    if n < 3:
        return wav
    env = np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-9)
    thr = max(env.max() * thr_rel, 1e-4)
    active = env > thr
    min_run = max(1, min_run_ms // 10)

    # scan backward for the end of the last active run of length >= min_run
    end_frame = None
    i = n - 1
    while i >= 0:
        if active[i]:
            j = i
            while j >= 0 and active[j]:
                j -= 1
            if (i - j) >= min_run:      # real speech
                end_frame = i + 1
                break
            i = j                        # short isolated burst -> skip past it
        else:
            i -= 1
    if end_frame is None:
        return wav

    cut = min(len(wav), end_frame * hop + int(pad_ms / 1000 * sr))
    out = wav[:cut].astype(np.float32).copy()
    f = int(fade_ms / 1000 * sr)
    if 0 < f < len(out):
        out[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return out
