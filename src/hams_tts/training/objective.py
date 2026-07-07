"""The VITS training objective: MPD/MSD discriminators + KL/mel/duration/feature-matching
losses + monotonic alignment.  GPU-host code (the ``train`` extra).

The *math* here is the canonical VITS/HiFi-GAN recipe (Kong et al.; Kim et al. 2021) and
is version-independent.  The only version-sensitive part is how we call the HF VitsModel
submodules (text_encoder / posterior_encoder / flow / decoder / duration_predictor); those
calls are pinned to the ``transformers`` version on the training host and mirror what
``ylacombe/finetune-hf-vits`` does.  We keep this module self-contained and heavily
commented so the objective is auditable.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

LRELU = 0.1


# ----------------------------------------------------------------------------------
# Discriminators (HiFi-GAN MPD + MSD) — standard, version-independent
# ----------------------------------------------------------------------------------
class _PeriodDisc(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect")
        x = x.view(b, c, x.shape[-1] // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class _ScaleDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class Discriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discs = nn.ModuleList([_ScaleDisc()] + [_PeriodDisc(p) for p in periods])

    def forward(self, y, y_hat):
        outs = [(d(y), d(y_hat)) for d in self.discs]
        yd_r = [o[0][0] for o in outs]
        yd_g = [o[1][0] for o in outs]
        fmap_r = [o[0][1] for o in outs]
        fmap_g = [o[1][1] for o in outs]
        return yd_r, yd_g, fmap_r, fmap_g


# ----------------------------------------------------------------------------------
# Loss terms (canonical)
# ----------------------------------------------------------------------------------
def discriminator_loss(real, fake):
    loss = 0.0
    for r, g in zip(real, fake):
        loss += torch.mean((1 - r) ** 2) + torch.mean(g ** 2)
    return loss


def generator_loss(fake):
    loss = 0.0
    for g in fake:
        loss += torch.mean((1 - g) ** 2)
    return loss


def feature_loss(fmap_r, fmap_g):
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for r, g in zip(dr, dg):
            loss += torch.mean(torch.abs(r.detach() - g))
    return loss * 2


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """KL between posterior N(m_q, s_q) (sampled z_p) and prior N(m_p, s_p)."""
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)


def mel_loss(y, y_hat, mel_fn, c_mel):
    return c_mel * F.l1_loss(mel_fn(y.squeeze(1)), mel_fn(y_hat.squeeze(1)))


def multi_resolution_stft_loss(y, y_hat, configs):
    """Spectral-convergence + log-magnitude L1 summed over several STFT resolutions
    (Parallel WaveGAN). Supervises fine spectral detail at multiple time/freq scales the
    single mel loss misses -> reduces HiFi-GAN's broadband 'air'/breathiness. fp32 in."""
    if y.dim() == 3:
        y = y.squeeze(1)
    if y_hat.dim() == 3:
        y_hat = y_hat.squeeze(1)
    total = 0.0
    for n_fft, hop, win in configs:
        window = torch.hann_window(win, device=y.device)
        Y = torch.stft(y, n_fft, hop, win, window, center=True, return_complex=True).abs()
        Yh = torch.stft(y_hat, n_fft, hop, win, window, center=True, return_complex=True).abs()
        sc = torch.norm(Y - Yh, p="fro") / (torch.norm(Y, p="fro") + 1e-7)         # spectral convergence
        mag = F.l1_loss(torch.log(Yh + 1e-5), torch.log(Y + 1e-5))                 # log magnitude
        total = total + sc + mag
    return total / len(configs)


# ----------------------------------------------------------------------------------
# Helpers: mel, segment slicing, monotonic alignment search
# ----------------------------------------------------------------------------------
def mel_spectrogram_fn(sample_rate=16000, n_fft=1024, hop=256, win=1024, n_mels=80):
    """Return a torchaudio-free log-mel closure (torch.stft + librosa mel filterbank)."""
    import librosa

    mel_fb = torch.from_numpy(librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)).float()
    window = torch.hann_window(win)

    def _mel(y):  # y: (b, samples) float32
        spec = torch.stft(y, n_fft, hop, win, window.to(y.device), center=True,
                          return_complex=True).abs()
        m = torch.matmul(mel_fb.to(y.device), spec)
        return torch.log(torch.clamp(m, min=1e-5))

    return _mel


def rand_slice_segments(x, lengths, seg_size):
    b, _, t = x.shape
    max_start = (lengths - seg_size).clamp(min=0)
    starts = (torch.rand(b, device=x.device) * (max_start + 1)).long()
    out = torch.stack([x[i, :, s:s + seg_size] for i, s in enumerate(starts)])
    return out, starts


import numpy as np

try:
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a else a[0]


@njit(cache=True)
def _mas_each(path, value, t_x, t_y, max_neg=-1e9):
    """Canonical VITS monotonic-alignment DP for one item (in-place on `value`/`path`)."""
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg if x == y else value[x, y - 1]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] = max(v_cur, v_prev) + value[x, y]
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1


def maximum_path(neg_cent, x_lengths, y_lengths):
    """Monotonic alignment search. neg_cent: torch (b, t_text, t_spec); lengths: torch (b,).

    Returns the hard alignment path (b, t_text, t_spec). Uses the canonical VITS DP,
    numba-jitted when available (≈100× the pure-python loop) — the throughput lever for
    training. Runs on CPU numpy then moves back to the input device."""
    dev = neg_cent.device
    value = neg_cent.detach().cpu().numpy().astype(np.float32).copy()
    b, t_x, t_y = value.shape
    path = np.zeros((b, t_x, t_y), dtype=np.float32)
    xl = x_lengths.detach().cpu().numpy().astype(np.int64)
    yl = y_lengths.detach().cpu().numpy().astype(np.int64)
    for i in range(b):
        _mas_each(path[i], value[i], int(xl[i]), int(yl[i]))
    return torch.from_numpy(path).to(dev)


# ----------------------------------------------------------------------------------
# Objective wrapper
# ----------------------------------------------------------------------------------
class VitsObjective(nn.Module):
    """Wraps the model + discriminator and runs one optimisation step.

    NOTE: ``_generator_forward`` calls the HF VitsModel submodules to produce the training
    outputs (reconstructed audio slice + prior/posterior stats + predicted durations).
    The exact submodule kwargs are pinned to the transformers build on the GPU host; the
    structure mirrors ``finetune-hf-vits``. Losses/discriminator above are framework-free.
    """

    def __init__(self, model, seg_size=8192, sample_rate=22050, c_mel=45.0, c_kl=1.0,
                 c_dur=1.0, c_fm=2.0, c_stft=0.0):
        super().__init__()
        self.model = model
        self.discriminator = Discriminator()
        self.seg_size = seg_size
        self.hop = 256
        self.c_mel, self.c_kl, self.c_dur, self.c_fm = c_mel, c_kl, c_dur, c_fm
        self.c_stft = c_stft
        # multi-resolution STFT configs (n_fft, hop, win) — small/mid/large windows
        self._stft_cfgs = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]
        self._mel = mel_spectrogram_fn(sample_rate)

    def _generator_forward(self, batch):
        """Return dict with y_slice, y_hat, kl terms, dur_loss. (HF-submodule glue.)"""
        return self.model.training_forward(  # implemented against the host transformers build
            phoneme_ids=batch["phoneme_ids"], phoneme_lengths=batch["phoneme_lengths"],
            language_ids=batch["language_ids"], spec=batch["spec"],
            spec_lengths=batch["spec_lengths"], speaker_id=batch["speaker_id"],
            wav=batch["wav"], seg_size=self.seg_size, hop=self.hop, maximum_path=maximum_path,
        )

    def step(self, batch, opt_g, opt_d, step=0) -> Dict[str, float]:
        # bf16 autocast (Ampere) — more stable than fp16+GradScaler for GAN training.
        amp = torch.autocast("cuda", dtype=torch.bfloat16)

        with amp:
            out = self._generator_forward(batch)
            y, y_hat = out["y_slice"], out["y_hat"]

        # ---- discriminator ----
        with amp:
            yd_r, yd_g, _, _ = self.discriminator(y, y_hat.detach())
            loss_d = discriminator_loss(yd_r, yd_g)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
        opt_d.step()

        # ---- generator ----
        with amp:
            yd_r, yd_g, fmap_r, fmap_g = self.discriminator(y, y_hat)
            loss_adv = generator_loss(yd_g)
            loss_fm = self.c_fm * feature_loss(fmap_r, fmap_g)
            loss_kl = self.c_kl * out["kl"]
            loss_dur = self.c_dur * out["dur_loss"]
        # mel + multi-res STFT losses in fp32 (STFT imprecise/unsupported in bf16)
        loss_mel = mel_loss(y.float(), y_hat.float(), self._mel, self.c_mel)
        if self.c_stft > 0:
            loss_stft = self.c_stft * multi_resolution_stft_loss(y.float(), y_hat.float(), self._stft_cfgs)
        else:
            loss_stft = torch.zeros((), device=y.device)
        loss_g = loss_adv + loss_fm + loss_mel + loss_kl + loss_dur + loss_stft
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 10.0)
        opt_g.step()

        return {"loss_d": float(loss_d), "loss_g": float(loss_g), "mel": float(loss_mel),
                "kl": float(loss_kl), "dur": float(loss_dur), "fm": float(loss_fm),
                "adv": float(loss_adv), "stft": float(loss_stft)}
