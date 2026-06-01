"""Torch Dataset for VITS fine-tuning (GPU/train extra). Imported lazily by data.py."""

from __future__ import annotations

import json
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


def _linear_spectrogram(wav: torch.Tensor, n_fft=1024, hop=256, win=1024) -> torch.Tensor:
    window = torch.hann_window(win, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                      center=True, return_complex=True)
    return torch.abs(spec) + 1e-9  # (freq, frames)


class VitsCodeSwitchDataset(Dataset):
    """Yields (phoneme_ids, language_ids, spec, wav, speaker_id) from a precomputed manifest.

    The manifest rows already carry ``phoneme_ids`` + ``language_ids`` from the front-end
    (see :func:`hams_tts.training.data.precompute_phonemes`), so the only per-item work is
    loading audio and computing the linear spectrogram VITS' posterior encoder consumes.
    """

    def __init__(self, manifest_jsonl: str, sample_rate: int = 22050, speaker_map: dict | None = None,
                 n_fft: int = 1024, hop: int = 256, win: int = 1024):
        self.rows = [json.loads(l) for l in open(manifest_jsonl, encoding="utf-8")]
        self.sample_rate = sample_rate
        self.n_fft, self.hop, self.win = n_fft, hop, win
        spk = speaker_map or {}
        for r in self.rows:
            spk.setdefault(r.get("speaker", "spk0"), len(spk))
        self.speaker_map = spk

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import soundfile as sf

        r = self.rows[i]
        wav_np, sr = sf.read(r["audio"], dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        wav = torch.from_numpy(wav_np)
        if sr != self.sample_rate:  # resample if needed (librosa on the train host)
            import librosa

            wav = torch.from_numpy(librosa.resample(wav_np, orig_sr=sr, target_sr=self.sample_rate))
        spec = _linear_spectrogram(wav, self.n_fft, self.hop, self.win)
        return {
            "phoneme_ids": torch.tensor(r["phoneme_ids"], dtype=torch.long),
            "language_ids": torch.tensor(r["language_ids"], dtype=torch.long),
            "spec": spec,
            "wav": wav,
            "speaker_id": torch.tensor(self.speaker_map[r.get("speaker", "spk0")], dtype=torch.long),
        }


def collate(batch: List[dict]) -> dict:
    """Pad to the longest item in the batch; return lengths for masking."""
    def pad_1d(xs, pad=0):
        m = max(x.shape[0] for x in xs)
        return torch.stack([torch.nn.functional.pad(x, (0, m - x.shape[0]), value=pad) for x in xs])

    def pad_2d(xs):  # (freq, frames)
        m = max(x.shape[1] for x in xs)
        return torch.stack([torch.nn.functional.pad(x, (0, m - x.shape[1])) for x in xs])

    return {
        "phoneme_ids": pad_1d([b["phoneme_ids"] for b in batch]),
        "phoneme_lengths": torch.tensor([b["phoneme_ids"].shape[0] for b in batch]),
        "language_ids": pad_1d([b["language_ids"] for b in batch]),
        "spec": pad_2d([b["spec"] for b in batch]),
        "spec_lengths": torch.tensor([b["spec"].shape[1] for b in batch]),
        "wav": pad_1d([b["wav"] for b in batch]),
        "wav_lengths": torch.tensor([b["wav"].shape[0] for b in batch]),
        "speaker_id": torch.stack([b["speaker_id"] for b in batch]),
    }
