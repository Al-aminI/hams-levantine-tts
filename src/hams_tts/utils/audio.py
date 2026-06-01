"""Audio helpers: float32<->int16, WAV I/O (stdlib only), light resampling, PCM/Opus.

Kept dependency-light (numpy + stdlib ``wave``) so the dev box and the server agree.
Opus encoding is optional (used by the server when the client asks for it); if the
``opuslib`` wheel is absent we transparently fall back to PCM and say so.
"""

from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np


def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2")


def int16_to_float(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Read a (mono/stereo) PCM WAV from bytes -> (float32 mono, sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return arr, sr


def write_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(float_to_int16(audio).tobytes())
    return buf.getvalue()


def save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    with open(path, "wb") as f:
        f.write(write_wav(audio, sample_rate))


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap high-quality-enough linear resampler (demo/telephony rates).

    For production-grade resampling prefer ``soxr``/``librosa``; this keeps the core
    dependency-free and is adequate at the 16/22.05/24 kHz rates we use.
    """
    if src_sr == dst_sr or audio.size == 0:
        return audio
    duration = audio.shape[0] / src_sr
    n_dst = int(round(duration * dst_sr))
    if n_dst <= 1:
        return audio
    x_src = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_dst = np.linspace(0.0, duration, num=n_dst, endpoint=False)
    return np.interp(x_dst, x_src, audio).astype(np.float32)


def to_pcm16(audio: np.ndarray) -> bytes:
    return float_to_int16(audio).tobytes()


class OpusEncoder:
    """Optional Opus encoder (20 ms frames). Falls back to raw PCM if opuslib absent."""

    def __init__(self, sample_rate: int = 24000, channels: int = 1, bitrate: int = 24000):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = 20
        self.frame_size = sample_rate * self.frame_ms // 1000
        self._enc = None
        try:  # pragma: no cover - optional dependency
            import opuslib  # type: ignore

            self._enc = opuslib.Encoder(sample_rate, channels, "voip")
            self._enc.bitrate = bitrate
            self.available = True
        except Exception:
            self.available = False

    def encode(self, audio: np.ndarray) -> list[bytes]:
        if not self.available:
            # fall back: hand back one PCM "frame"
            return [to_pcm16(audio)]
        pcm = float_to_int16(audio)
        out: list[bytes] = []
        for i in range(0, len(pcm) - self.frame_size + 1, self.frame_size):
            frame = pcm[i : i + self.frame_size].tobytes()
            out.append(self._enc.encode(frame, self.frame_size))
        return out
