"""Wire protocol for the streaming TTS server (REST + WebSocket).

Kept deliberately small and explicit so it is easy to validate and easy to integrate.
All inbound text is validated (length, type, format enum) before it ever reaches the
engine — basic abuse/DoS hygiene for a production endpoint.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 2000


class AudioFormat(str, Enum):
    pcm16 = "pcm16"  # raw little-endian int16 mono
    wav = "wav"      # WAV container (REST convenience)
    opus = "opus"    # Opus packets (falls back to pcm16 if opuslib absent)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    speaker_id: int = Field(0, ge=0, le=4096)
    sample_rate: int = Field(24000, ge=8000, le=48000)
    format: AudioFormat = AudioFormat.wav
    length_scale: float = Field(1.0, gt=0.3, le=3.0)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class WSClientMessage(BaseModel):
    type: Literal["speak", "flush", "close"] = "speak"
    text: Optional[str] = Field(None, max_length=MAX_TEXT_CHARS)
    speaker_id: int = Field(0, ge=0, le=4096)
    sample_rate: int = Field(24000, ge=8000, le=48000)
    format: AudioFormat = AudioFormat.pcm16
    length_scale: float = Field(1.0, gt=0.3, le=3.0)


class WSStart(BaseModel):
    type: Literal["start"] = "start"
    sample_rate: int
    format: AudioFormat
    channels: int = 1


class WSChunkHeader(BaseModel):
    type: Literal["chunk"] = "chunk"
    index: int
    bytes: int
    latency_ms: float
    is_first: bool
    is_last: bool
    text: str


class WSEnd(BaseModel):
    type: Literal["end"] = "end"
    ttfa_ms: float
    rtf: float
    audio_seconds: float
    total_ms: float


class WSError(BaseModel):
    type: Literal["error"] = "error"
    message: str
