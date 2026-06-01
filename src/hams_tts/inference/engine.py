"""Streaming TTS engine + backend abstraction.

The engine is backend-agnostic on purpose.  The *same* high-level streaming API drives:

  * ``EspeakBackend``   – CPU, dependency-free, produces **real audio on the dev Mac**
                          and even demonstrates code-switching (per-span voices).  Used
                          for development, CI, and exercising the server/Pipecat plumbing.
  * ``OnnxBackend``     – the deployable model: exported VITS run under ONNX Runtime
                          (CUDA EP on the L4, CPU EP elsewhere).
  * ``TensorRTBackend`` – the optimised production path (FP16/INT8 TensorRT engine).
  * ``TorchBackend``    – reference HamsVITS in PyTorch (training-time / parity checks).

Because the server, the Pipecat plugin and the benchmark all talk to
``StreamingTTSEngine``, swapping backends is a one-line change and every layer above is
testable on CPU today.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from ..text.frontend import TextFrontend, Utterance
from ..utils import audio as A
from .chunker import ChunkerConfig, chunk_text


# ======================================================================================
# Backends
# ======================================================================================
class TTSBackend(ABC):
    name: str = "base"
    sample_rate: int = 24000

    @abstractmethod
    def synthesize(self, utt: Utterance, speaker_id: int = 0) -> np.ndarray:
        """Return mono float32 audio in [-1, 1] for the whole utterance."""

    def warmup(self) -> None:
        """Run a tiny synth so the first real request isn't penalised by lazy init."""
        try:
            fe = TextFrontend()
            self.synthesize(fe.process("مرحبا hello"))
        except Exception:
            pass

    def vram_bytes(self) -> Optional[int]:
        return None


class EspeakBackend(TTSBackend):
    """Real audio on any machine; renders each code-switch span with its own voice.

    NOT the production-quality model — it is the dev/CI/demo backend that lets the entire
    streaming stack run end-to-end on CPU.  The neural backends below are the real thing.
    """

    name = "espeak"
    sample_rate = 22050  # espeak-ng native
    VOICES = {"ar": "ar", "en": "en-us"}

    def __init__(self, rate_wpm: int = 165):
        import shutil

        self._bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self._bin:
            raise RuntimeError("espeak-ng is required for EspeakBackend")
        self.rate_wpm = rate_wpm

    def _render(self, text: str, voice: str) -> np.ndarray:
        import subprocess
        import tempfile

        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            subprocess.run(
                [self._bin, "-v", voice, "-s", str(self.rate_wpm), "-w", tf.name, "--", text],
                capture_output=True,
                timeout=30,
            )
            tf.seek(0)
            data = tf.read()
        if not data:
            return np.zeros(0, dtype=np.float32)
        wav, sr = A.read_wav(data)
        return A.resample(wav, sr, self.sample_rate)

    def synthesize(self, utt: Utterance, speaker_id: int = 0) -> np.ndarray:
        parts: List[np.ndarray] = []
        gap = np.zeros(int(0.02 * self.sample_rate), dtype=np.float32)  # 20 ms span gap
        spans = utt.spans or []
        if not spans:  # fall back to raw text if no spans (e.g., pure punctuation)
            return self._render(utt.normalized, "en-us")
        for i, span in enumerate(spans):
            if i:
                parts.append(gap)
            parts.append(self._render(span.text, self.VOICES.get(span.lang, "en-us")))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class OnnxBackend(TTSBackend):
    """Deployable VITS under ONNX Runtime. Expects an engine exported by
    :mod:`hams_tts.models.optimize.export_onnx` with inputs (phoneme_ids, lang_ids,
    speaker_id, scales) and a single waveform output."""

    name = "onnx"

    def __init__(self, model_path: str, sample_rate: int = 24000, providers: Optional[list] = None):
        import onnxruntime as ort  # type: ignore

        self.sample_rate = sample_rate
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self._inames = {i.name for i in self.sess.get_inputs()}

    def synthesize(self, utt: Utterance, speaker_id: int = 0) -> np.ndarray:
        feeds = {
            "phoneme_ids": np.asarray([utt.phoneme_ids], dtype=np.int64),
            "phoneme_lengths": np.asarray([len(utt.phoneme_ids)], dtype=np.int64),
            "language_ids": np.asarray([utt.language_ids], dtype=np.int64),
        }
        if "speaker_id" in self._inames:
            feeds["speaker_id"] = np.asarray([speaker_id], dtype=np.int64)
        if "scales" in self._inames:
            # (noise_scale, length_scale, noise_scale_w) — length_scale<1 => faster speech
            feeds["scales"] = np.asarray([0.667, 1.0, 0.8], dtype=np.float32)
        feeds = {k: v for k, v in feeds.items() if k in self._inames}
        out = self.sess.run(None, feeds)[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)


class TensorRTBackend(TTSBackend):
    """Production path. Thin wrapper around a serialized TensorRT engine; see
    :mod:`hams_tts.models.optimize.build_tensorrt`. Implemented on the GPU host."""

    name = "tensorrt"

    def __init__(self, engine_path: str, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.engine_path = engine_path
        self._rt = None  # lazily built on the GPU host (pycuda/tensorrt)

    def synthesize(self, utt: Utterance, speaker_id: int = 0) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            "TensorRTBackend.synthesize runs on the GPU host; see build_tensorrt.py and "
            "the trt_runner module instantiated there."
        )


class TorchBackend(TTSBackend):
    """Reference HamsVITS in PyTorch (GPU). Used for training-time inference and as the
    numerical oracle when validating the ONNX/TensorRT exports."""

    name = "torch"

    def __init__(self, checkpoint: str, device: str = "cuda", sample_rate: int = 24000):
        import torch  # type: ignore

        from ..models.hams_vits import HamsVITS

        self.device = device
        self.torch = torch
        self.model = HamsVITS.from_checkpoint(checkpoint).to(device).eval()
        # the model dictates its native rate (mms-tts is 16 kHz); the engine resamples
        self.sample_rate = self.model.sample_rate

    def synthesize(self, utt: Utterance, speaker_id: int = 0) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            ids = torch.tensor([utt.phoneme_ids], device=self.device)
            langs = torch.tensor([utt.language_ids], device=self.device)
            spk = torch.tensor([speaker_id], device=self.device)
            wav = self.model.infer(ids, langs, spk)
        return wav.squeeze().float().cpu().numpy()

    def vram_bytes(self) -> Optional[int]:
        if self.device.startswith("cuda"):
            return int(self.torch.cuda.max_memory_allocated())
        return None


# ======================================================================================
# Engine
# ======================================================================================
@dataclass
class AudioChunk:
    pcm: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int
    chunk_index: int
    is_first: bool
    is_last: bool
    text: str
    latency_s: float  # wall-clock from stream() start to this chunk being ready


@dataclass
class SynthResult:
    audio: np.ndarray
    sample_rate: int
    ttfa_s: float
    total_s: float
    audio_s: float

    @property
    def rtf(self) -> float:
        return self.total_s / max(self.audio_s, 1e-9)


class StreamingTTSEngine:
    def __init__(
        self,
        backend: TTSBackend,
        frontend: Optional[TextFrontend] = None,
        chunker: Optional[ChunkerConfig] = None,
        output_sample_rate: Optional[int] = None,
    ):
        self.backend = backend
        self.frontend = frontend or TextFrontend()
        self.chunker = chunker or ChunkerConfig()
        self.output_sample_rate = output_sample_rate or backend.sample_rate

    def warmup(self) -> None:
        self.backend.warmup()

    def stream(self, text: str, speaker_id: int = 0) -> Iterator[AudioChunk]:
        """Yield audio chunks as they are synthesised (low TTFA via short first chunk)."""
        t0 = time.perf_counter()
        chunks = chunk_text(text, self.chunker)
        n = len(chunks)
        for i, ctext in enumerate(chunks):
            utt = self.frontend.process(ctext)
            audio = self.backend.synthesize(utt, speaker_id=speaker_id)
            if self.output_sample_rate != self.backend.sample_rate:
                audio = A.resample(audio, self.backend.sample_rate, self.output_sample_rate)
            yield AudioChunk(
                pcm=audio,
                sample_rate=self.output_sample_rate,
                chunk_index=i,
                is_first=(i == 0),
                is_last=(i == n - 1),
                text=ctext,
                latency_s=time.perf_counter() - t0,
            )

    def synthesize(self, text: str, speaker_id: int = 0) -> SynthResult:
        """Non-streaming convenience that still reports TTFA/RTF from the stream."""
        t0 = time.perf_counter()
        ttfa: Optional[float] = None
        parts: List[np.ndarray] = []
        for ch in self.stream(text, speaker_id=speaker_id):
            if ttfa is None:
                ttfa = ch.latency_s
            parts.append(ch.pcm)
        total = time.perf_counter() - t0
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        audio_s = audio.shape[0] / self.output_sample_rate if audio.size else 0.0
        return SynthResult(audio, self.output_sample_rate, ttfa or total, total, audio_s)


def build_engine(backend: str = "espeak", output_sample_rate: Optional[int] = None, **kw) -> StreamingTTSEngine:
    """Factory: ``backend`` in {espeak, onnx, tensorrt, torch}."""
    if backend == "espeak":
        be: TTSBackend = EspeakBackend()
    elif backend == "onnx":
        be = OnnxBackend(kw["model_path"], sample_rate=kw.get("sample_rate", 24000),
                         providers=kw.get("providers"))
    elif backend == "tensorrt":
        be = TensorRTBackend(kw["engine_path"], sample_rate=kw.get("sample_rate", 24000))
    elif backend == "torch":
        be = TorchBackend(kw["checkpoint"], device=kw.get("device", "cuda"),
                          sample_rate=kw.get("sample_rate", 24000))
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return StreamingTTSEngine(be, output_sample_rate=output_sample_rate)


if __name__ == "__main__":
    eng = build_engine("espeak")
    eng.warmup()
    res = eng.synthesize("مرحبا، بدي أحجز flight to London بكرا. Thank you!")
    print(f"backend={eng.backend.name} sr={res.sample_rate} ttfa={res.ttfa_s*1000:.0f}ms "
          f"rtf={res.rtf:.3f} audio={res.audio_s:.2f}s")
    A.save_wav("/tmp/hams_engine_demo.wav", res.audio, res.sample_rate)
    print("wrote /tmp/hams_engine_demo.wav")
