"""Pipecat TTS service for the Hams Levantine/English code-switching model.

Follows the standard Pipecat plugin contract: subclass ``TTSService``, implement
``run_tts`` as an async generator that emits ``TTSStartedFrame`` →
``TTSAudioRawFrame`` (chunked) → ``TTSStoppedFrame``, and drive TTFB/usage metrics.

Design choice — **in-process** synthesis (not a network call to our own server): for a
voice agent the TTS model runs on the same GPU as the rest of the pipeline, so calling
the model directly removes a network hop and shaves latency off TTFA.  (A WebSocket
variant that talks to the FastAPI server is trivial to add if you deploy TTS as a
separate microservice — see ``HamsWebSocketTTSService`` at the bottom.)

The blocking synthesis runs in a worker thread (``asyncio.to_thread``) per text chunk,
so the event loop stays responsive and the first short chunk is emitted fast.

Usage in a Pipecat pipeline::

    from hams_tts.pipecat_plugin import HamsTTSService
    tts = HamsTTSService(backend="onnx", model_path="hams_vits.onnx", sample_rate=24000)
    pipeline = Pipeline([... , llm, tts, transport.output()])

Requires the ``pipecat`` extra:  pip install -e '.[pipecat]'
"""

import asyncio
from typing import AsyncGenerator, Optional

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from ..inference.engine import StreamingTTSEngine, build_engine
from ..inference.chunker import ChunkerConfig, chunk_text
from ..utils import audio as A


class HamsTTSService(TTSService):
    """Streaming Levantine/English code-switching TTS as a Pipecat service."""

    def __init__(
        self,
        *,
        backend: str = "espeak",
        model_path: Optional[str] = None,
        sample_rate: int = 24000,
        speaker_id: int = 0,
        **kwargs,
    ):
        # Pipecat manages output sample rate; we synthesise/resample to match it.
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._speaker_id = speaker_id
        self._chunker = ChunkerConfig()
        kw = {}
        if backend in ("onnx", "tensorrt", "torch"):
            key = {"onnx": "model_path", "tensorrt": "engine_path", "torch": "checkpoint"}[backend]
            kw[key] = model_path
        self._engine: StreamingTTSEngine = build_engine(
            backend, output_sample_rate=sample_rate, **kw
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame) -> None:
        await super().start(frame)
        # warm the model so the first real utterance isn't penalised by lazy init
        await asyncio.to_thread(self._engine.warmup)

    def _synth_chunk_pcm(self, text: str) -> bytes:
        """Synthesise one text chunk → int16 PCM bytes at the service sample rate."""
        utt = self._engine.frontend.process(text)
        audio = self._engine.backend.synthesize(utt, speaker_id=self._speaker_id)
        if self._engine.backend.sample_rate != self.sample_rate:
            audio = A.resample(audio, self._engine.backend.sample_rate, self.sample_rate)
        return A.to_pcm16(audio)

    async def run_tts(self, text: str, context_id: Optional[str] = None) -> AsyncGenerator[Frame, None]:
        """Emit audio frames for ``text``.

        ``context_id`` is accepted (current Pipecat passes it) but optional, so this same
        plugin works against older Pipecat releases whose ``run_tts`` took only ``text``.
        """
        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            chunks = chunk_text(text, self._chunker)
            first = True
            for ctext in chunks:
                pcm = await asyncio.to_thread(self._synth_chunk_pcm, ctext)
                if first:
                    await self.stop_ttfb_metrics()  # first audio ready -> TTFB done
                    first = False
                if pcm:
                    yield TTSAudioRawFrame(
                        audio=pcm,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
        except Exception as e:  # surface errors as frames; never crash the pipeline
            yield ErrorFrame(f"HamsTTSService synthesis error: {e}")
        finally:
            yield TTSStoppedFrame()


class HamsWebSocketTTSService(TTSService):
    """Variant that streams from the FastAPI server over WebSocket (TTS-as-a-service).

    Use this when TTS is deployed separately from the agent.  Connects per utterance,
    streams PCM chunks back as ``TTSAudioRawFrame``.
    """

    def __init__(self, *, url: str = "ws://localhost:8000/tts/stream", sample_rate: int = 24000,
                 speaker_id: int = 0, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url
        self._speaker_id = speaker_id

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: Optional[str] = None) -> AsyncGenerator[Frame, None]:
        import json

        import websockets

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()
            async with websockets.connect(self._url, max_size=None) as ws:
                await ws.send(json.dumps({
                    "type": "speak", "text": text, "format": "pcm16",
                    "sample_rate": self.sample_rate, "speaker_id": self._speaker_id,
                }))
                first = True
                async for msg in ws:
                    if isinstance(msg, bytes):
                        if first:
                            await self.stop_ttfb_metrics()
                            first = False
                        yield TTSAudioRawFrame(audio=msg, sample_rate=self.sample_rate, num_channels=1)
                    else:
                        if json.loads(msg).get("type") == "end":
                            break
        except Exception as e:
            yield ErrorFrame(f"HamsWebSocketTTSService error: {e}")
        finally:
            yield TTSStoppedFrame()
