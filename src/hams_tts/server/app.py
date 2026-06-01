"""FastAPI streaming TTS server.

Endpoints
---------
* ``GET  /healthz``          liveness + backend/info
* ``POST /tts``              one-shot synth → WAV/PCM/Opus (with X-TTFA-ms / X-RTF headers)
* ``WS   /tts/stream``       low-latency streaming: send text, receive audio chunks as
                             they are synthesised (first short chunk → low TTFA)

The server holds one warm :class:`StreamingTTSEngine`.  Backend is chosen by the
``HAMS_BACKEND`` env var (``espeak`` for the CPU dev box; ``onnx``/``tensorrt`` on L4 with
``HAMS_MODEL_PATH``).  Synthesis runs in a worker thread so the event loop stays free for
concurrent streams.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

# NOTE: imported at module scope (not inside create_app) on purpose. With
# `from __future__ import annotations`, FastAPI resolves endpoint annotations like
# `ws: WebSocket` via get_type_hints against the *module* globals — a local import would
# leave WebSocket unresolved and FastAPI would reject every WS upgrade with HTTP 403.
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect

from ..inference.engine import StreamingTTSEngine, build_engine
from ..utils import audio as A
from .protocol import (
    AudioFormat,
    TTSRequest,
    WSChunkHeader,
    WSClientMessage,
    WSEnd,
    WSError,
    WSStart,
)

_ENGINE: Optional[StreamingTTSEngine] = None


def get_engine() -> StreamingTTSEngine:
    global _ENGINE
    if _ENGINE is None:
        backend = os.environ.get("HAMS_BACKEND", "espeak")
        kw = {}
        if backend == "onnx":
            kw["model_path"] = os.environ["HAMS_MODEL_PATH"]
        elif backend == "tensorrt":
            kw["engine_path"] = os.environ["HAMS_MODEL_PATH"]
        elif backend == "torch":
            kw["checkpoint"] = os.environ["HAMS_MODEL_PATH"]
        _ENGINE = build_engine(backend, **kw)
        _ENGINE.warmup()
    return _ENGINE


def _encode(audio, sample_rate: int, fmt: AudioFormat) -> bytes:
    if fmt == AudioFormat.wav:
        return A.write_wav(audio, sample_rate)
    if fmt == AudioFormat.opus:
        enc = A.OpusEncoder(sample_rate)
        return b"".join(enc.encode(audio))
    return A.to_pcm16(audio)  # pcm16


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        get_engine()  # warm on startup
        yield

    app = FastAPI(title="Hams Levantine TTS", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        eng = get_engine()
        return {
            "status": "ok",
            "backend": eng.backend.name,
            "backend_sample_rate": eng.backend.sample_rate,
            "output_sample_rate": eng.output_sample_rate,
        }

    @app.post("/tts")
    async def tts(req: TTSRequest):
        eng = get_engine()
        try:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            res = await loop.run_in_executor(
                None, lambda: eng.synthesize(req.text, speaker_id=req.speaker_id)
            )
            audio = A.resample(res.audio, res.sample_rate, req.sample_rate)
            body = _encode(audio, req.sample_rate, req.format)
            media = {
                AudioFormat.wav: "audio/wav",
                AudioFormat.pcm16: "application/octet-stream",
                AudioFormat.opus: "audio/opus",
            }[req.format]
            return Response(
                content=body,
                media_type=media,
                headers={
                    "X-TTFA-ms": f"{res.ttfa_s * 1000:.1f}",
                    "X-RTF": f"{res.rtf:.4f}",
                    "X-Audio-Seconds": f"{res.audio_s:.3f}",
                    "X-Synthesis-ms": f"{(time.perf_counter() - t0) * 1000:.1f}",
                },
            )
        except Exception as e:  # robust error surface
            raise HTTPException(status_code=400, detail=f"synthesis failed: {e}")

    @app.websocket("/tts/stream")
    async def tts_stream(ws: WebSocket):
        await ws.accept()
        eng = get_engine()
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    raw = await ws.receive_json()
                except WebSocketDisconnect:
                    break
                try:
                    msg = WSClientMessage(**raw)
                except Exception as e:
                    await ws.send_json(WSError(message=f"bad message: {e}").model_dump())
                    continue
                if msg.type == "close":
                    break
                if msg.type != "speak" or not msg.text:
                    continue

                await ws.send_json(
                    WSStart(sample_rate=msg.sample_rate, format=msg.format).model_dump()
                )
                t0 = time.perf_counter()
                ttfa: Optional[float] = None
                audio_seconds = 0.0

                # synthesise chunks in a thread, hand them to the socket as they arrive
                def _produce():
                    return list(eng.stream(msg.text, speaker_id=msg.speaker_id))

                chunks = await loop.run_in_executor(None, _produce)
                for ch in chunks:
                    audio = A.resample(ch.pcm, ch.sample_rate, msg.sample_rate)
                    payload = _encode(audio, msg.sample_rate, msg.format)
                    if ttfa is None:
                        ttfa = ch.latency_s
                    audio_seconds += audio.shape[0] / msg.sample_rate
                    await ws.send_json(
                        WSChunkHeader(
                            index=ch.chunk_index,
                            bytes=len(payload),
                            latency_ms=ch.latency_s * 1000,
                            is_first=ch.is_first,
                            is_last=ch.is_last,
                            text=ch.text,
                        ).model_dump()
                    )
                    await ws.send_bytes(payload)

                total = time.perf_counter() - t0
                await ws.send_json(
                    WSEnd(
                        ttfa_ms=(ttfa or total) * 1000,
                        rtf=total / max(audio_seconds, 1e-9),
                        audio_seconds=audio_seconds,
                        total_ms=total * 1000,
                    ).model_dump()
                )
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await ws.send_json(WSError(message=str(e)).model_dump())
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    return app


app = None  # lazily created so importing this module doesn't require fastapi


def main() -> None:
    import uvicorn

    global app
    app = create_app()
    host = os.environ.get("HAMS_HOST", "0.0.0.0")
    port = int(os.environ.get("HAMS_PORT", "8000"))
    # Force an explicit WebSocket implementation. uvicorn's ws="auto" can resolve to a
    # disabled WS protocol inside an embedded `uvicorn.run(app_obj)` process (rejecting
    # upgrades with HTTP 403 before the app is ever called); "wsproto" is version-robust.
    ws_impl = os.environ.get("HAMS_WS_IMPL", "wsproto")
    uvicorn.run(app, host=host, port=port, log_level="info", ws=ws_impl)


if __name__ == "__main__":
    main()
