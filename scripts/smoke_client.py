"""Smoke test for the streaming server: waits for readiness, then hits REST + WS."""

import asyncio
import sys
import time

import httpx
import websockets

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
WS = BASE.replace("http", "ws") + "/tts/stream"


async def wait_ready(timeout=30):
    async with httpx.AsyncClient() as c:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = await c.get(BASE + "/healthz", timeout=2)
                if r.status_code == 200:
                    print("healthz:", r.json())
                    return True
            except Exception:
                await asyncio.sleep(0.4)
    return False


async def test_rest():
    async with httpx.AsyncClient() as c:
        r = await c.post(BASE + "/tts", json={
            "text": "مرحبا، بدي أحجز flight to London بكرا. Thank you!",
            "format": "wav", "sample_rate": 24000,
        }, timeout=30)
        print(f"REST /tts -> {r.status_code}  bytes={len(r.content)}  "
              f"TTFA={r.headers.get('X-TTFA-ms')}ms  RTF={r.headers.get('X-RTF')}")
        with open("/tmp/hams_rest.wav", "wb") as f:
            f.write(r.content)


async def test_ws():
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send('{"type":"speak","text":"مرحبا كيف حالك؟ I am streaming now عَ ال L4.","format":"pcm16","sample_rate":24000}')
        n_chunks, n_bytes = 0, 0
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                n_bytes += len(msg)
            else:
                import json
                m = json.loads(msg)
                if m["type"] == "start":
                    print("WS start:", m)
                elif m["type"] == "chunk":
                    n_chunks += 1
                    print(f"  chunk {m['index']} first={m['is_first']} last={m['is_last']} "
                          f"latency={m['latency_ms']:.0f}ms text={m['text'][:40]!r}")
                elif m["type"] == "end":
                    print(f"WS end: TTFA={m['ttfa_ms']:.0f}ms RTF={m['rtf']:.3f} "
                          f"audio={m['audio_seconds']:.2f}s")
                    break
                elif m["type"] == "error":
                    print("WS error:", m); break
        print(f"WS received {n_chunks} chunk headers, {n_bytes} audio bytes")


async def main():
    if not await wait_ready():
        print("server not ready"); sys.exit(1)
    await test_rest()
    await test_ws()
    print("SMOKE_OK")


if __name__ == "__main__":
    asyncio.run(main())
