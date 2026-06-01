"""Minimal Pipecat voice-agent pipeline using HamsTTSService.

This is the integration shown in the demo video: an LLM produces (possibly
code-switched) text, and our TTS service streams Levantine/English audio out.

Run (needs the `pipecat` extra and your transport creds):
    pip install -e '.[pipecat]'
    python examples/pipecat_bot.py

This is a no-LLM smoke of just the TTS service: it pushes code-switched TextFrames
straight through HamsTTSService so you can hear streamed Levantine/English audio.
"""

import asyncio
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import TextFrame, EndFrame

from hams_tts.pipecat_plugin import HamsTTSService


async def main():
    # Backend: 'espeak' runs anywhere (dev); 'onnx'/'tensorrt' on the GPU host.
    tts = HamsTTSService(
        backend=os.environ.get("HAMS_BACKEND", "espeak"),
        model_path=os.environ.get("HAMS_MODEL_PATH"),
        sample_rate=24000,
    )

    # In a real bot you'd wire transport.input() -> STT -> LLM -> tts -> transport.output().
    # Here we push a code-switched line straight to TTS to demonstrate the service.
    pipeline = Pipeline([tts])
    task = PipelineTask(pipeline)

    async def feed():
        await task.queue_frames([
            TextFrame("مرحبا! أنا Hams، مساعدك الصوتي. "),
            TextFrame("بقدر إحكي عربي و English بنفس الجملة، seamlessly. "),
            TextFrame("خليني إحجزلك flight من بيروت to London بكرا الساعة 9. "),
            EndFrame(),
        ])

    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), feed())


if __name__ == "__main__":
    asyncio.run(main())
