# CPU streaming-server image (front-end + FastAPI/WebSocket, espeak backend).
# For the GPU/TensorRT production image, base on nvcr.io/nvidia/pytorch:24.05-py3,
# `pip install -e '.[gpu]'`, copy the .plan engine, and set HAMS_BACKEND=tensorrt.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e '.[server]'

ENV HAMS_BACKEND=espeak HAMS_HOST=0.0.0.0 HAMS_PORT=8000
EXPOSE 8000
# For the neural model: mount the engine and set HAMS_BACKEND=onnx|tensorrt + HAMS_MODEL_PATH.
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
CMD ["python", "-m", "hams_tts.server.app"]
