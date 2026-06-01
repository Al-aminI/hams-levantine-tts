#!/usr/bin/env bash
# Set up Hams TTS on a CPU dev box (front-end + server). For GPU, see README "Train & deploy".
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== espeak-ng (English G2P + Arabic fallback) =="
if ! command -v espeak-ng >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then brew install espeak-ng
  elif command -v apt-get >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y espeak-ng
  else echo "!! install espeak-ng manually (no brew/apt found)"; fi
fi

echo "== python venv =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

echo "== install (core + server + dev) =="
pip install -e '.[server,dev]'

echo "== sanity: front-end + tests =="
python -m hams_tts.text.frontend | head -n 6
pytest -q

cat <<'EOF'

✅ Ready.
  • samples:   python scripts/make_samples.py --eval-set
  • server:    HAMS_BACKEND=espeak python -m hams_tts.server.app
  • benchmark: python -m hams_tts.eval.benchmark --backend espeak --eval-set data/eval_set/eval_utterances.json
  • GPU train/deploy: see README → "Train & deploy on GPU"
EOF
