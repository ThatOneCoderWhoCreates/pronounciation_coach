#!/usr/bin/env bash
# setup.sh - one-time setup on macOS or Linux. Run from the project folder:
#     bash setup.sh
# It needs internet once (Whisper models and gTTS reference voices).
set -e
cd "$(dirname "$0")"

# 1. Find Python 3.12 or 3.11 (3.13+ can break the audio libraries; 3.10 is too old for numpy 2.4+).
PY=""
for candidate in /opt/homebrew/bin/python3.12 python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12 or 3.11 not found. On macOS run: brew install python@3.12"
  exit 1
fi
echo "Using $($PY --version) at $(command -v $PY)"

# 2. ffmpeg is recommended for unusual audio formats.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "NOTE: ffmpeg not found. Recommended: brew install ffmpeg (the app still works with wav/mp3)."
fi

# 3. Virtual environment and packages.
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Download Whisper models (tiny.en, base.en, small.en; about 700 MB in total).
python scripts/prefetch_models.py --all

# 5. Reference voices for every sentence and accent (40 clips).
python scripts/pregenerate_references.py || echo "Some clips failed. Run: python scripts/pregenerate_references.py again later."

# 6. Starting calibration from the four accent voices.
python scripts/calibrate.py --auto --save

# 7. Demo recordings to upload if the microphone fails.
python scripts/make_demo_audio.py

# 8. Tests and experiments (results go to experiments/results/results.md).
python -m pytest -q || echo "Some tests failed. See the output above."
python experiments/run_experiments.py

echo ""
echo "Setup complete. Start the app with:"
echo "    source .venv/bin/activate"
echo "    python app.py"
