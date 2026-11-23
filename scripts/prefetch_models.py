"""
prefetch_models.py - download Whisper models into data/models/ ahead of the demo.

    python scripts/prefetch_models.py            # base.en (the default)
    python scripts/prefetch_models.py --all      # tiny.en, base.en and small.en

After this the app needs no internet for speech-to-text.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import stt  # noqa: E402


def main():
    """Download (or confirm) each requested model and time one load."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="download every model in WHISPER_CHOICES")
    args = parser.parse_args()
    sizes = config.WHISPER_CHOICES if args.all else [config.WHISPER_MODEL]
    for size in sizes:
        start = time.time()
        try:
            stt.get_model(size)
        except stt.SttUnavailable as exc:
            print(f"FAILED {size}: {exc}")
            sys.exit(1)
        print(f"OK {size}: ready in {time.time() - start:.1f} s (stored in {config.MODELS_DIR})")


if __name__ == "__main__":
    main()
