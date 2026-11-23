"""
pregenerate_references.py - fill data/cache/ with a gTTS reference for every
practice sentence in every accent, so the live demo needs no internet.

    python scripts/pregenerate_references.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import tts  # noqa: E402
from audio_io import CoachError  # noqa: E402


def main():
    """Synthesize every (sentence, accent) pair with gTTS and report the result."""
    sentences = json.loads(config.SENTENCES_FILE.read_text(encoding="utf-8"))
    ok, failed = 0, 0
    for accent in config.ACCENTS:
        for sentence in sentences:
            start = time.time()
            try:
                path, message = tts.synthesize(sentence, accent, backend="gtts")
                ok += 1
                print(f"OK   {accent:18s} {time.time() - start:4.1f} s  {sentence[:50]}")
            except CoachError as exc:
                failed += 1
                print(f"FAIL {accent:18s} {sentence[:50]}  ({exc})")
            time.sleep(0.3)  # be polite to the gTTS service
    print(f"\nDone: {ok} cached, {failed} failed. Cache folder: {config.CACHE_DIR}")
    if failed:
        print("Some clips failed. Check the internet connection and run this script again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
