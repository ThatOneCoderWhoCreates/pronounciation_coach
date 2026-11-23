"""
make_demo_audio.py - create "learner" recordings in demo/ to upload during the
demo if the microphone fails. Each file shows one behavior of the scores.

    python scripts/make_demo_audio.py

These are synthetic voices, not real learners. For the showcase, ALSO record
your own voices (see README, demo plan) and put them in demo/.
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_io  # noqa: E402
import config  # noqa: E402
import tts  # noqa: E402

TARGET = "The weather is lovely today, so let's walk to the park."
WRONG = "The feather is lovely today, so let's talk to the bark."
OTHER = "Could you please tell me the way to the railway station?"
ACCENT = "US English"


def voice(text, accent):
    """Return the 16 kHz waveform of text in the given accent (cached TTS)."""
    path, message = tts.synthesize(text, accent, backend="auto")
    print("  ", message)
    return audio_io.load_audio(path)[0], "gTTS" in message


def add_noise(y, snr_db, seed=0):
    """Add white noise so that the signal-to-noise ratio (SNR) is snr_db decibels."""
    noise = np.random.default_rng(seed).normal(0, 1, len(y))
    scale = np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return np.clip(y + scale * noise, -1, 1).astype(np.float32)


def main():
    """Build each demo clip, save it as wav, and write demo/demo_files.json."""
    config.DEMO_DIR.mkdir(exist_ok=True)
    us, online = voice(TARGET, "US English")
    uk, _ = voice(TARGET, "UK English")
    india, _ = voice(TARGET, "Indian English")
    wrong, _ = voice(WRONG, "US English")
    other, _ = voice(OTHER, "US English")
    clips = [
        ("demo_uk_voice.wav", uk, "Correct words in a UK voice: high intelligibility, lower reference match."),
        ("demo_indian_voice.wav", india, "Correct words in an Indian-English voice."),
        ("demo_wrong_words.wav", wrong, "Two words changed (feather, talk, bark): intelligibility drops."),
        ("demo_too_slow.wav", librosa.effects.time_stretch(us, rate=0.6),
         "The reference slowed to 0.6x speed: pace match drops, reference match stays high."),
        ("demo_noisy.wav", add_noise(uk, 10), "UK voice with background noise at 10 dB SNR."),
        ("demo_different_sentence.wav", other, "A different sentence: every score should be low."),
    ]
    rows = []
    for name, y, description in clips:
        audio_io.save_wav(y, config.DEMO_DIR / name)
        rows.append({"file": name, "sentence": TARGET, "accent": ACCENT, "description": description})
        print(f"wrote demo/{name}")
    (config.DEMO_DIR / "demo_files.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not online:
        print("\nWARNING: gTTS was not reachable, so every clip uses the same offline voice. "
              "Run this again with internet for real accent differences.")


if __name__ == "__main__":
    main()
