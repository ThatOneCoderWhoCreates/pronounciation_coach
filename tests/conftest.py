"""
conftest.py - shared helpers for the tests. Everything uses SYNTHETIC audio,
so the tests need no microphone, no internet and no Whisper model.

A "synthetic utterance" is a chain of vowel-like segments. Each segment is a
buzz of harmonics (like vocal folds) shaped by two formant peaks (like the
mouth), so different vowel sequences give different MFCC patterns, just like
different words do.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # let the tests import the project modules

SR = 16000

# (first formant, second formant) in Hz for a few vowels
VOWELS = {"a": (730, 1090), "i": (270, 2290), "u": (300, 870), "e": (530, 1840), "o": (570, 840)}


def synth_utterance(vowels="aiueo", seg_dur=0.25, f0_start=140.0, f0_end=110.0,
                    sr=SR, gain=0.5, gap=0.05):
    """Build a speech-like signal: one harmonic segment per vowel with short gaps."""
    total = len(vowels)
    pieces = []
    for k, v in enumerate(vowels):
        n = int(seg_dur * sr)
        t = np.arange(n) / sr
        # pitch glides from f0_start to f0_end across the whole utterance
        f0 = np.linspace(f0_start + (f0_end - f0_start) * k / total,
                         f0_start + (f0_end - f0_start) * (k + 1) / total, n)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        f1, f2 = VOWELS[v]
        y = np.zeros(n)
        for h in range(1, 30):
            freq = h * f0
            weight = np.exp(-((freq - f1) / 120) ** 2) + 0.6 * np.exp(-((freq - f2) / 160) ** 2) + 0.02
            y += weight * np.sin(h * phase)
        envelope = np.hanning(n) ** 0.5
        pieces += [y * envelope, np.zeros(int(gap * sr))]
    y = np.concatenate(pieces)
    return (gain * y / np.max(np.abs(y))).astype(np.float32)


def tone(freq=200.0, dur=1.0, sr=SR, gain=0.5):
    """A harmonic tone (fundamental + 4 harmonics) at a fixed pitch."""
    t = np.arange(int(dur * sr)) / sr
    y = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, 6))
    return (gain * y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture
def tmp_history(tmp_path):
    """A history file inside pytest's temporary folder (never the real one)."""
    return tmp_path / "history.csv"


@pytest.fixture
def fake_stt(monkeypatch):
    """Replace Whisper with a function that returns a fixed transcript."""
    import stt

    state = {"text": ""}
    monkeypatch.setattr(stt, "transcribe", lambda wav, size=None: (state["text"], []))
    return state
