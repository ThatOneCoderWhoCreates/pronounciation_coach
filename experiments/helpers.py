"""
helpers.py - small tools shared by the experiments: audio transforms, scoring
one (reference, learner) pair without Whisper, and Markdown tables.
"""

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_io  # noqa: E402
import compare  # noqa: E402
import features  # noqa: E402


def load(path):
    """Load, trim and normalize an audio file exactly like the app does."""
    return audio_io.preprocess(audio_io.load_audio(path)[0])


def stretch(y, speed):
    """Change the speed without changing the pitch (speed 2 = twice as fast)."""
    return audio_io.preprocess(librosa.effects.time_stretch(y, rate=speed))


def shift(y, semitones):
    """Change the pitch without changing the speed."""
    return audio_io.preprocess(librosa.effects.pitch_shift(y, sr=16000, n_steps=semitones))


def add_noise(y, snr_db, seed=0):
    """Add white noise at the given signal-to-noise ratio in dB (inf = no noise)."""
    if np.isinf(snr_db):
        return y
    noise = np.random.default_rng(seed).normal(0, 1, len(y))
    scale = np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return np.clip(y + scale * noise, -1, 1).astype(np.float32)


def score_pair(y_ref, y_lrn, accent="US English"):
    """All audio scores (no Whisper) for one pair of prepared waveforms."""
    f_ref, f_lrn = features.extract_all(y_ref), features.extract_all(y_lrn)
    D, path, cost = compare.dtw_align(f_ref["mfcc"], f_lrn["mfcc"])
    inton, pairs = compare.intonation_score(path, f_ref["semitones"], f_lrn["semitones"])
    pace, ratio = compare.pace_score(f_lrn["duration"], f_ref["duration"])
    return {"cost": cost, "reference_match": compare.reference_match_score(cost, accent),
            "intonation_match": inton, "pace_match": pace, "ratio": ratio, "pairs": pairs}


def raw_cost(y_ref, y_lrn):
    """DTW cost on MFCCs WITHOUT CMVN (for the ablation experiment)."""
    return compare.dtw_align(features.mfcc_raw(y_ref), features.mfcc_raw(y_lrn))[2]


def cmvn_cost(y_ref, y_lrn):
    """DTW cost on MFCCs WITH CMVN (what the app uses)."""
    return compare.dtw_align(features.mfcc_cmvn(y_ref), features.mfcc_cmvn(y_lrn))[2]


def fmt(v, digits=1):
    """Format a number for a table (None -> n/a)."""
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{digits}f}"


def md_table(headers, rows):
    """Return a Markdown table as a string."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(lines)


def separation(same, diff):
    """
    How well costs separate same-sentence pairs from different-sentence pairs.
    Returns (d_prime, accuracy). d' = (mean diff - mean same) / pooled std.
    accuracy = share of (same, diff) combinations where same < diff.
    """
    same, diff = np.asarray(same), np.asarray(diff)
    pooled = np.sqrt((same.var() + diff.var()) / 2) + 1e-9
    accuracy = np.mean([s < d for s in same for d in diff])
    return (diff.mean() - same.mean()) / pooled, accuracy
