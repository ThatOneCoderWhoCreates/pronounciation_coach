"""
audio_io.py - getting audio in and out of the app.

Every signal goes through the same steps so that the reference and the learner
are always compared under identical conditions:
    decode -> mono -> resample to 16 kHz -> trim silence -> peak-normalize
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import config

# On Windows, stop helper programs (ffmpeg, the offline voice) from flashing a
# black console window. On macOS and Linux this flag is simply 0.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class CoachError(Exception):
    """An error with a message that is safe and friendly to show to the learner."""


def ffmpeg_available():
    """Return True if the ffmpeg program can be found on the PATH."""
    return shutil.which("ffmpeg") is not None


def _decode_with_ffmpeg(path):
    """
    Ask ffmpeg to convert any audio file (webm, m4a, mp3...) into a temporary
    16 kHz mono wav file, then read that file with soundfile.
    """
    if not ffmpeg_available():
        raise CoachError(
            "This audio format needs ffmpeg, which is not installed. "
            "Install ffmpeg (see README) or upload a .wav file."
        )
    # ignore_cleanup_errors: on Windows an antivirus scan can briefly lock the file
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp) / "decoded.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
               "-ac", "1", "-ar", str(config.SAMPLE_RATE), str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
        if result.returncode != 0 or not out.exists():
            raise CoachError("Could not read this audio file. Please try a .wav or .mp3 file.")
        y, sr = sf.read(str(out), dtype="float32")
    return y, sr


def load_audio(path):
    """
    Read an audio file and return (y, sr) with y as mono float32 at 16 kHz.

    soundfile is tried first (it reads wav, flac, ogg and mp3). If it fails,
    ffmpeg is used. librosa 1.0 removed its own audioread decoder, so ffmpeg is
    the fallback for browser formats such as webm or m4a.
    """
    if path is None or not Path(path).exists():
        raise CoachError("No audio was received. Please record or upload a file.")
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        y, sr = _decode_with_ffmpeg(path)

    # Stereo has shape (samples, channels): average the channels to get mono.
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Resample so that every signal in the app shares one sample rate.
    if sr != config.SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return y.astype(np.float32), config.SAMPLE_RATE


def preprocess(y):
    """
    Trim leading and trailing silence, then scale the loudest sample to PEAK.

    Trimming makes durations comparable (pace score) and stops DTW from
    matching silence with silence. Normalizing removes loudness differences
    caused by the microphone distance.
    """
    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        raise CoachError("The recording is silent. Check your microphone and try again.")
    trimmed, _ = librosa.effects.trim(
        y, top_db=config.TRIM_TOP_DB, frame_length=config.FRAME_LEN * 2, hop_length=config.HOP
    )
    if trimmed.size == 0:
        trimmed = y
    peak = np.max(np.abs(trimmed))
    return (trimmed / peak * config.PEAK).astype(np.float32)


def check_length(y, sr=config.SAMPLE_RATE):
    """Raise a friendly CoachError if the speech is too short or too long."""
    seconds = len(y) / sr
    if seconds < config.MIN_DUR:
        raise CoachError(
            f"The speech is only {seconds:.1f} s long. Please say the whole sentence "
            f"(at least {config.MIN_DUR} s)."
        )
    if seconds > config.MAX_DUR:
        raise CoachError(
            f"The recording is {seconds:.1f} s long. Please keep it under {config.MAX_DUR:.0f} s."
        )


def save_wav(y, path, sr=config.SAMPLE_RATE):
    """Write y to path as a 16-bit PCM wav file and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path


def load_and_prepare(path):
    """Convenience: load, preprocess and length-check in one call. Returns y."""
    y, sr = load_audio(path)
    y = preprocess(y)
    check_length(y, sr)
    return y
