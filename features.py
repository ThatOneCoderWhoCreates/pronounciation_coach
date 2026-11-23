"""
features.py - turning a waveform into numbers we can compare.

Three features are extracted, all with a 10 ms hop so their frames line up:
    MFCC (with CMVN)  -> what was said and how it sounded (spectral shape)
    pitch (semitones) -> the melody of the sentence (intonation)
    RMS energy (dB)   -> loudness over time
"""

import librosa
import numpy as np

import config


def mfcc_cmvn(y, sr=config.SAMPLE_RATE):
    """
    Return MFCCs of shape (13, T) after cepstral mean and variance
    normalization (CMVN).

    MFCC: short frames -> FFT -> 40 mel filters -> log -> DCT -> keep 13 numbers.
    CMVN: for each of the 13 coefficients, subtract its average over time and
    divide by its standard deviation. A microphone or a voice adds a roughly
    constant offset to each coefficient, so CMVN removes much of that
    difference and leaves the pattern of the words.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.N_FFT,
        hop_length=config.HOP, win_length=config.FRAME_LEN,
        window="hamming", n_mels=config.N_MELS,
    )
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-8  # avoid dividing by zero
    return (mfcc - mean) / std


def mfcc_raw(y, sr=config.SAMPLE_RATE):
    """MFCCs without CMVN. Used only by the CMVN experiment for comparison."""
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.N_FFT,
        hop_length=config.HOP, win_length=config.FRAME_LEN,
        window="hamming", n_mels=config.N_MELS,
    )


def f0_semitones(y, sr=config.SAMPLE_RATE):
    """
    Estimate pitch with pyin and return (semitones, voiced, f0_hz).

    pyin finds the fundamental frequency f0 of each frame and says whether the
    frame is voiced (vocal folds vibrating) or not.
    Semitones = 12 * log2(f0 / median voiced f0). Measuring relative to the
    speaker's own median means a low voice and a high voice saying the
    sentence with the same melody give the same contour. Unvoiced frames are NaN.
    """
    f0, voiced, _ = librosa.pyin(
        y, fmin=config.F0_MIN, fmax=config.F0_MAX, sr=sr,
        frame_length=config.PYIN_FRAME, hop_length=config.HOP,
    )
    voiced = voiced & np.isfinite(f0)
    semitones = np.full(f0.shape, np.nan)
    if voiced.any():
        median_f0 = np.median(f0[voiced])
        semitones[voiced] = 12.0 * np.log2(f0[voiced] / median_f0)
    return semitones, voiced, f0


def rms_db(y):
    """Return the energy of each frame in decibels (dB), one value per 10 ms."""
    rms = librosa.feature.rms(y=y, frame_length=config.FRAME_LEN, hop_length=config.HOP)[0]
    return librosa.amplitude_to_db(rms, ref=1.0)


def log_mel(y, sr=config.SAMPLE_RATE):
    """Return a log-mel spectrogram in dB, shape (N_MELS, T). Used by the plots."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP,
        win_length=config.FRAME_LEN, window="hamming", n_mels=config.N_MELS,
    )
    return librosa.power_to_db(mel, ref=1.0)


def extract_all(y):
    """Compute every feature for one signal and return them in a dictionary."""
    semitones, voiced, f0 = f0_semitones(y)
    return {
        "mfcc": mfcc_cmvn(y),
        "semitones": semitones,
        "voiced": voiced,
        "f0": f0,
        "energy_db": rms_db(y),
        "duration": len(y) / config.SAMPLE_RATE,
    }
