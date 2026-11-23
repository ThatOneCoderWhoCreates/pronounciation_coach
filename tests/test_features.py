"""Tests for features.py: MFCC with CMVN, pitch in semitones, energy."""

import numpy as np

import config
import features
from conftest import synth_utterance, tone


def test_mfcc_shape_and_cmvn():
    y = synth_utterance()
    m = features.mfcc_cmvn(y)
    assert m.shape[0] == config.N_MFCC
    assert m.shape[1] == 1 + len(y) // config.HOP  # one frame every 10 ms
    assert np.allclose(m.mean(axis=1), 0, atol=1e-5)  # CMVN: zero mean
    assert np.allclose(m.std(axis=1), 1, atol=1e-3)   # CMVN: unit variance


def test_cmvn_removes_loudness_difference():
    y = synth_utterance()
    a, b = features.mfcc_cmvn(y), features.mfcc_cmvn(0.2 * y)
    assert np.max(np.abs(a - b)) < 0.05


def test_pitch_of_steady_tone():
    semitones, voiced, f0 = features.f0_semitones(tone(200.0, dur=1.0))
    assert voiced.mean() > 0.8
    assert abs(np.median(f0[voiced]) - 200.0) < 4.0
    assert np.nanmax(np.abs(semitones)) < 0.5  # flat contour around 0


def test_pitch_contour_follows_a_rising_voice():
    y = synth_utterance("aaaa", f0_start=100, f0_end=180)
    semitones, voiced, _ = features.f0_semitones(y)
    v = semitones[voiced]
    first, last = np.mean(v[: len(v) // 4]), np.mean(v[-len(v) // 4:])
    assert last - first > 6  # 100 -> 180 Hz is about 10 semitones


def test_semitones_ignore_voice_height():
    low = features.f0_semitones(synth_utterance(f0_start=110, f0_end=90))[0]
    high = features.f0_semitones(synth_utterance(f0_start=220, f0_end=180))[0]
    both = np.isfinite(low) & np.isfinite(high)
    assert np.nanmedian(np.abs(low[both] - high[both])) < 0.5


def test_noise_is_mostly_unvoiced():
    rng = np.random.default_rng(0)
    _, voiced, _ = features.f0_semitones(rng.normal(0, 0.3, 16000).astype(np.float32))
    assert voiced.mean() < 0.2


def test_energy_frames_match_mfcc_frames():
    y = synth_utterance()
    assert len(features.rms_db(y)) == features.mfcc_cmvn(y).shape[1]
