"""Tests for audio_io.py: decoding, resampling, trimming, normalizing, length checks."""

import numpy as np
import pytest
import soundfile as sf

import audio_io
import config
from audio_io import CoachError
from conftest import synth_utterance, tone


def test_load_resamples_stereo_44k_to_mono_16k(tmp_path):
    y = tone(dur=1.0, sr=44100)
    stereo = np.stack([y, y], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, 44100)
    out, sr = audio_io.load_audio(path)
    assert sr == config.SAMPLE_RATE
    assert out.ndim == 1
    assert abs(len(out) - 16000) <= 2


def test_load_mp3(tmp_path):
    path = tmp_path / "clip.mp3"
    sf.write(str(path), tone(dur=1.0, sr=22050), 22050, format="MP3")
    out, sr = audio_io.load_audio(path)
    assert sr == 16000 and 0.9 < len(out) / sr < 1.2


def test_missing_file_is_friendly():
    with pytest.raises(CoachError):
        audio_io.load_audio(None)


def test_preprocess_trims_silence_and_normalizes():
    speech = synth_utterance(gain=0.3)
    padded = np.concatenate([np.zeros(16000), speech, np.zeros(16000)]).astype(np.float32)
    out = audio_io.preprocess(padded)
    assert len(out) < len(padded) - 25000  # most of the 2 s of silence is gone
    assert abs(np.max(np.abs(out)) - config.PEAK) < 1e-4


def test_silent_recording_is_friendly():
    with pytest.raises(CoachError, match="silent"):
        audio_io.preprocess(np.zeros(16000, dtype=np.float32))


def test_length_limits():
    with pytest.raises(CoachError, match="only"):
        audio_io.check_length(np.ones(int(0.2 * 16000)))
    with pytest.raises(CoachError, match="under"):
        audio_io.check_length(np.ones(int(16 * 16000)))
    audio_io.check_length(np.ones(16000))  # 1 s is fine


def test_save_and_reload_round_trip(tmp_path):
    y = tone(dur=0.5)
    path = audio_io.save_wav(y, tmp_path / "x.wav")
    info = sf.info(str(path))
    assert info.subtype == "PCM_16" and info.samplerate == 16000
    back, _ = audio_io.load_audio(path)
    assert np.max(np.abs(back - y)) < 1e-3
