"""
End-to-end tests of pipeline.analyze(), history.py, tts.py and plots.py.
Whisper and gTTS are replaced by stand-ins so no internet is needed.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

import audio_io
import config
import history
import pipeline
import plots
import tts
from audio_io import CoachError
from conftest import synth_utterance

SENTENCE = "The weather is lovely today."


@pytest.fixture
def wavs(tmp_path):
    """Reference, a slower copy, and a different 'sentence', all as wav files."""
    y = synth_utterance("aiueoa")
    slow = np.repeat(y, 2)[: int(len(y) * 1.6)]  # crude slow-down: lower pitch, longer
    return {
        "ref": audio_io.save_wav(y, tmp_path / "ref.wav"),
        "same": audio_io.save_wav(y * 0.5, tmp_path / "same.wav"),
        "slow": audio_io.save_wav(slow, tmp_path / "slow.wav"),
        "other": audio_io.save_wav(synth_utterance("uoiaeu"), tmp_path / "other.wav"),
    }


def run(wavs, key, fake_stt, tmp_history, text=SENTENCE):
    fake_stt["text"] = text
    return pipeline.analyze(SENTENCE, wavs["ref"], wavs[key], "US English", history_path=tmp_history)


def test_self_comparison_scores_100(wavs, fake_stt, tmp_history):
    s = run(wavs, "same", fake_stt, tmp_history)["scores"]
    assert s["intelligibility"] == 100 and s["reference_match"] == 100 and s["pace_match"] == 100
    assert s["intonation_match"] > 99 and s["overall"] > 99.5


def test_wrong_words_lower_intelligibility(wavs, fake_stt, tmp_history):
    r = run(wavs, "same", fake_stt, tmp_history, text="The feather is lovely today.")
    assert r["scores"]["intelligibility"] == pytest.approx(80)
    assert ("wrong", "weather", "feather") in r["diff"]


def test_different_sentence_lower_reference_match(wavs, fake_stt, tmp_history):
    same = run(wavs, "same", fake_stt, tmp_history)
    other = run(wavs, "other", fake_stt, tmp_history)
    assert other["dtw_cost"] > same["dtw_cost"]


def test_slow_speech_lowers_pace(wavs, fake_stt, tmp_history):
    r = run(wavs, "slow", fake_stt, tmp_history)
    assert r["rate_ratio"] > 1.4 and r["scores"]["pace_match"] < 80


def test_history_is_written_without_audio(wavs, fake_stt, tmp_history, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RECORDINGS_DIR", tmp_path / "rec")
    run(wavs, "same", fake_stt, tmp_history)
    run(wavs, "slow", fake_stt, tmp_history)
    df = history.load_history(tmp_history)
    assert len(df) == 2 and list(df.columns) == history.COLUMNS
    assert not (tmp_path / "rec").exists()  # privacy: nothing saved by default
    assert history.clear_history(tmp_history) == 2 and len(history.load_history(tmp_history)) == 0


def test_empty_sentence_is_friendly(wavs, fake_stt, tmp_history):
    with pytest.raises(CoachError):
        pipeline.analyze("  ", wavs["ref"], wavs["same"], "US English", history_path=tmp_history)


def test_every_plot_has_title_and_labels(wavs, fake_stt, tmp_history):
    r = run(wavs, "slow", fake_stt, tmp_history)
    figs = pipeline.make_figures(r) + [plots.plot_history(history.load_history(tmp_history))]
    assert len(figs) == 6
    for fig in figs:
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_title() and (ax.get_ylabel() or fig.axes[-1].get_ylabel())


# ----------------------------------------------------------------- tts --
def test_cache_name_depends_on_accent_and_backend():
    a = tts.cache_path("hello", "US English", "gtts")
    b = tts.cache_path("hello", "UK English", "gtts")
    c = tts.cache_path("hello", "US English", "pyttsx3")
    assert len({a, b, c}) == 3


def test_gtts_gets_explicit_lang_and_tld(monkeypatch, tmp_path):
    seen = {}

    class FakeGTTS:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def save(self, path):
            import soundfile as sf
            sf.write(path, synth_utterance(), 16000, format="WAV")  # stands in for the MP3

    import gtts
    monkeypatch.setattr(gtts, "gTTS", FakeGTTS)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    path, msg = tts.synthesize("Hello there", "Indian English", "gtts")
    assert seen["lang"] == "en" and seen["tld"] == "co.in"
    assert path.exists() and "gTTS" in msg


def test_auto_falls_back_to_offline_voice_and_says_so(monkeypatch, tmp_path):
    def broken(*args):
        raise ConnectionError("no internet")

    def fake_offline(text, out_path):
        audio_io.save_wav(synth_utterance(), out_path)
        return "offline system voice: Test Voice"

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(tts, "_gtts", broken)
    monkeypatch.setattr(tts, "_pyttsx3", fake_offline)
    path, msg = tts.synthesize("Hello there", "UK English", "auto")
    assert path.exists() and "NOT applied" in msg and "Test Voice" in msg
    with pytest.raises(CoachError, match="internet"):
        tts.synthesize("Another sentence", "UK English", "gtts")


def test_unknown_accent_is_rejected():
    with pytest.raises(CoachError):
        tts.synthesize("Hello", "Martian English")


def test_locked_history_file_gives_friendly_error(monkeypatch, tmp_history):
    """On Windows, Excel locks an open CSV file; the learner should get a clear message."""
    import pandas as pd

    def locked(*args, **kwargs):
        raise PermissionError("file is locked")

    monkeypatch.setattr(pd.DataFrame, "to_csv", locked)
    with pytest.raises(CoachError, match="open in"):
        history.append_attempt({"sentence": "x"}, tmp_history)
