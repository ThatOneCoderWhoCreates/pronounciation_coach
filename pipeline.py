"""
pipeline.py - the full analysis of one attempt, from audio files to scores.

app.py only calls the two public functions here, get_reference() and
analyze(), so the user interface never contains signal processing code.

    reference wav ---> load + trim + normalize ---> features --+
                                                               +--> DTW --> scores --> history
    learner wav -----> load + trim + normalize ---> features --+
                              |
                              +--> Whisper --> transcript --> WER + word diff
"""

import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import audio_io
import compare
import config
import features
import history
import plots
import stt
import tts


def get_reference(sentence, accent, backend="auto", own_reference=None):
    """
    Return (reference_path, message, source) for the chosen accent.
    If the learner chose "My own reference recording", their upload is used.
    """
    if accent == config.OWN_REFERENCE:
        if not own_reference:
            raise audio_io.CoachError("Upload your own reference recording first.")
        return Path(own_reference), "Reference: your uploaded recording.", "upload"
    path, message = tts.synthesize(sentence, accent, backend)
    source = "gtts" if "gTTS" in message else "pyttsx3"
    return path, message, source


@lru_cache(maxsize=32)
def _reference_features(path_str, mtime):
    """
    Load and analyze a reference once, then remember the result. The file's
    modification time is part of the key, so a changed file is analyzed again.
    """
    y = audio_io.load_and_prepare(path_str)
    return y, features.extract_all(y)


def analyze(sentence, ref_path, learner_path, accent, model_size=config.WHISPER_MODEL,
            learner_accent="Prefer not to say", reference_source="gtts",
            tts_backend="auto", save_recording=False, history_path=None):
    """
    Score one attempt and save it to the history. Returns a dictionary with the
    scores, the transcript, the word diff and everything the plots need.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        raise audio_io.CoachError("Please choose or type a sentence first.")

    # 1. Same preprocessing for both signals.
    y_ref, f_ref = _reference_features(str(ref_path), Path(ref_path).stat().st_mtime)
    y_lrn = audio_io.load_and_prepare(learner_path)
    f_lrn = features.extract_all(y_lrn)

    # 2. Speech-to-text on the cleaned 16 kHz learner audio.
    with tempfile.TemporaryDirectory() as tmp:
        wav = audio_io.save_wav(y_lrn, Path(tmp) / "learner.wav")
        transcript, words = stt.transcribe(wav, model_size)

    # 3. Text comparison.
    wer = compare.word_error_rate(sentence, transcript)
    diff = compare.word_diff(sentence, transcript)

    # 4. Audio comparison: DTW on MFCCs, then pitch and duration.
    D, path, cost = compare.dtw_align(f_ref["mfcc"], f_lrn["mfcc"])
    intonation, n_pairs = compare.intonation_score(path, f_ref["semitones"], f_lrn["semitones"])
    pace, rate_ratio = compare.pace_score(f_lrn["duration"], f_ref["duration"])
    calib_key = config.OWN_REFERENCE if reference_source == "upload" else accent

    scores = {
        "intelligibility": compare.intelligibility_score(wer),
        "reference_match": compare.reference_match_score(cost, calib_key),
        "intonation_match": intonation,
        "pace_match": pace,
    }
    scores["overall"] = compare.overall_score(scores)

    # 5. Save the attempt (numbers and text only, unless the learner opted in).
    stamp = datetime.now().isoformat(timespec="seconds")
    history.append_attempt({
        "timestamp": stamp, "sentence": sentence, "target_accent": accent,
        "learner_accent": learner_accent, "reference_source": reference_source,
        "tts_backend": tts_backend, "transcript": transcript, "wer": round(wer, 4),
        **{k: (None if v is None else round(v, 2)) for k, v in scores.items()},
        "ref_duration": round(f_ref["duration"], 3), "learner_duration": round(f_lrn["duration"], 3),
    }, history_path)
    if save_recording:
        name = stamp.replace(":", "-") + ".wav"  # ":" is not allowed in Windows file names
        audio_io.save_wav(y_lrn, config.RECORDINGS_DIR / name)

    return {
        "scores": scores, "transcript": transcript, "words": words, "wer": wer, "diff": diff,
        "dtw_cost": cost, "voiced_pairs": n_pairs, "rate_ratio": rate_ratio,
        "y_ref": y_ref, "y_lrn": y_lrn, "f_ref": f_ref, "f_lrn": f_lrn, "D": D, "path": path,
    }


def make_figures(result):
    """Build the five per-attempt figures from an analyze() result."""
    lrn_on_ref = compare.map_to_reference_time(
        result["path"], result["f_lrn"]["semitones"], len(result["f_ref"]["semitones"]))
    return [
        plots.plot_waveforms(result["y_ref"], result["y_lrn"]),
        plots.plot_spectrograms(result["y_ref"], result["y_lrn"]),
        plots.plot_pitch_overlay(result["f_ref"]["semitones"], lrn_on_ref),
        plots.plot_dtw(result["D"], result["path"]),
        plots.plot_scores(result["scores"]),
    ]


def system_status():
    """Short list of setup checks shown at the top of the app (Section 11)."""
    import sys
    cached = len(list(config.CACHE_DIR.glob("*.wav")))
    return {
        "python_ok": (3, 10) <= sys.version_info[:2] <= (3, 12),
        "python": ".".join(map(str, sys.version_info[:3])),
        "ffmpeg": audio_io.ffmpeg_available(),
        "whisper_cached": stt.is_model_cached(config.WHISPER_MODEL),
        "cached_references": cached,
    }
