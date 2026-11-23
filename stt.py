"""
stt.py - speech-to-text with Whisper.

Whisper is a neural network trained on 680,000 hours of speech. We use the
faster-whisper version (same model, faster CPU engine called CTranslate2).
The model is downloaded once into data/models/ and loaded from there after,
so the app works offline.
"""

import os
from functools import lru_cache

# Windows without Developer Mode cannot make symlinks, so the model download
# copies files instead and prints a long warning. The copy works fine.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import config  # noqa: E402
from audio_io import load_audio  # noqa: E402


class SttUnavailable(Exception):
    """Raised when no Whisper model can be loaded (not downloaded and offline)."""


@lru_cache(maxsize=3)
def get_model(model_size=config.WHISPER_MODEL):
    """
    Load a Whisper model once and keep it in memory (lru_cache remembers it).

    First try the copy already on disk (works offline). Only if it is missing,
    download it. CPU with int8 numbers keeps it fast on a laptop without a GPU.
    """
    from faster_whisper import WhisperModel

    root = str(config.MODELS_DIR)
    try:
        return WhisperModel(model_size, device="cpu", compute_type="int8",
                            download_root=root, local_files_only=True)
    except Exception:
        pass  # not on disk yet: fall through and download it
    try:
        return WhisperModel(model_size, device="cpu", compute_type="int8", download_root=root)
    except Exception as exc:
        raise SttUnavailable(
            f"The Whisper model '{model_size}' is not downloaded and could not be fetched. "
            "Connect to the internet once and run: python scripts/prefetch_models.py"
        ) from exc


def is_model_cached(model_size=config.WHISPER_MODEL):
    """Return True if the model files are already in data/models/ (no download needed)."""
    folder = config.MODELS_DIR / f"models--Systran--faster-whisper-{model_size}"
    return folder.exists() and any(folder.rglob("model.bin"))


def _transcribe_faster_whisper(wav_path, model_size):
    """Run faster-whisper and return (text, words)."""
    model = get_model(model_size)
    # We decode the audio ourselves (soundfile, 16 kHz mono) and pass the samples.
    # If faster-whisper got the file path, it would decode it with PyAV, and
    # faster-whisper 1.2.1 uses an argument that PyAV 19 no longer accepts.
    samples, _sr = load_audio(wav_path)
    segments, _info = model.transcribe(
        samples,
        language="en",
        beam_size=1,                       # greedy decoding: fast and repeatable
        temperature=0.0,                   # no random sampling
        condition_on_previous_text=False,  # each segment judged on its own
        word_timestamps=True,
        # IMPORTANT: no initial_prompt. Giving Whisper the target sentence would
        # push it toward the right words and hide mispronunciations.
    )
    # segments is a generator: the real work happens only while we loop over it.
    segments = list(segments)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": float(w.start),
                          "end": float(w.end), "probability": float(w.probability)})
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, words


def _transcribe_openai_whisper(wav_path, model_size):
    """Fallback with the original openai-whisper package, same return format."""
    import whisper  # only imported if faster-whisper is not installed

    model = whisper.load_model(model_size, download_root=str(config.MODELS_DIR))
    result = model.transcribe(str(wav_path), language="en", temperature=0.0,
                              condition_on_previous_text=False, word_timestamps=True)
    words = [{"word": w["word"].strip(), "start": float(w["start"]),
              "end": float(w["end"]), "probability": float(w.get("probability", 0.0))}
             for seg in result["segments"] for w in seg.get("words", [])]
    return result["text"].strip(), words


def transcribe(wav_path, model_size=config.WHISPER_MODEL):
    """
    Transcribe a 16 kHz wav file. Returns (text, words) where words is a list
    of {"word", "start", "end", "probability"} dictionaries (times in seconds).
    """
    try:
        import faster_whisper  # noqa: F401  (only checking that it is installed)
    except ImportError:
        try:
            return _transcribe_openai_whisper(wav_path, model_size)
        except ImportError as exc:
            raise SttUnavailable("Neither faster-whisper nor openai-whisper is installed.") from exc
    return _transcribe_faster_whisper(wav_path, model_size)
