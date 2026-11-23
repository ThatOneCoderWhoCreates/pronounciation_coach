"""
tts.py - text-to-speech for the reference voice.

Two engines:
    gTTS     Google Translate voice, online, the accent can be chosen (tld).
    pyttsx3  the operating system's own voice, offline, accent cannot be chosen.
Every result is saved in data/cache/ so a sentence is synthesized only once
and the demo works offline afterwards.
"""

import hashlib
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import config
from audio_io import NO_WINDOW, CoachError, load_audio, save_wav

# This small program runs in a SEPARATE Python process. pyttsx3 is known to
# hang when it is called again from Gradio worker threads on Windows, so we
# start a fresh process with a fresh engine for every sentence.
_PYTTSX3_WORKER = """
import sys, pyttsx3
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
text, out = sys.argv[1], sys.argv[2]
engine = pyttsx3.init()
voice_id = engine.getProperty("voice")
name = str(voice_id)
for v in engine.getProperty("voices"):
    if v.id == voice_id:
        name = v.name
engine.save_to_file(text, out)
engine.runAndWait()
print(name)
"""


def cache_path(text, accent, backend):
    """
    File name for a cached reference: SHA-1 hash of text + accent + backend.
    The accent is part of the name so a UK and a US version never overwrite
    each other.
    """
    key = f"{text.strip()}|{accent}|{backend}".encode("utf-8")
    return config.CACHE_DIR / f"{hashlib.sha1(key).hexdigest()}.wav"


def _gtts(text, accent, out_path):
    """Synthesize with gTTS (explicit lang and tld), decode the MP3, save as wav."""
    from gtts import gTTS

    tld = config.ACCENTS[accent]  # KeyError here would mean an unknown accent
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        mp3 = Path(tmp) / "ref.mp3"
        gTTS(text=text, lang="en", tld=tld).save(str(mp3))
        y, sr = load_audio(mp3)  # soundfile first, then ffmpeg
    save_wav(y, out_path, sr)
    return f"gTTS voice, accent {accent} (tld '{tld}')"


def _pyttsx3(text, out_path):
    """Synthesize offline with the operating system voice in a subprocess."""
    suffix = ".aiff" if platform.system() == "Darwin" else ".wav"  # macOS writes AIFF
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        raw = Path(tmp) / f"ref{suffix}"
        voice = None
        try:
            run = subprocess.run([sys.executable, "-c", _PYTTSX3_WORKER, text, str(raw)],
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=60, creationflags=NO_WINDOW)
            if run.returncode == 0 and raw.exists() and raw.stat().st_size > 1000:
                voice = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else "system voice"
        except subprocess.TimeoutExpired:
            pass
        # Last resort on macOS: the built-in "say" command.
        if voice is None and platform.system() == "Darwin" and shutil.which("say"):
            raw = Path(tmp) / "say.aiff"
            subprocess.run(["say", "-o", str(raw), text], check=False, timeout=60)
            if raw.exists():
                voice = "macOS 'say' default voice"
        if voice is None:
            raise CoachError("The offline voice (pyttsx3) failed. Upload your own reference recording.")
        y, sr = load_audio(raw)
    save_wav(y, out_path, sr)
    return f"offline system voice: {voice}"


def synthesize(text, accent=config.DEFAULT_ACCENT, backend="auto"):
    """
    Return (wav_path, message) for the reference sentence.

    backend "gtts": online voice with the chosen accent.
    backend "pyttsx3": offline system voice (the accent is NOT applied).
    backend "auto": use a cached gTTS file if we have one, else try gTTS,
                    and if that fails (no internet) fall back to pyttsx3.
    The message tells the learner honestly which voice was used.
    """
    text = (text or "").strip()
    if not text:
        raise CoachError("Please choose or type a sentence first.")
    if accent not in config.ACCENTS:
        raise CoachError(f"Unknown accent '{accent}'. Choose one from the list.")

    gtts_file = cache_path(text, accent, "gtts")
    py_file = cache_path(text, accent, "pyttsx3")

    if backend in ("auto", "gtts"):
        if gtts_file.exists():
            return gtts_file, f"Reference: gTTS voice, {accent} (from cache)."
        try:
            note = _gtts(text, accent, gtts_file)
            return gtts_file, f"Reference: {note}."
        except Exception as exc:
            gtts_file.unlink(missing_ok=True)
            if backend == "gtts":
                raise CoachError("gTTS needs internet and could not be reached. "
                                 "Choose backend 'auto' or 'pyttsx3'.") from exc

    # pyttsx3 (chosen directly, or as the auto fallback)
    warning = (f" The chosen accent ({accent}) was NOT applied, because the offline "
               "voice cannot change accent.")
    if py_file.exists():
        return py_file, "Reference: offline system voice (from cache)." + warning
    note = _pyttsx3(text, py_file)
    return py_file, f"Reference: {note}." + warning
