"""
config.py - every constant of Pronunciation Coach in one place.

If you want to change how the app behaves (sample rate, frame size, score
weights, calibration values), change it here and nowhere else.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders. pathlib keeps the paths correct on Windows, macOS and Linux.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"            # cached reference audio
MODELS_DIR = DATA_DIR / "models"          # downloaded Whisper models
RECORDINGS_DIR = DATA_DIR / "recordings"  # only used if the learner opts in
HISTORY_FILE = DATA_DIR / "history.csv"
SENTENCES_FILE = DATA_DIR / "sentences.json"
CALIBRATION_FILE = DATA_DIR / "calibration.json"
DEMO_DIR = ROOT_DIR / "demo"

# ---------------------------------------------------------------------------
# Signal processing settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000   # every signal is resampled to 16 kHz (what Whisper uses)
FRAME_LEN = 400       # 25 ms analysis window at 16 kHz
HOP = 160             # 10 ms step between frames
N_FFT = 512           # FFT size (next power of two above FRAME_LEN)
N_MELS = 40           # number of mel filters
N_MFCC = 13           # number of MFCC coefficients kept
F0_MIN = 65           # lowest pitch searched by pyin, in Hz
F0_MAX = 500          # highest pitch searched by pyin, in Hz
PYIN_FRAME = 1024     # pyin needs a longer window to see low pitches
TRIM_TOP_DB = 30      # anything 30 dB below the peak counts as silence
PEAK = 0.9            # peak level after normalization
MIN_DUR = 0.5         # shortest accepted recording, in seconds
MAX_DUR = 15.0        # longest accepted recording, in seconds
MIN_VOICED_PAIRS = 10 # fewer voiced pairs than this means no intonation score

# ---------------------------------------------------------------------------
# Speech-to-text (Whisper) settings
# ---------------------------------------------------------------------------
WHISPER_MODEL = "base.en"
WHISPER_CHOICES = ["tiny.en", "base.en", "small.en"]

# ---------------------------------------------------------------------------
# Accents. UI label -> gTTS top-level domain (tld). We ALWAYS pass the tld,
# because the gTTS default ("com") picks an accent from the network location.
# ---------------------------------------------------------------------------
ACCENTS = {
    "US English": "us",
    "UK English": "co.uk",
    "Indian English": "co.in",
    "Australian English": "com.au",
}
DEFAULT_ACCENT = "US English"
OWN_REFERENCE = "My own reference recording"
LEARNER_ACCENTS = ["US", "UK", "Indian", "Australian", "Other", "Prefer not to say"]
TTS_BACKENDS = ["auto", "gtts", "pyttsx3"]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
DTW_METRIC = "cosine"  # frame-to-frame distance used by DTW

# (good, bad) normalized DTW costs. "good" maps to 100, "bad" maps to 0.
# These are placeholders until scripts/calibrate.py is run (README, Section 9).
DTW_FALLBACK = (0.35, 0.90)
DTW_CALIBRATION = {
    "US English": DTW_FALLBACK,
    "UK English": DTW_FALLBACK,
    "Indian English": DTW_FALLBACK,
    "Australian English": DTW_FALLBACK,
    OWN_REFERENCE: DTW_FALLBACK,
}

# calibrate.py --save writes data/calibration.json; its values win over the
# placeholders above, so nobody has to edit this file by hand.
if CALIBRATION_FILE.exists():
    try:
        for name, pair in json.loads(CALIBRATION_FILE.read_text()).items():
            DTW_CALIBRATION[name] = (float(pair[0]), float(pair[1]))
    except (ValueError, KeyError, IndexError, TypeError):
        pass  # a broken file must never stop the app; placeholders are used

WEIGHTS = {
    "intelligibility": 0.50,
    "reference_match": 0.25,
    "intonation_match": 0.15,
    "pace_match": 0.10,
}

# Pace: a speaking-rate ratio inside [PACE_OK_LOW, PACE_OK_HIGH] scores 100,
# and the score falls to 0 at half speed or double speed.
PACE_OK_HIGH = 1.25
PACE_ZERO_AT = 2.0

# ---------------------------------------------------------------------------
# Privacy and display
# ---------------------------------------------------------------------------
SAVE_RECORDINGS = False  # learner audio is NOT kept unless the user opts in
PLOT_FONT = 12           # readable on a projector

# Okabe-Ito colors: safe for the most common kinds of color blindness.
COLORS = {
    "reference": "#0072B2",  # blue
    "learner": "#D55E00",    # vermilion
    "accent": "#E69F00",     # orange
    "good": "#009E73",       # green
    "neutral": "#999999",    # gray
}
