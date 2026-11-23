"""
ui_text.py - fixed text and styling for the user interface.

Kept apart from app.py so that app.py stays short and only describes the
layout and the button actions.
"""

import config

# One-line explanation shown under each score card (honest labeling, Section 7).
SCORE_INFO = {
    "intelligibility": ("Intelligibility", "Did Whisper hear the right words? Does not depend on the accent."),
    "reference_match": ("Reference match", "How close your sound is to the chosen reference voice, not correctness."),
    "intonation_match": ("Intonation match", "How similar your pitch melody is to the reference."),
    "pace_match": ("Pace match", "Your speaking rate compared with the reference (0.8x to 1.25x is full marks)."),
}

DIFF_COLORS = {
    "correct": config.COLORS["good"],
    "wrong": config.COLORS["learner"],
    "missing": config.COLORS["accent"],
    "extra": "#56B4E9",  # Okabe-Ito sky blue
}

CSS = """
.cards {display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;}
.card {background: #ffffff; color: #111111; border: 2px solid #0072B2; border-radius: 10px;
       padding: 10px; text-align: center;}
.card.overall {border-color: #009E73; background: #eefaf5;}
.card .label {font-size: 15px; font-weight: 700;}
.card .value {font-size: 34px; font-weight: 800; margin: 4px 0;}
.card .info {font-size: 13px; color: #333333;}
@media (max-width: 900px) {.cards {grid-template-columns: repeat(2, 1fr);}}
"""


def score_cards_html(scores):
    """Return HTML for the five score cards (four sub-scores and the overall score)."""
    parts = []
    for key, (label, info) in SCORE_INFO.items():
        value = scores.get(key)
        shown = "n/a" if value is None else f"{value:.0f}"
        if value is None and key == "intonation_match":
            info = "Not enough voiced speech to compare pitch."
        parts.append(f'<div class="card"><div class="label">{label}</div>'
                     f'<div class="value">{shown}</div><div class="info">{info}</div></div>')
    parts.append(f'<div class="card overall"><div class="label">Overall coach score</div>'
                 f'<div class="value">{scores["overall"]:.0f}</div>'
                 f'<div class="info">Weighted: 50% intelligibility, 25% reference, '
                 f'15% intonation, 10% pace.</div></div>')
    return '<div class="cards">' + "".join(parts) + "</div>"


def diff_to_highlights(diff):
    """Turn the word diff into (text, category) pairs for gr.HighlightedText."""
    out = []
    for status, ref_word, heard in diff:
        if status == "correct":
            out.append((ref_word + " ", "correct"))
        elif status == "wrong":
            out.append((f"{ref_word} (heard: {heard}) ", "wrong"))
        elif status == "missing":
            out.append((ref_word + " ", "missing"))
        else:
            out.append((heard + " ", "extra"))
    return out


ABOUT = f"""
## How Pronunciation Coach works

1. **Reference.** The target sentence is spoken by a text-to-speech (TTS) voice in the chosen
   accent (gTTS), or you upload your own reference recording. It is cached in `data/cache/`.
2. **Preprocessing.** Both recordings are converted to mono, resampled to {config.SAMPLE_RATE} Hz,
   trimmed (silence below {config.TRIM_TOP_DB} dB) and peak-normalized to {config.PEAK}.
3. **Features.** Mel-frequency cepstral coefficients (MFCC, {config.N_MFCC} per 25 ms frame,
   10 ms hop) with cepstral mean and variance normalization (CMVN); pitch from pyin in semitones
   relative to each speaker's median.
4. **Alignment.** Dynamic time warping (DTW) with cosine distance matches every frame of your
   speech to a frame of the reference.
5. **Speech-to-text.** Whisper (faster-whisper, CPU, int8) transcribes your speech **without**
   being told the target sentence, so mispronounced words are not hidden.

## What the scores mean
| Score | Measures |
|---|---|
| Intelligibility | 100 x (1 - word error rate). The fairest score across accents. |
| Reference match | Normalized DTW cost mapped linearly: calibrated "good" cost = 100, "bad" = 0. |
| Intonation match | Pearson correlation of the aligned pitch contours (voiced frames only). |
| Pace match | 100 when your duration is 0.8x to 1.25x the reference, 0 at 0.5x or 2x. |

## Limitations (please read)
- No model is trained on accents. "Reference match" means close to **one** reference voice,
  not "correct". Accents near the reference score higher.
- Whisper accuracy changes with accent, microphone and background noise.
- Intonation and pace differ naturally between accents.
- This is a practice tool, not a high-stakes assessment.

## Privacy
Only scores, transcripts and timestamps are stored in `data/history.csv`. Your audio is saved
only if you tick **Save my recordings**.
"""
