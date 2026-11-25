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
    "correct": "#009E73",  # Okabe-Ito green
    "wrong": "#D55E00",    # vermilion
    "missing": "#999999",  # gray
    "extra": "#56B4E9",    # sky blue
}

# Score colors on the dark background: green = good, amber = fair, red = needs work.
LEVELS = [(80, "#34D399", "Good"), (50, "#FBBF24", "Fair"), (0, "#F87171", "Needs work")]

# The app always uses Gradio's dark theme (Gradio switches theme with ?__theme=dark).
FORCE_DARK_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "dark") {
        url.searchParams.set("__theme", "dark");
        window.location.replace(url.href);
    }
}
"""



def restart_js(elem_id):
    """
    JavaScript for a "play from the start" button. Gradio's player keeps its real
    <audio> element inside a shadow root, so we search there too, jump to 0 s and play.
    """
    return f"""
() => {{
    const root = document.querySelector('#{elem_id}');
    if (!root) return;
    const players = [...root.querySelectorAll('audio')];
    root.querySelectorAll('*').forEach(el => {{
        if (el.shadowRoot) players.push(...el.shadowRoot.querySelectorAll('audio'));
    }});
    players.filter(a => a.src).forEach(a => {{ a.currentTime = 0; a.play(); }});
}}
"""


CSS = """
/* ---------- score cards: dark cards, bright text ---------- */
.cards {display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;}
.card {background: #111827; color: #F9FAFB; border: 2px solid var(--level); border-radius: 12px;
       padding: 14px 10px; text-align: center;}
.card.overall {background: #0B2A22; border-width: 3px;}
.card .label {font-size: 16px; font-weight: 700; color: #F9FAFB;}
.card .value {font-size: 40px; font-weight: 800; margin: 4px 0 0 0; color: var(--level);}
.card .level {font-size: 13px; font-weight: 700; color: var(--level); margin-bottom: 6px;}
.card .info {font-size: 13px; color: #D1D5DB;}
@media (max-width: 900px) {.cards {grid-template-columns: repeat(2, 1fr);}}

/* ---------- recording status banner ---------- */
.rec-status {text-align: center; font-size: 18px; font-weight: 700; padding: 12px;
             border-radius: 12px; border: 2px solid; margin: 4px auto;}
.rec-idle {color: #E5E7EB; border-color: #4B5563; background: #111827;}
.rec-live {color: #FECACA; border-color: #EF4444; background: #3B0D0D;}
.rec-ready {color: #A7F3D0; border-color: #10B981; background: #052E22;}
.rec-dot {display: inline-block; width: 14px; height: 14px; border-radius: 50%;
          background: #EF4444; margin-right: 10px; vertical-align: middle;
          animation: rec-pulse 1s ease-in-out infinite;}
@keyframes rec-pulse {0%, 100% {opacity: 1; transform: scale(1);} 50% {opacity: 0.25; transform: scale(0.75);}}

/* ---------- centered, larger record controls ---------- */
#learner-audio .controls {display: flex; flex-direction: column; align-items: center; gap: 10px;}
#learner-audio .controls .wrapper {justify-content: center;}
#learner-audio .record-button, #learner-audio .stop-button, #learner-audio .stop-button-paused {
    font-size: 18px; font-weight: 700; padding: 12px 34px; border-radius: 999px;
    width: auto; min-width: 150px; height: auto; justify-content: center;}
#learner-audio .record-button {background: #DC2626; color: #FFFFFF; border: none;}
#learner-audio .stop-button, #learner-audio .stop-button-paused {
    background: #111827; color: #FCA5A5; border: 2px solid #EF4444;}
#learner-audio .mic-select {max-width: 360px;}
#analyze-btn {font-size: 20px;}
#restart-ref {max-width: 280px; margin: 0 auto;}

/* ---------- Progress plot legend (inside a section that opens and closes) ---------- */
.legend {color: #F3F4F6; font-size: 15px;}
.legend-row {padding: 3px 0;}
.legend-mark {display: inline-block; width: 28px; font-size: 20px; text-align: center;}
"""


def score_level(value):
    """Return (color, word) for a score: Good (80+), Fair (50 to 79), Needs work (below 50)."""
    for threshold, color, word in LEVELS:
        if value >= threshold:
            return color, word
    return LEVELS[-1][1], LEVELS[-1][2]


def _card(label, value, info, extra_class=""):
    """HTML for one score card. The number and the border take the level color."""
    if value is None:
        color, word, shown = "#9CA3AF", "Not available", "n/a"
    else:
        (color, word), shown = score_level(value), f"{value:.0f}"
    return (f'<div class="card {extra_class}" style="--level: {color}"><div class="label">{label}</div>'
            f'<div class="value">{shown}</div><div class="level">{word}</div>'
            f'<div class="info">{info}</div></div>')


def score_cards_html(scores):
    """Return HTML for the five score cards (four sub-scores and the overall score)."""
    parts = []
    for key, (label, info) in SCORE_INFO.items():
        if scores.get(key) is None and key == "intonation_match":
            info = "Not enough voiced speech to compare pitch."
        parts.append(_card(label, scores.get(key), info))
    parts.append(_card("Overall coach score", scores["overall"],
                       "Weighted: 50% intelligibility, 25% reference, 15% intonation, 10% pace.",
                       "overall"))
    return '<div class="cards">' + "".join(parts) + "</div>"


def legend_html(items):
    """HTML legend for the Progress plot: the marker symbol in its color, then the sentence."""
    if not items:
        return '<div class="legend">No attempts yet.</div>'
    rows = "".join(f'<div class="legend-row"><span class="legend-mark" style="color: {color}">{symbol}</span>'
                   f"{sentence}</div>" for symbol, color, sentence in items)
    return f'<div class="legend">{rows}</div>'


def recording_status(state, seconds=None):
    """HTML banner that shows whether the microphone is recording right now."""
    if state == "live":
        return ('<div class="rec-status rec-live"><span class="rec-dot"></span>'
                "Recording... read the sentence, then press Stop</div>")
    if state == "ready":
        length = f" ({seconds:.1f} s)" if seconds else ""
        return f'<div class="rec-status rec-ready">Attempt ready{length}. Press Analyze.</div>'
    return ('<div class="rec-status rec-idle">Not recording. Press Record and read the sentence aloud, '
            "or upload a file.</div>")


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
