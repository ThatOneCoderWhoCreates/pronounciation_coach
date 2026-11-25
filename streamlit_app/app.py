"""
streamlit_app/app.py - the same Pronunciation Coach, with a Streamlit interface, for free
public hosting on Streamlit Community Cloud. Run locally with:

    streamlit run streamlit_app/app.py

Like app.py (the Gradio version), this file is interface only: all signal processing and
AI work is done by pipeline.py and the modules it calls.
"""

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project modules live one folder up

import config  # noqa: E402
import history  # noqa: E402
import pipeline  # noqa: E402
import plots  # noqa: E402
import stt  # noqa: E402
import ui_text  # noqa: E402
from audio_io import CoachError  # noqa: E402
from stt import SttUnavailable  # noqa: E402

st.set_page_config(page_title="Pronunciation Coach", page_icon="🗣️", layout="wide")
st.markdown(f"<style>{ui_text.CSS}</style>", unsafe_allow_html=True)
SENTENCES = json.loads(config.SENTENCES_FILE.read_text(encoding="utf-8"))
ACCENTS = list(config.ACCENTS)


@st.cache_resource(show_spinner="Loading the Whisper speech recognition model (first visit only)...")
def whisper():
    """Load Whisper once per server; every visitor shares it."""
    return stt.get_model(config.WHISPER_MODEL)


@st.cache_data(show_spinner=False)
def reference(sentence, accent):
    """Reference voice for a sentence and accent (cached)."""
    path, message, source = pipeline.get_reference(sentence, accent, "auto", None)
    return str(path), message, source


def save_upload(data, suffix):
    """Write uploaded or recorded audio bytes to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def transcript_html(diff):
    """Colored word chips: green correct, red wrong, gray missing, blue extra."""
    colors = ui_text.DIFF_COLORS
    chips = []
    for text, status in ui_text.diff_to_highlights(diff):
        chips.append(f'<span style="border:2px solid {colors[status]};border-radius:8px;padding:3px 8px;'
                     f'margin:3px;display:inline-block">{text.strip()}</span>')
    return "<div>" + "".join(chips) + "</div>"


st.title("Pronunciation Coach")
st.caption("Pick a sentence and an accent, listen, record yourself, then press Analyze. "
           "Speech Processing & Recognition (MAI522), CIA III project.")
practice, progress, about = st.tabs(["Practice", "Progress", "How it works"])

with practice:
    left, right = st.columns([3, 2])
    with left:
        sentence = st.selectbox("Practice sentence", SENTENCES)
        custom = st.text_input("Or type your own sentence (press Enter)")
        sentence = custom.strip() or sentence
    with right:
        accent = st.radio("Target accent", ACCENTS, horizontal=True)

    try:
        ref_path, ref_message, ref_source = reference(sentence, accent)
        st.markdown(f"**Reference:** {sentence}")
        st.audio(ref_path)
        st.caption(ref_message)
    except CoachError as exc:
        ref_path = None
        st.error(str(exc))

    st.subheader("Your attempt")
    recorded = st.audio_input("Press the microphone, read the sentence, press stop", sample_rate=16000)
    uploaded = st.file_uploader("Or upload a recording", type=["wav", "mp3", "m4a", "ogg", "webm", "flac"])
    demos = {Path(r[0]).name: r for r in pipeline.demo_examples()}
    demo_name = st.selectbox("Or use a demo recording (for the weather sentence)", ["(none)"] + list(demos))

    if st.button("Analyze", type="primary", use_container_width=True, disabled=ref_path is None):
        if demo_name != "(none)":
            learner, sentence, accent = demos[demo_name][0], demos[demo_name][1], demos[demo_name][2]
            ref_path, ref_message, ref_source = reference(sentence, accent)
        elif recorded is not None:
            learner = save_upload(recorded.getvalue(), ".wav")
        elif uploaded is not None:
            learner = save_upload(uploaded.getvalue(), Path(uploaded.name).suffix or ".wav")
        else:
            learner = None
            st.warning("Record or upload your attempt first.")
        if learner:
            try:
                whisper()
                with st.spinner("Analyzing..."):
                    r = pipeline.analyze(sentence, ref_path, learner, accent, config.WHISPER_MODEL,
                                         "Prefer not to say", ref_source, "auto", False)
                st.session_state["result"] = r
            except (CoachError, SttUnavailable) as exc:
                st.error(str(exc))
            except Exception as exc:  # friendly message, never a stack trace
                st.error(f"Something unexpected went wrong ({type(exc).__name__}). Please try again.")

    r = st.session_state.get("result")
    if r:
        st.markdown(ui_text.score_cards_html(r["scores"]), unsafe_allow_html=True)
        st.markdown("**Transcript** (green correct, red wrong, gray missing, blue extra)")
        st.markdown(transcript_html(r["diff"]), unsafe_allow_html=True)
        st.markdown(f"**Whisper heard:** {r['transcript'] or '(nothing)'}  \n"
                    f"Word error rate {r['wer']:.2f} | DTW cost {r['dtw_cost']:.3f} | "
                    f"voiced pitch pairs {r['voiced_pairs']} | speaking-rate ratio {r['rate_ratio']:.2f}x")
        names = ["Waveforms", "Spectrograms", "Pitch contours", "DTW alignment", "Scores"]
        for tab, fig in zip(st.tabs(names), pipeline.make_figures(r)):
            with tab:
                st.pyplot(fig)

with progress:
    df = history.load_history()
    if len(df):
        st.markdown(f"**{len(df)} attempts** on this server (shared by all visitors). "
                    f"Best overall {df['overall'].max():.0f}, latest {df['overall'].iloc[-1]:.0f}.")
        st.pyplot(plots.plot_history(df))
        with st.expander("Legend: which marker is which sentence"):
            st.markdown(ui_text.legend_html(plots.history_legend(df)), unsafe_allow_html=True)
        cols = ["timestamp", "sentence", "target_accent", "intelligibility", "reference_match",
                "intonation_match", "pace_match", "overall"]
        st.dataframe(df[cols].iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No attempts yet. Go to Practice and press Analyze.")

with about:
    st.markdown(ui_text.ABOUT)
