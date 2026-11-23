"""
app.py - the Gradio web interface. Run with:  python app.py

This file contains ONLY the user interface: layout and button actions.
All signal processing and AI work happens in pipeline.py and the modules it calls.
"""

import argparse
import functools
import json
import traceback

import gradio as gr

import config
import history
import pipeline
import plots
import ui_text
from audio_io import CoachError
from stt import SttUnavailable

SENTENCES = json.loads(config.SENTENCES_FILE.read_text(encoding="utf-8"))
ACCENT_CHOICES = list(config.ACCENTS) + [config.OWN_REFERENCE]


def friendly(fn):
    """Turn any error into a short message box instead of a stack trace (Section 12)."""
    @functools.wraps(fn)
    def wrapper(*args):
        try:
            return fn(*args)
        except (CoachError, SttUnavailable) as exc:
            raise gr.Error(str(exc))
        except gr.Error:
            raise
        except Exception as exc:
            traceback.print_exc()  # full details go to the terminal for the developers
            raise gr.Error(f"Something unexpected went wrong ({type(exc).__name__}). "
                           "Please try again, or upload one of the demo files.")
    return wrapper


def status_banner():
    """One line that tells the presenter if the setup is ready."""
    s = pipeline.system_status()
    tick = lambda ok: "OK" if ok else "MISSING"  # noqa: E731
    return (f"**Setup check:** Python {s['python']} ({tick(s['python_ok'])}) | "
            f"ffmpeg {tick(s['ffmpeg'])} | Whisper {config.WHISPER_MODEL} "
            f"{'downloaded' if s['whisper_cached'] else 'NOT downloaded'} | "
            f"cached reference clips: {s['cached_references']}")


@friendly
def on_get_reference(sentence, accent, backend, own_ref):
    """Button 1: produce (or load from cache) the reference audio so it can be played."""
    path, message, _source = pipeline.get_reference(sentence, accent, backend, own_ref)
    return str(path), message


@friendly
def on_analyze(sentence, accent, backend, own_ref, learner_audio, model, learner_accent, save):
    """Button 2: score the attempt and fill in every output."""
    if not learner_audio:
        raise CoachError("Record or upload your attempt first.")
    ref_path, message, source = pipeline.get_reference(sentence, accent, backend, own_ref)
    gr.Info("Analyzing... the first run loads Whisper and can take a few seconds.")
    r = pipeline.analyze(sentence, ref_path, learner_audio, accent, model, learner_accent,
                         source, backend, save)
    details = (f"**Whisper heard:** {r['transcript'] or '(nothing)'}\n\n"
               f"Word error rate: {r['wer']:.2f} | normalized DTW cost: {r['dtw_cost']:.3f} | "
               f"voiced pitch pairs: {r['voiced_pairs']} | speaking-rate ratio: {r['rate_ratio']:.2f}x\n\n"
               f"{message}")
    return (str(ref_path), message, ui_text.score_cards_html(r["scores"]),
            ui_text.diff_to_highlights(r["diff"]), details, *pipeline.make_figures(r))


@friendly
def on_refresh():
    """Progress tab: reload the history file."""
    df = history.load_history()
    if len(df):
        summary = (f"**{len(df)} attempts.** Best overall: {df['overall'].max():.0f}. "
                   f"Latest: {df['overall'].iloc[-1]:.0f}. First: {df['overall'].iloc[0]:.0f}.")
    else:
        summary = "No attempts yet. Go to the Practice tab and press Analyze."
    cols = ["timestamp", "sentence", "target_accent", "intelligibility", "reference_match",
            "intonation_match", "pace_match", "overall"]
    return plots.plot_history(df), df[cols].iloc[::-1], summary


@friendly
def on_clear():
    """Progress tab: delete all saved attempts."""
    n = history.clear_history()
    gr.Info(f"Removed {n} attempts.")
    return on_refresh()


def demo_examples():
    """Rows for gr.Examples from demo/demo_files.json (made by scripts/make_demo_audio.py)."""
    index = config.DEMO_DIR / "demo_files.json"
    if not index.exists():
        return []
    rows = json.loads(index.read_text(encoding="utf-8"))
    return [[str(config.DEMO_DIR / r["file"]), r["sentence"], r["accent"], r["description"]]
            for r in rows if (config.DEMO_DIR / r["file"]).exists()]


def build_ui():
    """Lay out the three tabs and connect the buttons to their functions."""
    with gr.Blocks(title="Pronunciation Coach") as demo:
        gr.Markdown("# Pronunciation Coach\nPick a sentence and an accent, listen, record "
                    "yourself, then press **Analyze**.")
        gr.Markdown(status_banner())
        with gr.Tab("Practice"):
            with gr.Row():
                sentence = gr.Dropdown(SENTENCES, value=SENTENCES[0], allow_custom_value=True,
                                       label="Practice sentence (pick one or type your own)", scale=3)
                accent = gr.Dropdown(ACCENT_CHOICES, value=config.DEFAULT_ACCENT, label="Target accent")
            with gr.Row():
                backend = gr.Radio(config.TTS_BACKENDS, value="auto", label="Reference voice engine")
                model = gr.Dropdown(config.WHISPER_CHOICES, value=config.WHISPER_MODEL,
                                    label="Whisper model (larger = slower, copes better with accents)")
                learner_accent = gr.Dropdown(config.LEARNER_ACCENTS, value="Prefer not to say",
                                             label="Your accent (optional, stored locally)")
            own_ref = gr.Audio(sources=["upload"], type="filepath", visible=False,
                               label="Your own reference recording (teacher or native speaker)")
            with gr.Row():
                ref_btn = gr.Button("1. Get reference", variant="secondary")
                ref_audio = gr.Audio(type="filepath", label="Reference audio", interactive=False)
            ref_msg = gr.Markdown()
            learner = gr.Audio(sources=["microphone", "upload"], type="filepath",
                               label="2. Your attempt (record or upload)")
            save = gr.Checkbox(value=config.SAVE_RECORDINGS, label="Save my recordings (off by default)")
            analyze_btn = gr.Button("3. Analyze", variant="primary", size="lg")
            examples = demo_examples()
            if examples:
                note = gr.Textbox(label="What this demo file shows", interactive=False)
                gr.Examples(examples, inputs=[learner, sentence, accent, note],
                            label="Demo recordings (use these if the microphone fails)")
            cards = gr.HTML()
            transcript = gr.HighlightedText(label="Transcript (green correct, red wrong, "
                                            "gray missing, blue extra)",
                                            color_map=ui_text.DIFF_COLORS, show_legend=True)
            details = gr.Markdown()
            figs = [gr.Plot(label=name) for name in
                    ["Waveforms", "Spectrograms", "Pitch contours", "DTW alignment", "Scores"]]
        with gr.Tab("Progress"):
            summary = gr.Markdown()
            hist_plot = gr.Plot(label="Progress")
            table = gr.Dataframe(label="All attempts (newest first)", interactive=False)
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                clear_btn = gr.Button("Clear history", variant="stop")
        with gr.Tab("How it works"):
            gr.Markdown(ui_text.ABOUT)

        accent.change(lambda a: gr.update(visible=a == config.OWN_REFERENCE), accent, own_ref)
        ref_btn.click(on_get_reference, [sentence, accent, backend, own_ref], [ref_audio, ref_msg])
        analyze_btn.click(on_analyze,
                          [sentence, accent, backend, own_ref, learner, model, learner_accent, save],
                          [ref_audio, ref_msg, cards, transcript, details, *figs]
                          ).then(on_refresh, None, [hist_plot, table, summary])
        refresh_btn.click(on_refresh, None, [hist_plot, table, summary])
        clear_btn.click(on_clear, None, [hist_plot, table, summary])
        demo.load(on_refresh, None, [hist_plot, table, summary])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pronunciation Coach")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    build_ui().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser,
                      theme=gr.themes.Soft(), css=ui_text.CSS,
                      allowed_paths=[str(config.DATA_DIR), str(config.DEMO_DIR)])
