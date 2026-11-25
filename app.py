"""
app.py - the Gradio web interface. Run with:  python app.py

This file contains ONLY the user interface: layout and button actions.
All signal processing and AI work happens in pipeline.py and the modules it calls.
"""

import argparse
import functools
import json
import os
import threading
import traceback

import gradio as gr
import soundfile as sf

import config
import history
import pipeline
import plots
import ui_text
from audio_io import CoachError
from stt import SttUnavailable

# True when running on Hugging Face Spaces (the platform sets SPACE_ID).
ON_SPACE = bool(os.environ.get("SPACE_ID"))
SENTENCES = json.loads(config.SENTENCES_FILE.read_text(encoding="utf-8"))
ACCENT_CHOICES = list(config.ACCENTS) + [config.OWN_REFERENCE]


def friendly(fn):
    """Turn any error into a short message box instead of a stack trace."""
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


def on_reference(sentence, accent, backend, own_ref):
    """
    Runs by itself whenever the sentence, accent, engine or own reference changes,
    so there is no button to press. Problems are shown as text under the player
    instead of a pop-up, because this runs without the learner asking for it.
    """
    try:
        path, message, _source = pipeline.get_reference(sentence, accent, backend, own_ref)
        return str(path), message
    except CoachError as exc:
        return None, f"**Reference not ready:** {exc}"
    except Exception as exc:
        traceback.print_exc()
        return None, f"**Reference not ready:** unexpected {type(exc).__name__}."


def on_recording_changed(audio_path):
    """Show 'ready' (with the length) when an attempt has been recorded or uploaded."""
    if not audio_path:
        return gr.update()  # keep the current banner (for example "Recording...")
    try:
        seconds = sf.info(audio_path).duration
    except Exception:
        seconds = None
    return ui_text.recording_status("ready", seconds)


@friendly
def on_analyze(sentence, accent, backend, own_ref, learner_audio, model, learner_accent, save):
    """Analyze button: score the attempt and fill in every output."""
    if not learner_audio:
        raise CoachError("Record or upload your attempt first.")
    ref_path, message, source = pipeline.get_reference(sentence, accent, backend, own_ref)
    gr.Info("Analyzing... the first run loads Whisper and can take a few seconds.")
    r = pipeline.analyze(sentence, ref_path, learner_audio, accent, model, learner_accent,
                         source, backend, save)
    saved = (f"\n\nRecording saved as `{r['recording_path'].name}` (see **Progress > Saved recordings**)."
             if r["recording_path"] else "")
    details = (f"**Whisper heard:** {r['transcript'] or '(nothing)'}\n\n"
               f"Word error rate: {r['wer']:.2f} | normalized DTW cost: {r['dtw_cost']:.3f} | "
               f"voiced pitch pairs: {r['voiced_pairs']} | speaking-rate ratio: {r['rate_ratio']:.2f}x"
               f"\n\n{message}{saved}")
    return (str(ref_path), message, ui_text.score_cards_html(r["scores"]),
            ui_text.diff_to_highlights(r["diff"]), details, *pipeline.make_figures(r))


@friendly
def on_refresh():
    """Progress tab: reload the history file and the list of saved recordings."""
    df = history.load_history()
    if len(df):
        summary = (f"**{len(df)} attempts.** Best overall: {df['overall'].max():.0f}. "
                   f"Latest: {df['overall'].iloc[-1]:.0f}. First: {df['overall'].iloc[0]:.0f}.")
    else:
        summary = "No attempts yet. Go to the Practice tab and press Analyze."
    cols = ["timestamp", "sentence", "target_accent", "intelligibility", "reference_match",
            "intonation_match", "pace_match", "overall"]
    recs = history.list_recordings()
    first = recs[0][1] if recs else None
    legend = ui_text.legend_html(plots.history_legend(df))
    return (plots.plot_history(df), legend, df[cols].iloc[::-1], summary,
            gr.update(choices=recs, value=first), first)


@friendly
def on_clear():
    """Progress tab: delete all saved attempts (saved recordings are kept)."""
    n = history.clear_history()
    gr.Info(f"Removed {n} attempts. Saved recordings were kept in data/recordings/.")
    return on_refresh()


def build_ui():
    """Lay out the three tabs and connect the controls to their functions."""
    with gr.Blocks(title="Pronunciation Coach") as demo:
        gr.Markdown("# Pronunciation Coach\nPick a sentence and an accent, listen to the reference, "
                    "record yourself, then press **Analyze**.")
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
            ref_audio = gr.Audio(type="filepath", interactive=False, elem_id="ref-audio",
                                 label="Reference audio (updates automatically when you change the sentence or accent)")
            restart_btn = gr.Button("Play reference from the start", size="sm", elem_id="restart-ref")
            ref_msg = gr.Markdown()
            with gr.Row():
                gr.Column(scale=1, min_width=0)
                with gr.Column(scale=3):
                    rec_status = gr.HTML(ui_text.recording_status("idle"))
                    learner = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                       elem_id="learner-audio", label="1. Your attempt (record or upload)")
                    # Hidden on the public server, so visitors' audio is never kept there.
                    save = gr.Checkbox(value=False if ON_SPACE else config.SAVE_RECORDINGS, visible=not ON_SPACE,
                                       label="Save my recordings (off by default). They are kept in "
                                             "data/recordings/ and listed in the Progress tab.")
                    analyze_btn = gr.Button("2. Analyze", variant="primary", size="lg", elem_id="analyze-btn")
                gr.Column(scale=1, min_width=0)
            examples = pipeline.demo_examples()
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
            with gr.Accordion("Legend: which marker is which sentence", open=False):
                legend_box = gr.HTML()
            table = gr.Dataframe(label="All attempts (newest first)", interactive=False)
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                clear_btn = gr.Button("Clear history", variant="stop")
            gr.Markdown("### Saved recordings\nOnly attempts analyzed with **Save my recordings** "
                        "ticked appear here. The files are in `data/recordings/`.")
            rec_pick = gr.Dropdown(label="Choose a recording", choices=[])
            rec_player = gr.Audio(type="filepath", interactive=False, label="Saved recording")
        with gr.Tab("How it works"):
            gr.Markdown(ui_text.ABOUT)

        # The reference updates itself whenever one of its inputs changes.
        ref_inputs, ref_outputs = [sentence, accent, backend, own_ref], [ref_audio, ref_msg]
        accent.change(lambda a: gr.update(visible=a == config.OWN_REFERENCE), accent, own_ref)
        for control in (sentence, accent, backend, own_ref):
            control.change(on_reference, ref_inputs, ref_outputs)

        # Runs only in the browser (fn=None): rewinds the reference player and plays it.
        restart_btn.click(None, None, None, js=ui_text.restart_js("ref-audio"))

        # Recording indicator.
        learner.start_recording(lambda: ui_text.recording_status("live"), None, rec_status)
        learner.change(on_recording_changed, learner, rec_status)
        learner.clear(lambda: ui_text.recording_status("idle"), None, rec_status)

        progress_outputs = [hist_plot, legend_box, table, summary, rec_pick, rec_player]
        analyze_btn.click(on_analyze,
                          [sentence, accent, backend, own_ref, learner, model, learner_accent, save],
                          [ref_audio, ref_msg, cards, transcript, details, *figs]
                          ).then(on_refresh, None, progress_outputs)
        refresh_btn.click(on_refresh, None, progress_outputs)
        clear_btn.click(on_clear, None, progress_outputs)
        rec_pick.change(lambda path: path, rec_pick, rec_player)
        demo.load(on_reference, ref_inputs, ref_outputs)
        demo.load(on_refresh, None, progress_outputs)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pronunciation Coach")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--share", action="store_true", help="also create a temporary public link")
    args = parser.parse_args()
    if ON_SPACE:
        # Public server: listen on all addresses and download Whisper now, not on the first click.
        import stt
        threading.Thread(target=stt.get_model, args=(config.WHISPER_MODEL,), daemon=True).start()
    build_ui().launch(server_name="0.0.0.0" if ON_SPACE else "127.0.0.1",
                      server_port=7860 if ON_SPACE else args.port,
                      inbrowser=not (args.no_browser or ON_SPACE), share=args.share and not ON_SPACE,
                      theme=gr.themes.Soft(), css=ui_text.CSS, js=ui_text.FORCE_DARK_JS,
                      allowed_paths=[str(config.DATA_DIR), str(config.DEMO_DIR)])
