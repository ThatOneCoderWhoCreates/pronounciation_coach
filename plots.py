"""
plots.py - every figure shown in the app.

Each function returns a matplotlib Figure. We create figures with
matplotlib.figure.Figure instead of pyplot, so they are never registered in
pyplot's global list: nothing has to be closed and memory does not leak when
the app runs for a long time. Agg is the non-interactive backend used on servers.
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import config  # noqa: E402
import features  # noqa: E402

C = config.COLORS
SR, HOP = config.SAMPLE_RATE, config.HOP

# One (matplotlib marker, same shape as a text symbol, color) per sentence in the
# Progress plot. The text symbol lets the app show the legend outside the plot.
HISTORY_STYLES = [
    ("o", "\u25CF", "#56B4E9"), ("s", "\u25A0", "#E69F00"), ("^", "\u25B2", "#2BC48A"),
    ("D", "\u25C6", "#F0E442"), ("v", "\u25BC", "#CC79A7"), ("P", "\u271A", "#FF7F50"),
    ("X", "\u2716", "#F3F4F6"), ("*", "\u2605", "#B39DDB"), ("h", "\u2B22", "#4DD0E1"),
    ("<", "\u25C0", "#FCA5A5"),
]


def _figure(rows=1, height=3.6, sharex=False):
    """Create a wide dark-themed figure with projector-friendly font sizes."""
    matplotlib.rcParams.update({
        "font.size": config.PLOT_FONT, "axes.titlesize": config.PLOT_FONT + 2,
        "axes.labelsize": config.PLOT_FONT, "legend.fontsize": config.PLOT_FONT,
        # dark background with bright text, so the plots match the app
        "figure.facecolor": config.PLOT_BG, "axes.facecolor": config.PLOT_PANEL,
        "text.color": config.PLOT_TEXT, "axes.labelcolor": config.PLOT_TEXT,
        "axes.titlecolor": config.PLOT_TEXT, "xtick.color": config.PLOT_TEXT,
        "ytick.color": config.PLOT_TEXT, "axes.edgecolor": config.PLOT_GRID,
        "legend.facecolor": config.PLOT_PANEL, "legend.edgecolor": config.PLOT_GRID,
    })
    fig = Figure(figsize=(10, height * rows), dpi=100, layout="constrained")
    axes = fig.subplots(rows, 1, sharex=sharex)
    return fig, axes


def _frame_times(n):
    """Time in seconds of each 10 ms frame."""
    return np.arange(n) * HOP / SR


def plot_waveforms(y_ref, y_lrn):
    """1. Reference and learner waveforms, one row each, time in seconds."""
    fig, (a1, a2) = _figure(rows=2, height=2.2)
    for ax, y, name, color in [(a1, y_ref, "Reference", C["reference"]), (a2, y_lrn, "You", C["learner"])]:
        ax.plot(np.arange(len(y)) / SR, y, color=color, linewidth=0.6)
        ax.set_title(f"{name}: waveform ({len(y) / SR:.2f} s after trimming)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1, 1)
    a2.set_xlabel("Time (s)")
    return fig


def plot_spectrograms(y_ref, y_lrn):
    """2. Log-mel spectrograms of both, drawn with the SAME color scale."""
    S_ref, S_lrn = features.log_mel(y_ref), features.log_mel(y_lrn)
    vmax = max(S_ref.max(), S_lrn.max())
    vmin = vmax - 80  # show an 80 dB range
    fig, (a1, a2) = _figure(rows=2, height=2.6)
    for ax, S, name in [(a1, S_ref, "Reference"), (a2, S_lrn, "You")]:
        img = ax.imshow(S, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                        extent=[0, S.shape[1] * HOP / SR, 0, S.shape[0]])
        ax.set_title(f"{name}: log-mel spectrogram")
        ax.set_ylabel("Mel bin")
    a2.set_xlabel("Time (s)")
    fig.colorbar(img, ax=[a1, a2], label="Power (dB)")
    return fig


def plot_pitch_overlay(ref_semitones, lrn_on_ref):
    """3. Pitch contours in semitones on the reference time axis (learner mapped by DTW)."""
    fig, ax = _figure(height=3.8)
    t = _frame_times(len(ref_semitones))
    ax.plot(t, ref_semitones, color=C["reference"], linewidth=2.5, label="Reference")
    ax.plot(t, lrn_on_ref, color=C["learner"], linewidth=2.0, linestyle="--", label="You (aligned by DTW)")
    ax.axhline(0, color=C["neutral"], linewidth=0.8)
    ax.set_title("Intonation: pitch contour relative to each speaker's median")
    ax.set_xlabel("Reference time (s)")
    ax.set_ylabel("Pitch (semitones)")
    ax.legend(loc="upper right")
    return fig


def plot_dtw(D, path):
    """4. DTW accumulated cost matrix with the warping path drawn on top."""
    fig, ax = _figure(height=4.6)
    n_ref, n_lrn = D.shape
    img = ax.imshow(D, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, n_lrn * HOP / SR, 0, n_ref * HOP / SR])
    ax.plot(path[:, 1] * HOP / SR, path[:, 0] * HOP / SR, color="white", linewidth=2.2, label="Warping path")
    longest = max(n_ref, n_lrn) * HOP / SR
    ax.plot([0, longest], [0, longest], color=C["accent"], linestyle=":", linewidth=1.5, label="Same speed")
    ax.set_xlim(0, n_lrn * HOP / SR)
    ax.set_ylim(0, n_ref * HOP / SR)
    ax.set_title("DTW alignment (steeper than the dotted line = you were faster there)")
    ax.set_xlabel("Your time (s)")
    ax.set_ylabel("Reference time (s)")
    ax.legend(loc="lower right")
    fig.colorbar(img, ax=ax, label="Accumulated cosine cost")
    return fig


def plot_scores(scores):
    """5. Bar chart of the four sub-scores and the overall score (0 to 100)."""
    labels = ["Intelligibility", "Reference match", "Intonation match", "Pace match", "Overall"]
    keys = ["intelligibility", "reference_match", "intonation_match", "pace_match", "overall"]
    values = [scores.get(k) for k in keys]
    fig, ax = _figure(height=3.8)
    colors = [C["reference"]] * 4 + [C["good"]]  # overall score stands out in green
    bars = ax.bar(labels, [v if v is not None else 0 for v in values], color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, (v or 0) + 2,
                "n/a" if v is None else f"{v:.0f}", ha="center", fontsize=config.PLOT_FONT + 1)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (0 to 100)")
    ax.set_title("Your scores for this attempt")
    return fig


def plot_history(df):
    """6. Overall score by attempt number, one marker shape per sentence."""
    fig, ax = _figure(height=4.2)
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "No attempts yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Progress")
        ax.set_xlabel("Attempt number")
        ax.set_ylabel("Overall score (0 to 100)")
        return fig
    attempt = np.arange(1, len(df) + 1)
    ax.plot(attempt, df["overall"], color=C["neutral"], linewidth=1.2, zorder=1)
    for k, (sentence, idx) in enumerate(df.groupby("sentence", sort=False).groups.items()):
        pos = np.asarray(df.index.get_indexer(idx))
        marker, _symbol, color = HISTORY_STYLES[k % len(HISTORY_STYLES)]
        ax.scatter(attempt[pos], df["overall"].iloc[pos], marker=marker, color=color, s=80, zorder=2)
    ax.set_ylim(0, 105)
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))  # whole attempt numbers
    ax.set_title("Progress: overall score per attempt")
    ax.set_xlabel("Attempt number")
    ax.set_ylabel("Overall score (0 to 100)")
    # No legend inside the plot: it would hide points. The app shows it below the
    # plot in a section that opens and closes (see history_legend).
    return fig


def history_legend(df):
    """(symbol, color, sentence) for each sentence, in the same order and style as plot_history."""
    if df is None or len(df) == 0:
        return []
    sentences = list(df.groupby("sentence", sort=False).groups)
    return [(HISTORY_STYLES[k % len(HISTORY_STYLES)][1], HISTORY_STYLES[k % len(HISTORY_STYLES)][2], s)
            for k, s in enumerate(sentences)]
