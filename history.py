"""
history.py - saving every attempt to a CSV file for the Progress tab.

Only numbers, text and timestamps are stored. Learner audio is never written
here (it is saved separately, and only if the learner opts in).
"""

import pandas as pd

import config
from audio_io import CoachError

COLUMNS = [
    "timestamp", "sentence", "target_accent", "learner_accent", "reference_source",
    "tts_backend", "transcript", "wer", "intelligibility", "reference_match",
    "intonation_match", "pace_match", "overall", "ref_duration", "learner_duration",
]


def append_attempt(row, path=None):
    """Add one attempt (a dictionary with the COLUMNS keys) to the CSV file."""
    path = path or config.HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{c: row.get(c) for c in COLUMNS}], columns=COLUMNS)
    try:
        frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
    except PermissionError as exc:
        # Windows locks a CSV file while Excel has it open.
        raise CoachError("Could not save this attempt because data/history.csv is open in "
                         "another program (for example Excel). Close it and press Analyze again.") from exc


def load_history(path=None):
    """Return all attempts as a DataFrame (empty, with the right columns, if none)."""
    path = path or config.HISTORY_FILE
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=COLUMNS)


def recording_name(timestamp):
    """File name of a saved recording: the attempt's timestamp with ':' replaced
    by '-' (Windows does not allow ':' in file names)."""
    return str(timestamp).replace(":", "-") + ".wav"


def list_recordings(folder=None, path=None):
    """
    Return the saved recordings, newest first, as (label, file path) pairs.
    The label shows the time, the sentence and the overall score of the
    attempt the recording belongs to (matched through the timestamp).
    """
    folder = folder or config.RECORDINGS_DIR
    if not folder.exists():
        return []
    attempts = {recording_name(r.timestamp): r for r in load_history(path).itertuples()}
    out = []
    for wav in sorted(folder.glob("*.wav"), reverse=True):
        row = attempts.get(wav.name)
        when = wav.stem[:10] + " " + wav.stem[11:].replace("-", ":")
        if row is not None:
            sentence = row.sentence if len(row.sentence) <= 45 else row.sentence[:42] + "..."
            out.append((f"{when} | {sentence} | overall {row.overall:.0f}", str(wav)))
        else:
            out.append((when, str(wav)))
    return out


def clear_history(path=None):
    """Delete the history file. Returns the number of attempts that were removed."""
    path = path or config.HISTORY_FILE
    n = len(load_history(path))
    try:
        path.unlink(missing_ok=True)
    except PermissionError as exc:
        raise CoachError("data/history.csv is open in another program. Close it and try again.") from exc
    return n
