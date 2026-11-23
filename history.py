"""
history.py - saving every attempt to a CSV file for the Progress tab.

Only numbers, text and timestamps are stored. Learner audio is never written
here (it is saved separately, and only if the learner opts in).
"""

import pandas as pd

import config

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
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_history(path=None):
    """Return all attempts as a DataFrame (empty, with the right columns, if none)."""
    path = path or config.HISTORY_FILE
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=COLUMNS)


def clear_history(path=None):
    """Delete the history file. Returns the number of attempts that were removed."""
    path = path or config.HISTORY_FILE
    n = len(load_history(path))
    path.unlink(missing_ok=True)
    return n
