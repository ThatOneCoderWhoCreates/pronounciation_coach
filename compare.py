"""
compare.py - comparing the learner with the target sentence and the reference.

Two kinds of comparison happen here:
    1. Text: target sentence vs what Whisper heard -> WER and word diff.
    2. Audio: reference features vs learner features -> DTW alignment and scores.
"""

import difflib
import math
import re

import jiwer
import librosa
import numpy as np

import config


# ----------------------------------------------------------------- text --
def normalize_text(s):
    """
    Make text comparable: lower-case, hyphens to spaces, drop punctuation
    except apostrophes (so "don't" stays one word), collapse spaces.
    """
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def word_error_rate(ref, hyp):
    """
    Word Error Rate (WER) = (substitutions + deletions + insertions) / words
    in the target. 0 means perfect; it can go above 1 if many words are added.
    """
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    if not ref_n:
        return 0.0
    if not hyp_n:
        return 1.0
    return float(jiwer.wer(ref_n, hyp_n))


def word_diff(ref, hyp):
    """
    Line up target words and heard words, and return a list of
    (status, ref_word, heard_word) where status is correct, wrong, missing or extra.

    difflib.SequenceMatcher finds the longest matching runs of words. Its
    opcodes are mapped as: equal -> correct, replace -> wrong (paired one to
    one; leftovers become missing or extra), delete -> missing, insert -> extra.
    """
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if tag == "equal":
            out += [("correct", r[i], h[j]) for i, j in zip(range(i1, i2), range(j1, j2))]
        elif tag == "delete":
            out += [("missing", r[i], "") for i in range(i1, i2)]
        elif tag == "insert":
            out += [("extra", "", h[j]) for j in range(j1, j2)]
        else:  # replace
            n = min(i2 - i1, j2 - j1)
            out += [("wrong", r[i1 + k], h[j1 + k]) for k in range(n)]
            out += [("missing", r[i], "") for i in range(i1 + n, i2)]
            out += [("extra", "", h[j]) for j in range(j1 + n, j2)]
    return out


# ---------------------------------------------------------------- audio --
def dtw_align(mfcc_ref, mfcc_lrn):
    """
    Dynamic Time Warping (DTW) between two MFCC sequences shaped (13, T).

    DTW finds the cheapest way to stretch and squeeze the learner in time so
    each of their frames is matched to a reference frame. Returns:
        D     accumulated cost matrix, shape (T_ref, T_learner)
        path  array of (ref_frame, learner_frame) pairs from START to END
        cost  D[-1, -1] / path length (average cost per matched pair)
    librosa returns the path from end to start, so it is reversed here.
    """
    D, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_lrn, metric=config.DTW_METRIC)
    path = wp[::-1]
    cost = float(D[-1, -1] / len(path))
    return D, path, cost


def map_to_reference_time(path, learner_values, n_ref):
    """
    Put a learner contour on the reference time axis using the DTW path.

    For each reference frame, take the median of the learner values that the
    path matched to it (ignoring NaN, i.e. unvoiced frames). Frames with no
    valid value stay NaN.
    """
    buckets = [[] for _ in range(n_ref)]
    n_lrn = len(learner_values)
    for i, j in path:
        if i < n_ref and j < n_lrn and np.isfinite(learner_values[j]):
            buckets[i].append(learner_values[j])
    return np.array([np.median(b) if b else np.nan for b in buckets])


# --------------------------------------------------------------- scores --
def intelligibility_score(wer):
    """100 x (1 - min(WER, 1)): how much of the sentence Whisper understood."""
    return 100.0 * (1.0 - min(wer, 1.0))


def reference_match_score(cost, accent):
    """
    Map the normalized DTW cost to 0..100 with a straight line: the calibrated
    good cost for this accent gives 100 and the bad cost gives 0.
    """
    good, bad = config.DTW_CALIBRATION.get(accent, config.DTW_FALLBACK)
    score = 100.0 * (bad - cost) / (bad - good)
    return float(np.clip(score, 0.0, 100.0))


def intonation_score(path, ref_semitones, lrn_semitones):
    """
    Pearson correlation between the two pitch contours after DTW alignment,
    using only frames where both are voiced. Score = 100 x max(0, r).
    Returns (score or None, number of voiced pairs used).
    """
    mapped = map_to_reference_time(path, lrn_semitones, len(ref_semitones))
    both = np.isfinite(ref_semitones) & np.isfinite(mapped)
    n_pairs = int(both.sum())
    if n_pairs < config.MIN_VOICED_PAIRS:
        return None, n_pairs
    a, b = ref_semitones[both], mapped[both]
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0, n_pairs  # a flat contour has no shape to correlate
    r = float(np.corrcoef(a, b)[0, 1])
    return 100.0 * max(0.0, r), n_pairs


def pace_score(learner_dur, ref_dur):
    """
    Speaking-rate ratio r = learner duration / reference duration.
    100 when r is between 0.8 and 1.25, falling to 0 at r = 0.5 or r = 2.
    """
    r = learner_dur / ref_dur
    lo, hi = math.log(config.PACE_OK_HIGH), math.log(config.PACE_ZERO_AT)
    penalty = max(0.0, abs(math.log(r)) - lo) / (hi - lo)
    return 100.0 * float(np.clip(1.0 - penalty, 0.0, 1.0)), r


def overall_score(scores, weights=config.WEIGHTS):
    """
    Weighted sum of the sub-scores. If a score is None (for example no
    intonation score), its weight is dropped and the others are rescaled.
    """
    used = {k: w for k, w in weights.items() if scores.get(k) is not None}
    total = sum(used.values())
    return sum(scores[k] * w for k, w in used.items()) / total
