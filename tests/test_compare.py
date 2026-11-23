"""Tests for compare.py: text comparison, DTW orientation and every score formula."""

import math

import librosa
import numpy as np
import pytest

import compare
import config
import features
from conftest import synth_utterance


# ----------------------------------------------------------------- text --
def test_normalize_text():
    assert compare.normalize_text("  Let's GO -- to the well-known park!! ") == "let's go to the well known park"


def test_wer():
    assert compare.word_error_rate("the cat sat", "The cat, sat.") == 0.0
    assert compare.word_error_rate("the cat sat", "the bat sat") == pytest.approx(1 / 3)
    assert compare.word_error_rate("the cat sat", "") == 1.0


def test_word_diff_statuses():
    diff = compare.word_diff("I want three red apples", "I want tree apples now")
    assert ("wrong", "three", "tree") in diff
    assert ("missing", "red", "") in diff
    assert ("extra", "", "now") in diff
    assert diff[0] == ("correct", "i", "i")


def test_word_diff_replace_with_leftovers():
    diff = compare.word_diff("a b c d", "a x d")
    assert diff == [("correct", "a", "a"), ("wrong", "b", "x"), ("missing", "c", ""), ("correct", "d", "d")]


# ---------------------------------------------------------------- audio --
def test_dtw_identical_has_zero_cost_and_diagonal_path():
    m = features.mfcc_cmvn(synth_utterance())
    D, path, cost = compare.dtw_align(m, m)
    assert cost < 1e-6
    assert np.all(path[:, 0] == path[:, 1])


def test_dtw_path_orientation():
    """Path must go from (0, 0) to (last ref frame, last learner frame), never backwards."""
    ref = features.mfcc_cmvn(synth_utterance(seg_dur=0.2))
    lrn = features.mfcc_cmvn(synth_utterance(seg_dur=0.35))
    D, path, _ = compare.dtw_align(ref, lrn)
    assert D.shape == (ref.shape[1], lrn.shape[1])
    assert tuple(path[0]) == (0, 0)
    assert tuple(path[-1]) == (ref.shape[1] - 1, lrn.shape[1] - 1)
    assert np.all(np.diff(path, axis=0) >= 0)


def test_dtw_same_words_cheaper_than_different_words():
    y = synth_utterance("aiueo")
    ref = features.mfcc_cmvn(y)
    slow = features.mfcc_cmvn(librosa.effects.time_stretch(y, rate=0.8))
    other = features.mfcc_cmvn(synth_utterance("uoiae"))
    assert compare.dtw_align(ref, slow)[2] < compare.dtw_align(ref, other)[2]


def test_map_to_reference_time():
    path = np.array([[0, 0], [0, 1], [1, 2], [2, 3]])
    out = compare.map_to_reference_time(path, np.array([1.0, 3.0, np.nan, 5.0]), 3)
    assert out[0] == 2.0 and np.isnan(out[1]) and out[2] == 5.0


# --------------------------------------------------------------- scores --
def test_intelligibility():
    assert compare.intelligibility_score(0.0) == 100
    assert compare.intelligibility_score(0.25) == 75
    assert compare.intelligibility_score(1.7) == 0


def test_reference_match_is_linear_and_clipped(monkeypatch):
    monkeypatch.setitem(config.DTW_CALIBRATION, "US English", (0.2, 0.6))
    assert compare.reference_match_score(0.2, "US English") == 100
    assert compare.reference_match_score(0.4, "US English") == pytest.approx(50)
    assert compare.reference_match_score(0.9, "US English") == 0
    assert compare.reference_match_score(0.1, "US English") == 100


def test_intonation_identical_inverted_and_unvoiced():
    contour = np.sin(np.linspace(0, 3, 100)) * 3
    path = np.stack([np.arange(100), np.arange(100)], axis=1)
    assert compare.intonation_score(path, contour, contour)[0] == pytest.approx(100)
    assert compare.intonation_score(path, contour, -contour)[0] == 0
    score, pairs = compare.intonation_score(path, contour, np.full(100, np.nan))
    assert score is None and pairs == 0


@pytest.mark.parametrize("ratio, expected", [(1.0, 100), (1.25, 100), (0.8, 100), (2.0, 0), (0.5, 0), (3.0, 0)])
def test_pace(ratio, expected):
    assert compare.pace_score(ratio * 2.0, 2.0)[0] == pytest.approx(expected, abs=1e-6)


def test_pace_halfway():
    mid = math.exp((math.log(1.25) + math.log(2.0)) / 2)
    assert compare.pace_score(mid, 1.0)[0] == pytest.approx(50)


def test_overall_weights_and_renormalization():
    all_100 = {"intelligibility": 100, "reference_match": 100, "intonation_match": 100, "pace_match": 100}
    assert compare.overall_score(all_100) == pytest.approx(100)
    mixed = {"intelligibility": 100, "reference_match": 0, "intonation_match": None, "pace_match": 0}
    assert compare.overall_score(mixed) == pytest.approx(100 * 0.5 / 0.85)
