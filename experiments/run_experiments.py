"""
run_experiments.py - measure how the scores behave and write the tables to
experiments/results/results.md (copy them into the README).

    python experiments/run_experiments.py

E1-E5 need only reference audio. E6 needs the gTTS cache for several accents,
E7 needs downloaded Whisper models, E8 needs attempts in data/history.csv.
Experiments whose inputs are missing are skipped and say why.
"""

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as H  # noqa: E402

import compare  # noqa: E402
import config  # noqa: E402
import history  # noqa: E402
import stt  # noqa: E402
import tts  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SENTENCES = json.loads(config.SENTENCES_FILE.read_text(encoding="utf-8"))
targets = []  # (target, measured, pass?) rows for the acceptance table


def check(name, measured, ok):
    """Record one acceptance target."""
    targets.append([name, measured, "PASS" if ok else "FAIL"])


def reference(sentence, accent="US English"):
    """Prepared reference waveform plus the engine that produced it."""
    path, message = tts.synthesize(sentence, accent, backend="auto")
    return H.load(path), ("gTTS" if "gTTS" in message else "offline voice")


def e1_e4(y):
    """E1 self-comparison, E2 speed, E3 pitch, E4 noise, on sentence 1."""
    out = []
    s = H.score_pair(y, y)
    out.append("## E1. Reference compared with itself\n" + H.md_table(
        ["DTW cost", "Reference match", "Intonation match", "Pace match"],
        [[f"{s['cost']:.4f}", s["reference_match"], s["intonation_match"], s["pace_match"]]]))
    check("E1 self: reference 100, intonation >= 99, pace 100",
          f"{s['reference_match']:.0f} / {s['intonation_match']:.1f} / {s['pace_match']:.0f}",
          s["reference_match"] == 100 and s["intonation_match"] >= 99 and s["pace_match"] == 100)

    rows, by_speed = [], {}
    for speed in [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0]:
        r = H.score_pair(y, H.stretch(y, speed))
        by_speed[speed] = r
        rows.append([f"{speed}x", r["ratio"], f"{r['cost']:.3f}", r["reference_match"], r["intonation_match"], r["pace_match"]])
    out.append("## E2. Speaking speed (time stretch, pitch unchanged)\n" + H.md_table(
        ["Speed", "Duration ratio", "DTW cost", "Reference match", "Intonation", "Pace"], rows))
    check("E2 pace 100 at 0.8x and 1.25x speed", f"{by_speed[0.8]['pace_match']:.0f} / {by_speed[1.25]['pace_match']:.0f}",
          by_speed[0.8]["pace_match"] >= 99 and by_speed[1.25]["pace_match"] >= 99)
    check("E2 pace <= 10 at 0.5x and 2x speed", f"{by_speed[0.5]['pace_match']:.0f} / {by_speed[2.0]['pace_match']:.0f}",
          by_speed[0.5]["pace_match"] <= 10 and by_speed[2.0]["pace_match"] <= 10)
    check("E2 reference match >= 80 at 0.8x and 1.25x (DTW absorbs tempo)",
          f"{by_speed[0.8]['reference_match']:.0f} / {by_speed[1.25]['reference_match']:.0f}",
          min(by_speed[0.8]["reference_match"], by_speed[1.25]["reference_match"]) >= 80)

    rows, by_st = [], {}
    for st in [-4, -2, 0, 2, 4]:
        r = H.score_pair(y, H.shift(y, st))
        by_st[st] = r
        rows.append([f"{st:+d}", f"{r['cost']:.3f}", r["reference_match"], r["intonation_match"], r["pairs"]])
    out.append("## E3. Voice pitch (pitch shift, speed unchanged)\n" + H.md_table(
        ["Semitones", "DTW cost", "Reference match", "Intonation", "Voiced pairs"], rows))
    worst = min(by_st[-2]["intonation_match"] or 0, by_st[2]["intonation_match"] or 0)
    check("E3 intonation >= 90 at +/-2 semitones (semitone normalization)", f"{worst:.1f}", worst >= 90)

    rows = []
    for snr in [np.inf, 30, 20, 10, 5, 0]:
        r = H.score_pair(y, H.add_noise(y, snr))
        rows.append(["clean" if np.isinf(snr) else f"{snr} dB", f"{r['cost']:.3f}", r["reference_match"], r["intonation_match"], r["pairs"]])
    out.append("## E4. Background noise (white noise at a signal-to-noise ratio)\n" + H.md_table(
        ["SNR", "DTW cost", "Reference match", "Intonation", "Voiced pairs"], rows))
    return out


def e5(n=6):
    """E5: does CMVN help tell same-sentence pairs from different-sentence pairs?"""
    refs = [reference(s)[0] for s in SENTENCES[:n]]
    same_c, same_r, diff_c, diff_r = [], [], [], []
    for i, y in enumerate(refs):
        uk = tts.cache_path(SENTENCES[i], "UK English", "gtts")
        # "same sentence, another take": a real UK voice if cached, otherwise a slower
        # copy with background noise (a phase-vocoder pitch shift also moves the
        # formants, see E3, so it would not be a fair stand-in for a new speaker).
        other_voice = H.load(uk) if uk.exists() else H.add_noise(H.stretch(y, 0.85), 20)
        same_c.append(H.cmvn_cost(y, other_voice))
        same_r.append(H.raw_cost(y, other_voice))
        for j, z in enumerate(refs):
            if j != i:
                diff_c.append(H.cmvn_cost(y, z))
                diff_r.append(H.raw_cost(y, z))
    rows = []
    for name, s, d in [("MFCC + CMVN (app)", same_c, diff_c), ("MFCC without CMVN", same_r, diff_r)]:
        dp, acc = H.separation(s, d)
        rows.append([name, f"{np.mean(s):.3f}", f"{np.mean(d):.3f}", f"{dp:.2f}", f"{100 * acc:.0f}%"])
    source = "real UK gTTS voice" if tts.cache_path(SENTENCES[0], "UK English", "gtts").exists() else "slower noisy copy"
    dp_c, acc_c = H.separation(same_c, diff_c)
    check("E5 same-sentence cost < different-sentence cost (CMVN)", f"{100 * acc_c:.0f}% of pairs", acc_c >= 0.95)
    return ["## E5. CMVN ablation\nSame sentence by another voice or take (" + source + ") vs different sentences, "
            f"{n} sentences. Higher d' and accuracy mean better separation.\n" + H.md_table(
                ["Features", "Mean cost, same", "Mean cost, different", "d'", "Accuracy"], rows)]


def e6(n=6):
    """E6: cross-accent matrix of reference match (needs gTTS cache)."""
    accents = [a for a in config.ACCENTS if tts.cache_path(SENTENCES[0], a, "gtts").exists()]
    if len(accents) < 2:
        return ["## E6. Cross-accent matrix\nSkipped: needs gTTS clips for 2+ accents (run pregenerate_references.py)."]
    rows = []
    for ra in accents:
        row = [ra]
        for la in accents:
            costs = [H.cmvn_cost(H.load(tts.cache_path(s, ra, "gtts")), H.load(tts.cache_path(s, la, "gtts")))
                     for s in SENTENCES[:n] if tts.cache_path(s, ra, "gtts").exists() and tts.cache_path(s, la, "gtts").exists()]
            row.append(f"{np.mean(costs):.3f} ({compare.reference_match_score(float(np.mean(costs)), ra):.0f})")
        rows.append(row)
    return ["## E6. Cross-accent: mean DTW cost (Reference match) with the row as reference\n"
            + H.md_table(["Reference \\ learner voice"] + accents, rows)]


def e7():
    """E7: Whisper models on the demo clips: word error rate and time."""
    index = config.DEMO_DIR / "demo_files.json"
    models = [m for m in config.WHISPER_CHOICES if stt.is_model_cached(m)]
    if not models or not index.exists():
        return ["## E7. Whisper model comparison\nSkipped: needs downloaded models and demo files."]
    clips = json.loads(index.read_text())
    rows = []
    for m in models:
        stt.get_model(m)  # load once so loading time is not counted
        for c in clips:
            start = time.time()
            text, _ = stt.transcribe(config.DEMO_DIR / c["file"], m)
            rows.append([m, c["file"], f"{compare.word_error_rate(c['sentence'], text):.2f}", f"{time.time() - start:.2f}", text])
    by = {(r[0], r[1]): float(r[2]) for r in rows}
    good, bad = by.get((models[0], "demo_uk_voice.wav")), by.get((models[0], "demo_wrong_words.wav"))
    if good is not None and bad is not None:
        check("E7 wrong-words clip has higher WER than correct UK clip", f"{bad:.2f} vs {good:.2f}", bad > good)
    return ["## E7. Whisper models on the demo clips (target: the weather sentence)\n"
            + H.md_table(["Model", "Clip", "WER", "Seconds", "Transcript"], rows)]


def e8():
    """E8: average scores by the learner's self-reported accent (from history.csv)."""
    df = history.load_history()
    if len(df) == 0:
        return ["## E8. Scores by learner accent\nSkipped: no attempts in data/history.csv yet."]
    g = df.groupby("learner_accent")[["intelligibility", "reference_match", "intonation_match", "overall"]].mean()
    rows = [[idx, *[float(v) if v == v else None for v in r], int((df["learner_accent"] == idx).sum())] for idx, r in g.iterrows()]
    return ["## E8. Scores by learner accent (real attempts)\n" + H.md_table(
        ["Learner accent", "Intelligibility", "Reference match", "Intonation", "Overall", "Attempts"], rows)]


def main():
    """Run everything and write results.md."""
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    y, engine = reference(SENTENCES[0])
    sections = e1_e4(y) + e5() + e6() + e7() + e8()
    header = (f"# Experiment results\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} on {platform.system()}, "
              f"reference voice: {engine}, calibration US English: {config.DTW_CALIBRATION['US English']}.\n"
              f"Sentence for E1 to E4: \"{SENTENCES[0]}\"\n")
    accept = "## Acceptance targets\n" + H.md_table(["Target", "Measured", "Result"], targets)
    text = "\n\n".join([header, accept] + sections) + f"\n\nTotal time: {time.time() - start:.0f} s\n"
    (OUT / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
