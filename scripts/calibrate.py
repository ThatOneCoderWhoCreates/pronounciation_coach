"""
calibrate.py - choose the (good, bad) DTW costs that map to 100 and 0 for
"Reference match" (spec Section 9).

MANUAL mode (recommended, with your own voices):
    python scripts/calibrate.py --accent "US English" --sentence "The weather is lovely today, so let's walk to the park." --good me1.wav me2.wav me3.wav --bad mumble.wav --other other_sentence.wav --save
    good cost = median cost of the good attempts; bad cost = cost of the other sentence.
    Use --reference my_ref.wav instead of --sentence for "My own reference recording".

AUTO mode (a starting point when you have no recordings yet):
    python scripts/calibrate.py --auto --save
    For each accent, "good" = the same sentence spoken by the OTHER accents' voices,
    "bad" = different sentences. Needs the gTTS cache (pregenerate_references.py).

--save writes data/calibration.json, which config.py loads at start-up.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import pipeline  # noqa: E402
import tts  # noqa: E402


def save(results):
    """Merge {accent: (good, bad)} into data/calibration.json."""
    current = json.loads(config.CALIBRATION_FILE.read_text()) if config.CALIBRATION_FILE.exists() else {}
    current.update({k: [round(g, 4), round(b, 4)] for k, (g, b) in results.items()})
    config.CALIBRATION_FILE.write_text(json.dumps(current, indent=2))
    print(f"Saved to {config.CALIBRATION_FILE}")


def manual(args):
    """Measure the costs of the learner's own good, bad and other-sentence recordings."""
    if args.reference:
        ref = Path(args.reference)
    else:
        ref, message = tts.synthesize(args.sentence, args.accent, backend="auto")
        print(message)
    good = [pipeline.dtw_cost_between(ref, f) for f in args.good]
    for f, c in zip(args.good, good):
        print(f"(a) good   {Path(f).name:30s} cost = {c:.4f}")
    for f in args.bad or []:
        print(f"(b) bad    {Path(f).name:30s} cost = {pipeline.dtw_cost_between(ref, f):.4f}")
    other = pipeline.dtw_cost_between(ref, args.other)
    print(f"(c) other  {Path(args.other).name:30s} cost = {other:.4f}")
    pair = (statistics.median(good), other)
    print(f"\nSuggested for '{args.accent}': good = {pair[0]:.4f}, bad = {pair[1]:.4f}")
    if pair[0] >= pair[1]:
        print("WARNING: good cost is not below bad cost. Check the recordings before saving.")
        return {}
    return {args.accent: pair}


def auto(n_sentences):
    """Estimate (good, bad) per accent from the cached gTTS voices."""
    sentences = json.loads(config.SENTENCES_FILE.read_text(encoding="utf-8"))[:n_sentences]
    results = {}
    for accent in config.ACCENTS:
        good, bad = [], []
        for i, s in enumerate(sentences):
            ref = tts.cache_path(s, accent, "gtts")
            if not ref.exists():
                continue
            for other_accent in config.ACCENTS:
                same = tts.cache_path(s, other_accent, "gtts")
                if other_accent != accent and same.exists():
                    good.append(pipeline.dtw_cost_between(ref, same))
            diff = tts.cache_path(sentences[(i + 1) % len(sentences)], accent, "gtts")
            if diff.exists():
                bad.append(pipeline.dtw_cost_between(ref, diff))
        if good and bad:
            results[accent] = (statistics.median(good), statistics.median(bad))
            print(f"{accent:18s} good = {results[accent][0]:.4f} (n={len(good)})  "
                  f"bad = {results[accent][1]:.4f} (n={len(bad)})")
        else:
            print(f"{accent:18s} skipped: no cached gTTS clips (run pregenerate_references.py)")
    return results


def main():
    """Parse the arguments and run the chosen mode."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--auto", action="store_true")
    p.add_argument("--sentences", type=int, default=6, help="auto mode: how many sentences to use")
    p.add_argument("--accent", default=config.DEFAULT_ACCENT)
    p.add_argument("--sentence")
    p.add_argument("--reference")
    p.add_argument("--good", nargs="+")
    p.add_argument("--bad", nargs="*")
    p.add_argument("--other")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    if args.auto:
        results = auto(args.sentences)
    else:
        if not (args.good and args.other and (args.sentence or args.reference)):
            p.error("manual mode needs --good, --other and --sentence (or --reference)")
        if args.reference:
            args.accent = config.OWN_REFERENCE if args.accent == config.DEFAULT_ACCENT else args.accent
        results = manual(args)
    if args.save and results:
        save(results)


if __name__ == "__main__":
    main()
