# Experiment results

Generated 2026-11-23 13:38 on Linux, reference voice: offline voice, calibration US English: (0.35, 0.9).
Sentence for E1 to E4: "The weather is lovely today, so let's walk to the park."


## Acceptance targets
| Target | Measured | Result |
|---|---|---|
| E1 self: reference 100, intonation >= 99, pace 100 | 100 / 100.0 / 100 | PASS |
| E2 pace 100 at 0.8x and 1.25x speed | 100 / 100 | PASS |
| E2 pace <= 10 at 0.5x and 2x speed | 3 / 0 | PASS |
| E2 reference match >= 80 at 0.8x and 1.25x (DTW absorbs tempo) | 100 / 100 | PASS |
| E3 intonation >= 90 at +/-2 semitones (semitone normalization) | 93.3 | PASS |
| E5 same-sentence cost < different-sentence cost (CMVN) | 100% of pairs | PASS |

## E1. Reference compared with itself
| DTW cost | Reference match | Intonation match | Pace match |
|---|---|---|---|
| 0.0000 | 100.0 | 100.0 | 100.0 |

## E2. Speaking speed (time stretch, pitch unchanged)
| Speed | Duration ratio | DTW cost | Reference match | Intonation | Pace |
|---|---|---|---|---|---|
| 0.5x | 2.0 | 0.178 | 100.0 | 96.1 | 2.5 |
| 0.67x | 1.5 | 0.215 | 100.0 | 96.4 | 62.9 |
| 0.8x | 1.2 | 0.244 | 100.0 | 95.6 | 100.0 |
| 1.0x | 1.0 | 0.000 | 100.0 | 100.0 | 100.0 |
| 1.25x | 0.8 | 0.292 | 100.0 | 93.1 | 100.0 |
| 1.5x | 0.7 | 0.342 | 100.0 | 92.7 | 61.2 |
| 2.0x | 0.5 | 0.386 | 93.4 | 95.4 | 0.0 |

## E3. Voice pitch (pitch shift, speed unchanged)
| Semitones | DTW cost | Reference match | Intonation | Voiced pairs |
|---|---|---|---|---|
| -4 | 0.684 | 39.4 | 79.9 | 131 |
| -2 | 0.452 | 81.5 | 97.6 | 155 |
| +0 | 0.000 | 100.0 | 100.0 | 163 |
| +2 | 0.404 | 90.1 | 93.3 | 160 |
| +4 | 0.713 | 34.0 | 81.0 | 161 |

## E4. Background noise (white noise at a signal-to-noise ratio)
| SNR | DTW cost | Reference match | Intonation | Voiced pairs |
|---|---|---|---|---|
| clean | 0.000 | 100.0 | 100.0 | 163 |
| 30 dB | 0.054 | 100.0 | 100.0 | 163 |
| 20 dB | 0.136 | 100.0 | 99.9 | 163 |
| 10 dB | 0.280 | 100.0 | 99.6 | 162 |
| 5 dB | 0.364 | 97.5 | 99.3 | 78 |
| 0 dB | 0.466 | 78.9 | n/a | 0 |

## E5. CMVN ablation
Same sentence by another voice or take (slower noisy copy) vs different sentences, 6 sentences. Higher d' and accuracy mean better separation.
| Features | Mean cost, same | Mean cost, different | d' | Accuracy |
|---|---|---|---|---|
| MFCC + CMVN (app) | 0.292 | 0.622 | 13.57 | 100% |
| MFCC without CMVN | 0.023 | 0.057 | 6.35 | 100% |

## E6. Cross-accent matrix
Skipped: needs gTTS clips for 2+ accents (run pregenerate_references.py).

## E7. Whisper model comparison
Skipped: needs downloaded models and demo files.

## E8. Scores by learner accent
Skipped: no attempts in data/history.csv yet.

Total time: 9 s
