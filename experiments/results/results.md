# Experiment results

Generated 2026-11-23 19:29 on Darwin, reference voice: gTTS, calibration US English: (0.3564, 0.6243).
Sentence for E1 to E4: "The weather is lovely today, so let's walk to the park."


## Acceptance targets
| Target | Measured | Result |
|---|---|---|
| E1 self: reference 100, intonation >= 99, pace 100 | 100 / 100.0 / 100 | PASS |
| E2 pace 100 at 0.8x and 1.25x speed | 100 / 100 | PASS |
| E2 pace <= 10 at 0.5x and 2x speed | 0 / 0 | PASS |
| E2 reference match >= 60 at 0.8x and 1.25x (DTW absorbs tempo) | 100 / 100 | PASS |
| E3 intonation >= 80 at +/-2 semitones (semitone normalization) | 98.9 | PASS |
| E5 same-sentence cost < different-sentence cost (CMVN) | 100% of pairs | PASS |
| E5 calibration: same sentence >= 60 and different sentence <= 20 reference match | 93 / 6 | PASS |
| E7 wrong-words clip has higher WER than correct UK clip | 0.27 vs 0.00 | PASS |

## E1. Reference compared with itself
| DTW cost | Reference match | Intonation match | Pace match |
|---|---|---|---|
| 0.0000 | 100.0 | 100.0 | 100.0 |

## E2. Speaking speed (time stretch, pitch unchanged)
| Speed | Duration ratio | DTW cost | Reference match | Intonation | Pace |
|---|---|---|---|---|---|
| 0.5x | 2.0 | 0.172 | 100.0 | 99.4 | 0.3 |
| 0.67x | 1.5 | 0.210 | 100.0 | 99.0 | 62.3 |
| 0.8x | 1.2 | 0.223 | 100.0 | 99.1 | 100.0 |
| 1.0x | 1.0 | 0.000 | 100.0 | 100.0 | 100.0 |
| 1.25x | 0.8 | 0.264 | 100.0 | 97.8 | 100.0 |
| 1.5x | 0.7 | 0.298 | 100.0 | 97.3 | 61.2 |
| 2.0x | 0.5 | 0.330 | 100.0 | 97.4 | 0.0 |

## E3. Voice pitch (pitch shift, speed unchanged)
| Semitones | DTW cost | Reference match | Intonation | Voiced pairs |
|---|---|---|---|---|
| -4 | 0.647 | 0.0 | 98.5 | 247 |
| -2 | 0.375 | 92.9 | 98.9 | 247 |
| +0 | 0.000 | 100.0 | 100.0 | 261 |
| +2 | 0.370 | 94.9 | 99.1 | 243 |
| +4 | 0.628 | 0.0 | 98.7 | 259 |

## E4. Background noise (white noise at a signal-to-noise ratio)
| SNR | DTW cost | Reference match | Intonation | Voiced pairs |
|---|---|---|---|---|
| clean | 0.000 | 100.0 | 100.0 | 261 |
| 30 dB | 0.091 | 100.0 | 99.9 | 259 |
| 20 dB | 0.190 | 100.0 | 99.9 | 260 |
| 10 dB | 0.331 | 100.0 | 99.8 | 243 |
| 5 dB | 0.410 | 80.1 | 99.6 | 202 |
| 0 dB | 0.503 | 45.3 | 98.2 | 37 |

## E5. CMVN ablation
Same sentence by another voice or take (real UK gTTS voice) vs different sentences, 6 sentences. Higher d' and accuracy mean better separation.
| Features | Mean cost, same | Mean cost, different | d' | Accuracy |
|---|---|---|---|---|
| MFCC + CMVN (app) | 0.367 | 0.620 | 6.78 | 100% |
| MFCC without CMVN | 0.023 | 0.035 | 2.26 | 91% |

## E6. Cross-accent: mean DTW cost (Reference match) with the row as reference
| Reference \ learner voice | US English | UK English | Indian English | Australian English |
|---|---|---|---|---|
| US English | 0.000 (100) | 0.367 (96) | 0.414 (78) | 0.301 (100) |
| UK English | 0.367 (90) | 0.000 (100) | 0.405 (74) | 0.276 (100) |
| Indian English | 0.414 (87) | 0.405 (92) | 0.000 (100) | 0.387 (100) |
| Australian English | 0.301 (99) | 0.276 (100) | 0.387 (70) | 0.000 (100) |

## E7. Whisper models on the demo clips (target: the weather sentence)
| Model | Clip | WER | Seconds | Transcript |
|---|---|---|---|---|
| tiny.en | demo_uk_voice.wav | 0.00 | 0.16 | The weather is lovely today, so let's walk to the park. |
| tiny.en | demo_indian_voice.wav | 0.00 | 0.13 | The weather is lovely today, so let's walk to the park. |
| tiny.en | demo_wrong_words.wav | 0.27 | 0.13 | The feather is lovely today, so let's talk to the bark. |
| tiny.en | demo_too_slow.wav | 0.00 | 0.13 | The weather is lovely today, so let's walk to the park. |
| tiny.en | demo_noisy.wav | 0.00 | 0.13 | The weather is lovely today, so let's walk to the park. |
| tiny.en | demo_different_sentence.wav | 0.91 | 0.13 | Could you please tell me the way to the railway station? |
| base.en | demo_uk_voice.wav | 0.00 | 0.27 | The weather is lovely today, so let's walk to the park. |
| base.en | demo_indian_voice.wav | 0.00 | 0.27 | The weather is lovely today, so let's walk to the park. |
| base.en | demo_wrong_words.wav | 0.27 | 0.26 | The feather is lovely today, so let's talk to the bark. |
| base.en | demo_too_slow.wav | 0.00 | 0.27 | The weather is lovely today, so let's walk to the park. |
| base.en | demo_noisy.wav | 0.00 | 0.27 | The weather is lovely today, so let's walk to the park. |
| base.en | demo_different_sentence.wav | 0.91 | 0.26 | Could you please tell me the way to the railway station? |
| small.en | demo_uk_voice.wav | 0.00 | 0.91 | The weather is lovely today, so let's walk to the park. |
| small.en | demo_indian_voice.wav | 0.00 | 0.91 | The weather is lovely today, so let's walk to the park. |
| small.en | demo_wrong_words.wav | 0.27 | 0.91 | The feather is lovely today, so let's talk to the bark. |
| small.en | demo_too_slow.wav | 0.00 | 0.91 | The weather is lovely today, so let's walk to the park. |
| small.en | demo_noisy.wav | 0.00 | 0.91 | The weather is lovely today, so let's walk to the park. |
| small.en | demo_different_sentence.wav | 0.91 | 0.90 | Could you please tell me the way to the railway station? |

## E8. Scores by learner accent
Skipped: no attempts in data/history.csv yet.

Total time: 12 s
