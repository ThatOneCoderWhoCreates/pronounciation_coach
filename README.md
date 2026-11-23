# Pronunciation Coach

Speech Processing and Recognition (MAI 522-5), CIA III mini-project.

A local web app. The learner picks a sentence and a target accent, listens to a reference voice,
records an attempt, and gets an overall score, four sub-scores, a color-coded transcript and six
plots. Every attempt is saved so the Progress tab shows improvement over time.

## 1. Quick start

### macOS (zsh)

Run these one at a time from the folder that contains this README. The first command is only
needed if Homebrew does not have Python 3.12 or ffmpeg yet.

```
brew install python@3.12 ffmpeg
```

```
bash setup.sh
```

```
source .venv/bin/activate
```

```
python app.py
```

`setup.sh` needs internet once. It creates the virtual environment, installs the packages,
downloads the Whisper models (about 700 MB), creates the reference voices for all 10 sentences in
all 4 accents, calibrates the scores, makes the demo clips, runs the tests and runs the
experiments. How long it takes depends mostly on the download speed. After that the app works offline.

### Windows 10/11

1. Install Python 3.12 from https://www.python.org/downloads/ and tick **Add python.exe to PATH**
   on the first screen of the installer.
2. Optional: install ffmpeg. In PowerShell run `winget install Gyan.FFmpeg`, then open a new window.
3. Put the project in a short local folder such as `C:\pronunciation_coach`. Avoid OneDrive-synced
   folders (Desktop and Documents are often synced): OneDrive can lock files while the models download.
4. Double-click `setup_windows.bat` and wait for "Setup complete".
5. Double-click `run_app.bat` whenever you want to start the app.

From Command Prompt or PowerShell, the same two steps are:

```
.\setup_windows.bat
```

```
.\run_app.bat
```

The scripts call `.venv\Scripts\python.exe` directly, so there is no need to "activate" the
environment (PowerShell blocks `Activate.ps1` by default). To run a single script by hand, use the
same Python, for example `.venv\Scripts\python.exe -m pytest -q`.

The app opens at http://127.0.0.1:7860. Use `python app.py --no-browser` to stop it from opening
a tab, or `--port 7861` to change the port.

## 2. Using the app (user flow)

1. Choose a practice sentence from the list, or type your own.
2. Choose the target accent (US, UK, Indian, Australian English), or "My own reference recording"
   and upload a teacher's or native speaker's recording.
3. Press **1. Get reference** to hear the reference (cached after the first time).
4. Record with the microphone, or upload a file, in **2. Your attempt**.
5. Press **3. Analyze**. You get the score cards, the colored transcript and the plots.
6. The attempt is stored. The **Progress** tab shows the trend.

If the microphone fails, click any row under **Demo recordings**, then press Analyze.

## 3. Rubric mapping

| Required part | Where it is |
|---|---|
| 1. Speech and audio data acquisition | Microphone and file upload (`app.py`), text-to-speech (TTS) or uploaded reference (`tts.py`) |
| 2. Back-end speech signal processing | `audio_io.py`, `features.py`, `compare.py` (resampling, trimming, framing, MFCC, pitch, DTW) |
| 3. Front-end and application | Gradio web app (`app.py`, `ui_text.py`) |
| 4. APIs, AI models and agents | Speech-to-text (STT) with faster-whisper (`stt.py`); TTS with gTTS and pyttsx3 (`tts.py`) |
| 5. Visualization and analytics | Six plots (`plots.py`), colored transcript, score cards, Progress tab, experiment tables |
| 6. Live demonstration | Demo plan in Section 13 |

## 4. Project structure

```
pronunciation_coach/
  app.py                 Gradio interface only (layout and button actions)
  ui_text.py             fixed text, score explanations, CSS for the interface
  pipeline.py            runs one full analysis: reference -> features -> DTW -> scores -> history
  config.py              every constant in one place
  audio_io.py            load, decode, resample, trim, normalize, save, length check
  features.py            MFCC with CMVN, pitch in semitones, energy, log-mel spectrogram
  stt.py                 faster-whisper wrapper with a cached model (openai-whisper fallback)
  tts.py                 gTTS and pyttsx3 wrapper, accent selection, disk cache
  compare.py             WER, word diff, DTW, the four scores and the overall score
  plots.py               every matplotlib figure
  history.py             CSV storage of attempts
  data/sentences.json    practice sentences
  data/cache/            cached reference audio (filled by setup)
  data/models/           downloaded Whisper models (filled by setup)
  data/calibration.json  calibrated (good, bad) costs (written by calibrate.py)
  demo/                  clips to upload if the microphone fails (made by setup)
  tests/                 pytest tests with synthetic audio (45 tests, no internet needed)
  experiments/           run_experiments.py, helpers.py, results/results.md
  scripts/               calibrate.py, pregenerate_references.py, prefetch_models.py, make_demo_audio.py
  setup.sh               one-time setup for macOS and Linux
  setup_windows.bat      one-time setup for Windows
  run_app.bat            starts the app on Windows (double-click)
```

## 5. The signal processing chain (for the viva)

1. **Decode.** `soundfile` reads wav, flac, ogg and mp3. Anything else (for example webm) goes
   through `ffmpeg`. librosa 1.0 removed its old audioread decoder, so ffmpeg is called directly.
2. **Mono and resample** to 16 kHz, the rate Whisper uses. Both signals must share one rate,
   otherwise frames would cover different amounts of time.
3. **Trim** leading and trailing silence (anything 30 dB below the peak), so silence is not
   aligned with silence and durations are comparable.
4. **Peak-normalize** to 0.9 so loudness does not matter.
5. **Mel-frequency cepstral coefficients (MFCC):** 25 ms Hamming-windowed frames every 10 ms,
   512-point FFT, 40 mel filters, log, discrete cosine transform, keep 13 coefficients.
6. **Cepstral mean and variance normalization (CMVN):** for each coefficient, subtract its mean
   over time and divide by its standard deviation. This removes much of the constant difference
   caused by the voice and microphone.
7. **Pitch:** probabilistic YIN (pyin) between 65 and 500 Hz gives f0 and a voiced flag per
   frame. We convert to semitones relative to the speaker's own median f0:
   `12 * log2(f0 / median)`, so a low and a high voice with the same melody give the same contour.
8. **Dynamic time warping (DTW)** with cosine distance aligns the learner's MFCC frames with the
   reference frames. librosa returns the path from end to start; we reverse it so it runs from
   (0, 0) to (last reference frame, last learner frame). Normalized cost = `D[-1, -1] / path length`.
9. **Whisper** transcribes the cleaned learner audio. We never give it the target sentence as a
   prompt, because that would push it toward the right words and hide mistakes.

## 6. Accent handling

The app trains no model and uses no accent dataset. It compares the learner with one reference
voice, so the chosen reference accent defines what "close" means. Intelligibility is measured
separately by Whisper.

| Option in the app | gTTS setting | Notes |
|---|---|---|
| US English | lang en, tld us | default |
| UK English | lang en, tld co.uk | |
| Indian English | lang en, tld co.in | |
| Australian English | lang en, tld com.au | |
| My own reference recording | none | upload any accent |

- `lang` and `tld` are always passed. The gTTS default (`com`) picks an accent from the network
  location, so results would change between machines.
- The cache file name is the SHA-1 hash of text + accent + engine, so accents never overwrite
  each other.
- pyttsx3 (offline) uses the operating system voice and cannot change accent. When it is used,
  the app says which voice was used and that the chosen accent was **not** applied.
- Calibration is stored per accent.

## 7. Scores

| Score in the app | Definition | Range |
|---|---|---|
| Intelligibility | 100 x (1 - min(WER, 1)), word error rate between the target and Whisper's transcript. Does not depend on the reference accent. | 0 to 100 |
| Reference match | Straight-line map of the normalized DTW cost: 100 at the calibrated "good" cost, 0 at the "bad" cost, clipped. Means closeness to the chosen reference voice, not correctness. | 0 to 100 |
| Intonation match | After DTW, Pearson correlation of the two semitone contours on frames where both are voiced. 100 x max(0, r). Fewer than 10 voiced pairs gives "n/a". | 0 to 100 or n/a |
| Pace match | r = learner duration / reference duration. 100 for r between 0.8 and 1.25, falling to 0 at r = 0.5 or 2 (on a log scale). | 0 to 100 |
| Overall coach score | 50% intelligibility, 25% reference match, 15% intonation, 10% pace. If intonation is n/a, the other weights are rescaled. | 0 to 100 |

Intelligibility has the largest weight because it is the fairest measure across accents.

### Calibration

`scripts/calibrate.py --auto --save` (run by setup) gives a starting point: for each accent,
"good" is the median cost of the same sentence spoken by the other three accent voices, and "bad"
is the median cost of a different sentence. Better values come from your own voices:

```
python scripts/calibrate.py --accent "US English" --sentence "The weather is lovely today, so let's walk to the park." --good me1.wav me2.wav me3.wav --bad mumble.wav --other other_sentence.wav --save
```

Good cost = median of the good attempts, bad cost = cost of the other sentence. The values are
saved in `data/calibration.json` and loaded by `config.py`. The placeholders before calibration
are good = 0.35 and bad = 0.90.

## 8. User interface (assumed Section 10)

Three tabs: **Practice** (inputs, reference player, attempt recorder, score cards with a one-line
explanation each, colored transcript, five plots, demo recordings), **Progress** (history plot,
table, refresh and clear buttons) and **How it works** (method, scores, limitations, privacy).
A setup-check line at the top shows the Python version, ffmpeg, whether Whisper is downloaded and
how many reference clips are cached. Plots use font size 12 or larger and the Okabe-Ito
color-blind-safe palette.

Transcript colors: green correct, red wrong (shows what Whisper heard), gray missing, blue extra.

## 9. Platform notes (assumed Section 11)

- **Python:** 3.12 or 3.11. librosa 1.0 needs 3.12, so `requirements.txt` installs librosa 0.11
  on 3.11 (both were tested). Python 3.13+ is not supported by some audio packages. Python 3.10
  is too old for numpy 2.4+.
- **ffmpeg:** recommended, not required. wav and mp3 work without it.
- **pyttsx3:** runs in a separate process with a fresh engine for every sentence, because calling
  it from Gradio's worker threads hangs on Windows. On Windows it uses the SAPI5 voices; the helper
  process starts without a console window. On macOS, if pyttsx3 fails, the built-in `say` command
  is used. macOS writes AIFF files, which soundfile reads.
- **Whisper audio:** the app decodes audio itself and passes the samples to Whisper, so
  faster-whisper never needs PyAV to open files (faster-whisper 1.2.1 and PyAV 19 disagree on an
  argument).
- **Model download on Windows:** without Developer Mode, Windows cannot create symbolic links, so
  the model files are copied instead. This works; the warning about it is switched off.
- **Text encoding:** every file is read and written as UTF-8, and the scripts set `PYTHONUTF8=1`,
  so a Windows console that is not UTF-8 cannot crash when Whisper prints unusual characters.
- **Temporary files:** temporary folders ignore cleanup errors, because antivirus software on
  Windows can lock a file for a moment.
- **History file:** Windows locks `data/history.csv` while Excel has it open. The app then asks you
  to close it instead of crashing.
- **Line endings:** `.gitattributes` keeps `.bat` files in Windows (CRLF) format, which `cmd`
  needs to read labels correctly.
- **File names:** recordings use `-` instead of `:` in timestamps (Windows does not allow `:`).
- **Paths:** pathlib everywhere, no POSIX-only code.

## 10. Friendly errors (assumed Section 12)

Every button is wrapped so the learner sees a short message, never a stack trace (the full
details still go to the terminal for the developers).

| Situation | Message |
|---|---|
| Analyze with no recording | Record or upload your attempt first. |
| Silent recording | The recording is silent. Check your microphone and try again. |
| Shorter than 0.5 s or longer than 15 s | The speech is only X s long... / Please keep it under 15 s. |
| Unreadable format without ffmpeg | This audio format needs ffmpeg... or upload a .wav file. |
| gTTS offline with engine "gtts" | gTTS needs internet... Choose backend 'auto' or 'pyttsx3'. |
| Own reference chosen but not uploaded | Upload your own reference recording first. |
| history.csv open in Excel (Windows) | Could not save this attempt because data/history.csv is open in another program... |
| Whisper not downloaded and offline | Connect to the internet once and run: python scripts/prefetch_models.py |
| Anything unexpected | Something unexpected went wrong (ErrorType). Please try again, or upload one of the demo files. |

## 11. Tests, experiments and acceptance targets (assumed Section 13)

```
python -m pytest -q
```

45 tests with synthetic, speech-like audio (harmonic "vowels" with formant peaks), so they need no
microphone, internet or Whisper model. They cover decoding and resampling, trimming and
normalization, CMVN statistics, pitch on known tones, DTW path orientation, every score formula,
the word diff, the full pipeline with a stand-in transcriber, privacy (no audio saved by default),
the TTS cache and fallback, and that every plot has a title and labels.

```
python experiments/run_experiments.py
```

writes `experiments/results/results.md`:

| Experiment | Question |
|---|---|
| E1 | Does the reference compared with itself score 100? |
| E2 | Speed changes (0.5x to 2x): does DTW absorb tempo while pace match reacts? |
| E3 | Pitch shifts (-4 to +4 semitones): does the semitone normalization keep intonation high? |
| E4 | White noise at 30 to 0 dB signal-to-noise ratio (SNR): how robust are the scores? |
| E5 | CMVN ablation: does CMVN separate same-sentence pairs from different-sentence pairs better? |
| E6 | Cross-accent matrix: reference match of each accent voice against each reference accent |
| E7 | Whisper tiny.en, base.en and small.en on the demo clips: word error rate and time |
| E8 | Average scores by the learner's self-reported accent (from real attempts) |

Acceptance targets:

| Target | Why |
|---|---|
| Reference vs itself: reference match 100, intonation 99 or more, pace 100 | Sanity check of the whole chain |
| Pace 100 at 0.8x and 1.25x speed, 10 or less at 0.5x and 2x | Pace formula behaves as defined |
| Reference match 60 or more at 0.8x and 1.25x speed | DTW absorbs moderate tempo changes |
| Intonation 80 or more at +/-2 semitones | Semitone normalization makes voice height irrelevant |
| Same-sentence cost below different-sentence cost for 95% of pairs | Features carry the words |
| After calibration: same sentence 60 or more, different sentence 20 or less reference match | Calibration makes the score meaningful |
| Wrong-words clip has a higher WER than the correct UK clip | Whisper catches the changed words |

### Results so far (replace with your own run)

These numbers come from the build environment, which had no internet. The four accents were
therefore played by four eSpeak voices instead of gTTS, and Whisper could not be downloaded, so
E7 and E8 did not run there. **After `setup.sh` finishes on your laptop, copy the tables from
`experiments/results/results.md` into this section.**

| Target | Measured | Result |
|---|---|---|
| E1 self: reference 100, intonation >= 99, pace 100 | 100 / 100.0 / 100 | PASS |
| E2 pace 100 at 0.8x and 1.25x speed | 100 / 100 | PASS |
| E2 pace <= 10 at 0.5x and 2x speed | 2 / 0 | PASS |
| E2 reference match >= 60 at 0.8x and 1.25x | 89 / 76 | PASS |
| E3 intonation >= 80 at +/-2 semitones | 86.4 | PASS |
| E5 same-sentence cost < different-sentence cost (CMVN) | 100% of pairs | PASS |
| E5 calibration: same >= 60 and different <= 20 reference match | 100 / 4 | PASS |

With the placeholder calibration (0.35, 0.90) the last target failed (a different sentence still
scored 50), which is why calibration matters. Other findings from that run:

- **CMVN (E5):** d' (separation between same and different sentences) was 12.9 with CMVN and
  8.5 without.
- **Pitch shift (E3):** a phase-vocoder pitch shift of 2 semitones drops reference match to about
  40, because it also moves the formants, so MFCCs change. Intonation stays above 86.
- **Noise (E4):** reference match stays at 100 down to 20 dB SNR and falls to 25 at 0 dB, while
  intonation stays above 96.
- **Cross-accent (E6):** costs between different voices were 0.09 to 0.24, against about 0.59 for
  a different sentence.

## 12. Build phases (assumed Section 14)

The project is a git repository with one commit per phase (`git log`):

1. Core DSP: config, audio input and output, features, comparison and scores.
2. AI and interface: speech-to-text, text-to-speech, history, plots, pipeline, Gradio app.
3. Tests, scripts and experiments.
4. Verification, setup scripts, README.

## 13. Demo plan (assumed Section 15)

**Before the showcase**

1. Run the setup and check that the setup-check line says OK, Whisper downloaded, 40 clips.
2. Each member records three attempts of sentence 1 (QuickTime or Voice Memos, export as .m4a or
   .wav) and saves them in `demo/`. Use them for manual calibration (Section 7).
3. Turn off Wi-Fi once and run through the demo to prove it works offline.
4. Press Analyze once before presenting, so Whisper is already loaded.

**Live (about 6 minutes)**

1. Show the sentence and play the reference in US English, then UK English: the accent changes.
2. One member records live and presses Analyze. Walk through the cards, the transcript and the
   plots: waveforms, spectrograms (same color scale), pitch overlay, DTW path, scores.
3. Click the demo rows: `demo_wrong_words.wav` (intelligibility drops), `demo_too_slow.wav` (pace
   drops, reference match stays high), `demo_different_sentence.wav` (everything low).
4. Show the Progress tab.
5. Show the experiment tables and the limitations.

**If something fails:** the microphone does not work, so use the demo rows. No internet: the
references and Whisper are cached. Whisper too slow: choose tiny.en.

### Team split

| Member | Owns and explains | Rubric parts |
|---|---|---|
| A: signal processing | `audio_io.py`, `features.py`, `plots.py`, experiments E2 to E5 | 2, 5 |
| B: comparison and scoring | `compare.py`, `pipeline.py`, `history.py`, `scripts/calibrate.py`, `tests/` | 2, 5 |
| C: AI models and front end | `stt.py`, `tts.py`, `app.py`, `ui_text.py`, other scripts, demo | 1, 3, 4, 6 |

Everyone should be able to explain the chain in Section 5 and the scores in Section 7.

## 14. Assumptions

The specification (v2) given to us stopped at Section 9, so Sections 10 to 15 here are our own
choices. Other assumptions and changes:

- American spelling is used everywhere, including function names (`normalize_text` instead of `normalise_text`).
- `pipeline.py` and `ui_text.py` were added so that `app.py` contains only the interface and no
  file goes above about 200 lines.
- `tts.synthesize` returns the path **and** a message, so the app can say honestly which voice
  was used.
- Calibration values are saved to `data/calibration.json` by the script instead of being edited
  into `config.py` by hand; `config.py` keeps the placeholders and loads the file.
- Whisper models are stored in `data/models/` (not the user's global cache), so it is clear the
  demo does not need a download.
- Figures are built with `matplotlib.figure.Figure` instead of pyplot, so they never need closing
  and do not leak memory.
- The decoding fallback is ffmpeg, because librosa 1.0 no longer has its own decoder.
- Python 3.11 and 3.12 are supported (3.10 is too old for the current numpy and pandas).
- The demo clips are synthetic voices. They show how the scores behave, not real learners.

## 15. Limitations

- No accent training. "Reference match" means close to **one** reference voice, not correct.
  Accents near the reference voice score higher.
- Whisper accuracy changes with accent, microphone and background noise.
- Intonation and pace legitimately differ between accents.
- A wrong word that sounds similar (weather and feather) barely changes the DTW cost. Only
  Whisper's transcript catches it, which is another reason intelligibility has the largest weight.
- gTTS needs internet the first time each sentence and accent is used. Custom sentences typed
  offline fall back to the offline voice, without the accent.
- This is a practice tool, not a high-stakes assessment.

## 16. Privacy

Only scores, transcripts and timestamps are stored, in `data/history.csv`. Learner audio is saved
in `data/recordings/` only if **Save my recordings** is ticked (off by default). The learner's
accent is optional and stays on the computer. **Clear history** in the Progress tab deletes the
file.
