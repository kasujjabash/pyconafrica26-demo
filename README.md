# Working with Audio in Python (A Pythonic Approach)

## Presentation Link: https://docs.google.com/presentation/d/1r3ayF5cF1eQLxxkmOSTtFGqqhGTt-Fv-/edit?usp=sharing&ouid=104013299534540640003&rtpof=true&sd=true

## Getting Started

### Prerequisites

- **Python 3.9+**
- A working **microphone** (built-in laptop mic, USB mic, or Bluetooth headset — see Troubleshooting if a Bluetooth mic doesn't show up)
- macOS, Linux, or Windows (these demos are built and tested primarily on macOS)

### 1. Get the code onto your computer

```bash
git clone <your-repo-url>
cd <repo-folder-name>
```

(Replace `<your-repo-url>` with this repository's actual GitHub URL once it's published.)

### 2. Install system-level audio dependencies

**macOS:**

```bash
brew install portaudio ffmpeg
```

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get update && sudo apt-get install -y portaudio19-dev ffmpeg
```

**Windows:** no extra system package needed — `pip install pyaudio` pulls in a prebuilt wheel. If it fails to build, see Troubleshooting.

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

You'll know it worked when your terminal prompt shows `(.venv)` at the start.

### 4. Install Python packages (and download models)

```bash
bash setup.sh
```

This installs everything in `requirements.txt`, downloads the small offline Vosk speech model (~50 MB), and pre-downloads the tiny Whisper model. Or, to do it manually without the model downloads:

```bash
pip install -r requirements.txt
```

### 5. Run it

```bash
python main.py
```

This opens a menu-driven launcher — type a number (`1`–`7`) to run a demo, `d` to check which libraries are installed, or `q` to quit.

> **First run on macOS:** your system will prompt for microphone access the first time a demo tries to record — click **Allow**. If you miss the prompt or denied it, go to **System Settings → Privacy & Security → Microphone** and enable it for your terminal app.

---

## 📖 Demo Guide

All 7 demos are reachable from the `python main.py` menu, or runnable directly as standalone scripts. Press `Ctrl+C` to stop any running demo (Demo 4's matplotlib window can also just be closed).

| #   | Demo                     | What it shows                                                    |
| --- | ------------------------ | ----------------------------------------------------------------- |
| 1   | **Microphone Input**     | Device listing · live level meter · record & playback            |
| 2   | **Live Transcription**   | Continuous speech-to-text via Whisper / Vosk / Google STT        |
| 3   | **Wake Word Detection**  | Passive listener · triggers on "hey python" · command capture    |
| 4   | **Sound Visualization**  | Real-time waveform + FFT spectrum + spectrogram waterfall        |
| 5   | **Voice Assistant**      | Wake word + natural language commands + TTS responses            |
| 6   | **Noise Detection**      | dBFS monitoring · threshold alerts · silence detection · summary |
| 7   | **Voice Transformation** | Record a clip · pitch shift up/down · echo — pure numpy/scipy    |

### Demo 1 — Microphone Input

**What it does:** Lists every audio device your system can see, shows a live input-level meter for 5 seconds, records a 5-second clip with a visual countdown, then plays that clip straight back.

**How it works:** Opens a `sounddevice.InputStream`; every incoming chunk is converted to dBFS (`utils/audio_utils.py`) to drive the live level bar, then the full recording is buffered and handed to `sd.play()`.

**Run it:**
```bash
python demos/mic_input.py
```

**Try it:**
- Check the device table — confirm your mic is marked `● default` under "In"
- During the 5-second monitor phase, talk or tap near the mic and watch the bar react
- When the countdown recording starts, say something — you'll hear it played back immediately

### Demo 2 — Live Transcription

**What it does:** Continuously listens and prints a transcript of what you say, using whichever speech engine is available (priority order: Whisper → Vosk → Google STT).

**How it works:** Streams audio into a buffer, waits until it has a few seconds of non-silent audio (skipping low-RMS chunks so it doesn't waste time transcribing nothing), converts the buffer to WAV in memory, and sends it to the active recognizer.

**Run it:**
```bash
python demos/live_transcription.py
```

Force a specific engine:
```bash
FORCE_ENGINE=google python demos/live_transcription.py
FORCE_ENGINE=whisper python demos/live_transcription.py
FORCE_ENGINE=vosk python demos/live_transcription.py
```

**Try it:** Just talk naturally — a transcript appears a second or two after each phrase. `Ctrl+C` to stop and see a summary.

### Demo 3 — Wake Word Detection

**What it does:** Listens passively in the background at low CPU cost, and only reacts once it hears a wake word — then records and transcribes your follow-up command.

**How it works:** Monitors RMS energy continuously; only attempts a (more expensive) transcription when the signal crosses a minimum energy threshold. Once a wake word is recognized in that transcription, it records a few more seconds as your "command" and transcribes that separately.

**Run it:**
```bash
python demos/wake_word.py
```

**Wake words:** say any of — **"hey python"**, **"python"**, **"computer"**, **"listen"**, **"wake up"**

**Try it:** Say a wake word, wait for the chime/alert, then speak a short command. It'll print what it heard.

### Demo 4 — Sound Visualization

**What it does:** Opens a live matplotlib window with three real-time panels — waveform, frequency spectrum (with peak-hold), and a scrolling spectrogram waterfall — plus a terminal VU meter running alongside it.

**How it works:** An `InputStream` callback feeds a rolling numpy buffer. Every frame recomputes an FFT (`np.fft.rfft`) for the spectrum and spectrogram panels, redrawn via `matplotlib.animation.FuncAnimation`.

**Run it:**
```bash
python demos/sound_visualization.py
```

**Try it, in this order:**
1. **Stay silent** — watch the waveform flatten to near-zero
2. **Talk normally** — see a broad, messy spread across the spectrum (speech uses many frequencies at once)
3. **Whistle a steady note, then slide the pitch up or down** — watch a single sharp peak move across the spectrum panel in real time
4. **Clap once** — see an instant bright flash across the *entire* spectrum and a line on the spectrogram (a clap has no single pitch — it's broadband noise for an instant)
5. Close the window, or `Ctrl+C` in the terminal, to exit

### Demo 5 — Voice Assistant

**What it does:** The same wake-word listener as Demo 3, plus simple command handling — time/date, a joke, basic math, a canned weather reply — spoken back out loud via `pyttsx3` (or printed if TTS isn't installed).

**How it works:** Once a wake word triggers a recording, the transcribed command is matched against a small set of keyword rules, and a response is generated and spoken.

**Run it:**
```bash
python demos/voice_assistant.py
```

**Commands:**

| Say               | Response                 |
| ----------------- | ------------------------ |
| time / date / day  | Current time / date      |
| weather            | Sample weather readout   |
| joke               | Random programmer joke   |
| calculate 2 + 2    | Evaluates the expression |
| what can you do    | Lists commands           |
| stop / bye         | Ends the session         |

**Try it:** Say a wake word (see Demo 3), then one of the commands above.

### Demo 6 — Noise Detection

**What it does:** Monitors ambient noise level continuously, flashing an alert whenever it gets too loud, logging silence periods, and printing a session summary when you stop it.

**How it works:** The same RMS/dBFS calculation used throughout the project, compared against two thresholds — above −20 dBFS logs a noise alert, below −50 dBFS counts as silence — tracked over a rolling 10-second min/avg/max window.

**Run it:**
```bash
python demos/noise_detection.py
```

**Try it:**
- Stay quiet — watch the silence-period counter increase
- Talk or clap loudly — see the alert flash red and get logged with a timestamp
- `Ctrl+C` to stop and view the session summary (duration, alert count, peak level)

### Demo 7 — Voice Transformation

**What it does:** Records a few seconds of your voice, then plays it back four ways: the original, pitched up ("chipmunk"), pitched down ("deep voice"), and with echo added.

**How it works:** Pitch shifting resamples the array to fewer or more points with `scipy.signal.resample` — the same principle as speeding up or slowing down a cassette tape, so pitch and duration change *together*. Echo is a delayed, progressively quieter copy of the signal summed back onto the original, normalized afterward so it never clips.

**Run it:**
```bash
python demos/voice_transformation.py
```

**Try it:** Speak a short phrase when the countdown starts, then listen to all four versions play back in order. The chipmunk version will sound noticeably faster/shorter and the deep-voice version slower/longer than the original — that's expected (same effect as tape speed), not a bug.

---

## Libraries

| Library             | Purpose                                     |
| ------------------- | -------------------------------------------- |
| `sounddevice`       | Cross-platform audio I/O (primary)          |
| `pyaudio`           | Low-level PortAudio bindings                |
| `SpeechRecognition` | Speech-to-text wrapper (Google, offline)    |
| `openai-whisper`    | OpenAI Whisper local transcription          |
| `vosk`              | Offline streaming speech recognition        |
| `numpy`             | Audio buffer processing & FFT               |
| `matplotlib`        | Real-time waveform / spectrum visualisation |
| `scipy`             | Signal processing utilities                 |
| `rich`              | Terminal UI panels, live displays, tables   |
| `pyttsx3`           | Cross-platform text-to-speech               |

---

## Project Structure

```
demo/
├── main.py                     # Menu-driven launcher
├── requirements.txt
├── setup.sh                    # One-shot setup script
├── demos/
│   ├── mic_input.py             # Demo 1
│   ├── live_transcription.py    # Demo 2
│   ├── wake_word.py             # Demo 3
│   ├── sound_visualization.py   # Demo 4
│   ├── voice_assistant.py       # Demo 5
│   ├── noise_detection.py       # Demo 6
│   └── voice_transformation.py  # Demo 7
├── utils/
│   └── audio_utils.py          # Shared helpers
└── models/                     # Downloaded automatically by setup.sh
    └── vosk-model-small-en-us-0.15/
```

---

## Engine Priority

Demos that do transcription auto-select the best available engine:

```
Whisper (offline, tiny model)  →  Vosk (offline, streaming)  →  Google STT (online)
```

You can override auto-detection and force a specific engine using the `FORCE_ENGINE` environment variable:

```bash
# Force Google STT (shows SpeechRecognition tuning settings live)
FORCE_ENGINE=google python demos/live_transcription.py

# Force Whisper
FORCE_ENGINE=whisper python demos/live_transcription.py

# Force Vosk
FORCE_ENGINE=vosk python demos/live_transcription.py
```

> **Note:** `energy_threshold`, `dynamic_energy_threshold`, and `pause_threshold` (the `Recognizer` tuning settings) only apply to the Google engine. If Whisper is installed, auto-detection skips Google entirely — use `FORCE_ENGINE=google` to exercise that code path.

---

## Troubleshooting

**No audio input detected**

- Check microphone permissions: System Settings → Privacy & Security → Microphone (macOS)
- Run Demo 1 first to verify device detection, or use the standalone sanity-check script (records 3 seconds, shows a live level meter, no other setup needed): `python mic_test.py`
- If using Bluetooth headphones/AirPods, make sure they're selected as the **input** device in your OS sound settings, not just the output

**`portaudio` not found / `pyaudio` fails to install**

```bash
# macOS
brew install portaudio

# Linux
sudo apt-get install portaudio19-dev
```

Then re-run `pip install -r requirements.txt`.

**`vosk` version not found by pip**

Make sure `requirements.txt` pins a version that's actually published (check available versions with `pip index versions vosk`) — a pin on a version that was never released will make the entire `pip install -r requirements.txt` fail, even for unrelated packages.

**Whisper model slow on first run**

- The tiny model (~39 MB) downloads automatically on first use
- Pre-download with: `python -c "import whisper; whisper.load_model('tiny')"`

**Visualization window doesn't open**

- Make sure `matplotlib` is installed: `pip install matplotlib`
- On macOS the `MacOSX` backend is used by default; change to `TkAgg` in `sound_visualization.py` if needed

**`python: command not found` even after activating a virtual environment**

- Make sure you actually ran `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows) in the *same* terminal session you're running commands in — the `(.venv)` prefix should show in your prompt
- If the venv was created, then the project folder was later renamed or moved, the venv's internal paths go stale. Recreate it: `rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

---

## Learn More — Package Docs

| Package | What it's for here | Docs |
| --- | --- | --- |
| `sounddevice` | Mic/speaker I/O | https://python-sounddevice.readthedocs.io/ |
| `numpy` | Audio as arrays, FFT, math | https://numpy.org/doc/ |
| `scipy` | Signal processing (resample, filters) | https://docs.scipy.org/doc/scipy/ |
| `rich` | Terminal UI (panels, tables, live meters) | https://rich.readthedocs.io/ |
| `matplotlib` | Real-time waveform/spectrum visualization | https://matplotlib.org/stable/ |
| `SpeechRecognition` | Speech-to-text wrapper (Google engine) | https://pypi.org/project/SpeechRecognition/ |
| `openai-whisper` | Offline speech-to-text | https://github.com/openai/whisper |
| `vosk` | Offline streaming speech-to-text | https://alphacephei.com/vosk/ |
| `pyttsx3` | Text-to-speech | https://pyttsx3.readthedocs.io/ |
| `pyaudio` | Low-level PortAudio bindings | https://pypi.org/project/PyAudio/ |


