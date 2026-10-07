import datetime
import io
import math
import os
import queue
import random
import sys
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.audio_utils import SAMPLE_RATE, CHANNELS, DTYPE, rms, level_bar

console = Console()

# ── Configuration ──────────────────────────────────────────────────────────
WAKE_WORDS       = {"hey python", "python", "computer", "assistant"}
ENERGY_THRESHOLD = 0.008
COMMAND_SECS     = 5.0

JOKES = [
    "Why do Python programmers wear glasses? Because they can't C!",
    "I tried to catch some fog earlier. I mist.",
    "Why did the function break up with the loop? It said 'you're too controlling'.",
    "There are 10 kinds of people: those who understand binary, and those who don't.",
    "A programmer's partner says 'go to the store, get a litre of milk, if they have eggs get a dozen'. "
    "Programmer returns with 12 litres of milk. They had eggs.",
    "Why don't scientists trust atoms? Because they make up everything.",
    "Why did the developer go broke? Because he used up all his cache.",
]

WEATHER_REPORTS = [
    "Currently 22°C and partly cloudy. Perfect day for coding outdoors!",
    "Overcast with a chance of afternoon thunderstorms. Pack an umbrella.",
    "Sunny and 27°C with a light breeze. Ideal conference weather!",
    "Foggy morning, clearing by noon. Highs around 18°C.",
]


# ── TTS helper ─────────────────────────────────────────────────────────────

_tts_engine = None
_tts_available = False

def _init_tts() -> None:
    global _tts_engine, _tts_available
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate",   160)
        engine.setProperty("volume", 0.9)
        _tts_engine    = engine
        _tts_available = True
    except Exception:
        _tts_available = False


def _speak(text: str) -> None:
    if _tts_available and _tts_engine:
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
            return
        except Exception:
            pass
    # Silent fallback — text is already displayed in the panel


# ── Command processor ──────────────────────────────────────────────────────

def _handle_command(text: str) -> str:
    t = text.lower().strip().rstrip(".")

    if any(g in t for g in ("hello", "hi ", "hey ", "hi$", "greet")):
        return f"Hello! I'm Python, your voice assistant. How can I help?"

    if "time" in t:
        return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."

    if "date" in t or "today" in t:
        return f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."

    if "day" in t:
        return f"Today is {datetime.datetime.now().strftime('%A')}."

    if "weather" in t:
        return random.choice(WEATHER_REPORTS)

    if "joke" in t:
        return random.choice(JOKES)

    if any(c in t for c in ("calculat", "what is", "how much")):
        # Extract a simple math expression
        expr = t
        for prefix in ("calculate", "what is", "how much is", "compute",
                       "what's", "whats"):
            expr = expr.replace(prefix, "").strip()
        try:
            # Safe eval: only allow numbers and basic operators
            allowed = set("0123456789+-*/(). ")
            if all(c in allowed for c in expr) and expr:
                result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — restricted
                return f"{expr.strip()} equals {result}"
        except Exception:
            pass
        return "Sorry, I couldn't parse that calculation."

    if any(k in t for k in ("what can you do", "help", "commands",
                             "capabilities")):
        return ("I can tell you the time, date, weather, tell jokes, "
                "perform calculations, and have a basic conversation. "
                "Just say my wake word and ask!")

    if any(k in t for k in ("stop", "exit", "bye", "goodbye", "quit")):
        return "__STOP__"

    if "name" in t and ("your" in t or "you" in t):
        return "I'm Python, your real-time voice assistant!"

    if "python" in t and "language" in t:
        return ("Python is a high-level, interpreted programming language "
                "known for its readability and versatility. It was created "
                "by Guido van Rossum in 1991.")

    return f"I heard: \"{text}\". I'm not sure how to respond to that yet!"


# ── Transcription ──────────────────────────────────────────────────────────

_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model


def _transcribe(audio: np.ndarray) -> str:
    try:
        model  = _get_whisper()
        result = model.transcribe(audio.astype(np.float32), fp16=False,
                                  language="en")
        return result["text"].strip().lower()
    except Exception:
        pass
    try:
        import speech_recognition as sr
        pcm = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        rec  = sr.Recognizer()
        data = sr.AudioData(buf.read(), SAMPLE_RATE, 2)
        return rec.recognize_google(data).lower()
    except Exception:
        return ""


# ── Chime ──────────────────────────────────────────────────────────────────

def _chime(freq: float = 660.0, dur: float = 0.12) -> None:
    t    = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    tone = 0.25 * np.sin(2 * np.pi * freq * t) * np.exp(-5 * t / dur)
    sd.play(tone.astype(np.float32), samplerate=SAMPLE_RATE)
    sd.wait()


# ── State ──────────────────────────────────────────────────────────────────
_state         = "idle"
_log: list[str]= []
_current_db    = -96.0
_running       = True


def _panel() -> Panel:
    content = Text()

    state_map = {
        "idle":     ("[cyan]● Waiting for wake word …[/cyan]",),
        "detected": ("[bold yellow]🔔  Wake word detected![/bold yellow]",),
        "listen":   ("[bold red]🎙  Listening for command …[/bold red]",),
        "think":    ("[bold magenta]⚙  Processing …[/bold magenta]",),
        "speak":    ("[bold green]🔊  Speaking …[/bold green]",),
    }
    label = state_map.get(_state, ("",))[0]
    content.append(f"  {label}\n\n")
    content.append(f"  {level_bar(_current_db)}\n\n")

    for entry in _log[-10:]:
        content.append(f"  {entry}\n")

    content.append(
        "\n  [dim]Wake words: " +
        ", ".join(f'"{w}"' for w in sorted(WAKE_WORDS)) + "[/dim]"
    )
    return Panel(
        content,
        title="[bold]🤖  Voice Assistant[/bold]",
        border_style="bright_magenta",
        padding=(1, 2),
    )


# ── Main loop ──────────────────────────────────────────────────────────────

def _assistant_loop(stop_event: threading.Event) -> None:
    global _state, _current_db, _running

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)

    def callback(indata, frames, t, status):
        global _current_db
        chunk = indata[:, 0].copy()
        _current_db = 20 * np.log10(max(rms(chunk), 1e-10))
        try:
            audio_q.put_nowait(chunk)
        except queue.Full:
            pass

    chunk_secs = 1.5
    target     = int(SAMPLE_RATE * chunk_secs)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=callback,
                        blocksize=int(SAMPLE_RATE * 0.05)):
        buf: list[np.ndarray] = []
        buf_len = 0

        while not stop_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            buf.append(chunk); buf_len += len(chunk)

            if buf_len < target:
                continue

            audio = np.concatenate(buf)
            buf.clear(); buf_len = 0

            if rms(audio) < ENERGY_THRESHOLD:
                _state = "idle"
                continue

            # Transcribe detection window
            text = _transcribe(audio)
            if not text:
                _state = "idle"
                continue

            ts = time.strftime("%H:%M:%S")

            if any(ww in text for ww in WAKE_WORDS):
                _state = "detected"
                _log.append(f"[{ts}] [yellow]WAKE[/yellow] — \"{text}\"")
                threading.Thread(target=_chime, daemon=True).start()
                time.sleep(0.35)

                # Collect command
                _state = "listen"
                cmd_buf: list[np.ndarray] = []
                deadline = time.time() + COMMAND_SECS
                while time.time() < deadline and not stop_event.is_set():
                    try:
                        c = audio_q.get(timeout=0.3)
                        cmd_buf.append(c)
                    except queue.Empty:
                        break

                if not cmd_buf:
                    _state = "idle"
                    continue

                _state = "think"
                cmd_audio = np.concatenate(cmd_buf)
                cmd_text  = _transcribe(cmd_audio)
                if not cmd_text:
                    _log.append(f"[{ts}] [dim]Command: (unclear)[/dim]")
                    _state = "idle"
                    continue

                _log.append(f"[{ts}] [bold]You:[/bold] \"{cmd_text}\"")

                reply = _handle_command(cmd_text)
                if reply == "__STOP__":
                    _log.append(f"[{ts}] [bold green]Assistant:[/bold green] Goodbye!")
                    _state = "speak"
                    _speak("Goodbye!")
                    stop_event.set()
                    break

                _log.append(f"[{ts}] [bold green]Assistant:[/bold green] {reply}")
                _state = "speak"
                _speak(reply)
                _state = "idle"

            elif len(text) > 3:
                # Background speech, not a wake word
                pass


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global _state, _log, _current_db
    _state = "idle"; _log = []; _current_db = -96.0

    console.rule("[bold bright_magenta]Demo 5 — Voice Assistant[/bold bright_magenta]")
    console.print()
    console.print("  [bold]Wake words:[/bold]  " +
                  "  ".join(f'[yellow]{w}[/yellow]' for w in sorted(WAKE_WORDS)))
    console.print("  [bold]Commands:[/bold]   time · date · weather · joke · "
                  "calculate · hello · what can you do · stop")
    console.print("\n  [dim]Say a wake word, then your command. "
                  "Say 'stop' to end.[/dim]\n")

    # Initialise TTS in background
    threading.Thread(target=_init_tts, daemon=True).start()
    threading.Thread(target=_get_whisper, daemon=True).start()

    stop_event = threading.Event()
    worker     = threading.Thread(target=_assistant_loop,
                                  args=(stop_event,), daemon=True)
    worker.start()

    try:
        with Live(_panel(), refresh_per_second=8, console=console) as live:
            while not stop_event.is_set():
                live.update(_panel())
                time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    console.print(f"\n[bold green]Session ended — {len([l for l in _log if 'You:' in l])} command(s) processed.[/bold green]\n")


if __name__ == "__main__":
    main()
