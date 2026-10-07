"""
demos/wake_word.py
──────────────────────────────────────────────────────────────────────────────
Demo 3 — Wake Word Detection
  • Continuously listens in the background (energy-gated, low CPU)
  • Detects configurable wake words: "python", "hey python", "computer", "listen"
  • On detection: plays a chime, shows an alert, records the follow-up command
  • Transcribes and displays the command
  • Press Ctrl+C to stop
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import queue
import threading
import time
import io
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
WAKE_WORDS       = {"python", "hey python", "computer", "listen", "wake up"}
ENERGY_THRESHOLD = 0.01      # minimum RMS to even attempt transcription
LISTEN_SECS      = 3.0       # record for this long after hearing something
COMMAND_SECS     = 4.0       # record follow-up command for this long


# ── State ──────────────────────────────────────────────────────────────────
_state: str          = "idle"   # idle | triggered | listening_command
_event_log: list[str] = []
_detections: int     = 0
_current_db: float   = -96.0


# ── Transcription helper ───────────────────────────────────────────────────

def _transcribe(audio: np.ndarray) -> str:
    """Try Whisper → Google; return text or empty string."""
    # Try Whisper first (offline)
    try:
        import whisper
        model = _get_whisper_model()
        result = model.transcribe(audio.astype(np.float32), fp16=False,
                                  language="en")
        return result["text"].strip().lower()
    except Exception:
        pass

    # Fallback to Google STT
    try:
        import speech_recognition as sr
        pcm = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        rec = sr.Recognizer()
        data = sr.AudioData(buf.read(), SAMPLE_RATE, 2)
        return rec.recognize_google(data).lower()
    except Exception:
        return ""


_whisper_model = None
def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model


# ── Chime (generated tone) ─────────────────────────────────────────────────

def _play_chime(freq: float = 880.0, duration: float = 0.15,
                sr: int = SAMPLE_RATE) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-3 * t / duration)
    sd.play(tone.astype(np.float32), samplerate=sr)
    sd.wait()


# ── Panel renderer ─────────────────────────────────────────────────────────

def _panel() -> Panel:
    content = Text()

    # State indicator
    if _state == "idle":
        content.append("  ◉ Listening for wake word …\n\n", style="cyan")
    elif _state == "triggered":
        content.append("  🔔  WAKE WORD DETECTED!\n\n", style="bold yellow")
    elif _state == "listening_command":
        content.append("  🎙  Recording your command …\n\n", style="bold red")

    # Level bar
    content.append(f"  {level_bar(_current_db)}\n\n")

    # Event log
    content.append(f"  [bold]Detections:[/bold] {_detections}\n\n")
    for entry in _event_log[-8:]:
        content.append(f"  {entry}\n")

    content.append("\n  [dim]Wake words: " +
                   ", ".join(f'"{w}"' for w in sorted(WAKE_WORDS)) +
                   "[/dim]")

    return Panel(
        content,
        title="[bold]👂  Wake Word Detection[/bold]",
        border_style="bright_yellow",
        padding=(1, 2),
    )


# ── Core detection loop ────────────────────────────────────────────────────

def _detection_loop() -> None:
    global _state, _detections, _current_db

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    chunk_secs = 1.5    # capture in 1.5-second windows

    def callback(indata, frames, t, status):
        _current_db = 20 * np.log10(max(rms(indata[:, 0]), 1e-10))
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=callback,
                        blocksize=int(SAMPLE_RATE * 0.05)):

        buf: list[np.ndarray] = []
        buf_len = 0
        target  = int(SAMPLE_RATE * chunk_secs)

        while True:
            chunk = audio_q.get()
            buf.append(chunk)
            buf_len += len(chunk)

            if buf_len < target:
                continue

            audio = np.concatenate(buf)
            buf.clear(); buf_len = 0

            # Energy gate — skip silence
            if rms(audio) < ENERGY_THRESHOLD:
                _state = "idle"
                continue

            # Transcribe the window
            text = _transcribe(audio)
            if not text:
                _state = "idle"
                continue

            ts = time.strftime("%H:%M:%S")

            # Check for wake word
            if any(ww in text for ww in WAKE_WORDS):
                _detections += 1
                _state = "triggered"
                _event_log.append(
                    f"[{ts}] [bold yellow]WAKE WORD[/bold yellow] — heard: \"{text}\""
                )
                threading.Thread(target=_play_chime, daemon=True).start()
                time.sleep(0.3)

                # Record follow-up command
                _state = "listening_command"
                cmd_chunks: list[np.ndarray] = []
                deadline = time.time() + COMMAND_SECS
                while time.time() < deadline:
                    try:
                        c = audio_q.get(timeout=0.5)
                        cmd_chunks.append(c)
                    except queue.Empty:
                        break

                if cmd_chunks:
                    cmd_audio  = np.concatenate(cmd_chunks)
                    cmd_text   = _transcribe(cmd_audio)
                    if cmd_text:
                        _event_log.append(
                            f"[{ts}] [bold green]Command:[/bold green] \"{cmd_text}\""
                        )
                    else:
                        _event_log.append(f"[{ts}] [dim]Command: (unclear)[/dim]")

                _state = "idle"
            else:
                # Log non-wake speech at lower verbosity
                if len(text) > 3:
                    _event_log.append(f"[{ts}] [dim]heard: \"{text}\"[/dim]")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global _state, _detections, _event_log, _current_db
    _state = "idle"; _detections = 0; _event_log = []; _current_db = -96.0

    console.rule("[bold bright_yellow]Demo 3 — Wake Word Detection[/bold bright_yellow]")
    console.print()
    console.print(f"  Say one of the wake words to trigger detection:")
    for ww in sorted(WAKE_WORDS):
        console.print(f"    [bold yellow]•  \"{ww}\"[/bold yellow]")
    console.print("\n  [dim]Press Ctrl+C to stop.[/dim]\n")

    # Pre-load model in background
    preload = threading.Thread(target=_get_whisper_model, daemon=True)
    preload.start()

    t = threading.Thread(target=_detection_loop, daemon=True)
    t.start()

    try:
        with Live(_panel(), refresh_per_second=8, console=console) as live:
            while True:
                live.update(_panel())
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    console.print(f"\n[bold green]Session ended — {_detections} wake word(s) detected.[/bold green]\n")


if __name__ == "__main__":
    main()
