"""
demos/live_transcription.py
──────────────────────────────────────────────────────────────────────────────
Demo 2 — Live Transcription
  • Engine A: Google Speech Recognition (online, fast, no model download)
  • Engine B: OpenAI Whisper  (offline, tiny model, ~39 MB)
  • Engine C: Vosk            (offline, streaming, ~50 MB)

  Engines are tried in priority order; whichever is available runs.
  Press Ctrl+C to stop.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import queue
import threading
import time
import io
import struct
import wave

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.audio_utils import SAMPLE_RATE, CHANNELS, DTYPE

console = Console()

VOSK_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "vosk-model-small-en-us-0.15"
)


# ── Engine detection ───────────────────────────────────────────────────────

def _detect_engine() -> str:
    try:
        import whisper  # noqa: F401
        return "whisper"
    except ImportError:
        pass
    try:
        import vosk  # noqa: F401
        if os.path.isdir(VOSK_MODEL_PATH):
            return "vosk"
    except ImportError:
        pass
    return "google"  # fallback — requires internet


# ── Whisper transcription ──────────────────────────────────────────────────

class WhisperEngine:
    def __init__(self):
        import whisper
        console.print("[cyan]Loading Whisper tiny model …[/cyan]")
        self.model = whisper.load_model("tiny")
        self.sr    = 16_000

    def transcribe(self, audio: np.ndarray) -> str:
        audio = audio.astype(np.float32)
        result = self.model.transcribe(audio, fp16=False, language="en")
        return result["text"].strip()


# ── Vosk transcription (streaming) ────────────────────────────────────────

class VoskEngine:
    def __init__(self):
        import vosk
        vosk.SetLogLevel(-1)
        console.print("[cyan]Loading Vosk model …[/cyan]")
        self.model = vosk.Model(VOSK_MODEL_PATH)
        self.rec   = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        self.sr    = SAMPLE_RATE

    def feed(self, audio_int16: bytes) -> str | None:
        """Feed raw PCM bytes; return final text when sentence ends."""
        import json
        if self.rec.AcceptWaveform(audio_int16):
            res = json.loads(self.rec.Result())
            return res.get("text", "").strip() or None
        return None

    def partial(self) -> str:
        import json
        res = json.loads(self.rec.PartialResult())
        return res.get("partial", "")


# ── Google STT (using SpeechRecognition) ──────────────────────────────────

class GoogleEngine:
    def __init__(self):
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        # Tune for accuracy: lower threshold = more sensitive in quiet rooms
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True  # auto-adjusts to background noise
        self.recognizer.pause_threshold = 0.8            # seconds of silence = end of phrase
        self.sr = SAMPLE_RATE

    def transcribe(self, audio: np.ndarray) -> str:
        import speech_recognition as sr
        pcm = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        audio_data = sr.AudioData(buf.read(), SAMPLE_RATE, 2)
        try:
            return self.recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return f"[Google error: {e}]"


# ── Shared state ───────────────────────────────────────────────────────────
_transcript_lines: list[str] = []
_status: str = "Listening …"
_partial: str = ""


def _panel() -> Panel:
    content = Text()
    for line in _transcript_lines[-12:]:  # last 12 utterances
        content.append(f"  {line}\n", style="white")
    if _partial:
        content.append(f"  [dim]… {_partial}[/dim]\n")
    content.append(f"\n  [bold cyan]{_status}[/bold cyan]")
    return Panel(
        content,
        title="[bold]📝  Live Transcription[/bold]",
        border_style="bright_green",
        padding=(1, 2),
    )


# ── Runners ────────────────────────────────────────────────────────────────

def _run_whisper_or_google(engine, chunk_secs: float = 3.0) -> None:
    global _status, _partial

    chunk_size = int(SAMPLE_RATE * chunk_secs)
    buf: list[np.ndarray] = []
    buf_len: int = 0

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, t, status):
        audio_q.put(indata[:, 0].copy())

    _status = f"Using [bold]{'Whisper' if hasattr(engine, 'model') else 'Google'}[/bold] — speak freely"

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=callback):
        while True:
            chunk = audio_q.get()
            buf.append(chunk)
            buf_len += len(chunk)
            if buf_len >= chunk_size:
                audio = np.concatenate(buf)
                buf.clear(); buf_len = 0
                # Silence detection: skip transcription if audio is too quiet
                rms = np.sqrt(np.mean(audio ** 2))
                if rms < 0.01:
                    _status = "Listening … (silence detected)"
                    continue
                _status = "Transcribing …"
                text = engine.transcribe(audio)
                if text:
                    ts = time.strftime("%H:%M:%S")
                    _transcript_lines.append(f"[{ts}]  {text}")
                _status = "Listening …"


def _run_vosk(engine: VoskEngine) -> None:
    global _status, _partial

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, t, status):
        # Vosk wants int16 PCM
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        audio_q.put(pcm.tobytes())

    _status = "Using [bold]Vosk[/bold] (offline streaming) — speak freely"

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=callback,
                        blocksize=4000):
        while True:
            raw = audio_q.get()
            text = engine.feed(raw)
            if text:
                ts = time.strftime("%H:%M:%S")
                _transcript_lines.append(f"[{ts}]  {text}")
                _partial = ""
            else:
                _partial = engine.partial()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global _transcript_lines, _status, _partial
    _transcript_lines = []
    _status = "Initialising …"
    _partial = ""

    console.rule("[bold bright_green]Demo 2 — Live Transcription[/bold bright_green]")

    engine_name = os.environ.get("FORCE_ENGINE") or _detect_engine()
    console.print(f"[green]Engine detected:[/green] [bold]{engine_name}[/bold]\n")

    if engine_name == "whisper":
        engine = WhisperEngine()
        runner = lambda: _run_whisper_or_google(engine)
    elif engine_name == "vosk":
        engine = VoskEngine()
        runner = lambda: _run_vosk(engine)
    else:
        console.print("[yellow]Whisper/Vosk not found — falling back to Google STT (requires internet)[/yellow]")
        engine = GoogleEngine()
        runner = lambda: _run_whisper_or_google(engine)

    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    try:
        with Live(_panel(), refresh_per_second=8, console=console) as live:
            while True:
                live.update(_panel())
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    console.print(f"\n[bold green]Captured {len(_transcript_lines)} utterance(s).[/bold green]\n")


if __name__ == "__main__":
    main()
