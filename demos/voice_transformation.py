"""
demos/voice_transformation.py
──────────────────────────────────────────────────────────────────────────────
Demo 7 — Voice Transformation
  • Records a short clip of your voice
  • Applies pitch shifting (up = "chipmunk", down = "deep voice") via resampling
  • Applies echo via a delay-and-mix
  • Plays each transformed version back
  • Press Ctrl+C during recording to abort
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time

import numpy as np
from scipy.signal import resample
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.audio_utils import SAMPLE_RATE, record_audio, play_audio, level_bar, db

console = Console()

RECORD_SECS = 4.0


# ── Effects ──────────────────────────────────────────────────────────────────

def pitch_shift(audio: np.ndarray, factor: float) -> np.ndarray:
    """
    factor > 1.0 = higher pitch ("chipmunk"), < 1.0 = lower ("deep voice").

    Resampling to fewer samples describes the same waveform with fewer points,
    so playing them back at SAMPLE_RATE reads the wave faster: the pitch goes
    up AND the clip gets shorter. That is exactly what speeding up a tape does.

    NOTE: resampling a second time back to len(audio) does NOT keep the pitch
    change -- it undoes it. Down then up is a round trip, and you get the
    original pitch, the original length, and a slightly duller sound.
    Verified: a 200 Hz tone at factor 1.6 came back out at 200 Hz, not 320 Hz.
    Changing pitch while KEEPING the duration needs time-stretching
    (a phase vocoder), which is a different job.
    """
    return resample(audio, max(1, int(len(audio) / factor)))


def echo(audio: np.ndarray, delay_secs: float = 0.25,
         decay: float = 0.5, repeats: int = 3) -> np.ndarray:
    """Classic delay-and-mix echo."""
    delay_samples = int(delay_secs * SAMPLE_RATE)
    out = np.copy(audio)
    for i in range(1, repeats + 1):
        pad = np.zeros(delay_samples * i)
        echoed = np.concatenate([pad, audio * (decay ** i)])
        if len(echoed) > len(out):
            out = np.pad(out, (0, len(echoed) - len(out)))
        else:
            echoed = np.pad(echoed, (0, len(out) - len(echoed)))
        out = out + echoed
    # Only turn it DOWN if it would clip -- never boost a quiet recording,
    # or the echo plays back much louder than the original (measured 3.3x).
    return out / max(1.0, np.max(np.abs(out)))     # normalize, avoid clipping


# ── Recording UI ───────────────────────────────────────────────────────────

def _countdown_panel(remaining: float, current_db: float) -> Panel:
    bar = level_bar(current_db)
    content = (
        f" [bold red]⏺  RECORDING …  {remaining:.1f}s remaining[/bold red]\n\n"
        f" [bold]Level[/bold]  {bar}"
    )
    return Panel(
        content,
        title="[bold]🎚  Voice Transformation — Recording[/bold]",
        border_style="bright_magenta",
        padding=(1, 2),
    )


def _record_with_countdown(duration: float) -> np.ndarray:
    """Records `duration` seconds while showing a live countdown + level bar."""
    import queue
    import sounddevice as sd
    from utils.audio_utils import CHANNELS, DTYPE

    q: "queue.Queue[np.ndarray]" = queue.Queue()
    frames: list = []
    state = {"db": -96.0}

    def callback(indata, _frames, _time, _status):
        chunk = indata[:, 0].copy()
        state["db"] = db(chunk)
        frames.append(chunk)

    console.print(f"[dim]Get ready — recording starts in a moment …[/dim]")
    time.sleep(1.0)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=callback):
        with Live(_countdown_panel(duration, -96.0), refresh_per_second=20,
                  console=console) as live:
            start = time.time()
            while True:
                elapsed   = time.time() - start
                remaining = max(0.0, duration - elapsed)
                live.update(_countdown_panel(remaining, state["db"]))
                if elapsed >= duration:
                    break
                time.sleep(0.05)

    return np.concatenate(frames) if frames else np.zeros(int(SAMPLE_RATE * duration))


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold bright_magenta]Demo 7 — Voice Transformation[/bold bright_magenta]")
    console.print(
        "\n  Record a short phrase, then hear it [bold]pitched up[/bold], "
        "[bold]pitched down[/bold], and [bold]echoed[/bold] — all pure numpy/scipy.\n"
    )

    try:
        audio = _record_with_countdown(RECORD_SECS)
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording aborted.[/yellow]\n")
        return

    if np.max(np.abs(audio)) < 1e-4:
        console.print(
            "\n[bold red]No signal detected — check mic permissions/input device "
            "(see Demo 1).[/bold red]\n"
        )
        return

    console.print("\n[bold green]✓ Recorded.[/bold green] Playing back the original …")
    play_audio(audio)

    effects = [
        ("🐿  Pitch UP (chipmunk)", lambda a: pitch_shift(a, 1.6)),
        ("🐻  Pitch DOWN (deep voice)", lambda a: pitch_shift(a, 0.7)),
        ("🌊 Echo", lambda a: echo(a)),
    ]

    for label, fn in effects:
        console.print(f"\n[bold cyan]{label}[/bold cyan]")
        with console.status("[dim]processing …[/dim]", spinner="dots"):
            transformed = fn(audio)
        play_audio(transformed)

    console.print("\n[bold green]Done — that's pitch shifting and echo in a few lines "
                  "of numpy/scipy.[/bold green]\n")


if __name__ == "__main__":
    main()
