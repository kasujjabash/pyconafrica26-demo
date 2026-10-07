import queue
import sys
import time
import wave
import os

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.audio_utils import (
    SAMPLE_RATE, CHANNELS, DTYPE,
    list_audio_devices, db, level_bar, save_wav, play_audio,
)

console = Console()

# ── Internal state ─────────────────────────────────────────────────────────
_q: "queue.Queue[np.ndarray]" = queue.Queue()
_current_db: float = -96.0
_peak_db:    float = -96.0
_frames:     list  = []


def _callback(indata: np.ndarray, frames: int, t, status) -> None:
    global _current_db, _peak_db
    chunk = indata[:, 0].copy()
    _current_db = db(chunk)
    if _current_db > _peak_db:
        _peak_db = _current_db
    _q.put(chunk)
    _frames.append(chunk)


# ── Renderers ──────────────────────────────────────────────────────────────

def _make_panel(phase: str, countdown: int | None = None) -> Panel:
    lines = []

    bar  = level_bar(_current_db)
    peak = level_bar(_peak_db)
    lines.append(f" [bold]Level[/bold]  {bar}")
    lines.append(f" [dim]Peak [/dim]  {peak}")
    lines.append("")

    if phase == "monitor":
        lines.append(" [bold cyan]● LIVE MONITORING[/bold cyan]  (press Ctrl+C to stop)")
    elif phase == "record":
        lines.append(f" [bold red]⏺  RECORDING …  {countdown}s remaining[/bold red]")
    elif phase == "done":
        lines.append(" [bold green]✓  Recording complete![/bold green]")

    return Panel(
        "\n".join(lines),
        title="[bold]🎙  Microphone Input[/bold]",
        border_style="bright_blue",
        padding=(1, 2),
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global _peak_db, _current_db
    _frames.clear()
    _peak_db    = -96.0
    _current_db = -96.0

    console.rule("[bold bright_blue]Demo 1 — Microphone Input[/bold bright_blue]")
    console.print()

    # Show device table
    list_audio_devices()
    console.print()

    dev = sd.query_devices(kind="input")
    console.print(f"[green]Using:[/green] [bold]{dev['name']}[/bold]  "
                  f"@ [cyan]{SAMPLE_RATE} Hz[/cyan]  mono\n")

    # ── Phase 1: Live level monitor (5 s) ─────────────────────────────────
    console.print("[dim]Monitoring mic for 5 seconds …[/dim]")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=_callback):
        with Live(_make_panel("monitor"), refresh_per_second=20,
                  console=console) as live:
            deadline = time.time() + 5
            while time.time() < deadline:
                live.update(_make_panel("monitor"))
                time.sleep(0.05)

    _peak_db    = -96.0          # reset peak before recording
    _current_db = -96.0
    _frames.clear()

    console.print()
    console.print("[bold yellow]Starting 5-second recording …[/bold yellow]")
    time.sleep(0.5)

    # ── Phase 2: Record 5 s with countdown ────────────────────────────────
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=_callback):
        with Live(_make_panel("record", 5), refresh_per_second=20,
                  console=console) as live:
            start = time.time()
            while True:
                elapsed   = time.time() - start
                remaining = max(0, int(5 - elapsed))
                live.update(_make_panel("record", remaining))
                if elapsed >= 5:
                    break
                time.sleep(0.05)
        live.update(_make_panel("done"))

    # ── Save & playback ───────────────────────────────────────────────────
    audio = np.concatenate(_frames) if _frames else np.zeros(SAMPLE_RATE * 5)
    out_path = "/tmp/python_can_hear_recording.wav"
    save_wav(out_path, audio)
    console.print(f"\n[green]Saved recording → [bold]{out_path}[/bold][/green]")

    console.print("[yellow]Playing back …[/yellow]")
    play_audio(audio)
    console.print("[bold green]Playback done.[/bold green]\n")


if __name__ == "__main__":
    main()
