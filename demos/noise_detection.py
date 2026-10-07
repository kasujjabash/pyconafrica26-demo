import os
import sys
import queue
import statistics
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.audio_utils import SAMPLE_RATE, CHANNELS, DTYPE, db, rms

console = Console()

# ── Configuration ──────────────────────────────────────────────────────────
ALERT_THRESHOLD_DB  = -20.0    # dBFS — above this → alert
SILENCE_THRESHOLD_DB= -50.0    # dBFS — below this → silence
WINDOW_SECS         = 10.0     # statistics window
EVENT_COOLDOWN_SECS = 2.0      # min gap between noise events logged

# ── Shared state ───────────────────────────────────────────────────────────
_current_db  = -96.0
_db_window   = deque(maxlen=int(WINDOW_SECS * 20))  # ~20 samples/sec
_events: list[str] = []
_alerts      = 0
_silence_periods = 0
_last_event_time = 0.0
_is_alert    = False
_is_silence  = False
_start_time  = time.time()

_audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=30)


# ── Audio callback ─────────────────────────────────────────────────────────

def _callback(indata: np.ndarray, frames: int, t, status) -> None:
    global _current_db
    chunk = indata[:, 0].copy()
    _current_db = db(chunk)
    try:
        _audio_q.put_nowait(chunk)
    except queue.Full:
        pass


# ── Level bar (with alert colouring) ──────────────────────────────────────

def _db_bar(value: float, width: int = 44,
            low: float = -80.0, high: float = 0.0) -> str:
    span    = high - low
    clamped = max(low, min(high, value))
    filled  = int((clamped - low) / span * width)

    # Colour zones
    alert_pos   = int((ALERT_THRESHOLD_DB - low) / span * width)
    silence_pos = int((SILENCE_THRESHOLD_DB - low) / span * width)

    bar = ""
    for i in range(width):
        ch = "█" if i < filled else "░"
        if i >= alert_pos:
            bar += f"[bold red]{ch}[/bold red]"
        elif i >= silence_pos:
            bar += f"[green]{ch}[/green]"
        else:
            bar += f"[dim]{ch}[/dim]"

    return bar + f" [{'bold red' if _is_alert else 'white'}]{value:+.1f} dBFS[/]"


# ── Stat helpers ───────────────────────────────────────────────────────────

def _stats():
    if not _db_window:
        return -96.0, -96.0, -96.0
    vals = list(_db_window)
    return min(vals), statistics.mean(vals), max(vals)


def _elapsed() -> str:
    s = int(time.time() - _start_time)
    return f"{s // 60:02d}:{s % 60:02d}"


# ── Panel renderer ─────────────────────────────────────────────────────────

def _panel() -> Panel:
    content = Text()

    # Level bar
    content.append("  Level    ")
    content.append(_db_bar(_current_db))
    content.append("\n\n")

    # Threshold indicators
    content.append(
        f"  Alert threshold   [bold red]{ALERT_THRESHOLD_DB:+.0f} dBFS[/bold red]"
        f"   Silence threshold  [dim]{SILENCE_THRESHOLD_DB:+.0f} dBFS[/dim]\n\n"
    )

    # Rolling stats
    mn, avg, mx = _stats()
    content.append(
        f"  [bold]Rolling {WINDOW_SECS:.0f}s stats[/bold]   "
        f"min [cyan]{mn:+.1f}[/cyan]  "
        f"avg [yellow]{avg:+.1f}[/yellow]  "
        f"max [red]{mx:+.1f}[/red] dBFS\n\n"
    )

    # Counters
    content.append(
        f"  Noise alerts: [bold red]{_alerts}[/bold red]   "
        f"Silence periods: [dim]{_silence_periods}[/dim]   "
        f"Elapsed: [white]{_elapsed()}[/white]\n\n"
    )

    # State badge
    if _is_alert:
        content.append("  [bold reverse red]  ⚠  NOISE ALERT  ⚠  [/bold reverse red]\n\n")
    elif _is_silence:
        content.append("  [dim]  🤫  Silence detected  [/dim]\n\n")
    else:
        content.append("  [cyan]  ● Monitoring …[/cyan]\n\n")

    # Event log
    content.append("  [bold]Events:[/bold]\n")
    for entry in _events[-8:]:
        content.append(f"    {entry}\n")

    return Panel(
        content,
        title="[bold]🔊  Noise Detection & Monitoring[/bold]",
        border_style="bright_red",
        padding=(1, 2),
    )


# ── Monitoring thread ──────────────────────────────────────────────────────

def _monitor_loop(stop_event: threading.Event) -> None:
    global _is_alert, _is_silence, _alerts, _silence_periods, _last_event_time

    while not stop_event.is_set():
        try:
            chunk = _audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        level = db(chunk)
        _db_window.append(level)

        now = time.time()
        ts  = time.strftime("%H:%M:%S")

        # Noise alert
        if level > ALERT_THRESHOLD_DB:
            if not _is_alert:
                _is_alert = True
            if now - _last_event_time > EVENT_COOLDOWN_SECS:
                _alerts += 1
                _last_event_time = now
                _events.append(
                    f"[{ts}] [bold red]NOISE  {level:+.1f} dBFS[/bold red]"
                )
        else:
            _is_alert = False

        # Silence detection
        if level < SILENCE_THRESHOLD_DB:
            if not _is_silence:
                _is_silence = True
                _silence_periods += 1
                _events.append(
                    f"[{ts}] [dim]Silence  ({level:+.1f} dBFS)[/dim]"
                )
        else:
            _is_silence = False


# ── Final summary ──────────────────────────────────────────────────────────

def _print_summary() -> None:
    mn, avg, mx = _stats()
    elapsed = int(time.time() - _start_time)

    table = Table(title="Session Summary", show_lines=True)
    table.add_column("Metric",   style="cyan")
    table.add_column("Value",    style="white", justify="right")

    table.add_row("Duration",         f"{elapsed // 60}m {elapsed % 60}s")
    table.add_row("Noise alerts",     str(_alerts))
    table.add_row("Silence periods",  str(_silence_periods))
    table.add_row("Min level",        f"{mn:+.1f} dBFS")
    table.add_row("Avg level",        f"{avg:+.1f} dBFS")
    table.add_row("Peak level",       f"{mx:+.1f} dBFS")
    table.add_row("Alert threshold",  f"{ALERT_THRESHOLD_DB:+.0f} dBFS")
    table.add_row("Silence threshold",f"{SILENCE_THRESHOLD_DB:+.0f} dBFS")

    console.print(table)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global _current_db, _is_alert, _is_silence, _alerts, _silence_periods
    global _last_event_time, _start_time, _events
    _current_db      = -96.0
    _is_alert        = False
    _is_silence      = False
    _alerts          = 0
    _silence_periods = 0
    _last_event_time = 0.0
    _start_time      = time.time()
    _events          = []
    _db_window.clear()

    console.rule("[bold bright_red]Demo 6 — Noise Detection[/bold bright_red]")
    console.print(
        f"\n  [bold]Alert[/bold] when level exceeds [red]{ALERT_THRESHOLD_DB:+.0f} dBFS[/red]   "
        f"[bold]Silence[/bold] when below [dim]{SILENCE_THRESHOLD_DB:+.0f} dBFS[/dim]\n"
        "  [dim]Press Ctrl+C to stop and view session summary.[/dim]\n"
    )

    stop_event = threading.Event()
    worker     = threading.Thread(target=_monitor_loop,
                                  args=(stop_event,), daemon=True)
    worker.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=_callback,
                        blocksize=int(SAMPLE_RATE * 0.05)):
        try:
            with Live(_panel(), refresh_per_second=12, console=console) as live:
                while True:
                    live.update(_panel())
                    time.sleep(0.08)
        except KeyboardInterrupt:
            stop_event.set()

    console.print()
    _print_summary()
    console.print()


if __name__ == "__main__":
    main()
