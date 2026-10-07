#!/usr/bin/env python3
"""Quick mic test — records 3 seconds and shows live level meter."""

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

console = Console()
SAMPLE_RATE = 16_000
DURATION = 3

console.print("\n[bold cyan]Mic Test[/bold cyan] — checking your microphone\n")

# Show available devices
devices = sd.query_devices()
default_in = sd.default.device[0]
console.print(f"[green]Default input device:[/green] {devices[default_in]['name']}\n")

# Record 3 seconds with live level bar
console.print("[yellow]Recording for 3 seconds — say something![/yellow]\n")

recorded = []

def callback(indata, frames, time, status):
    recorded.append(indata.copy())

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
    with Live(console=console, refresh_per_second=20) as live:
        import time
        start = time.time()
        while time.time() - start < DURATION:
            if recorded:
                chunk = recorded[-1][:, 0]
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                db = 20 * np.log10(rms) if rms > 0 else -96
                filled = max(0, int((db + 60) / 60 * 40))
                colour = "red" if db > -6 else "yellow" if db > -18 else "green"
                bar = f"[{colour}]{'█' * filled}{'░' * (40 - filled)}[/{colour}] {db:+.1f} dBFS"
                elapsed = time.time() - start
                live.update(Panel(bar, title=f"[bold]🎙 Level — {elapsed:.1f}s / {DURATION}s[/bold]", border_style="cyan"))
            time.sleep(0.05)

audio = np.concatenate(recorded)[:, 0]
peak = float(np.max(np.abs(audio)))
console.print(f"\n[bold green]✓ Recorded {len(audio)/SAMPLE_RATE:.1f}s of audio[/bold green]")
console.print(f"  Peak level: [cyan]{peak:.4f}[/cyan]")

if peak < 0.001:
    console.print("\n[bold red]⚠ Very low signal — check mic permissions or device selection[/bold red]")
else:
    console.print("\n[bold green]✓ Mic is working![/bold green]")
