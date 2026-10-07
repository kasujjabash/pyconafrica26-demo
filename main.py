#!/usr/bin/env python3
"""
main.py — "Python Can Hear: Building Real-Time Audio Applications"
Conference demo launcher.

Run:  python main.py
──────────────────────────────────────────────────────────────────────────────
"""

import importlib
import os
import sys
import time

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# ── Demo registry ──────────────────────────────────────────────────────────

DEMOS = [
    {
        "key":    "1",
        "module": "demos.mic_input",
        "title":  "Microphone Input",
        "desc":   "Live level meter · 5-second recording · instant playback",
        "icon":   "🎙",
        "libs":   ["sounddevice", "numpy"],
    },
    {
        "key":    "2",
        "module": "demos.live_transcription",
        "title":  "Live Transcription",
        "desc":   "Continuous speech-to-text via Whisper / Vosk / Google STT",
        "icon":   "📝",
        "libs":   ["sounddevice", "whisper / vosk / speech_recognition"],
    },
    {
        "key":    "3",
        "module": "demos.wake_word",
        "title":  "Wake Word Detection",
        "desc":   'Triggers on "hey python" — then records your command',
        "icon":   "👂",
        "libs":   ["sounddevice", "whisper"],
    },
    {
        "key":    "4",
        "module": "demos.sound_visualization",
        "title":  "Sound Visualization",
        "desc":   "Real-time waveform · FFT spectrum · spectrogram waterfall",
        "icon":   "📊",
        "libs":   ["sounddevice", "matplotlib", "numpy"],
    },
    {
        "key":    "5",
        "module": "demos.voice_assistant",
        "title":  "Voice Assistant",
        "desc":   'Wake word + commands: time · date · weather · jokes · maths',
        "icon":   "🤖",
        "libs":   ["sounddevice", "whisper", "pyttsx3"],
    },
    {
        "key":    "6",
        "module": "demos.noise_detection",
        "title":  "Noise Detection",
        "desc":   "dBFS monitor · threshold alerts · silence detection · summary",
        "icon":   "🔊",
        "libs":   ["sounddevice", "numpy"],
    },
    {
        "key":    "7",
        "module": "demos.voice_transformation",
        "title":  "Voice Transformation",
        "desc":   "Record a clip · pitch shift up/down · echo — pure numpy/scipy",
        "icon":   "🎚",
        "libs":   ["sounddevice", "numpy", "scipy"],
    },
]


# ── Dependency check ───────────────────────────────────────────────────────

def _check_imports() -> dict[str, bool]:
    checks = {
        "sounddevice":      "sounddevice",
        "numpy":            "numpy",
        "matplotlib":       "matplotlib",
        "rich":             "rich",
        "speech_recognition": "speech_recognition",
        "whisper":          "whisper",
        "vosk":             "vosk",
        "pyttsx3":          "pyttsx3",
        "scipy":            "scipy",
    }
    results = {}
    for name, pkg in checks.items():
        try:
            importlib.import_module(pkg)
            results[name] = True
        except ImportError:
            results[name] = False
    return results


# ── Header banner ──────────────────────────────────────────────────────────

BANNER = r"""
[bold bright_cyan]
  ██████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ███╗   ██╗
  ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔═══██╗████╗  ██║
  ██████╔╝ ╚████╔╝    ██║   ███████║██║   ██║██╔██╗ ██║
  ██╔═══╝   ╚██╔╝     ██║   ██╔══██║██║   ██║██║╚██╗██║
  ██║        ██║      ██║   ██║  ██║╚██████╔╝██║ ╚████║
  ╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
[/bold bright_cyan]
[bold white]              C A N   H E A R[/bold white]

[dim]    Building Real-Time Audio Applications with Python[/dim]
"""


def _print_header() -> None:
    console.print(BANNER)
    console.rule(style="bright_blue dim")


# ── Main menu ──────────────────────────────────────────────────────────────

def _print_menu(deps: dict[str, bool]) -> None:
    table = Table(
        show_header=True,
        header_style="bold bright_blue",
        show_lines=False,
        box=None,
        padding=(0, 2),
        min_width=72,
    )
    table.add_column("#",     style="bold cyan", width=4, justify="center")
    table.add_column("Icon",  width=4,           justify="center")
    table.add_column("Demo",  style="bold white", width=26)
    table.add_column("Description", style="dim white")

    for demo in DEMOS:
        table.add_row(
            demo["key"],
            demo["icon"],
            demo["title"],
            demo["desc"],
        )

    table.add_row("", "", "", "")
    table.add_row("[bold]d[/bold]", "🔍", "[bold]Dependencies[/bold]",
                  "Show library install status")
    table.add_row("[bold]q[/bold]", "🚪", "[bold]Quit[/bold]",
                  "Exit the launcher")

    console.print(
        Panel(table, title="[bold]Choose a Demo[/bold]",
              border_style="bright_blue", padding=(1, 2))
    )


def _print_deps(deps: dict[str, bool]) -> None:
    table = Table(title="Library Status", show_lines=True)
    table.add_column("Package",       style="cyan")
    table.add_column("Status",        justify="center")
    table.add_column("Install hint",  style="dim")

    hints = {
        "sounddevice":       "pip install sounddevice",
        "numpy":             "pip install numpy",
        "matplotlib":        "pip install matplotlib",
        "rich":              "pip install rich",
        "speech_recognition":"pip install SpeechRecognition",
        "whisper":           "pip install openai-whisper",
        "vosk":              "pip install vosk  (+ download model)",
        "pyttsx3":           "pip install pyttsx3",
        "scipy":             "pip install scipy",
    }

    for name, ok in deps.items():
        status = "[bold green]✓  installed[/bold green]" if ok \
                 else "[bold red]✗  missing[/bold red]"
        table.add_row(name, status, hints.get(name, ""))

    console.print(table)
    console.print("\n  [dim]Run [bold]bash setup.sh[/bold] to install everything automatically.[/dim]\n")


# ── Runner ─────────────────────────────────────────────────────────────────

def _run_demo(module_path: str) -> None:
    console.print()
    try:
        mod = importlib.import_module(module_path)
        if hasattr(mod, "main"):
            mod.main()
        else:
            console.print(f"[red]Module {module_path} has no main() function.[/red]")
    except ImportError as exc:
        console.print(f"\n[bold red]Import error:[/bold red] {exc}")
        console.print("[dim]Run [bold]bash setup.sh[/bold] to install missing packages.[/dim]\n")
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        import traceback
        console.print_exception(show_locals=False)


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure demo/ and utils/ are importable from any working directory
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    deps = _check_imports()
    _print_header()

    key_map = {d["key"]: d["module"] for d in DEMOS}

    while True:
        _print_menu(deps)
        try:
            choice = console.input(
                "\n  [bold bright_blue]>[/bold bright_blue] Enter demo number (or d / q): "
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ("q", "quit", "exit"):
            break
        elif choice in ("d", "deps", "dependencies"):
            _print_deps(deps)
        elif choice in key_map:
            module = key_map[choice]
            _run_demo(module)
            console.print()
            console.rule(style="bright_blue dim")
            input("\n  Press Enter to return to the menu …\n")
            console.clear()
            _print_header()
        else:
            console.print(f"  [red]Unknown option:[/red] [bold]{choice}[/bold]\n")

    console.print("\n  [bold bright_cyan]Thanks for watching — Python Can Hear![/bold bright_cyan]\n")


if __name__ == "__main__":
    main()
