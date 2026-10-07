"""
utils/audio_utils.py
Shared audio helpers used across all demos.
"""

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.table import Table

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000   # Hz  — good for speech (Whisper/Vosk native rate)
CHANNELS    = 1        # mono
DTYPE       = "float32"


# ── Device helpers ─────────────────────────────────────────────────────────

def list_audio_devices() -> None:
    """Print a rich table of available audio devices."""
    devices = sd.query_devices()
    default_in  = sd.default.device[0]
    default_out = sd.default.device[1]

    table = Table(title="Audio Devices", show_lines=True)
    table.add_column("ID",   style="cyan",  justify="right", no_wrap=True)
    table.add_column("Name", style="white", min_width=30)
    table.add_column("In",   style="green", justify="center")
    table.add_column("Out",  style="yellow", justify="center")
    table.add_column("Rate", style="magenta", justify="right")

    for idx, dev in enumerate(devices):
        marker_in  = "● default" if idx == default_in  else ("✓" if dev["max_input_channels"]  > 0 else "")
        marker_out = "● default" if idx == default_out else ("✓" if dev["max_output_channels"] > 0 else "")
        table.add_row(
            str(idx),
            dev["name"],
            marker_in,
            marker_out,
            str(int(dev["default_samplerate"])),
        )

    console.print(table)


def get_default_input_device() -> dict:
    """Return info dict for the default input device."""
    idx = sd.default.device[0]
    return sd.query_devices(idx)


# ── Level helpers ──────────────────────────────────────────────────────────

def rms(data: np.ndarray) -> float:
    """Root-mean-square amplitude (0–1 float32 range)."""
    return float(np.sqrt(np.mean(data ** 2)))


def db(data: np.ndarray, ref: float = 1.0) -> float:
    """Convert audio chunk to dBFS."""
    r = rms(data)
    if r == 0:
        return -96.0
    return 20.0 * np.log10(r / ref)


def level_bar(value_db: float, width: int = 40, low: float = -60.0,
              high: float = 0.0) -> str:
    """Return a coloured ASCII bar representing a dBFS level."""
    span    = high - low
    clamped = max(low, min(high, value_db))
    filled  = int((clamped - low) / span * width)

    if value_db > -6:
        colour = "red"
    elif value_db > -18:
        colour = "yellow"
    else:
        colour = "green"

    bar = "█" * filled + "░" * (width - filled)
    return f"[{colour}]{bar}[/{colour}] {value_db:+.1f} dBFS"


# ── Recording helpers ──────────────────────────────────────────────────────

def record_audio(duration: float, sr: int = SAMPLE_RATE,
                 channels: int = CHANNELS) -> np.ndarray:
    """Block and record `duration` seconds of audio. Returns float32 array."""
    frames = int(duration * sr)
    recording = sd.rec(frames, samplerate=sr, channels=channels,
                       dtype=DTYPE, blocking=True)
    return recording.flatten()


def save_wav(path: str, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save a float32 numpy array as a 16-bit WAV file."""
    import wave, struct
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def play_audio(data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Play back a float32 audio array (blocks until done)."""
    sd.play(data, samplerate=sr)
    sd.wait()


# ── FFT helper ─────────────────────────────────────────────────────────────

def compute_fft(data: np.ndarray, sr: int = SAMPLE_RATE):
    """Return (frequencies, magnitudes) for a real FFT of `data`."""
    n    = len(data)
    mag  = np.abs(np.fft.rfft(data * np.hanning(n))) / n
    freq = np.fft.rfftfreq(n, d=1.0 / sr)
    return freq, mag
