import os
import sys
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("MacOSX")          # native macOS backend — change to TkAgg if needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from rich.console import Console
from rich.live import Live

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.audio_utils import SAMPLE_RATE, CHANNELS, DTYPE, db, level_bar

console = Console()

# ── Parameters ─────────────────────────────────────────────────────────────
BLOCK_SIZE      = 1024            # samples per callback
HISTORY_SECS    = 2.0             # waveform history length
SPEC_ROWS       = 80              # rows in spectrogram waterfall
FFT_SIZE        = 2048
FREQ_MAX        = 8000            # Hz — upper limit for display

HISTORY_LEN     = int(SAMPLE_RATE * HISTORY_SECS)
FREQ_BINS       = FFT_SIZE // 2 + 1

# ── Shared buffers (lock-free via deque + numpy) ───────────────────────────
_wave_buf  = np.zeros(HISTORY_LEN, dtype=np.float32)
_fft_mag   = np.zeros(FREQ_BINS,   dtype=np.float32)
_fft_peak  = np.zeros(FREQ_BINS,   dtype=np.float32)
_spec_buf  = np.zeros((SPEC_ROWS, FREQ_BINS), dtype=np.float32)
_current_db: float = -96.0

_audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=20)


# ── Audio callback ─────────────────────────────────────────────────────────

def _audio_callback(indata: np.ndarray, frames: int, t, status) -> None:
    global _current_db
    chunk = indata[:, 0].copy()
    _current_db = db(chunk)
    try:
        _audio_q.put_nowait(chunk)
    except queue.Full:
        pass   # drop frame rather than block


# ── Buffer-update thread ───────────────────────────────────────────────────

_peak_decay = 0.95   # per-frame peak hold decay

def _update_buffers() -> None:
    global _wave_buf, _fft_mag, _fft_peak, _spec_buf
    spec_row_idx = 0

    while True:
        try:
            chunk = _audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        n = len(chunk)

        # Roll waveform buffer
        _wave_buf = np.roll(_wave_buf, -n)
        _wave_buf[-n:] = chunk

        # Compute FFT
        window      = np.hanning(FFT_SIZE)
        seg         = _wave_buf[-FFT_SIZE:]
        mag         = np.abs(np.fft.rfft(seg * window)) / FFT_SIZE
        _fft_mag[:] = mag

        # Peak hold with decay
        _fft_peak = np.where(mag > _fft_peak, mag, _fft_peak * _peak_decay)

        # Roll spectrogram
        _spec_buf = np.roll(_spec_buf, -1, axis=0)
        _spec_buf[-1, :] = 20 * np.log10(np.maximum(mag, 1e-10))

        spec_row_idx = (spec_row_idx + 1) % SPEC_ROWS


# ── Matplotlib setup ───────────────────────────────────────────────────────

def _build_figure():
    plt.rcParams.update({
        "figure.facecolor": "#0d0d0d",
        "axes.facecolor":   "#111111",
        "axes.edgecolor":   "#333333",
        "axes.labelcolor":  "#aaaaaa",
        "xtick.color":      "#666666",
        "ytick.color":      "#666666",
        "grid.color":       "#222222",
        "text.color":       "#cccccc",
    })

    fig = plt.figure(figsize=(13, 8), facecolor="#0d0d0d")
    fig.canvas.manager.set_window_title("Python Can Hear — Sound Visualization")
    gs  = GridSpec(3, 1, figure=fig, hspace=0.4)

    freq_axis = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    freq_mask = freq_axis <= FREQ_MAX
    freq_display = freq_axis[freq_mask]

    t_axis = np.linspace(-HISTORY_SECS, 0, HISTORY_LEN)

    # ── Panel 1: Waveform ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Waveform", color="#00e5ff", fontsize=11, pad=6)
    ax1.set_xlim(-HISTORY_SECS, 0)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("Time (s)", fontsize=8)
    ax1.set_ylabel("Amplitude", fontsize=8)
    ax1.grid(True, alpha=0.3)
    [line_wave] = ax1.plot(t_axis, _wave_buf, color="#00e5ff", lw=0.7)

    # ── Panel 2: Spectrum ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Frequency Spectrum", color="#ff6f00", fontsize=11, pad=6)
    ax2.set_xlim(0, FREQ_MAX)
    ax2.set_ylim(0, 0.05)
    ax2.set_xlabel("Frequency (Hz)", fontsize=8)
    ax2.set_ylabel("Magnitude", fontsize=8)
    ax2.grid(True, alpha=0.3)
    [line_fft]  = ax2.plot(freq_display, _fft_mag[freq_mask],
                           color="#ff6f00", lw=0.9)
    [line_peak] = ax2.plot(freq_display, _fft_peak[freq_mask],
                           color="#ff0000", lw=0.6, linestyle="--", alpha=0.6,
                           label="peak hold")
    ax2.legend(loc="upper right", fontsize=7, facecolor="#1a1a1a",
               edgecolor="#333333", labelcolor="#cccccc")

    # ── Panel 3: Spectrogram ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("Spectrogram (Waterfall)", color="#76ff03", fontsize=11, pad=6)
    ax3.set_xlabel("Frequency (Hz)", fontsize=8)
    ax3.set_ylabel("Time →", fontsize=8)
    ax3.set_yticks([])
    im = ax3.imshow(
        _spec_buf[:, freq_mask],
        aspect="auto", origin="lower",
        extent=[0, FREQ_MAX, 0, SPEC_ROWS],
        cmap="inferno", vmin=-80, vmax=0,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax3, label="dBFS", shrink=0.8)

    db_text = ax1.text(0.01, 0.92, "", transform=ax1.transAxes,
                       color="#ffffff", fontsize=9,
                       bbox=dict(facecolor="#1a1a1a", edgecolor="#333333",
                                 alpha=0.7, pad=3))

    return fig, line_wave, line_fft, line_peak, im, db_text, t_axis, freq_mask


# ── Animation update ───────────────────────────────────────────────────────

def _make_update(line_wave, line_fft, line_peak, im, db_text,
                 t_axis, freq_mask):
    def update(frame):
        line_wave.set_ydata(_wave_buf)
        line_fft.set_ydata(_fft_mag[freq_mask])
        line_peak.set_ydata(_fft_peak[freq_mask])
        im.set_data(_spec_buf[:, freq_mask])
        db_text.set_text(f"{_current_db:+.1f} dBFS")
        return line_wave, line_fft, line_peak, im, db_text

    return update


# ── Terminal VU thread ─────────────────────────────────────────────────────

_stop_vu = threading.Event()

def _vu_thread() -> None:
    while not _stop_vu.is_set():
        bar = level_bar(_current_db, width=32)
        console.print(f"\r  {bar}  ", end="")
        time.sleep(0.1)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global _wave_buf, _fft_mag, _fft_peak, _spec_buf, _current_db
    _wave_buf[:] = 0; _fft_mag[:] = 0; _fft_peak[:] = 0
    _spec_buf[:] = 0; _current_db = -96.0
    _stop_vu.clear()

    console.rule("[bold bright_cyan]Demo 4 — Sound Visualization[/bold bright_cyan]")
    console.print("\n  [dim]A matplotlib window will open. Close it (or Ctrl+C) to return.[/dim]\n")

    # Start buffer-update thread
    buf_thread = threading.Thread(target=_update_buffers, daemon=True)
    buf_thread.start()

    # Start VU meter in terminal
    vu = threading.Thread(target=_vu_thread, daemon=True)
    vu.start()

    # Open audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS,
        dtype=DTYPE, blocksize=BLOCK_SIZE,
        callback=_audio_callback,
    )

    fig, line_wave, line_fft, line_peak, im, db_text, t_axis, freq_mask = \
        _build_figure()

    update_fn = _make_update(line_wave, line_fft, line_peak, im,
                             db_text, t_axis, freq_mask)

    ani = animation.FuncAnimation(fig, update_fn, interval=50,
                                  blit=True, cache_frame_data=False)

    try:
        with stream:
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        _stop_vu.set()
        plt.close("all")

    console.print("\n\n[bold green]Visualization closed.[/bold green]\n")


if __name__ == "__main__":
    main()
