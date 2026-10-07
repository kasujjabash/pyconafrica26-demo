#!/usr/bin/env bash
# setup.sh — One-shot environment setup for "Python Can Hear" demo
set -e

echo "=== Python Can Hear — Setup ==="

# ── Homebrew dependencies (macOS) ──────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &>/dev/null; then
        echo "[!] Homebrew not found. Install it from https://brew.sh first."
        exit 1
    fi
    echo "[+] Installing portaudio and ffmpeg via Homebrew..."
    brew install portaudio ffmpeg 2>/dev/null || true
fi

# ── Python packages ────────────────────────────────────────────────────────
echo "[+] Installing Python packages..."
pip install -r requirements.txt

# ── Vosk model download ────────────────────────────────────────────────────
MODEL_DIR="models/vosk-model-small-en-us-0.15"
if [ ! -d "$MODEL_DIR" ]; then
    echo "[+] Downloading Vosk small English model (~50 MB)..."
    mkdir -p models
    curl -L -o models/vosk-model.zip \
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    cd models && unzip -q vosk-model.zip && rm vosk-model.zip && cd ..
    echo "[+] Vosk model ready at $MODEL_DIR"
else
    echo "[✓] Vosk model already present."
fi

# ── Whisper — tiny model pre-download ─────────────────────────────────────
echo "[+] Pre-downloading Whisper 'tiny' model (~39 MB)..."
python -c "import whisper; whisper.load_model('tiny')" 2>/dev/null && \
    echo "[✓] Whisper tiny model ready." || \
    echo "[!] Whisper model download skipped (will download on first use)."

echo ""
echo "=== Setup complete! Run: python main.py ==="
