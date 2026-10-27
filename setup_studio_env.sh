#!/usr/bin/env bash
# Automated Zero-Touch Environment Bootstrapper for Creative Editors
# Antigravity runs this script automatically so the editor never has to touch the terminal.

set -euo pipefail

echo "=== [1/4] Checking Homebrew & FFmpeg ==="
export PATH="/opt/homebrew/bin:/usr/local/bin:/opt/homebrew/share/google-cloud-sdk/bin:/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin:$PATH"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing ffmpeg via Homebrew..."
    brew install ffmpeg
  else
    echo "WARNING: ffmpeg not found and Homebrew is not installed."
  fi
else
  echo "FFmpeg ready: $(command -v ffmpeg)"
fi

echo "=== [2/4] Checking Google Cloud SDK (gcloud) ==="
if ! command -v gcloud >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing Google Cloud SDK via Homebrew..."
    brew install --cask google-cloud-sdk || true
  fi
else
  echo "gcloud ready: $(command -v gcloud)"
fi

echo "=== [3/4] Setting Up Python Virtual Environment (.venv) ==="
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
echo "Python environment ready: $(.venv/bin/python --version)"

echo "=== [4/4] Checking Google Cloud Authentication ==="
if command -v gcloud >/dev/null 2>&1; then
  if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "Opening browser for Google Cloud Sign-In (Editor: just click your Google account in the browser window)..."
    gcloud auth login
  fi
  if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    echo "Opening browser for Application Default Credentials..."
    gcloud auth application-default login
  fi
  echo "Active GCP Account: $(gcloud config get-value account 2>/dev/null || echo 'not set')"
  echo "Active GCP Project: $(gcloud config get-value project 2>/dev/null || echo 'not set')"
fi

echo "=== STUDIO ENVIRONMENT READY ==="
