#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

OS="$(uname -s)"
case "$OS" in
    Darwin)
        command -v brew >/dev/null 2>/dev/null || { echo "Instalá Homebrew: https://brew.sh"; exit 1; }
        brew install portaudio 2>/dev/null || echo "No se pudo instalar portaudio. Instalá manualmente: brew install portaudio"
        ;;
    Linux)
        sudo apt-get install -y -qq portaudio19-dev libasound2-dev build-essential unzip alsa-utils 2>/dev/null || echo "No se pudieron instalar dependencias del sistema. Instalá manualmente: sudo apt install portaudio19-dev libasound2-dev build-essential unzip alsa-utils"
        ;;
esac

PYTHON="python3"
command -v python3.12 >/dev/null && PYTHON="python3.12"
[ -d .venv ] || $PYTHON -m venv .venv
.venv/bin/pip install -q -r requirements.txt

if ! [ -f models/es_ES-sharvard-medium.onnx ] || ! [ -d models/vosk-model-es-0.42 ]; then
    bash scripts/download_models.sh
fi

.venv/bin/python -m src.main
