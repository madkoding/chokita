#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

sudo apt-get install -y -qq portaudio19-dev libasound2-dev build-essential unzip alsa-utils 2>/dev/null || echo "No se pudieron instalar dependencias del sistema. Instalá manualmente: sudo apt install portaudio19-dev libasound2-dev build-essential unzip alsa-utils"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

if ! [ -f models/es_ES-davefx-medium.onnx ] || ! [ -d models/vosk-model-small-es-0.42 ]; then
    bash scripts/download_models.sh
fi

if ! command -v piper &>/dev/null && [ ! -f bin/piper ]; then
    mkdir -p bin
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64) a=amd64 ;;
        aarch64|arm64) a=arm64 ;;
        *) echo "Arquitectura no soportada: $ARCH"; exit 1 ;;
    esac
    echo "Descargando piper para $ARCH..."
    curl -L "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_${a}.tar.gz" -o /tmp/piper.tar.gz
    tar -xzf /tmp/piper.tar.gz -C bin --strip-components=1
    rm /tmp/piper.tar.gz
fi

export PATH="$PWD/bin:$PATH"
.venv/bin/python -m src.main
