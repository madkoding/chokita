#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

VOSK_URL="https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
VOSK_DIR="$MODELS_DIR/vosk-model-small-es-0.42"
if [ ! -d "$VOSK_DIR" ]; then
    echo "Descargando modelo Vosk..."
    curl -L "$VOSK_URL" -o /tmp/vosk.zip
    unzip -q /tmp/vosk.zip -d "$MODELS_DIR"
    rm /tmp/vosk.zip
else
    echo "Vosk ya existe, saltando."
fi

PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/sharvard/medium"
PIPER_ONNX="$MODELS_DIR/es_ES-sharvard-medium.onnx"
PIPER_JSON="$MODELS_DIR/es_ES-sharvard-medium.onnx.json"
if [ ! -f "$PIPER_ONNX" ]; then
    echo "Descargando voz Piper ES (sharvard-medium, femenina)..."
    curl -L "$PIPER_BASE/es_ES-sharvard-medium.onnx?download=true" -o "$PIPER_ONNX"
else
    echo "Voz Piper ES ya existe, saltando."
fi
if [ ! -f "$PIPER_JSON" ]; then
    curl -L "$PIPER_BASE/es_ES-sharvard-medium.onnx.json?download=true" -o "$PIPER_JSON"
else
    echo "Config Piper ES ya existe, saltando."
fi

echo "Modelos listos en $MODELS_DIR/"
