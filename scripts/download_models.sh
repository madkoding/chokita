#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

# Qwen3-ASR se descarga automáticamente desde HuggingFace al primer uso.
# No necesita descarga manual.

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

echo "Verificando modelo de embeddings..."
if ! ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    echo "Descargando modelo de embeddings (nomic-embed-text)..."
    ollama pull nomic-embed-text
else
    echo "Modelo de embeddings ya existe, saltando."
fi
