# chokita

Asistente de IA de escritorio para terminal, diseñado para correr en consola sin entorno gráfico.

## Arquitectura

- **STT**: `vosk` + `pyaudio` en hilo dedicado (opcional, si hay micrófono).
- **LLM**: cliente HTTP a Ollama (`/api/chat`).
- **TTS**: `piper` CLI oficial, con fallback a stdout si el binario no está.
- **TUI**: `textual` con cara de gato animada (parpadeo + boca al hablar), input de texto integrado.
- **Concurrencia**: comunicación entre hilos solo por `queue.Queue`.

## Estructura

```text
src/
  main.py
  ui.py        <- TUI con textual
  audio.py
  llm.py
  tts.py
  config.py
scripts/
  download_models.sh
  run.sh
requirements.txt
Dockerfile
docker-compose.yml
```

## Requisitos

- Python 3.11+
- Ollama ejecutándose (`ollama serve`)
- Micrófono (opcional, sin él funciona en modo texto)
- Piper (opcional, sin él imprime respuestas por stdout)

## Rápido

```bash
./run.sh
```

Descarga modelos, instala dependencias, descarga piper si no está, y arranca la TUI.

## Build y ejecución (Docker)

```bash
docker compose build
docker compose up
```

## Variables de entorno relevantes

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `ornith:9b`)
- `VOSK_MODEL_PATH`
- `PIPER_MODEL_PATH`
- `PIPER_CONFIG_PATH`
- `PLAYBACK_COMMAND` (`aplay` o `ffplay`)
- `TTS_FALLBACK_STDOUT` (`0` para desactivar fallback, default `1`)
- `USER_ID` / `GROUP_ID` para mapear el usuario del host en `docker-compose.yml`

## Desarrollo y pruebas

```bash
python -m pytest -q
```

Compilación sintáctica:

```bash
python -m compileall src
```
