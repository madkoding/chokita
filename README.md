# chokita

Asistente de IA de escritorio para terminal, diseñado para correr en consola sin entorno gráfico.

## Arquitectura

- **STT**: `vosk` + `pyaudio` en hilo dedicado (opcional, si hay micrófono). Wake word "chokita" con aliases para Vosk small-es.
- **LLM**: cliente HTTP a Ollama (`/api/chat`) usando stdlib `urllib.request`. Sin deps externas para HTTP.
- **Memoria**: SQLite + WAL con embeddings (Ollama `/api/embeddings`). Sesiones, mensajes, chunks RAG, árbol RAPTOR.
- **RAG**: búsqueda coseno brute-force sobre embeddings almacenados. Recupera chunks relevantes al mensaje del usuario.
- **RAPTOR**: clustering jerárquico (k-means stdlib) + summarization. Se construye en REM sleep y se consulta en cada turno de chat (top-3 resúmenes en system prompt).
- **Alma**: hilo de reflexión idle. 3 voces freudianas (YO/SUPERYO/ELLO) que reflexionan sobre su personalidad y sintetizan un delta → RAG.
- **REM sleep**: hilo que reindexa el RAG construyendo el árbol RAPTOR cada 30 min de idle.
- **Tools**: 6 herramientas (read/list/glob/grep/write/bash) sandboxeadas a `CHOKITA_WORKDIR`.
- **Contexto**: compactación automática al 80% del context window. Extracción de memorias episódicas cada N mensajes.
- **TTS**: `piper` CLI oficial, con fallback a stdout si el binario no está.
- **TUI**: `textual` con cara de gato animada (parpadeo + boca al hablar), input de texto integrado.
- **Concurrencia**: comunicación entre hilos solo por `queue.Queue`.

## Estructura

```text
src/
  main.py       <- loop principal + threads (soul, REM, assistant, STT)
  ui.py         <- TUI con textual (FaceApp, KaomojiFace, StatusPill, TokenBar)
  audio.py      <- STT (vosk + pyaudio), wake word parser
  llm.py        <- cliente Ollama + tool loop + RAG + compactación
  memory.py     <- SQLite + WAL + embeddings + RAPTOR
  soul.py       <- reflexión idle (YO/SUPERYO/ELLO)
  sleep.py      <- REM sleep (RAPTOR reindex)
  tools.py      <- 6 tools sandboxeadas a CHOKITA_WORKDIR
  tts.py        <- Piper TTS + playback (aplay/paplay/ffplay/powershell)
  config.py     <- settings dataclass con defaults via env vars
scripts/
  download_models.sh
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

### Ollama
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `ornith:9b`)
- `OLLAMA_TIMEOUT_SECONDS` (default: `15`)
- `OLLAMA_KEEP_ALIVE` (default: `-1`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)

### Audio / STT
- `VOSK_MODEL_PATH`
- `AUDIO_SAMPLE_RATE` (default: `16000`)
- `AUDIO_CHUNK_SIZE` (default: `4000`)
- `STT_RETRY_DELAY_SECONDS` (default: `1.5`)
- `WAKE_COMMAND_TIMEOUT_SECONDS` (default: `12.0`)

### TTS
- `PIPER_BIN` (default: `piper`)
- `PIPER_MODEL_PATH`
- `PIPER_CONFIG_PATH`
- `PIPER_SPEAKER`
- `PLAYBACK_COMMAND` (`aplay`, `paplay`, `ffplay`, `powershell`, o `auto`)
- `TTS_FALLBACK_STDOUT` (`0` para desactivar fallback, default `1`)

### Memoria + RAG
- `CHOKITA_DB_PATH`
- `RAG_TOP_K` (default: `6`)
- `HISTORY_WINDOW` (default: `20`)
- `MAX_TOOL_ITERATIONS` (default: `5`)
- `CHOKITA_WORKDIR` (default: `.`)

### Alma (reflexión idle)
- `SOUL_IDLE_THRESHOLD_SECONDS` (default: `30`)
- `SOUL_REFLECT_MIN_SECONDS` (default: `300`, 5 min)
- `SOUL_REFLECT_MAX_SECONDS` (default: `900`, 15 min)
- `SOUL_REFLECT_MAX_CHARS` (default: `600`)

### REM sleep + RAPTOR
- `REM_IDLE_THRESHOLD_SECONDS` (default: `600`, 10 min)
- `REM_RAPTOR_INTERVAL_SECONDS` (default: `1800`, 30 min)
- `RAPTOR_CLUSTER_K` (default: `8`)
- `RAPTOR_MAX_LEVELS` (default: `4`)
- `RAPTOR_SUMMARY_MAX_CHARS` (default: `400`)

### Contexto / compactación
- `CONTEXT_WINDOW_TOKENS` (default: `262144`)
- `COMPACT_THRESHOLD` (default: `0.80`)
- `CHARS_PER_TOKEN` (default: `4`)
- `MEMORY_EXTRACT_INTERVAL` (default: `10`)

### Docker
- `USER_ID` / `GROUP_ID` para mapear el usuario del host en `docker-compose.yml`

## Desarrollo y pruebas

```bash
python -m pytest -q
```

Compilación sintáctica:

```bash
python -m compileall src
```
