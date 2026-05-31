# chokita

Asistente de IA de escritorio para hardware limitado (netbooks antiguas), diseñado para Ubuntu Server/Ubuntu WSL con Docker.

## Arquitectura

- **STT**: `vosk` + `pyaudio` en hilo dedicado.
- **Reconocimiento facial**: `opencv-contrib-python` con `cv2.face.LBPHFaceRecognizer_create()` + Haar cascade.
- **LLM**: cliente HTTP a Ollama (`/api/chat`).
- **TTS**: `piper` CLI oficial, WAV temporal + reproducción (`aplay`/`ffplay`) con limpieza automática.
- **UI**: `tkinter` con kaomojis por estado.
- **Concurrencia**: comunicación entre hilos solo por `queue.Queue`.

## Estructura

```text
src/
  main.py
  ui.py
  audio.py
  vision.py
  llm.py
  tts.py
  config.py
train_face.py
requirements.txt
Dockerfile
docker-compose.yml
.dockerignore
```

## Requisitos de host

- Docker Engine + Docker Compose v2
- Cámara accesible en `/dev/video0`
- Audio accesible en `/dev/snd`
- Servidor X11 activo (para `tkinter`)
- Ollama ejecutándose en host (`http://host.docker.internal:11434`)

## Preparación de modelos

Crear carpeta `models/` en la raíz del proyecto y añadir:

- Vosk ES pequeño: `models/vosk-model-small-es-0.42/`
- Piper ES ONNX + JSON:
  - `models/es_ES-mls_10246-medium.onnx`
  - `models/es_ES-mls_10246-medium.onnx.json`
- Modelo facial LBPH:
  - `models/lbph_model.yml`
  - `models/face_labels.json`

### Entrenar modelo facial

Estructura del dataset:

```text
dataset/
  persona_1/
    foto1.jpg
    foto2.jpg
  persona_2/
    foto1.jpg
```

Entrenamiento:

```bash
python train_face.py --dataset ./dataset --output models/lbph_model.yml --labels models/face_labels.json
```

## Build y ejecución

```bash
docker compose build
docker compose up
```

Si necesitas invocación explícita de dispositivos:

```bash
docker compose up --build --force-recreate
```

## Variables de entorno relevantes

- `OLLAMA_BASE_URL` (default: `http://host.docker.internal:11434`)
- `OLLAMA_MODEL` (default: `llama3.2:3b`)
- `VOSK_MODEL_PATH`
- `PIPER_MODEL_PATH`
- `PIPER_CONFIG_PATH`
- `FACE_MODEL_PATH`
- `FACE_LABELS_PATH`
- `PLAYBACK_COMMAND` (`aplay` o `ffplay`)
- `USER_ID` / `GROUP_ID` para mapear el usuario del host en `docker-compose.yml`

## Desarrollo y pruebas

Pruebas rápidas:

```bash
python -m pytest -q
```

Compilación sintáctica:

```bash
python -m compileall src train_face.py
```

## Notas Ubuntu Server / WSL

- Para WSLg (Windows 11), `DISPLAY` suele configurarse automáticamente.
- En Ubuntu Server sin entorno gráfico local, usa forwarding X11 desde cliente remoto antes de levantar el contenedor.
- Asegura permisos para `/dev/video0` y `/dev/snd` en el usuario host.
