# AGENTS.md — chokita

## Verificación
- `python -m compileall src && python -m pytest -q` — check obligatorio antes de commit.
- 31 tests (audio, llm, memory, raptor, tools, tts).

## Decisiones de arquitectura
- HTTP a Ollama: stdlib `urllib.request` (no `requests`). Sin deps externas para HTTP.
- RAPTOR se construye en REM sleep y se consulta en cada turno de chat (top-3 resúmenes en system prompt).
- Alma: 3 voces (YO/SUPERYO/ELLO) en hilo idle, sintetiza delta → RAG.
- Tools: 6 herramientas sandboxeadas a `CHOKITA_WORKDIR`, registry explícito `_TOOLS` dict.
- Concurrencia: comunicación entre hilos solo por `queue.Queue`.
- `from __future__ import annotations` en todos los módulos (compat 3.9+ runtime).

## Estilo
- Ponytail: mínimo código que funciona, stdlib primero, sin abstracciones con 1 implementación.
- Marcar deliberadas simplificaciones con `# ponytail: ...`.
- Sin comentarios explicativos salvo que se pidan.

## No tocar sin preguntar
- `SOUL.md` (personalidad de Chokita — es contenido, no código).
- `docker-compose.yml` paths (deben coincidir con `config.py` + `download_models.sh`).
