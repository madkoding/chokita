from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "ornith:9b")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
    ollama_fallback_message: str = os.getenv(
        "OLLAMA_FALLBACK_MESSAGE",
        "No pude contactar al modelo local en este momento.",
    )
    ollama_keep_alive: int = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))

    asr_model: str = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B-hf")
    sample_rate_hz: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

    # ponytail: web_host default 127.0.0.1. WAN exposure = decisión del operador.
    web_host: str = os.getenv("WEB_HOST", "127.0.0.1")
    web_port: int = int(os.getenv("WEB_PORT", "8080"))
    # ponytail: token simple para WS; sin token = warning. JWT sería over-engineering.
    ws_token: str = os.getenv("CHOKITA_WS_TOKEN", "")

    piper_model_path: Path = Path(os.getenv("PIPER_MODEL_PATH", "models/es_ES-sharvard-medium.onnx"))
    _piper_speaker_val = os.getenv("PIPER_SPEAKER", "1")
    piper_speaker: int | None = int(_piper_speaker_val) if _piper_speaker_val else None
    tts_fallback_stdout: bool = os.getenv("TTS_FALLBACK_STDOUT", "1") in ("1", "true", "yes")

    shutdown_join_timeout_seconds: float = float(os.getenv("SHUTDOWN_JOIN_TIMEOUT_SECONDS", "2"))

    # --- Memoria + RAG ---
    db_path: Path = Path(os.getenv(
        "CHOKITA_DB_PATH",
        str(Path.home() / ".local/share/chokita/chokita.db"),
    ))
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    # Historial de mensajes por sesión (ventana deslizante)
    history_window: int = int(os.getenv("HISTORY_WINDOW", "20"))
    # Iteraciones máximas del loop de tools
    max_tool_iterations: int = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))

    # --- Workdir para tools ---
    workdir: Path = Path(os.getenv("CHOKITA_WORKDIR", ".")).resolve()

    # --- Reflexión del alma (idle) ---
    soul_idle_threshold_seconds: float = float(os.getenv("SOUL_IDLE_THRESHOLD_SECONDS", "30"))
    soul_reflect_min_seconds: float = float(os.getenv("SOUL_REFLECT_MIN_SECONDS", "300"))   # 5 min
    soul_reflect_max_seconds: float = float(os.getenv("SOUL_REFLECT_MAX_SECONDS", "900"))  # 15 min
    # Límite de tokens/longitud de una reflexión del alma
    soul_reflect_max_chars: int = int(os.getenv("SOUL_REFLECT_MAX_CHARS", "600"))

    # --- Fase REM (sueño) + RAPTOR ---
    # Chokita entra en REM si está idle este tiempo (default 10 min).
    rem_idle_threshold_seconds: float = float(os.getenv("REM_IDLE_THRESHOLD_SECONDS", "600"))
    # Cada cuánto reindexa el RAG y reconstruye RAPTOR (default 30 min).
    rem_raptor_interval_seconds: float = float(os.getenv("REM_RAPTOR_INTERVAL_SECONDS", "1800"))
    # Tamaño de cluster para k-means en RAPTOR (número de grupos por nivel).
    raptor_cluster_k: int = int(os.getenv("RAPTOR_CLUSTER_K", "8"))
    # Profundidad máxima del árbol RAPTOR.
    raptor_max_levels: int = int(os.getenv("RAPTOR_MAX_LEVELS", "4"))
    # Chars máximos por resumen de cluster.
    raptor_summary_max_chars: int = int(os.getenv("RAPTOR_SUMMARY_MAX_CHARS", "400"))
    # Semilla para k-means (RAPTOR determinista entre runs).
    raptor_seed: int = int(os.getenv("RAPTOR_SEED", "42"))


    # --- Memoria largo plazo ---
    # Extraer memorias episódicas cada N mensajes del usuario.
    memory_extract_interval: int = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "10"))
    # --- GC de RAG ---
    # Cap duro de chunks (reflection + memory, soul nunca se poda).
    rag_max_chunks: int = int(os.getenv("RAG_MAX_CHUNKS", "5000"))
    # Días de retención por fuente de chunk.
    rag_reflection_retention_days: int = int(os.getenv("RAG_REFLECTION_RETENTION_DAYS", "14"))
    rag_memory_retention_days: int = int(os.getenv("RAG_MEMORY_RETENTION_DAYS", "90"))
    # Paginas de WAL antes de autocheckpoint (default 1000 ≈ 4MB).
    wal_autocheckpoint_pages: int = int(os.getenv("WAL_AUTOCHECKPOINT_PAGES", "1000"))

    # --- Contexto / compactación ---
    # Tamaño del contexto del modelo en tokens (ornith:9b = 256K).
    context_window_tokens: int = int(os.getenv("CONTEXT_WINDOW_TOKENS", "262144"))
    # Umbral de uso para disparar compactación (0-1).
    compact_threshold: float = float(os.getenv("COMPACT_THRESHOLD", "0.80"))
    # Estimación: 1 token ≈ 4 chars. Sin tiktoken.




SETTINGS = Settings()
