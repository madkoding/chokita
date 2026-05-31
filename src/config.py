"""Runtime configuration for Chokita assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_haarcascade_path() -> str:
    try:
        import cv2  # pylint: disable=import-error

        return str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    except Exception:
        return "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"


KAOMOJIS = {
    "IDLE": "(=^･ω･^=)",
    "LISTENING": "(=^･ｪ･^=))ﾉ彡☆",
    "RECOGNIZED": "(๑˃ᴗ˂)ﻭ",
    "THINKING": "(・・ ) ?",
    "SPEAKING": "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧",
    "ERROR": "(╥﹏╥)",
}


@dataclass(frozen=True)
class Settings:
    """Centralized app settings resolved from environment variables."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_chat_path: str = os.getenv("OLLAMA_CHAT_PATH", "/api/chat")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "15"))
    ollama_fallback_message: str = os.getenv(
        "OLLAMA_FALLBACK_MESSAGE",
        "No pude contactar al modelo local en este momento.",
    )

    vosk_model_path: Path = Path(os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-es-0.42"))
    sample_rate_hz: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    audio_chunk_size: int = int(os.getenv("AUDIO_CHUNK_SIZE", "4000"))
    stt_retry_delay_seconds: float = float(os.getenv("STT_RETRY_DELAY_SECONDS", "1.5"))

    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    haarcascade_path: Path = Path(
        os.getenv(
            "HAAR_CASCADE_PATH",
            _default_haarcascade_path(),
        )
    )
    face_model_path: Path = Path(os.getenv("FACE_MODEL_PATH", "models/lbph_model.yml"))
    face_labels_path: Path = Path(os.getenv("FACE_LABELS_PATH", "models/face_labels.json"))
    face_confidence_threshold: float = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "65.0"))
    face_unknown_confidence: float = float(os.getenv("FACE_UNKNOWN_CONFIDENCE", "999.0"))
    face_detection_max_frames: int = int(os.getenv("FACE_DETECTION_MAX_FRAMES", "20"))
    face_detect_scale_factor: float = float(os.getenv("FACE_DETECT_SCALE_FACTOR", "1.2"))
    face_detect_min_neighbors: int = int(os.getenv("FACE_DETECT_MIN_NEIGHBORS", "5"))

    piper_bin: str = os.getenv("PIPER_BIN", "piper")
    piper_model_path: Path = Path(os.getenv("PIPER_MODEL_PATH", "models/es_ES-mls_10246-medium.onnx"))
    piper_config_path: Path = Path(os.getenv("PIPER_CONFIG_PATH", "models/es_ES-mls_10246-medium.onnx.json"))
    playback_command: str = os.getenv("PLAYBACK_COMMAND", "auto")

    ui_poll_interval_ms: int = int(os.getenv("UI_POLL_INTERVAL_MS", "40"))
    shutdown_join_timeout_seconds: float = float(os.getenv("SHUTDOWN_JOIN_TIMEOUT_SECONDS", "2"))


SETTINGS = Settings()
