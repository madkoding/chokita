"""Audio capture thread using PyAudio + Vosk STT."""

from __future__ import annotations

import json
import logging
import queue
import re
import struct
import threading
import time
import unicodedata
from collections.abc import Callable
from typing import Any

from src.config import SETTINGS

LOGGER = logging.getLogger(__name__)

# Wake word: "chokita" anywhere in the utterance.
# ponytail: Vosk es-0.42 (1.4GB) is accurate enough that the wake word
# should be recognized cleanly. Aliases kept as fallback for edge cases.
_WAKE_RE = re.compile(
    r"\b(chokita|choquita|chiquita|chiquitita|chaquita|jaquita|chocita|choki|chiqui)\b\s*(.*)",
    re.DOTALL,
)
# Stop-voice commands (after wake word): "para", "detente", "callate", ...
_STOP_WORDS = re.compile(r"^(para|par\u00e1|detente|callate|c\u00e1llate|silencio|alto|stop)\b")


def _normalize(text: str) -> str:
    """Lowercase + strip accents for robust matching."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def parse_wake(text: str) -> tuple[bool, str]:
    """Returns (matched, remainder). remainder is the command after the wake word.

    Wake word may appear anywhere in the utterance (Vosk small-es often inserts
    filler/garbage before a misheard "chokita"). Anything said *before* the wake
    word is dropped — only the text after the wake word is the command.
    """
    n = _normalize(text)
    m = _WAKE_RE.search(n)
    if not m:
        return False, ""
    return True, m.group(2).strip()


def is_stop_command(remainder: str) -> bool:
    """Check if the post-wake-word text is a stop command."""
    return bool(_STOP_WORDS.match(_normalize(remainder)))


class SpeechRecognizerThread(threading.Thread):
    """Continuously captures microphone audio and emits recognized text."""

    def __init__(
        self,
        text_queue: queue.Queue[str],
        ui_queue: queue.Queue[dict[str, Any]],
        stop_event: threading.Event,
        on_stop_command: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.on_stop_command = on_stop_command
        self._model: object | None = None

    def _notify(self, state: str, message: str) -> None:
        self.ui_queue.put({"type": "state", "state": state, "message": message})

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._run_once()
            except FileNotFoundError:
                LOGGER.exception("STT model not found")
                self.ui_queue.put({"type": "log", "message": f"❌ Modelo Vosk no encontrado en {SETTINGS.vosk_model_path}. Ejecutá: bash scripts/download_models.sh"})
                return
            except Exception:
                LOGGER.exception("STT thread failure; retrying")
                self._notify("ERROR", "Error de audio, reintentando...")
                self.ui_queue.put({"type": "log", "message": "⚠ Error de STT, reintentando..."})
                time.sleep(SETTINGS.stt_retry_delay_seconds)

    def _run_once(self) -> None:
        if not SETTINGS.vosk_model_path.exists():
            raise FileNotFoundError(f"Vosk model not found: {SETTINGS.vosk_model_path}")

        if self._model is None:
            import os as _os

            from vosk import KaldiRecognizer, Model
            _sv = _os.dup(2)
            _dn = _os.open(_os.devnull, _os.O_WRONLY)
            _os.dup2(_dn, 2)
            _os.close(_dn)
            try:
                self._model = Model(str(SETTINGS.vosk_model_path))
            finally:
                _os.dup2(_sv, 2)
                _os.close(_sv)
        recognizer = KaldiRecognizer(self._model, SETTINGS.sample_rate_hz)

        # suppress ALSA/JACK noise during pyaudio init (background thread, fd 2)
        import os as _os
        _saved_stderr = _os.dup(2)
        _dn = _os.open(_os.devnull, _os.O_WRONLY)
        _os.dup2(_dn, 2)
        _os.close(_dn)
        try:
            import pyaudio
            mic = pyaudio.PyAudio()
            stream: pyaudio.Stream | None = None
            try:
                stream = mic.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=SETTINGS.sample_rate_hz,
                    input=True,
                    frames_per_buffer=SETTINGS.audio_chunk_size,
                )
            except Exception:
                mic.terminate()
                raise
        finally:
            _os.dup2(_saved_stderr, 2)
            _os.close(_saved_stderr)

        self._notify("LISTENING", "Escuchando...")

        wake_at: float = 0.0  # timestamp of last bare "chokita" wake (0 = none pending)
        try:
            while not self.stop_event.is_set():
                chunk = stream.read(SETTINGS.audio_chunk_size, exception_on_overflow=False)
                count = len(chunk) // 2
                if count:
                    shorts = struct.unpack(f"<{count}h", chunk)
                    rms = (sum(s * s for s in shorts) / count) ** 0.5
                    level = min(int(rms / 32768 * 20), 20)
                    self.ui_queue.put({"type": "audio_level", "level": level})
                if recognizer.AcceptWaveform(chunk):
                    payload = json.loads(recognizer.Result())
                    text = payload.get("text", "").strip()
                    if text:
                        LOGGER.info("Recognized text: %s", text)
                        now = time.time()
                        # If a bare wake is pending and this utterance arrives
                        # within the timeout, treat the whole utterance as the
                        # command continuation (merge). Otherwise re-evaluate wake.
                        pending = wake_at and (now - wake_at) < SETTINGS.wake_command_timeout_seconds
                        wake_at = 0.0
                        if pending:
                            # Continuation after bare "chokita": use raw text as command.
                            if is_stop_command(text):
                                LOGGER.info("Stop command detected (continuation): %s", text)
                                self._notify("RECOGNIZED", f"[STOP] {text}")
                                if self.on_stop_command:
                                    self.on_stop_command()
                                self._notify("LISTENING", "Escuchando...")
                                continue
                            self.text_queue.put(text)
                            self._notify("RECOGNIZED", text)
                            self._notify("LISTENING", "Escuchando...")
                            continue
                        matched, remainder = parse_wake(text)
                        if not matched:
                            # Not addressed to chokita — ignore.
                            continue
                        if remainder and is_stop_command(remainder):
                            LOGGER.info("Stop command detected: %s", text)
                            self._notify("RECOGNIZED", f"[STOP] {text}")
                            if self.on_stop_command:
                                self.on_stop_command()
                            self._notify("LISTENING", "Escuchando...")
                            continue
                        if not remainder:
                            # Just "chokita" with nothing after — start the timeout window.
                            wake_at = now
                            self._notify("RECOGNIZED", f"[WAKE] {text}")
                            self._notify("LISTENING", "Escuchando...")
                            continue
                        self.text_queue.put(remainder)
                        self._notify("RECOGNIZED", text)
                        self._notify("LISTENING", "Escuchando...")
        finally:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            mic.terminate()
