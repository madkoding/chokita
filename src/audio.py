"""Audio capture thread using PyAudio + Vosk STT."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any

import pyaudio
from vosk import KaldiRecognizer, Model

from src.config import SETTINGS

LOGGER = logging.getLogger(__name__)


class SpeechRecognizerThread(threading.Thread):
    """Continuously captures microphone audio and emits recognized text."""

    def __init__(
        self,
        text_queue: "queue.Queue[str]",
        ui_queue: "queue.Queue[dict[str, Any]]",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.ui_queue = ui_queue
        self.stop_event = stop_event

    def _notify(self, state: str, message: str) -> None:
        self.ui_queue.put({"type": "state", "state": state, "message": message})

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._run_once()
            except Exception:
                LOGGER.exception("STT thread failure; retrying")
                self._notify("ERROR", "Error de audio, reintentando...")
                time.sleep(1.5)

    def _run_once(self) -> None:
        if not SETTINGS.vosk_model_path.exists():
            raise FileNotFoundError(f"Vosk model not found: {SETTINGS.vosk_model_path}")

        model = Model(str(SETTINGS.vosk_model_path))
        recognizer = KaldiRecognizer(model, SETTINGS.sample_rate_hz)
        mic = pyaudio.PyAudio()
        stream = mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SETTINGS.sample_rate_hz,
            input=True,
            frames_per_buffer=SETTINGS.audio_chunk_size,
        )

        self._notify("LISTENING", "Escuchando...")

        try:
            while not self.stop_event.is_set():
                chunk = stream.read(SETTINGS.audio_chunk_size, exception_on_overflow=False)
                if recognizer.AcceptWaveform(chunk):
                    payload = json.loads(recognizer.Result())
                    text = payload.get("text", "").strip()
                    if text:
                        LOGGER.info("Recognized text: %s", text)
                        self.text_queue.put(text)
                        self._notify("RECOGNIZED", text)
                        self._notify("LISTENING", "Escuchando...")
        finally:
            stream.stop_stream()
            stream.close()
            mic.terminate()
