"""Main entry point orchestrating UI, STT, face auth, LLM and TTS."""

from __future__ import annotations

import logging
import queue
import signal
import threading
from typing import Any

from src.audio import SpeechRecognizerThread
from src.llm import OllamaClient
from src.tts import PiperTTS
from src.ui import AssistantUI
from src.vision import FaceAuthenticator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def assistant_loop(
    text_queue: "queue.Queue[str]",
    ui_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
) -> None:
    llm = OllamaClient()
    tts = PiperTTS()

    while not stop_event.is_set():
        try:
            user_text = text_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            ui_queue.put({"type": "state", "state": "THINKING", "message": "Validando rostro..."})
            face = FaceAuthenticator().authenticate()
            if not face.authorized:
                msg = "Rostro no autorizado. No responderé a esta solicitud."
                ui_queue.put({"type": "state", "state": "ERROR", "message": msg})
                continue

            prompt = f"Usuario autenticado ({face.label}). Solicitud: {user_text}"
            ui_queue.put({"type": "state", "state": "THINKING", "message": "Consultando LLM..."})
            answer = llm.chat(prompt)

            ui_queue.put({"type": "state", "state": "SPEAKING", "message": answer})
            tts.speak(answer)
            ui_queue.put({"type": "state", "state": "IDLE", "message": "Esperando comando..."})
        except Exception:
            LOGGER.exception("Assistant loop iteration failed")
            ui_queue.put({"type": "state", "state": "ERROR", "message": "Fallo interno del asistente"})


def main() -> None:
    text_queue: "queue.Queue[str]" = queue.Queue()
    ui_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    stop_event = threading.Event()

    def _handle_signal(_sig: int, _frame: Any) -> None:
        stop_event.set()
        ui_queue.put({"type": "shutdown"})

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    stt_thread = SpeechRecognizerThread(text_queue=text_queue, ui_queue=ui_queue, stop_event=stop_event)
    worker_thread = threading.Thread(
        target=assistant_loop,
        args=(text_queue, ui_queue, stop_event),
        daemon=True,
        name="assistant-loop",
    )

    stt_thread.start()
    worker_thread.start()

    ui = AssistantUI(ui_queue=ui_queue)
    try:
        ui.run()
    finally:
        stop_event.set()
        stt_thread.join(timeout=2)
        worker_thread.join(timeout=2)


if __name__ == "__main__":
    main()
