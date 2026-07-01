from __future__ import annotations

import logging
import logging.handlers
import queue
import shutil
import signal
import sys
import threading
import time
from typing import Any

from src.audio import SpeechRecognizerThread
from src.config import SETTINGS
from src.llm import OllamaClient
from src.memory import Memory
from src.sleep import SleepThread
from src.soul import SoulThread
from src.tts import PiperTTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler("chokita.log", maxBytes=5_242_880, backupCount=2),
    ],
)
LOGGER = logging.getLogger(__name__)

_last_activity = time.time()
_activity_lock = threading.Lock()


def _activity_ping() -> None:
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def _seconds_idle() -> float:
    with _activity_lock:
        return time.time() - _last_activity


def _smoke_check() -> None:
    if not SETTINGS.vosk_model_path.exists():
        print(f"ERROR: Modelo Vosk no encontrado en {SETTINGS.vosk_model_path}")
        print("Ejecutá: bash scripts/download_models.sh")
        sys.exit(1)
    if not SETTINGS.tts_fallback_stdout:
        if not shutil.which(SETTINGS.piper_bin):
            print(f"ERROR: Piper no encontrado en PATH y TTS_FALLBACK_STDOUT=0.")
            sys.exit(1)
        if not SETTINGS.piper_model_path.exists():
            print(f"ERROR: Modelo de voz Piper no encontrado en {SETTINGS.piper_model_path}")
            print("Ejecutá: bash scripts/download_models.sh")
            sys.exit(1)
    try:
        import urllib.request
        url = f"{SETTINGS.ollama_base_url}/api/tags"
        urllib.request.urlopen(url, timeout=3)
    except Exception:
        print(f"ERROR: Ollama no responde en {SETTINGS.ollama_base_url}")
        print("Asegurate de tener Ollama corriendo:")
        print("  ollama serve")
        print(f"  ollama pull {SETTINGS.ollama_model}")
        sys.exit(1)


def assistant_loop(
    text_queue: queue.Queue[str],
    ui_queue: queue.Queue[dict[str, Any]],
    stop_event: threading.Event,
    memory: Memory,
    abort_event: threading.Event,
    tts: PiperTTS,
) -> None:
    llm = OllamaClient(memory=memory, ui_queue=ui_queue)

    while not stop_event.is_set():
        try:
            user_text = text_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        _activity_ping()
        abort_event.clear()
        try:
            ui_queue.put({"type": "state", "state": "THINKING", "message": "Consultando LLM..."})
            answer = llm.chat(user_text)

            # If a stop command arrived while we were thinking, drop the answer.
            if abort_event.is_set():
                LOGGER.info("Abort fired; dropping LLM answer.")
                ui_queue.put({"type": "state", "state": "IDLE", "message": "Detenido."})
                continue

            ui_queue.put({"type": "state", "state": "SPEAKING", "message": answer})
            tts.speak(answer)
            if abort_event.is_set():
                tts.stop()
            ui_queue.put({"type": "state", "state": "IDLE", "message": "Esperando comando..."})
            _activity_ping()  # speaking counts as activity for idle purposes

            # Long-term memory extraction: every N messages, extract & persist.
            if memory.session_message_count() >= SETTINGS.memory_extract_interval:
                try:
                    n = llm.extract_memories()
                    if n:
                        ui_queue.put({"type": "log", "message": f"🧠 {n} memorias guardadas a largo plazo."})
                except Exception:
                    LOGGER.warning("Memory extraction failed")
        except Exception as exc:
            LOGGER.exception("Assistant loop iteration failed")
            ui_queue.put({"type": "state", "state": "ERROR", "message": str(exc)[:80]})
            ui_queue.put({"type": "log", "message": str(exc)})


def _headless_loop(
    text_queue: queue.Queue[str],
    ui_queue: queue.Queue[dict[str, Any]],
    stop_event: threading.Event,
) -> None:
    def _stdin_reader() -> None:
        for line in sys.stdin:
            line = line.strip()
            if line:
                text_queue.put(line)

    reader = threading.Thread(target=_stdin_reader, daemon=True, name="stdin-reader")
    reader.start()

    while not stop_event.is_set():
        try:
            event = ui_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if event.get("type") == "shutdown":
            break
        if event.get("type") == "state":
            print(f"[{event['state']}] {event.get('message', '')}", flush=True)


def _has_microphone() -> bool:
    import os as _os

    _saved = _os.dup(2)
    _dn = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(_dn, 2)
    _os.close(_dn)
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        info = p.get_default_input_device_info()
        p.terminate()
        return info["maxInputChannels"] > 0
    except Exception:
        return False
    finally:
        _os.dup2(_saved, 2)
        _os.close(_saved)


def main() -> None:
    print("chokita — iniciando...", flush=True)
    _smoke_check()
    print("✓ Ollama listo", flush=True)

    # Memory + session
    memory = Memory()
    memory.start_session()
    print(f"✓ Memoria SQLite: {SETTINGS.db_path}", flush=True)

    # Import FaceApp BEFORE starting STT thread — pyaudio's import lock
    # would otherwise block textual from importing (ALSA init takes ~30s on WSL)
    from src.ui import FaceApp

    text_queue: queue.Queue[str] = queue.Queue()
    ui_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    stop_event = threading.Event()
    abort_event = threading.Event()
    tts = PiperTTS()

    def _handle_signal(_sig: int, _frame: Any) -> None:
        stop_event.set()
        ui_queue.put({"type": "shutdown"})

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def _on_stop_command() -> None:
        """Called from the STT thread when a voice stop command is heard."""
        LOGGER.info("Stop command -> abort + kill audio")
        abort_event.set()
        tts.stop()

    stt_thread: SpeechRecognizerThread | None = None

    def _start_stt() -> None:
        nonlocal stt_thread
        if _has_microphone():
            stt_thread = SpeechRecognizerThread(
                text_queue=text_queue,
                ui_queue=ui_queue,
                stop_event=stop_event,
                on_stop_command=_on_stop_command,
            )
            stt_thread.start()
        else:
            LOGGER.warning("Micrófono no detectado. Modo solo texto.")

    threading.Thread(target=_start_stt, daemon=True, name="stt-launcher").start()

    worker_thread = threading.Thread(
        target=assistant_loop,
        args=(text_queue, ui_queue, stop_event, memory, abort_event, tts),
        daemon=True,
        name="assistant-loop",
    )
    worker_thread.start()

    # Soul reflection thread: idle musings on her own personality.
    # Build a lightweight chat fn that calls Ollama directly (no tools, no memory writes).
    def _soul_chat(messages: list[dict[str, str]]) -> str:
        client = OllamaClient(memory=memory)
        return client.chat_raw(messages)

    soul_thread = SoulThread(
        memory=memory,
        chat_fn=_soul_chat,
        stop_event=stop_event,
        activity_fn=_seconds_idle,
    )
    soul_thread.start()
    LOGGER.info(
        "Soul thread iniciado (idle %ds, reflexion cada %g-%gs)",
        int(SETTINGS.soul_idle_threshold_seconds),
        SETTINGS.soul_reflect_min_seconds,
        SETTINGS.soul_reflect_max_seconds,
    )

    # REM sleep thread: reindexes the RAG (RAPTOR) while idle.
    def _summarize(text: str) -> str:
        client = OllamaClient(memory=memory)
        return client.chat_raw(
            [
                {"role": "system", "content": "Resumí los siguientes puntos en 2-3 lineas, en español rioplatense."},
                {"role": "user", "content": text},
            ],
        )

    sleep_thread = SleepThread(
        memory=memory,
        summarize_fn=_summarize,
        stop_event=stop_event,
        activity_fn=_seconds_idle,
        ui_queue=ui_queue,
    )
    sleep_thread.start()
    LOGGER.info(
        "REM sleep thread iniciado (idle %ds, RAPTOR cada %gs)",
        int(SETTINGS.rem_idle_threshold_seconds),
        SETTINGS.rem_raptor_interval_seconds,
    )

    print("Iniciando interfaz...", flush=True)
    # Clear terminal so TUI starts on a clean screen
    print("\033[2J\033[H", end="", flush=True)
    try:
        app = FaceApp(text_queue, ui_queue)
        app.run()
    except Exception as exc:
        LOGGER.warning("TUI no disponible (%s), modo headless", exc)
        _headless_loop(text_queue, ui_queue, stop_event)

    stop_event.set()
    if stt_thread:
        stt_thread.join(timeout=SETTINGS.shutdown_join_timeout_seconds)
    worker_thread.join(timeout=SETTINGS.shutdown_join_timeout_seconds)
    soul_thread.join(timeout=SETTINGS.shutdown_join_timeout_seconds + 1)
    sleep_thread.join(timeout=SETTINGS.shutdown_join_timeout_seconds + 1)
    # Final long-term memory extraction before closing.
    try:
        llm = OllamaClient(memory=memory)
        llm.extract_memories()
    except Exception:
        LOGGER.warning("Final memory extraction failed")
    memory.end_session()
    memory.close()


if __name__ == "__main__":
    main()