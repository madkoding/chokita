from __future__ import annotations

import asyncio
import base64
import json
import logging
import logging.handlers
import os
import queue
import subprocess
import sys
import threading
import time
import types
import urllib.request
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import SETTINGS
from src.llm import OllamaClient, load_soul_text
from src.memory import Memory
from src.sleep import SleepThread
from src.soul import SoulThread
from src.tts import PiperTTS

LOG_DIR = Path.home() / ".local/share/chokita"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(LOG_DIR / "chokita.log", maxBytes=5_242_880, backupCount=2),
    ],
)
LOGGER = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

text_queue: queue.Queue[str] = queue.Queue()
ui_queue: queue.Queue[dict[str, Any]] = queue.Queue()
stop_event = threading.Event()
abort_event = threading.Event()
mute_event = threading.Event()

memory: Memory | None = None
llm: OllamaClient | None = None
tts: PiperTTS | None = None
worker_thread: threading.Thread | None = None
soul_thread: SoulThread | None = None
sleep_thread: SleepThread | None = None
stt_proc: subprocess.Popen[bytes] | None = None
stt_stdout_thread: threading.Thread | None = None
stt_stdin_lock = threading.Lock()
stt_dead = threading.Event()  # ponytail: flag de EOF. _ensure_stt chequea además de poll().

_last_activity = time.time()
_activity_lock = threading.Lock()


def _activity_ping() -> None:
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def _seconds_idle() -> float:
    with _activity_lock:
        return time.time() - _last_activity


def _crash_handler(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: types.TracebackType | None,
) -> None:
    # ponytail: cleanup graceful > kill instantáneo. setea stop_event; el lifespan
    # post-yield corre el cleanup. si en 5s el main sigue vivo, kill duro.
    LOGGER.critical("Excepción no capturada", exc_info=(exc_type, exc_value, exc_tb))
    stderr = sys.__stderr__ or sys.stderr
    stderr.write("Ocurrió un error. Revisá: ~/.local/share/chokita/chokita.log\n")
    stop_event.set()
    time.sleep(5)
    if threading.current_thread() is threading.main_thread():
        os._exit(1)


def _thread_crash_handler(args: threading.ExceptHookArgs) -> None:
    thread_name = args.thread.name if args.thread else "?"
    LOGGER.critical("Excepción en hilo %s", thread_name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))  # type: ignore[arg-type]
    stderr = sys.__stderr__ or sys.stderr
    stderr.write("Ocurrió un error. Revisá: ~/.local/share/chokita/chokita.log\n")
    stop_event.set()


sys.excepthook = _crash_handler
threading.excepthook = _thread_crash_handler


def _smoke_check() -> None:
    if not SETTINGS.tts_fallback_stdout:
        if not SETTINGS.piper_model_path.exists():
            print(f"ERROR: Modelo de voz Piper no encontrado en {SETTINGS.piper_model_path}")
            print("Ejecutá: bash scripts/download_models.sh")
            sys.exit(1)
    try:
        url = f"{SETTINGS.ollama_base_url}/api/tags"
        urllib.request.urlopen(url, timeout=3)
    except Exception:
        print(f"ERROR: Ollama no responde en {SETTINGS.ollama_base_url}")
        print("Asegurate de tener Ollama corriendo: ollama serve")
        sys.exit(1)


_SPINNER = "|/-\\"

def _spinner(msg: str, done: threading.Event) -> None:
    i = 0
    while not done.is_set():
        print(f"\r{msg} {_SPINNER[i % len(_SPINNER)]}", end="", flush=True)
        i += 1
        time.sleep(0.15)
    print(f"\r{msg} ✓", flush=True)


def _preload_with_spinner(label: str, fn: Callable[[], None]) -> bool:
    done = threading.Event()
    t = threading.Thread(target=_spinner, args=(label, done), daemon=True)
    t.start()
    try:
        fn()
        return True
    except Exception as exc:
        LOGGER.warning("%s falló: %s", label, exc)
        return False
    finally:
        done.set()
        t.join(timeout=2)


def _warm_chat_model() -> None:
    """Pre-carga el modelo de chat en memoria con una inferencia real."""
    url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": SETTINGS.ollama_model,
        "stream": False,
        "keep_alive": SETTINGS.ollama_keep_alive,
        "messages": [{"role": "user", "content": "hi"}],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=SETTINGS.ollama_timeout_seconds).read()


def _warm_embed_model() -> None:
    """Pre-carga el modelo de embeddings con una inferencia real."""
    url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/embeddings"
    payload = {
        "model": SETTINGS.ollama_embed_model,
        "prompt": "test",
        "keep_alive": SETTINGS.ollama_keep_alive,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=SETTINGS.ollama_timeout_seconds).read()


def _pull_ollama_model(model: str) -> bool:
    """Descarga un modelo con `ollama pull`. Retorna True si OK."""
    print(f"  Descargando {model}...", flush=True)
    try:
        r = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            LOGGER.error("ollama pull %s falló (rc=%d): %s", model, r.returncode, r.stderr[:200])
            return False
        return True
    except FileNotFoundError:
        LOGGER.error("comando 'ollama' no encontrado en PATH")
        return False
    except subprocess.TimeoutExpired:
        LOGGER.error("ollama pull %s timeout (600s)", model)
        return False


def _ensure_model(label: str, model: str, warm_fn: Callable[[], None]) -> bool:
    """Pre-carga un modelo. Si falla, hace pull y reintenta."""
    if _preload_with_spinner(label, warm_fn):
        return True
    print(f"  {model} no está descargado. Haciendo pull...", flush=True)
    if not _pull_ollama_model(model):
        return False
    return _preload_with_spinner(label, warm_fn)


def assistant_loop() -> None:
    global memory, llm, tts
    assert memory is not None and llm is not None and tts is not None
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

            if abort_event.is_set():
                LOGGER.info("Abort fired; dropping LLM answer.")
                ui_queue.put({"type": "state", "state": "IDLE", "message": "Detenido."})
                continue

            ui_queue.put({"type": "state", "state": "SPEAKING", "message": answer})
            mute_event.set()
            try:
                wav_data = tts.speak(answer)
                if wav_data:
                    ui_queue.put({"type": "audio", "data": base64.b64encode(wav_data).decode()})
            finally:
                mute_event.clear()
            ui_queue.put({"type": "state", "state": "IDLE", "message": "Esperando comando..."})
            _activity_ping()

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


def _start_stt() -> bool:
    """Inicia el subprocess STT y espera a que cargue el modelo. Retorna True si OK."""
    global stt_proc, stt_stdout_thread
    LOGGER.debug("Starting STT subprocess...")

    env = dict(os.environ)
    if os.environ.get("CHOKITA_STT_TEST", "") == "1":
        env["CHOKITA_STT_TEST"] = "1"
    env["CHOKITA_ASR_MODEL"] = SETTINGS.asr_model
    env["CHOKITA_SAMPLE_RATE"] = str(SETTINGS.sample_rate_hz)

    try:
        stt_proc = subprocess.Popen(
            [sys.executable, "-m", "src.stt_subprocess"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except Exception:
        LOGGER.exception("Failed to start STT subprocess")
        ui_queue.put({"type": "log", "message": "❌ No se pudo iniciar STT"})
        return False

    LOGGER.debug("STT subprocess iniciado (pid=%d)", stt_proc.pid)

    # Esperar al evento "Modelo cargado" leyendo stdout línea por línea.
    stt_loaded = threading.Event()

    def _read_stdout() -> None:
        global stt_stdout_thread
        assert stt_proc and stt_proc.stdout
        for raw_line in iter(stt_proc.stdout.readline, b""):
            if stop_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("event")

            if etype == "status" and "cargado" in event.get("message", "").lower():
                stt_loaded.set()

            if etype == "listening":
                ui_queue.put({"type": "state", "state": "LISTENING", "message": "Escuchando..."})
            elif etype == "not_recognized":
                ui_queue.put({"type": "log", "message": "🔇 No te escuché bien. Probá hablar más cerca del mic."})
            elif etype == "audio_level":
                ui_queue.put({"type": "audio_level", "level": event.get("level", 0.0)})
            elif etype == "recognized":
                text = event.get("text", "")
                remainder = event.get("remainder", "")
                is_stop = event.get("stop", False)
                LOGGER.info("Recognized: %s (stop=%s)", text, is_stop)
                if is_stop:
                    ui_queue.put({"type": "state", "state": "RECOGNIZED", "message": f"[STOP] {text}"})
                    abort_event.set()
                    ui_queue.put({"type": "state", "state": "LISTENING", "message": "Escuchando..."})
                elif remainder:
                    ui_queue.put({"type": "voice_input", "text": text})
                    ui_queue.put({"type": "state", "state": "RECOGNIZED", "message": text})
                    ui_queue.put({"type": "state", "state": "LISTENING", "message": "Escuchando..."})
                else:
                    ui_queue.put({"type": "state", "state": "RECOGNIZED", "message": "Sí? Te escucho..."})
                    ui_queue.put({"type": "state", "state": "LISTENING", "message": "Escuchando..."})
            elif etype == "error":
                msg = event.get("message", "error desconocido")
                LOGGER.warning("STT subprocess error: %s", msg)
                ui_queue.put({"type": "log", "message": f"⚠ {msg}"})
            elif etype == "status":
                msg = event.get("message", "")
                if msg:
                    LOGGER.debug("STT status: %s", msg)
                    ui_queue.put({"type": "log", "message": f"🎤 {msg}"})

        # ponytail: si el for termina (EOF o stop_event), marcar muerto para que
        # _ensure_stt lo detecte en la próxima utterance.
        stt_dead.set()

    stt_stdout_thread = threading.Thread(target=_read_stdout, daemon=True, name="stt-stdout")
    stt_stdout_thread.start()

    done = threading.Event()
    t = threading.Thread(target=_spinner, args=("Cargando modelo de voz (STT)...", done), daemon=True)
    t.start()
    if not stt_loaded.wait(timeout=300):
        done.set()
        t.join(timeout=2)
        LOGGER.error("STT no cargó el modelo en 300s")
        return False
    done.set()
    t.join(timeout=2)
    return True


def _stop_stt() -> None:
    global stt_proc
    if stt_proc:
        try:
            with stt_stdin_lock:
                if stt_proc.stdin:
                    stt_proc.stdin.write(json.dumps({"cmd": "stop"}).encode() + b"\n")
                    stt_proc.stdin.flush()
        except Exception:
            pass
        try:
            stt_proc.wait(timeout=5)
        except Exception:
            stt_proc.kill()
        stt_proc = None


def _ensure_stt() -> None:
    global stt_proc, stt_stdout_thread
    # ponytail: chequear stt_dead ademas de poll() para detectar EOF.
    if stt_dead.is_set():
        stt_dead.clear()
        stt_proc = None
        stt_stdout_thread = None
    if stt_proc is not None and stt_proc.poll() is not None:
        LOGGER.warning("STT subprocess muerto (rc=%d), reiniciando...", stt_proc.returncode)
        stt_proc = None
        stt_stdout_thread = None
    if stt_proc is None:
        _start_stt()


def _stt_send_audio(audio_base64: str) -> None:
    # ponytail: lock global evita que dos WS concurrentes reinicien STT a la vez.
    with stt_stdin_lock:
        raw_len = len(audio_base64) * 3 // 4  # aprox decoded size
        LOGGER.debug("audio utterance recibida: ~%d bytes PCM", raw_len)
        _ensure_stt()
        if stt_proc and stt_proc.stdin:
            try:
                stt_proc.stdin.write(json.dumps({"audio": audio_base64}).encode() + b"\n")
                stt_proc.stdin.flush()
            except Exception:
                LOGGER.exception("Failed to send audio to STT subprocess")
                _ensure_stt()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global memory, llm, tts, worker_thread, soul_thread, sleep_thread

    print("chokita — iniciando...", flush=True)
    _smoke_check()

    memory = Memory()
    memory.start_session()
    try:
        memory.seed_soul(load_soul_text())
    except Exception:
        LOGGER.warning("No se pudo seedear SOUL en RAG")

    print(f"✓ Memoria SQLite: {SETTINGS.db_path}", flush=True)

    tts = PiperTTS()
    llm = OllamaClient(memory=memory, ui_queue=ui_queue, abort_event=abort_event)

    # --- Pre-carga de modelos con spinner en consola ---
    print("Precargando modelos en memoria...", flush=True)

    ok_chat = _ensure_model(f"  Modelo de chat ({SETTINGS.ollama_model})", SETTINGS.ollama_model, _warm_chat_model)
    ok_embed = _ensure_model(f"  Modelo de embeddings ({SETTINGS.ollama_embed_model})", SETTINGS.ollama_embed_model, _warm_embed_model)

    if not ok_chat:
        print("✗  Modelo de chat no disponible tras pull. Abortando.", flush=True)
        sys.exit(1)
    if not ok_embed:
        print("✗  Modelo de embeddings no disponible tras pull. Abortando.", flush=True)
        sys.exit(1)

    worker_thread = threading.Thread(
        target=assistant_loop, daemon=True, name="assistant-loop",
    )
    worker_thread.start()

    soul_thread = SoulThread(
        memory=memory, chat_fn=llm.chat_raw, stop_event=stop_event, activity_fn=_seconds_idle,
    )
    soul_thread.start()

    def _summarize(text: str) -> str:
        return llm.chat_raw([
            {"role": "system", "content": "Resumí los siguientes puntos en 2-3 lineas, en español rioplatense."},
            {"role": "user", "content": text},
        ])

    sleep_thread = SleepThread(
        memory=memory, summarize_fn=_summarize, stop_event=stop_event,
        activity_fn=_seconds_idle, ui_queue=ui_queue,
    )
    sleep_thread.start()

    # Iniciar STT y esperar a que cargue el modelo antes de levantar la UI.
    ok_stt = _start_stt()
    if not ok_stt:
        print("⚠  STT no cargó el modelo de voz. Entrada por voz deshabilitada.", flush=True)

    ui_queue.put({"type": "state", "state": "IDLE", "message": "Listo."})
    print("✓ Servidor listo. Abrí http://localhost:8080 en tu navegador.", flush=True)

    yield

    # ponytail: cleanup en try/finally. _stop_stt y memory.close deben correr
    # aunque un join lance, sino queda STT subprocess zombie y DB sin cerrar.
    try:
        stop_event.set()
        _stop_stt()
        if worker_thread:
            worker_thread.join(timeout=SETTINGS.shutdown_join_timeout_seconds)
        if soul_thread:
            soul_thread.join(timeout=SETTINGS.shutdown_join_timeout_seconds + 1)
        if sleep_thread:
            sleep_thread.join(timeout=SETTINGS.shutdown_join_timeout_seconds + 1)
        try:
            if llm:
                llm.extract_memories()
        except Exception:
            pass
    finally:
        if memory:
            try:
                memory.end_session()
            except Exception:
                pass
            try:
                memory.close()
            except Exception:
                pass


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    # ponytail: token simple desde env. si CHOKITA_WS_TOKEN está seteado, lo exigimos.
    if SETTINGS.ws_token:
        token_qs = websocket.query_params.get("token", "")
        auth_hdr = websocket.headers.get("authorization", "")
        token_hdr = auth_hdr[7:] if auth_hdr.lower().startswith("bearer ") else ""
        if token_qs != SETTINGS.ws_token and token_hdr != SETTINGS.ws_token:
            await websocket.close(code=1008, reason="invalid token")
            return
    await websocket.accept()

    ui_queue.put({"type": "state", "state": "IDLE", "message": "Conectado."})

    async def sender() -> None:
        while True:
            try:
                event = ui_queue.get_nowait()
                await websocket.send_json(event)
            except queue.Empty:
                await asyncio.sleep(0.05)

    async def receiver() -> None:
        # ponytail: cap de 5MB de base64 = ~30s @ 16kHz int16. evita OOM por audio gigante.
        while True:
            msg = await websocket.receive_json()
            t = msg.get("type")
            if t == "text":
                text = msg.get("text", "").strip()
                if text:
                    text_queue.put(text)
            elif t == "stop":
                abort_event.set()
            elif t == "audio_utterance":
                data = msg.get("data", "")
                if data and len(data) <= 5_000_000:
                    _stt_send_audio(data)

    sender_task = asyncio.create_task(sender())
    receiver_task = asyncio.create_task(receiver())

    try:
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except (WebSocketDisconnect, asyncio.CancelledError):
        sender_task.cancel()
        receiver_task.cancel()
