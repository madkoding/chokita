"""Subprocess: Qwen3-ASR STT via stdin audio + transformers inference.

Recibe utterances completas de audio PCM desde stdin como JSON lines:
  {"audio": "<base64 PCM int16 mono 16kHz>"}
  {"cmd": "mute"} / {"cmd": "unmute"} / {"cmd": "stop"}

Emite eventos por stdout (JSON lines):
  {"event": "listening"}
  {"event": "status", "message": "..."}
  {"event": "recognized", "text": "...", "wake": true, "remainder": "...", "stop": false}
  {"event": "error", "message": "..."}
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy

LOG_DIR = Path.home() / ".local/share/chokita"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] stt-sub: %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "stt-subprocess.log")],
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

_WAKE_RE = re.compile(
    r"\b(chokita|choquita|chiquita|chiquitita|chiquitán|chiquitan|chaquita|jaquita|chocita|choki|chiqui|chiquit|chiquitin|chiquitit)\b\s*(.*)",
    re.DOTALL,
)
_STOP_WORDS = re.compile(r"^(para|par\u00e1|detente|callate|c\u00e1llate|silencio|alto|stop)\b")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def parse_wake(text: str) -> tuple[bool, str]:
    """Returns (matched, remainder). remainder is the command after the wake word."""
    n = _normalize(text)
    m = _WAKE_RE.search(n)
    if not m:
        return False, ""
    return True, m.group(2).strip()


def is_stop_command(remainder: str) -> bool:
    """Check if the post-wake-word text is a stop command."""
    return bool(_STOP_WORDS.match(_normalize(remainder)))


def _emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()




def _pcm_to_array(raw_pcm: bytes, sample_rate: int) -> tuple[numpy.ndarray, int]:
    """Convierte raw PCM int16 mono directamente a array float32 normalizado."""
    audio = numpy.frombuffer(raw_pcm, dtype=numpy.int16).astype(numpy.float32) / 32768.0
    return audio, sample_rate


def _read_msg() -> tuple[str | None, Any]:
    """Lee un mensaje JSON de stdin. Retorna (kind, data):
    kind='audio' → data=bytes PCM
    kind='cmd' → data=dict comando
    kind=None → EOF (stdin cerrado)
    """
    try:
        line = sys.stdin.readline()
    except Exception:
        return None, None
    if not line:
        return None, None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None, None
    if "audio" in msg:
        try:
            return "audio", base64.b64decode(msg["audio"])
        except Exception:
            return None, None
    if "cmd" in msg:
        return "cmd", msg
    return None, None


def _process_cmd(msg: Any, muted: bool) -> tuple[bool, bool]:
    """Procesa un comando. Retorna (should_break, new_muted)."""
    cmd = msg.get("cmd", "")
    if cmd == "stop":
        return True, muted
    elif cmd == "mute":
        return False, True
    elif cmd == "unmute":
        return False, False
    return False, muted


def _transcribe(audio_buffer: bytes, sample_rate: int,
                processor: Any, model: Any) -> tuple[str, dict[str, float]]:
    """Transcribe audio buffer con Qwen3-ASR. Retorna (texto, timings)."""
    import torch
    timings: dict[str, float] = {}
    t0 = time.time()
    audio_array, sr = _pcm_to_array(audio_buffer, sample_rate)
    timings["pcm_convert"] = time.time() - t0
    LOGGER.debug("audio: len=%d samples, max=%.4f, min=%.4f, mean_abs=%.4f",
                 len(audio_array), audio_array.max(), audio_array.min(),
                 numpy.abs(audio_array).mean())

    t1 = time.time()
    inputs = processor.apply_transcription_request(
        audio=audio_array, language="Spanish",
    )
    timings["processor"] = time.time() - t1
    LOGGER.debug("input_features shape=%s", inputs["input_features"].shape)
    LOGGER.debug("input_features_mask shape=%s", inputs.get("input_features_mask", "N/A"))
    LOGGER.debug("input_ids decoded=%r", processor.decode(inputs["input_ids"][0]))

    t2 = time.time()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
    timings["to_device"] = time.time() - t2

    t3 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    timings["generate"] = time.time() - t3
    LOGGER.debug("output_ids shape=%s", output_ids.shape)

    t4 = time.time()
    gen = output_ids[:, inputs["input_ids"].shape[1]:]
    LOGGER.debug("gen tokens=%s", gen[0].tolist())
    raw = processor.decode(gen[0], skip_special_tokens=False)
    LOGGER.debug("gen decoded=%r", raw)
    result = processor.decode(gen, return_format="transcription_only")[0]
    timings["decode"] = time.time() - t4
    LOGGER.debug("decode result=%r", result)
    return result, timings


def main() -> None:
    LOGGER.debug("subprocess iniciado, pid=%d", os.getpid())

    sample_rate = int(os.environ.get("CHOKITA_SAMPLE_RATE", "16000"))

    model_id = os.environ.get("CHOKITA_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B-hf")
    LOGGER.debug("cargando modelo ASR: %s", model_id)
    _emit({"event": "status", "message": "Cargando modelo de voz..."})

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, device_map="cpu",  # ponytail: "cpu" evita la dep accelerate.
    )
    model.eval()
    LOGGER.info("modelo ASR cargado en %.1fs, device=%s, dtype=%s",
                time.time() - t0, model.device, model.dtype)
    _emit({"event": "status", "message": "Modelo cargado. Esperando audio..."})

    _emit({"event": "listening"})

    muted = False

    while True:
        try:
            kind, data = _read_msg()
            if kind is None:
                LOGGER.debug("stdin cerrado, saliendo")
                break
            if kind == "cmd":
                should_break, muted = _process_cmd(data, muted)
                if should_break:
                    break
                continue
            if kind != "audio":
                continue

            if muted:
                LOGGER.debug("audio ignorado (muted)")
                continue

            LOGGER.debug("utterance recibida: %d bytes", len(data))

            t1 = time.time()
            text, timings = _transcribe(data, sample_rate, processor, model)
            elapsed = time.time() - t1
            LOGGER.info("transcrito en %.2fs (pcm=%.3fs processor=%.3fs to_device=%.3fs generate=%.3fs decode=%.3fs): %r",
                        elapsed, timings["pcm_convert"], timings["processor"],
                        timings["to_device"], timings["generate"], timings["decode"], text)

            if not text:
                _emit({"event": "not_recognized"})
                _emit({"event": "listening"})
                continue

            # Filtrar transcripciones con caracteres no latinos (falso positivo de idioma)
            if not all(ord(c) < 0x250 or c.isspace() or c in ".,;:!?¡¿áéíóúñüÁÉÍÓÚÑÜ" for c in text):
                LOGGER.warning("transcripción descartada (caracteres no latinos): %r", text)
                _emit({"event": "not_recognized"})
                _emit({"event": "listening"})
                continue

            matched, remainder = parse_wake(text)
            if matched:
                _emit({"event": "recognized", "text": text, "wake": True,
                       "remainder": remainder, "stop": is_stop_command(remainder)})
            else:
                _emit({"event": "recognized", "text": text, "wake": False,
                       "remainder": text, "stop": False})

            _emit({"event": "listening"})
        except Exception:
            LOGGER.exception("Error en loop principal, continuando")
            _emit({"event": "listening"})

    LOGGER.debug("subprocess terminado")


if __name__ == "__main__":
    main()