from __future__ import annotations

import logging
import subprocess
import sys

from src.config import SETTINGS

LOGGER = logging.getLogger(__name__)


class PiperTTS:
    _PIPER_WORKER = (
        "import sys, wave, tempfile, os\n"
        "from pathlib import Path\n"
        "from piper import PiperVoice\n"
        "from piper.config import SynthesisConfig\n"
        "text = sys.stdin.buffer.read().decode()\n"
        "speaker_id = int(sys.argv[2]) if len(sys.argv) > 2 else None\n"
        "dn = os.open(os.devnull, os.O_WRONLY); os.dup2(dn, 2)\n"
        # ponytail: load() en cada speak (~1-3s overhead). mover a daemon
        # subprocess (stdin/stdout persistentes) cuando la latencia importe.
        "voice = PiperVoice.load(sys.argv[1])\n"
        "config = SynthesisConfig(speaker_id=speaker_id) if speaker_id is not None else None\n"
        "with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:\n"
        "    wav_path = Path(tmp.name)\n"
        "try:\n"
        "    with wave.open(str(wav_path), 'wb') as wav_file:\n"
        "        voice.synthesize_wav(text, wav_file, syn_config=config)\n"
        "    sys.stdout.buffer.write(wav_path.read_bytes())\n"
        "finally:\n"
        "    # ponytail: borra el WAV temp. sin esto, /tmp crece sin límite.\n"
        "    wav_path.unlink(missing_ok=True)\n"
    )

    def speak(self, text: str) -> bytes | None:
        if not SETTINGS.piper_model_path.exists():
            LOGGER.debug("TTS: modelo no encontrado")
            if SETTINGS.tts_fallback_stdout:
                print(f"[TTS] {text}", flush=True)
            return None

        try:
            args = [sys.executable, "-c", self._PIPER_WORKER, str(SETTINGS.piper_model_path)]
            if SETTINGS.piper_speaker is not None:
                args.append(str(SETTINGS.piper_speaker))
            p = subprocess.run(
                args,
                input=text.encode(), capture_output=True, timeout=30,
            )
            if p.returncode != 0:
                raise RuntimeError(p.stderr.decode()[:200])
            return p.stdout
        except subprocess.TimeoutExpired:
            LOGGER.warning("TTS: subprocess timeout (30s)")
            if SETTINGS.tts_fallback_stdout:
                print(f"[TTS] {text}", flush=True)
            return None
        except Exception as exc:
            LOGGER.warning("TTS: subprocess fallo: %s", exc)
            if SETTINGS.tts_fallback_stdout:
                print(f"[TTS] {text}", flush=True)
            return None
