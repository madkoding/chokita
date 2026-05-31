"""Piper CLI wrapper for speech synthesis and playback."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.config import SETTINGS

LOGGER = logging.getLogger(__name__)

PLAYBACK_APLAY = "aplay"
PLAYBACK_FFPLAY = "ffplay"
PLAYBACK_AUTO = "auto"


class PiperTTS:
    """Generate speech with Piper binary and play with aplay or ffplay."""

    def __init__(self) -> None:
        self.playback_cmd = SETTINGS.playback_command

    def _resolve_playback_command(self) -> list[str]:
        if self.playback_cmd == PLAYBACK_APLAY:
            return [PLAYBACK_APLAY, "-q"]
        if self.playback_cmd == PLAYBACK_FFPLAY:
            return [PLAYBACK_FFPLAY, "-v", "quiet", "-autoexit", "-nodisp"]
        if self.playback_cmd != PLAYBACK_AUTO:
            raise RuntimeError(f"Unsupported playback command: {self.playback_cmd}")

        if shutil.which(PLAYBACK_APLAY):
            return [PLAYBACK_APLAY, "-q"]
        if shutil.which(PLAYBACK_FFPLAY):
            return [PLAYBACK_FFPLAY, "-v", "quiet", "-autoexit", "-nodisp"]
        raise RuntimeError("No playback binary available (aplay/ffplay)")

    def speak(self, text: str) -> None:
        wav_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = Path(tmp.name)

            cmd = [
                SETTINGS.piper_bin,
                "--model",
                str(SETTINGS.piper_model_path),
                "--config",
                str(SETTINGS.piper_config_path),
                "--output_file",
                str(wav_path),
            ]
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)

            playback = self._resolve_playback_command() + [str(wav_path)]
            subprocess.run(playback, check=True)
        except Exception:
            LOGGER.exception("TTS failure")
            raise
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink()
