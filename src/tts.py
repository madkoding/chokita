from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

from src.config import SETTINGS

LOGGER = logging.getLogger(__name__)

PLAYBACK_APLAY = "aplay"
PLAYBACK_PAPLAY = "paplay"
PLAYBACK_FFPLAY = "ffplay"
PLAYBACK_AFPLAY = "afplay"
PLAYBACK_POWERSHELL = "powershell"
PLAYBACK_AUTO = "auto"


_PLAYBACK_BACKENDS: list[tuple[str, list[str] | None]] = []
if shutil.which(PLAYBACK_AFPLAY):
    _PLAYBACK_BACKENDS.append((PLAYBACK_AFPLAY, None))
if shutil.which(PLAYBACK_PAPLAY):
    _PLAYBACK_BACKENDS.append((PLAYBACK_PAPLAY, None))
if shutil.which(PLAYBACK_APLAY):
    _PLAYBACK_BACKENDS.append((PLAYBACK_APLAY, ["-q"]))
if shutil.which(PLAYBACK_FFPLAY):
    _PLAYBACK_BACKENDS.append((PLAYBACK_FFPLAY, ["-v", "quiet", "-autoexit", "-nodisp"]))
if shutil.which("powershell.exe"):
    _PLAYBACK_BACKENDS.append(("powershell", None))


class PiperTTS:
    def __init__(self) -> None:
        self.playback_cmd = SETTINGS.playback_command
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._voice: Any = None

    def _get_voice(self) -> Any:
        if self._voice is None:
            with self._lock:
                if self._voice is None:
                    from piper import PiperVoice
                    self._voice = PiperVoice.load(str(SETTINGS.piper_model_path))
        return self._voice

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    def _run_playback(self, args: list[str]) -> None:
        with self._lock:
            self._proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc = self._proc
        proc.wait()
        with self._lock:
            if self._proc is proc:
                self._proc = None

    def _play_wav(self, wav_path: Path) -> None:
        _NAMED = {
            PLAYBACK_APLAY: [PLAYBACK_APLAY, "-q"],
            PLAYBACK_PAPLAY: [PLAYBACK_PAPLAY],
            PLAYBACK_FFPLAY: [PLAYBACK_FFPLAY, "-v", "quiet", "-autoexit", "-nodisp"],
            PLAYBACK_AFPLAY: [PLAYBACK_AFPLAY],
        }
        if self.playback_cmd in _NAMED:
            self._run_playback([*_NAMED[self.playback_cmd], str(wav_path)])
            return
        if self.playback_cmd == PLAYBACK_POWERSHELL:
            self._play_via_powershell(wav_path)
            return
        if self.playback_cmd != PLAYBACK_AUTO:
            raise RuntimeError(f"Unsupported playback command: {self.playback_cmd}")

        if not _PLAYBACK_BACKENDS:
            raise RuntimeError("No playback binary available (aplay/ffplay/powershell)")

        last_error: Exception | None = None
        for name, extra in _PLAYBACK_BACKENDS:
            try:
                if name == "powershell":
                    self._play_via_powershell(wav_path)
                else:
                    args = [name]
                    if extra:
                        args.extend(extra)
                    args.append(str(wav_path))
                    self._run_playback(args)
                return
            except subprocess.CalledProcessError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError("All playback methods failed") from last_error

    def _play_via_powershell(self, wav_path: Path) -> None:
        win_path = subprocess.check_output(
            ["wslpath", "-w", str(wav_path)], stderr=subprocess.DEVNULL
        ).decode().strip()
        self._run_playback(
            ["powershell.exe", "-c",
             f"(New-Object Media.SoundPlayer '{win_path}').PlaySync()"]
        )

    def speak(self, text: str) -> None:
        fallback = SETTINGS.tts_fallback_stdout

        try:
            voice = self._get_voice()
        except Exception:
            if fallback:
                print(f"[TTS] {text}", flush=True)
            return

        wav_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = Path(tmp.name)

            with wave.open(str(wav_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)

            self._play_wav(wav_path)
        except Exception as exc:
            LOGGER.warning("TTS failure: %s — el texto ya se muestra en la UI", exc)
            if fallback:
                print(f"[TTS] {text}", flush=True)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink()
