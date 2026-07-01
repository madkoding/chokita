from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from src.config import SETTINGS

DEVNULL = subprocess.DEVNULL

LOGGER = logging.getLogger(__name__)

PLAYBACK_APLAY = "aplay"
PLAYBACK_PAPLAY = "paplay"
PLAYBACK_FFPLAY = "ffplay"
PLAYBACK_POWERSHELL = "powershell"
PLAYBACK_AUTO = "auto"


class PiperTTS:
    def __init__(self) -> None:
        self.playback_cmd = SETTINGS.playback_command
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        """Kill any in-flight playback. Safe to call from another thread."""
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    def _run_playback(self, args: list[str]) -> None:
        """Run a playback subprocess, stored so stop() can kill it."""
        with self._lock:
            self._proc = subprocess.Popen(args, stdout=DEVNULL, stderr=DEVNULL)
            proc = self._proc
        proc.wait()
        with self._lock:
            if self._proc is proc:
                self._proc = None

    def _play_wav(self, wav_path: Path) -> None:
        if self.playback_cmd == PLAYBACK_APLAY:
            self._run_playback([PLAYBACK_APLAY, "-q", str(wav_path)])
            return
        if self.playback_cmd == PLAYBACK_PAPLAY:
            self._run_playback([PLAYBACK_PAPLAY, str(wav_path)])
            return
        if self.playback_cmd == PLAYBACK_FFPLAY:
            self._run_playback(
                [PLAYBACK_FFPLAY, "-v", "quiet", "-autoexit", "-nodisp", str(wav_path)]
            )
            return
        if self.playback_cmd == PLAYBACK_POWERSHELL:
            self._play_via_powershell(wav_path)
            return

        if self.playback_cmd != PLAYBACK_AUTO:
            raise RuntimeError(f"Unsupported playback command: {self.playback_cmd}")

        attempts: list[list[str]] = []
        # ponytail: WSLg exposes audio via PulseAudio, not ALSA — aplay fails with
        # "no soundcards found" there. Prefer paplay when present.
        if shutil.which(PLAYBACK_PAPLAY):
            attempts.append([PLAYBACK_PAPLAY, str(wav_path)])
        if shutil.which(PLAYBACK_APLAY):
            attempts.append([PLAYBACK_APLAY, "-q", str(wav_path)])
        if shutil.which(PLAYBACK_FFPLAY):
            attempts.append(
                [PLAYBACK_FFPLAY, "-v", "quiet", "-autoexit", "-nodisp", str(wav_path)]
            )
        if shutil.which("powershell.exe"):
            attempts.append(["__powershell__", str(wav_path)])

        if not attempts:
            raise RuntimeError("No playback binary available (aplay/ffplay/powershell)")

        last_error: Exception | None = None
        for cmd in attempts:
            try:
                if cmd[0] == "__powershell__":
                    self._play_via_powershell(Path(cmd[1]))
                else:
                    self._run_playback(cmd)
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
            ["wslpath", "-w", str(wav_path)], stderr=DEVNULL
        ).decode().strip()
        self._run_playback(
            ["powershell.exe", "-c",
             f"(New-Object Media.SoundPlayer '{win_path}').PlaySync()"]
        )

    def speak(self, text: str) -> None:
        fallback = SETTINGS.tts_fallback_stdout

        if fallback and not shutil.which(SETTINGS.piper_bin):
            print(f"[TTS] {text}", flush=True)
            return

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
            if SETTINGS.piper_speaker is not None:
                cmd += ["--speaker", str(SETTINGS.piper_speaker)]
            with self._lock:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=DEVNULL,
                    stderr=DEVNULL,
                )
                proc = self._proc
            try:
                proc.communicate(input=text.encode("utf-8"))
            finally:
                with self._lock:
                    if self._proc is proc:
                        self._proc = None

            self._play_wav(wav_path)
        except Exception as exc:
            # ponytail: playback failure shouldn't crash the assistant loop —
            # the answer text is already shown in the UI via SPEAKING state.
            LOGGER.warning("TTS failure: %s — el texto ya se muestra en la UI", exc)
            if fallback:
                print(f"[TTS] {text}", flush=True)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink()
