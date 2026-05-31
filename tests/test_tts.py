from unittest.mock import patch

from src.tts import PiperTTS


def test_resolve_playback_command_with_fallback() -> None:
    tts = PiperTTS()
    tts.playback_cmd = "auto"
    with patch("src.tts.shutil.which", side_effect=lambda cmd: "/usr/bin/aplay" if cmd == "aplay" else None):
        cmd = tts._resolve_playback_command()
    assert cmd == ["aplay", "-q"]
