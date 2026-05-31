from unittest.mock import patch

from src.tts import PiperTTS


def test_resolve_playback_command_with_fallback() -> None:
    tts = PiperTTS()
    tts.playback_cmd = "auto"
    command_map = {"aplay": "/usr/bin/aplay", "ffplay": None}
    with patch("src.tts.shutil.which", side_effect=command_map.get):
        cmd = tts._resolve_playback_command()
    assert cmd == ["aplay", "-q"]
