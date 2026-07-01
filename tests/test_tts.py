import shutil
from unittest.mock import patch

from src.tts import PiperTTS


def test_speak_falls_back_when_piper_missing(capsys) -> None:
    tts = PiperTTS()
    with patch.object(tts, "_get_voice", side_effect=ImportError("no piper")):
        with patch("src.tts.SETTINGS") as mock_settings:
            mock_settings.tts_fallback_stdout = True
            tts.speak("hola mundo")
    captured = capsys.readouterr()
    assert "[TTS] hola mundo" in captured.out


def test_play_wav_auto_prefers_paplay_over_aplay(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")

    tts = PiperTTS()
    tts.playback_cmd = "auto"

    which_map = {"paplay": "/usr/bin/paplay", "aplay": "/usr/bin/aplay"}
    seen: list[list[str]] = []

    def fake_run(args: list[str]) -> None:
        seen.append(args)

    with patch.object(shutil, "which", side_effect=which_map.get):
        with patch.object(tts, "_run_playback", side_effect=fake_run):
            tts._play_wav(wav)

    assert seen and seen[0][0] == "paplay"
