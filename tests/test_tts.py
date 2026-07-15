from unittest.mock import patch

from src.tts import PiperTTS


def test_piper_tts_no_stop_no_proc_no_lock_no_voice() -> None:
    tts = PiperTTS()
    assert not hasattr(tts, "stop")
    assert not hasattr(tts, "_proc")
    assert not hasattr(tts, "_lock")
    assert not hasattr(tts, "_voice")


def test_piper_tts_init_takes_no_args() -> None:
    PiperTTS()


def test_speak_falls_back_when_model_missing(capsys) -> None:
    tts = PiperTTS()
    with patch("src.tts.SETTINGS") as mock_s:
        mock_s.piper_model_path.exists.return_value = False
        mock_s.tts_fallback_stdout = True
        result = tts.speak("hola mundo")
    assert result is None
    captured = capsys.readouterr()
    assert "[TTS] hola mundo" in captured.out


def test_speak_returns_bytes_when_model_available(tmp_path) -> None:
    tts = PiperTTS()
    wav_bytes = b"RIFF\x12\x34\x56\x78"
    with patch("src.tts.SETTINGS") as mock_s:
        mock_s.piper_model_path.exists.return_value = True
        mock_s.piper_model_path.__str__.return_value = "/fake/model.onnx"
        with patch("src.tts.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = wav_bytes
            result = tts.speak("hola")
    assert result == wav_bytes


def test_speak_fallback_on_subprocess_failure(capsys) -> None:
    tts = PiperTTS()
    with patch("src.tts.SETTINGS") as mock_s:
        mock_s.piper_model_path.exists.return_value = True
        mock_s.piper_model_path.__str__.return_value = "/fake/model.onnx"
        mock_s.tts_fallback_stdout = True
        with patch("src.tts.subprocess.run", side_effect=RuntimeError("no piper")):
            result = tts.speak("fallback text")
    assert result is None
    captured = capsys.readouterr()
    assert "[TTS] fallback text" in captured.out

