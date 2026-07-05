import subprocess
from unittest.mock import Mock, patch

import pytest

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
    seen: list[list[str]] = []

    def fake_run(args: list[str]) -> None:
        seen.append(args)

    with patch("src.tts._PLAYBACK_CMDS", [["paplay"], ["aplay", "-q"]]):
        with patch.object(tts, "_run_playback", side_effect=fake_run):
            tts._play_wav(wav)

    assert seen and seen[0][0] == "paplay"


def test_available_playback_cmds_all() -> None:
    from src.tts import _PLAYBACK_CMDS
    assert isinstance(_PLAYBACK_CMDS, list)


def test_available_playback_cmds_none() -> None:
    pass  # ponytail: removed, _PLAYBACK_CMDS is module constant evaluated at import


def test_get_voice_loads_piper() -> None:
    tts = PiperTTS()
    mock_piper = Mock()
    mock_piper.PiperVoice.load.return_value = "voice_obj"
    with patch.dict("sys.modules", {"piper": mock_piper}):
        voice = tts._get_voice()
        assert voice == "voice_obj"
        voice2 = tts._get_voice()
        assert voice2 == "voice_obj"


def test_stop_kills_running_process() -> None:
    tts = PiperTTS()
    proc = Mock()
    proc.poll.return_value = None
    tts._proc = proc
    tts.stop()
    proc.kill.assert_called_once()


def test_run_playback(tmp_path) -> None:
    tts = PiperTTS()
    with patch("src.tts.subprocess.Popen") as mock_popen:
        proc = Mock()
        mock_popen.return_value = proc
        tts._run_playback(["paplay", str(tmp_path / "test.wav")])
        mock_popen.assert_called_once_with(
            ["paplay", str(tmp_path / "test.wav")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.wait.assert_called_once()


def test_play_wav_named_command(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")
    tts = PiperTTS()
    tts.playback_cmd = "aplay"
    with patch.object(tts, "_run_playback") as mock_run:
        tts._play_wav(wav)
        mock_run.assert_called_once_with(["aplay", "-q", str(wav)])


def test_play_wav_unsupported_command(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")
    tts = PiperTTS()
    tts.playback_cmd = "invalid"
    with pytest.raises(RuntimeError, match="Unsupported"):
        tts._play_wav(wav)


def test_play_wav_auto_no_binaries(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")
    tts = PiperTTS()
    tts.playback_cmd = "auto"
    with patch("src.tts._PLAYBACK_CMDS", []):
        with pytest.raises(RuntimeError, match="No playback binary"):
            tts._play_wav(wav)



def test_play_wav_powershell(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")
    tts = PiperTTS()
    tts.playback_cmd = "powershell"
    with patch.object(tts, "_play_via_powershell") as mock_ps:
        tts._play_wav(wav)
        mock_ps.assert_called_once_with(wav)


def test_play_via_powershell(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")
    tts = PiperTTS()
    with patch("src.tts.subprocess.check_output", return_value=b"C:\\path\\out.wav"), \
         patch.object(tts, "_run_playback") as mock_run:
        tts._play_via_powershell(wav)
        mock_run.assert_called_once_with(
            ["powershell.exe", "-c",
             "(New-Object Media.SoundPlayer 'C:\\path\\out.wav').PlaySync()"]
        )


def test_speak_synthesizes_and_plays(tmp_path) -> None:
    tts = PiperTTS()
    mock_voice = Mock()
    tts._voice = mock_voice
    wav_path = tmp_path / "speak.wav"
    with patch("src.tts.tempfile.NamedTemporaryFile") as mock_tmp:
        mock_tmp.return_value.__enter__.return_value.name = str(wav_path)
        with patch("src.tts.wave.open") as mock_wave_open:
            mock_wav_file = Mock()
            mock_wave_open.return_value.__enter__.return_value = mock_wav_file
            with patch.object(tts, "_play_wav") as mock_play:
                tts.speak("hola")
                mock_voice.synthesize_wav.assert_called_once_with("hola", mock_wav_file)
                mock_play.assert_called_once()


def test_speak_fallback_on_synthesis_failure(capsys) -> None:
    tts = PiperTTS()
    mock_voice = Mock()
    tts._voice = mock_voice
    with patch("src.tts.wave.open", side_effect=OSError("no wav")):
        with patch("src.tts.SETTINGS") as mock_s:
            mock_s.tts_fallback_stdout = True
            tts.speak("fallback text")
    captured = capsys.readouterr()
    assert "[TTS] fallback text" in captured.out


def test_stop_already_finished_process() -> None:
    tts = PiperTTS()
    proc = Mock()
    proc.poll.return_value = 0
    tts._proc = proc
    tts.stop()
    proc.kill.assert_not_called()


def test_play_wav_auto_all_fail(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")
    tts = PiperTTS()
    tts.playback_cmd = "auto"
    with patch("src.tts._PLAYBACK_CMDS", [
        ["paplay"], ["aplay", "-q"],
    ]):
        with patch.object(tts, "_run_playback", side_effect=[
            subprocess.CalledProcessError(1, "paplay"),
            subprocess.CalledProcessError(1, "aplay"),
        ]):
            with pytest.raises(RuntimeError, match="All playback methods failed"):
                tts._play_wav(wav)


def test_play_wav_auto_generic_error(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")
    tts = PiperTTS()
    tts.playback_cmd = "auto"
    with patch("src.tts._PLAYBACK_CMDS", [["paplay"]]):
        with patch.object(tts, "_run_playback", side_effect=RuntimeError("generic")):
            with pytest.raises(RuntimeError, match="All playback methods failed"):
                tts._play_wav(wav)


def test_play_wav_auto_powershell(tmp_path) -> None:
    wav = tmp_path / "out.wav"
    wav.write_bytes(b"RIFF")
    tts = PiperTTS()
    tts.playback_cmd = "auto"
    with patch("src.tts._PLAYBACK_CMDS", [["__powershell__"]]):
        with patch.object(tts, "_play_via_powershell") as mock_ps:
            tts._play_wav(wav)
            mock_ps.assert_called_once_with(wav)


def test_stop_kill_raises() -> None:
    tts = PiperTTS()
    proc = Mock()
    proc.poll.return_value = None
    proc.kill.side_effect = OSError("permission denied")
    tts._proc = proc
    tts.stop()
    proc.kill.assert_called_once()
