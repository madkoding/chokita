from pathlib import Path

_SRC_WEB = Path(__file__).resolve().parent.parent / "src" / "web.py"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_web_does_not_call_tts_stop():
    src = _SRC_WEB.read_text(encoding="utf-8")
    assert "tts.stop()" not in src, "tts.stop() fue eliminado; no debe volver a llamarse"


def test_requirements_drops_librosa_and_soundfile():
    reqs = (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "librosa" not in reqs
    assert "soundfile" not in reqs