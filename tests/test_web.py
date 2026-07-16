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


def test_requirements_drops_accelerate():
    # ponytail: con device_map="cpu" no se necesita accelerate.
    reqs = (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "accelerate" not in reqs


def test_web_host_default_localhost():
    # ponytail: web_host default 127.0.0.1, no 0.0.0.0 (expone a LAN).
    from src.config import SETTINGS
    assert SETTINGS.web_host == "127.0.0.1"


def test_ws_enforces_token():
    # ponytail: WS rechaza conexiones sin token cuando CHOKITA_WS_TOKEN está seteado.
    src = _SRC_WEB.read_text(encoding="utf-8")
    assert "CHOKITA_WS_TOKEN" in src or "ws_token" in src
    assert "websocket.close(code=1008" in src


def test_audio_utterance_capped():
    # ponytail: audio_utterance > 5MB se descarta (DoS por OOM).
    src = _SRC_WEB.read_text(encoding="utf-8")
    assert "5_000_000" in src or "5000000" in src


def test_xss_thinking_uses_textcontent():
    # ponytail: thinking_chunk usa textContent, no innerHTML (XSS via RAG).
    html = (_REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "thoughtText.textContent" in html
    # Buscar la asignacion real (no en comentario). Si la linea tiene '=' con innerHTML, falla.
    for line in html.splitlines():
        if "thoughtText" in line and "=" in line and "innerHTML" in line and "// ponytail" not in line:
            raise AssertionError(f"innerHTML en asignacion real: {line}")