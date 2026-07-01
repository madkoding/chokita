import json
from unittest.mock import Mock, patch

from src.llm import OllamaClient


class DummyResponse:
    def __init__(self, content: str = "respuesta"):
        self._content = content

    def read(self) -> bytes:
        return json.dumps({"message": {"content": self._content}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_client() -> OllamaClient:
    memory = Mock()
    memory.recent_messages.return_value = []
    memory.retrieve.return_value = []
    memory.add_message.return_value = 1
    memory.seed_soul = Mock()
    client = OllamaClient(memory=memory, retries=0)
    return client, memory


@patch("urllib.request.urlopen", return_value=DummyResponse("respuesta"))
def test_ollama_client_parses_response(_mock_urlopen) -> None:
    client, memory = _make_client()
    out = client.chat("hola")
    assert out == "respuesta"
    roles = [c.args[0] for c in memory.add_message.call_args_list]
    assert "user" in roles and "assistant" in roles


@patch("urllib.request.urlopen")
def test_ollama_client_executes_tool_call(mock_urlopen) -> None:
    client, memory = _make_client()
    r1 = DummyResponse('Veo... <tool>{"name":"list","args":{"path":"."}}</tool>')
    r2 = DummyResponse("listo")
    mock_urlopen.side_effect = [r1, r2]
    out = client.chat("que hay aca?")
    assert out == "listo"
    tool_calls = [c for c in memory.add_message.call_args_list if c.kwargs.get("tool_name")]
    assert tool_calls, "esperaba al menos un mensaje de tool"


def test_token_estimation():
    assert OllamaClient._estimate_tokens("hola") >= 1
    assert OllamaClient._estimate_tokens("x" * 400) == 100


def test_emit_tokens():
    import queue as q
    ui = q.Queue()
    memory = Mock()
    memory.recent_messages.return_value = []
    memory.retrieve.return_value = []
    memory.seed_soul = Mock()
    client = OllamaClient(memory=memory, ui_queue=ui, retries=0)
    client._emit_tokens([{"role": "system", "content": "x" * 800}])
    event = ui.get_nowait()
    assert event["type"] == "tokens"
    assert event["used"] == 200
    assert event["total"] > 0
