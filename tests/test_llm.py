from unittest.mock import Mock

from src.llm import OllamaClient


class DummyResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"message": {"content": "respuesta"}}


def _make_client() -> OllamaClient:
    memory = Mock()
    memory.recent_messages.return_value = []
    memory.retrieve.return_value = []
    memory.add_message.return_value = 1
    # seed_soul must not blow up during __init__
    memory.seed_soul = Mock()
    client = OllamaClient(memory=memory, retries=0)
    session = Mock()
    session.post.return_value = DummyResponse()
    client._session = session
    return client, memory


def test_ollama_client_parses_response() -> None:
    client, memory = _make_client()
    out = client.chat("hola")
    assert out == "respuesta"
    # user message + assistant reply persisted
    roles = [c.args[0] for c in memory.add_message.call_args_list]
    assert "user" in roles and "assistant" in roles


def test_ollama_client_executes_tool_call() -> None:
    client, memory = _make_client()
    # first reply has a tool call, second is plain
    r1 = DummyResponse()
    r1.json = lambda: {"message": {"content": 'Veo... <tool>{"name":"list","args":{"path":"."}}</tool>'}}
    r2 = DummyResponse()
    r2.json = lambda: {"message": {"content": "listo"}}
    session = Mock()
    session.post.side_effect = [r1, r2]
    client._session = session
    out = client.chat("que hay aca?")
    assert out == "listo"
    # at least one tool message persisted
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