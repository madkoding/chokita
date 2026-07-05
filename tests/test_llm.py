import json
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from src.llm import OllamaClient, parse_model_line


class DummyResponse:
    def __init__(self, content: str = "respuesta"):
        self._content = content

    def read(self) -> bytes:
        return json.dumps({"message": {"content": self._content}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        yield (json.dumps({"message": {"content": self._content}, "done": True}) + "\n").encode()


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
    r1 = DummyResponse('{"type":"tool_call","name":"list","args":{"path":"."}}\n{"type":"response","content":"veo"}')
    r2 = DummyResponse('{"type":"response","content":"listo"}')
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


class EmptyFirstResponse:
    def read(self) -> bytes:
        return json.dumps({"message": {"content": "ok"}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        yield b"\n"
        yield (json.dumps({"message": {"content": "ok"}, "done": True}) + "\n").encode()


class StreamDummyResponse:
    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    def read(self) -> bytes:
        return json.dumps({"message": {"content": "".join(self._tokens)}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        for token in self._tokens:
            yield (json.dumps({"message": {"content": token}}) + "\n").encode()
        yield (json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode()


@patch("urllib.request.urlopen", return_value=DummyResponse("respuesta"))
def test_chat_raw(mock_urlopen) -> None:
    client, _memory = _make_client()
    out = client.chat_raw([{"role": "user", "content": "hola"}])
    assert out == "respuesta"


@patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail"))
def test_chat_fallback_on_network_error(mock_urlopen) -> None:
    from src.llm import SETTINGS
    client, _memory = _make_client()
    out = client.chat("hola")
    assert out == SETTINGS.ollama_fallback_message


@patch("urllib.request.urlopen")
def test_invalid_tool_call_name_skipped(mock_urlopen) -> None:
    client, memory = _make_client()
    r1 = DummyResponse('{"type":"tool_call","name":"","args":{}}\n{"type":"response","content":"final answer"}')
    mock_urlopen.side_effect = [r1]
    out = client.chat("test")
    assert out == "final answer"


@patch("urllib.request.urlopen")
def test_max_tool_iterations(mock_urlopen) -> None:
    client, memory = _make_client()
    tool_resp = DummyResponse('{"type":"tool_call","name":"list","args":{"path":"."}}\n{"type":"response","content":"ok"}')
    final_resp = DummyResponse('{"type":"response","content":"final answer"}')
    mock_urlopen.side_effect = [tool_resp] * 5 + [final_resp]
    out = client.chat("test")
    assert out == "final answer"


def test_maybe_compact_triggers() -> None:
    import queue as q
    client, memory = _make_client()
    client.ui_queue = q.Queue()
    client.memory.compact_history = Mock(return_value=5)
    big = "x" * 30  # 7 tokens each with chars_per_token=4, 8 msgs = 56 total
    messages = [
        {"role": "system", "content": big},
    ] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": big}
        for i in range(7)
    ]
    with patch("src.llm.SETTINGS") as mock_s:
        mock_s.context_window_tokens = 100
        mock_s.compact_threshold = 0.5
        with patch.object(client, "_raw_chat", return_value="resumen de todo"):
            result = client._maybe_compact(messages)
    assert len(result) < len(messages)


@patch("urllib.request.urlopen")
def test_extract_memories(mock_urlopen) -> None:
    client, memory = _make_client()
    memory.all_session_messages.return_value = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    mock_urlopen.return_value = DummyResponse("- le gusta programar\n- es paciente")
    count = client.extract_memories()
    assert count == 2


@patch("urllib.request.urlopen")
def test_extract_memories_short_conversation(mock_urlopen) -> None:
    client, memory = _make_client()
    memory.all_session_messages.return_value = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    count = client.extract_memories()
    assert count == 0


@patch("urllib.request.urlopen")
def test_extract_memories_no_bullets(mock_urlopen) -> None:
    client, memory = _make_client()
    memory.all_session_messages.return_value = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    mock_urlopen.return_value = DummyResponse("linea suelta\notra linea")
    count = client.extract_memories()
    assert count == 2


def test_build_system_prompt_rag_error() -> None:
    client, memory = _make_client()
    memory.retrieve.side_effect = RuntimeError("embed fail")
    prompt = client._build_system_prompt("hola")
    assert "## Memoria relevante" not in prompt
    assert "## Tools disponibles" in prompt


def test_build_system_prompt_with_raptor() -> None:
    client, memory = _make_client()
    memory.retrieve.return_value = [{"source": "soul", "kind": "soul", "text": "nucleo"}]
    memory.retrieve_raptor.return_value = [{"level": 1, "text": "raptor summary"}]
    prompt = client._build_system_prompt("hola")
    assert "## Contexto agregado (RAPTOR)" in prompt
    assert "raptor summary" in prompt


@patch("urllib.request.urlopen")
def test_empty_stream_line_skipped(mock_urlopen) -> None:
    client, memory = _make_client()
    mock_urlopen.return_value = EmptyFirstResponse()
    out = client.chat("hola")
    assert out == "ok"


@patch("urllib.request.urlopen")
def test_stream_response_emit_to_ui_queue(mock_urlopen) -> None:
    import queue as q
    ui = q.Queue()
    memory = Mock()
    memory.recent_messages.return_value = []
    memory.retrieve.return_value = []
    memory.seed_soul = Mock()
    memory.add_message.return_value = 1
    client = OllamaClient(memory=memory, ui_queue=ui, retries=0)
    mock_urlopen.return_value = StreamDummyResponse(["hola", " ", "mundo"])
    out = client.chat("test")
    assert out == "hola mundo"
    events = []
    while not ui.empty():
        events.append(ui.get_nowait())
    assert any(e["type"] == "response" for e in events)


@patch("urllib.request.urlopen")
def test_raw_chat_retry_on_url_error(mock_urlopen) -> None:
    client, memory = _make_client()
    client.retries = 1
    mock_urlopen.side_effect = [
        urllib.error.URLError("timeout"),
        DummyResponse("ok after retry"),
    ]
    out = client.chat("hola")
    assert out == "ok after retry"


def test_load_soul_text_fallback() -> None:
    from src.llm import load_soul_text
    load_soul_text.cache_clear()
    mock_path = Mock(spec=Path)
    mock_path.read_text.side_effect = OSError("no file")
    with patch("src.llm.SOUL_PATH", mock_path):
        text = load_soul_text()
    assert text == ""


def test_maybe_compact_too_few_messages() -> None:
    client, _memory = _make_client()
    messages = [
        {"role": "system", "content": "x" * 100},
        {"role": "user", "content": "hi"},
    ]
    with patch("src.llm.SETTINGS") as mock_s:
        mock_s.context_window_tokens = 100
        mock_s.compact_threshold = 0.5
        result = client._maybe_compact(messages)
    assert result == messages


def test_maybe_compact_summary_raises() -> None:
    import queue as q
    client, memory = _make_client()
    client.ui_queue = q.Queue()
    client.memory.compact_history = Mock()
    big = "x" * 100
    messages = [
        {"role": "system", "content": big},
    ] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": big}
        for i in range(7)
    ]
    with patch("src.llm.SETTINGS") as mock_s:
        mock_s.context_window_tokens = 100
        mock_s.compact_threshold = 0.5
        with patch.object(client, "_raw_chat", side_effect=RuntimeError("fail")):
            result = client._maybe_compact(messages)
    assert len(result) < len(messages)


@patch("urllib.request.urlopen")
def test_max_tool_iterations_fallback(mock_urlopen) -> None:
    client, memory = _make_client()
    tool_resp = DummyResponse('{"type":"tool_call","name":"list","args":{"path":"."}}\n{"type":"response","content":"ok"}')
    mock_urlopen.side_effect = [tool_resp] * 5 + [DummyResponse("")]
    out = client.chat("test")
    from src.config import SETTINGS
    assert out == SETTINGS.ollama_fallback_message


@patch("urllib.request.urlopen")
def test_extract_memories_short_bullets(mock_urlopen) -> None:
    client, memory = _make_client()
    memory.all_session_messages.return_value = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    mock_urlopen.return_value = DummyResponse("- a\n- bc\n- defgh")
    count = client.extract_memories()
    assert count == 1


@patch("urllib.request.urlopen")
def test_extract_memories_empty_raw(mock_urlopen) -> None:
    client, memory = _make_client()
    memory.all_session_messages.return_value = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    mock_urlopen.return_value = DummyResponse("")
    count = client.extract_memories()
    assert count == 0


@patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail"))
def test_extract_memories_chat_raises(mock_urlopen) -> None:
    client, memory = _make_client()
    memory.all_session_messages.return_value = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    count = client.extract_memories()
    assert count == 0


@patch("urllib.request.urlopen")
def test_extract_memories_add_chunk_fails(mock_urlopen) -> None:
    client, memory = _make_client()
    memory.all_session_messages.return_value = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    mock_urlopen.return_value = DummyResponse("- memoria guardar")
    memory.add_chunk.side_effect = RuntimeError("db fail")
    count = client.extract_memories()
    assert count == 0


class EmptyStreamResponse:
    def read(self) -> bytes:
        return json.dumps({"message": {"content": ""}}).encode()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __iter__(self):
        yield (json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode()


@patch("urllib.request.urlopen")
def test_raw_chat_stream_empty_then_retry(mock_urlopen) -> None:
    client, memory = _make_client()
    client.retries = 1
    mock_urlopen.side_effect = [EmptyStreamResponse(), DummyResponse("ok after empty")]
    out = client.chat("hola")
    assert out == "ok after empty"


@patch("urllib.request.urlopen", side_effect=ValueError("strange"))
def test_raw_chat_generic_exception(mock_urlopen) -> None:
    client, memory = _make_client()
    out = client.chat("hola")
    from src.config import SETTINGS
    assert out == SETTINGS.ollama_fallback_message


@patch("urllib.request.urlopen")
def test_extract_memories_only_tools(mock_urlopen) -> None:
    client, memory = _make_client()
    memory.all_session_messages.return_value = [
        {"role": "tool", "content": "x"},
        {"role": "tool", "content": "x"},
        {"role": "tool", "content": "x"},
        {"role": "tool", "content": "x"},
    ]
    count = client.extract_memories()
    assert count == 0


def test_parse_model_line_thinking() -> None:
    r = parse_model_line('{"type":"thinking","content":"pienso..."}')
    assert r == {"type": "thinking", "content": "pienso..."}


def test_parse_model_line_response() -> None:
    r = parse_model_line('{"type":"response","content":"hola"}')
    assert r == {"type": "response", "content": "hola"}


def test_parse_model_line_tool_call() -> None:
    r = parse_model_line('{"type":"tool_call","name":"read","args":{"path":"x"}}')
    assert r == {"type": "tool_call", "name": "read", "args": {"path": "x"}}


def test_parse_model_line_feeling() -> None:
    r = parse_model_line('{"type":"feeling","feeling":"curious"}')
    assert r == {"type": "feeling", "feeling": "curious"}


def test_parse_model_line_memory() -> None:
    r = parse_model_line('{"type":"memory","content":"recordar esto"}')
    assert r == {"type": "memory", "content": "recordar esto"}


def test_parse_model_line_plain_text_fallback() -> None:
    r = parse_model_line("hola mundo")
    assert r == {"type": "response", "content": "hola mundo"}


def test_parse_model_line_empty() -> None:
    assert parse_model_line("") is None
    assert parse_model_line("   ") is None


def test_parse_model_line_unknown_type() -> None:
    r = parse_model_line('{"type":"unknown","content":"x"}')
    assert r is None


def test_parse_model_line_no_type() -> None:
    r = parse_model_line('{"hello":"world"}')
    assert r is None


def test_parse_model_line_empty_after_json_parse() -> None:
    assert parse_model_line('{"type":"thinking","content":""}') is None
    assert parse_model_line('{"type":"tool_call","name":"","args":{}}') is None


@patch("urllib.request.urlopen")
def test_ndjson_parse_stream_response_only(mock_urlopen) -> None:
    client, memory = _make_client()
    mock_urlopen.return_value = DummyResponse('{"type":"response","content":"hola mundo"}')
    out = client.chat("test")
    assert out == "hola mundo"


@patch("urllib.request.urlopen")
def test_ndjson_parse_stream_thinking_and_response(mock_urlopen) -> None:
    import queue as q
    ui = q.Queue()
    memory = Mock()
    memory.recent_messages.return_value = []
    memory.retrieve.return_value = []
    memory.seed_soul = Mock()
    memory.add_message.return_value = 1
    client = OllamaClient(memory=memory, ui_queue=ui, retries=0)
    mock_urlopen.return_value = DummyResponse(
        '{"type":"thinking","content":"analizando..."}\n{"type":"response","content":"la respuesta"}'
    )
    out = client.chat("test")
    assert out == "la respuesta"
    events = []
    while not ui.empty():
        events.append(ui.get_nowait())
    thinking_events = [e for e in events if e["type"] == "thinking"]
    assert len(thinking_events) > 0


@patch("urllib.request.urlopen")
def test_ndjson_stream_feeling_and_response(mock_urlopen) -> None:
    import queue as q
    ui = q.Queue()
    memory = Mock()
    memory.recent_messages.return_value = []
    memory.retrieve.return_value = []
    memory.seed_soul = Mock()
    memory.add_message.return_value = 1
    client = OllamaClient(memory=memory, ui_queue=ui, retries=0)
    mock_urlopen.return_value = DummyResponse(
        '{"type":"feeling","feeling":"happy"}\n{"type":"response","content":"estoy feliz"}'
    )
    out = client.chat("test")
    assert out == "estoy feliz"
    events = []
    while not ui.empty():
        events.append(ui.get_nowait())
    assert any(e["type"] == "feeling" and e["feeling"] == "happy" for e in events)


@patch("urllib.request.urlopen")
def test_ndjson_stream_memory_persisted(mock_urlopen) -> None:
    client, memory = _make_client()
    mock_urlopen.return_value = DummyResponse(
        '{"type":"memory","content":"recordar esto"}\n{"type":"response","content":"ok"}'
    )
    out = client.chat("test")
    assert out == "ok"
    memory.add_chunk.assert_called_once_with("memory", "episode", "recordar esto")


@patch("urllib.request.urlopen")
def test_ndjson_stream_thinking_toolcall_response(mock_urlopen) -> None:
    import queue as q
    ui = q.Queue()
    memory = Mock()
    memory.recent_messages.return_value = []
    memory.retrieve.return_value = []
    memory.seed_soul = Mock()
    memory.add_message.return_value = 1
    client = OllamaClient(memory=memory, ui_queue=ui, retries=0)
    mock_urlopen.return_value = DummyResponse(
        '{"type":"thinking","content":"debo listar"}\n'
        '{"type":"tool_call","name":"list","args":{"path":"."}}\n'
        '{"type":"response","content":"listando..."}'
    )
    r2 = DummyResponse('{"type":"response","content":"final"}')
    mock_urlopen.side_effect = [mock_urlopen.return_value, r2]
    out = client.chat("test")
    assert out == "final"
    tool_calls = [c for c in memory.add_message.call_args_list if c.kwargs.get("tool_name")]
    assert tool_calls


@patch("urllib.request.urlopen")
def test_ndjson_plain_text_fallback_stream(mock_urlopen) -> None:
    client, memory = _make_client()
    mock_urlopen.return_value = DummyResponse("texto plano")
    out = client.chat("test")
    assert out == "texto plano"
