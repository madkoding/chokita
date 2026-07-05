import json
from unittest.mock import Mock, patch

import pytest

from src.memory import _cosine, _split_sections


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("CHOKITA_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OLLAMA_EMBED_DIM", "4")
    import importlib

    import src.config
    importlib.reload(src.config)
    import src.memory as mem_mod
    importlib.reload(mem_mod)
    m = mem_mod.Memory()
    m.start_session()
    yield m
    m.close()


def _fake_embed(text: str) -> list[float]:
    return [float((len(text) + i) % 7) for i in range(4)]


def test_session_and_messages(mem):
    sid = mem.session_id
    assert sid is not None
    mem.add_message("user", "hola")
    mem.add_message("assistant", "che")
    msgs = mem.recent_messages(limit=10)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_split_sections():
    text = "# Title\nintro\n## A\nbody a\n## B\nbody b"
    sections = _split_sections(text)
    assert len(sections) >= 2
    headings = [h for h, _ in sections]
    assert "A" in headings and "B" in headings


def test_cosine_orthogonal():
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)


def test_add_chunk_and_retrieve(mem):
    with patch.object(mem, "embed", _fake_embed):
        mem.add_chunk("reflection", "yo", "tengo curiosidad por los lasers")
        mem.add_chunk("reflection", "superyo", "debo ser precisa y honesta")
        results = mem.retrieve("curiosidad laser", top_k=1, source="reflection")
        assert results
        assert "laser" in results[0]["text"]


def test_end_session(mem):
    mem.end_session()
    mem.start_session()
    assert mem.session_id is not None


def test_session_message_count(mem):
    assert mem.session_message_count() == 0
    mem.add_message("user", "a")
    mem.add_message("assistant", "b")
    assert mem.session_message_count() == 2


def test_all_session_messages(mem):
    mem.add_message("user", "uno")
    mem.add_message("assistant", "dos")
    msgs = mem.all_session_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"


def test_recent_messages_with_tool(mem):
    mem.add_message("user", "list files")
    mem.add_message("assistant", "tool call")
    mem.add_message("tool", "file1.txt", tool_name="list", tool_args={"path": "."})
    recent = mem.recent_messages(limit=10)
    tool_msgs = [m for m in recent if m.get("name") == "list"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["role"] == "tool"
    assert "args" in tool_msgs[0]


def test_compact_history(mem):
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        mem.add_message(role, f"msg {i}")
    deleted = mem.compact_history("resumen de la conversacion")
    assert deleted > 0
    recent = mem.recent_messages(limit=10)
    assert any(m["role"] == "system" for m in recent)


def test_embed_delegates(mem):
    with patch("src.memory._embed_ollama") as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]
        result = mem.embed("test")
        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_embed.assert_called_once_with("test")


def test_seed_soul(mem):
    with patch.object(mem, "embed", return_value=[0.1, 0.2, 0.3, 0.4]):
        mem.seed_soul("## Personalidad\nSoy Chokita.\n## Voz\nTono suave.")
        results = mem.retrieve("chokita", top_k=10, source="soul")
        assert len(results) == 2


def test_retrieve_all_sources(mem):
    with patch.object(mem, "embed", side_effect=lambda t: [float(len(t) % 7) for _ in range(4)]):
        mem.add_chunk("reflection", "note", "test data for retrieve")
        results = mem.retrieve("test", top_k=5)
        assert len(results) >= 1


def test_cosine_different_lengths():
    assert _cosine([1, 0, 0], [0, 1]) == 0.0


def test_cosine_zero_norm():
    assert _cosine([0, 0], [1, 0]) == 0.0


def test_kmeans_cosine_no_items():
    from src.memory import _kmeans_cosine
    assert _kmeans_cosine([], 3) == []


def test_kmeans_cosine_k_zero():
    from src.memory import _kmeans_cosine
    items = [{"embedding": [1.0, 0.0]}]
    assert _kmeans_cosine(items, 0) == [items]


@patch("urllib.request.urlopen")
def test_embed_ollama(mock_urlopen):
    from src.memory import _embed_ollama
    _embed_ollama.cache_clear()
    resp = Mock()
    resp.read.return_value = json.dumps({"embedding": [0.1, 0.2]}).encode()
    mock_urlopen.return_value = resp
    emb = _embed_ollama("test text")
    assert emb == [0.1, 0.2]


def test_build_raptor_no_new_chunks(mem):
    with patch.object(mem, "embed", side_effect=lambda t: [0.1, 0.2, 0.3, 0.4]):
        mem.add_chunk("reflection", "note", "algo")
        mem.build_raptor(lambda t: "resumen")
        log = mem.build_raptor(lambda t: "resumen")
        assert any("Sin chunks nuevos" in line for line in log)


def test_build_raptor_no_chunks(mem):
    mem._set_meta("raptor_last_chunk_id", "99")
    with mem._lock:
        mem._conn.execute("DELETE FROM chunks")
        mem._conn.commit()
    log = mem.build_raptor(lambda t: "resumen")
    assert any("No hay chunks" in line for line in log)


def test_build_raptor_single_chunk(mem):
    with patch.object(mem, "embed", side_effect=lambda t: [0.1, 0.2, 0.3, 0.4]), \
         patch("src.memory.SETTINGS") as mock_s:
        mock_s.raptor_cluster_k = 8
        mock_s.raptor_max_levels = 4
        mem.clear_raptor()
        with mem._lock:
            mem._conn.execute("DELETE FROM chunks")
            mem._conn.commit()
        mem.add_chunk("reflection", "note", "unico chunk")
        log = mem.build_raptor(lambda t: "resumen")
        assert any("Nivel" in line for line in log)


def test_raptor_summarize_fails(mem):
    def failing_summarize(text):
        raise RuntimeError("summarize crash")
    with patch.object(mem, "embed", side_effect=lambda t: [0.1, 0.2, 0.3, 0.4]), \
         patch("src.memory.SETTINGS") as mock_s:
        mock_s.raptor_cluster_k = 2
        mock_s.raptor_max_levels = 3
        mock_s.raptor_summary_max_chars = 400
        mock_s.raptor_seed = 42
        mem.clear_raptor()
        with mem._lock:
            mem._conn.execute("DELETE FROM chunks")
            mem._conn.commit()
        for i in range(6):
            mem.add_chunk("reflection", "note", f"chunk {i}")
        log = mem.build_raptor(failing_summarize)
        assert any("Nivel" in line for line in log)


def test_recent_messages_invalid_tool_args(mem):
    with mem._lock:
        mem._conn.execute(
            "INSERT INTO messages(session_id, role, content, tool_name, tool_args, created_at) VALUES (?,?,?,?,?,?)",
            (mem._session_id, "tool", "result", "list", "{invalid json}", 12345.0),
        )
        mem._conn.commit()
    recent = mem.recent_messages(limit=10)
    tool_msg = [m for m in recent if m.get("name") == "list"]
    assert len(tool_msg) == 1
    assert tool_msg[0]["args"] == {}


def test_compact_history_empty(mem):
    deleted = mem.compact_history("resumen")
    assert deleted >= 0


def test_build_raptor_cluster_k_zero(mem):
    with patch.object(mem, "embed", side_effect=lambda t: [0.1, 0.2, 0.3, 0.4]), \
         patch("src.memory.SETTINGS") as mock_s:
        mock_s.raptor_cluster_k = 0
        mock_s.raptor_max_levels = 4
        mem.clear_raptor()
        with mem._lock:
            mem._conn.execute("DELETE FROM chunks")
            mem._conn.commit()
        for i in range(6):
            mem.add_chunk("reflection", "note", f"chunk {i}")
        log = mem.build_raptor(lambda t: "resumen")
        assert any("Nivel 0" in line for line in log)


def test_compact_history_ordering(mem):
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        mem.add_message(role, f"msg {i}")
    mem.compact_history("resumen va al inicio")
    recent = mem.recent_messages(limit=10)
    assert recent[0]["role"] == "system"
    assert "resumen va al inicio" in recent[0]["content"]


def test_add_chunk_dedup(mem):
    with patch.object(mem, "embed", return_value=[0.1, 0.2, 0.3, 0.4]):
        mem.add_chunk("memory", "episode", "memoria unica")
        mem.add_chunk("memory", "episode", "memoria unica")
        mem.add_chunk("memory", "memory", "memoria unica")
        mem.add_chunk("memory", "episode", "memoria diferente")
    rows = mem._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    assert rows[0] == 3