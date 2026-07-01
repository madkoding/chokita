import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.memory import Memory, _cosine, _split_sections


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("CHOKITA_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OLLAMA_EMBED_DIM", "4")
    # Reload settings so the env vars take effect.
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
    # deterministic pseudo-embedding from text length
    base = len(text) % 4
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