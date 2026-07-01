from unittest.mock import patch

import pytest

from src.memory import _kmeans_cosine


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("CHOKITA_DB_PATH", str(tmp_path / "raptor.db"))
    monkeypatch.setenv("OLLAMA_EMBED_DIM", "4")
    monkeypatch.setenv("RAPTOR_CLUSTER_K", "3")
    monkeypatch.setenv("RAPTOR_MAX_LEVELS", "3")
    import importlib

    import src.config
    importlib.reload(src.config)
    import src.memory as mem_mod
    importlib.reload(mem_mod)
    m = mem_mod.Memory()
    m.start_session()
    yield m
    m.close()


def _fake_embed(text):
    return [float(len(text) % 7), float(len(text) // 3 % 5),
            float(len(text) // 7 % 3), float(len(text) % 11)]


def _fake_summarize(text):
    return "resumen: " + text[:40]


def test_kmeans_cosine_partitions():
    # Two well-separated groups in 4D so k-means splits them.
    items = []
    for i in range(3):
        items.append({"id": i, "embedding": [1.0, 0.0, 0.0, 0.0]})
    for i in range(3):
        items.append({"id": i + 3, "embedding": [0.0, 1.0, 0.0, 0.0]})
    clusters = _kmeans_cosine(items, 2, iters=10)
    assert len(clusters) == 2
    total = sum(len(c) for c in clusters)
    assert total == 6


def test_raptor_build_and_retrieve(mem):
    with patch.object(mem, "embed", _fake_embed):
        for i in range(10):
            mem.add_chunk("reflection", "note", f"chunk {i} sobre tema {i % 3}")
        log = mem.build_raptor(_fake_summarize)
        assert any("Nivel 0" in line for line in log)
        stats = mem.raptor_stats()
        assert stats[0] == 10  # 10 leaves
        assert 1 in stats  # at least one summary level
        r = mem.retrieve_raptor("tema 1", top_k=3)
        assert len(r) == 3
        assert all("level" in x for x in r)


def test_raptor_clear(mem):
    with patch.object(mem, "embed", _fake_embed):
        mem.add_chunk("reflection", "note", "algo")
        mem.build_raptor(_fake_summarize)
        assert mem.raptor_stats()
        mem.clear_raptor()
        assert not mem.raptor_stats()