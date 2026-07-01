import os

from src.config import SETTINGS


def test_defaults():
    assert SETTINGS.ollama_base_url == "http://localhost:11434"
    assert SETTINGS.ollama_model == "ornith:9b"
    assert SETTINGS.rag_top_k == 6
    assert SETTINGS.history_window == 20
    assert SETTINGS.max_tool_iterations == 5
    assert SETTINGS.raptor_cluster_k == 8
    assert SETTINGS.raptor_max_levels == 4
    assert SETTINGS.context_window_tokens == 262144
    assert SETTINGS.compact_threshold == 0.80
    assert SETTINGS.chars_per_token == 4
    assert SETTINGS.tts_fallback_stdout is True
    assert SETTINGS.playback_command == "auto"


def test_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("TTS_FALLBACK_STDOUT", "0")
    monkeypatch.setenv("PLAYBACK_COMMAND", "ffplay")
    import importlib
    import src.config
    importlib.reload(src.config)
    s = src.config.SETTINGS
    assert s.ollama_base_url == "http://ollama:11434"
    assert s.ollama_model == "llama3"
    assert s.rag_top_k == 3
    assert s.tts_fallback_stdout is False
    assert s.playback_command == "ffplay"
