from src.config import KAOMOJIS


def test_kaomoji_states_present() -> None:
    expected = {"IDLE", "LISTENING", "RECOGNIZED", "THINKING", "SPEAKING", "ERROR"}
    assert expected.issubset(set(KAOMOJIS.keys()))
