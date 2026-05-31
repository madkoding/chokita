from unittest.mock import Mock

from src.llm import OllamaClient


class DummyResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"message": {"content": "respuesta"}}


def test_ollama_client_parses_response() -> None:
    client = OllamaClient(retries=0)
    session = Mock()
    session.post.return_value = DummyResponse()
    client._session = session

    out = client.chat("hola")
    assert out == "respuesta"
