"""Ollama client with retries and robust HTTP exception handling."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.config import SETTINGS

LOGGER = logging.getLogger(__name__)


class OllamaClient:
    """Simple Ollama /api/chat client."""

    def __init__(self, retries: int = 2, retry_delay_seconds: float = 0.8) -> None:
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._session = requests.Session()

    def chat(self, user_message: str) -> str:
        url = f"{SETTINGS.ollama_base_url.rstrip('/')}{SETTINGS.ollama_chat_path}"
        payload = {
            "model": SETTINGS.ollama_model,
            "stream": False,
            "messages": [{"role": "user", "content": user_message}],
        }

        for attempt in range(self.retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=SETTINGS.ollama_timeout_seconds)
                response.raise_for_status()
                body: dict[str, Any] = response.json()
                msg = body.get("message", {})
                text = msg.get("content", "").strip()
                if not text:
                    raise ValueError("Empty Ollama response")
                return text
            except requests.Timeout as exc:
                LOGGER.error("Ollama timeout: %s", exc)
            except requests.ConnectionError as exc:
                LOGGER.error("Ollama connection error: %s", exc)
            except requests.HTTPError as exc:
                LOGGER.error("Ollama HTTP error: %s", exc)
            except Exception as exc:
                LOGGER.exception("Unexpected Ollama error: %s", exc)

            if attempt < self.retries:
                time.sleep(self.retry_delay_seconds)

        return "No pude contactar al modelo local en este momento."
