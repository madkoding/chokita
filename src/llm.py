"""Ollama client with memory, RAG soul retrieval, and a tool-use loop.

Flow per user turn:
  1. Build system prompt: SOUL.md + RAG soul chunks + tools doc.
  2. Append history (window) + new user message.
  3. Ask the model. If it emits a <tool> tag, execute it, append result, loop.
  4. After max iterations or a normal reply, persist everything to memory.
"""

from __future__ import annotations

import functools
import json
import logging
import queue
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.config import SETTINGS
from src.memory import Memory
from src.tools import call_tool, tools_system_doc

LOGGER = logging.getLogger(__name__)

SOUL_PATH = Path(__file__).resolve().parent.parent / "SOUL.md"

_TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)
TOOL_FORMAT_HINT = (
    'Para usar una tool respondé con EXACTAMENTE: <tool>{"name":"read","args":{"path":"src/main.py"}}</tool>\n'
    "y esperá el resultado antes de seguir. Si no necesitás tools, respondé normalmente."
)


@functools.lru_cache(maxsize=1)
def load_soul_text() -> str:
    try:
        return SOUL_PATH.read_text(encoding="utf-8")
    except Exception:
        LOGGER.warning("No se pudo cargar SOUL.md")
        return ""


class OllamaClient:
    """Ollama /api/chat client with memory + RAG + tools + context compaction."""

    def __init__(
        self,
        memory: Memory,
        ui_queue: queue.Queue[dict[str, Any]] | None = None,
        retries: int = 2,
        retry_delay_seconds: float = 0.8,
    ) -> None:
        self.memory = memory
        self.ui_queue = ui_queue
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._soul = load_soul_text()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """ponytail: 1 token ≈ 4 chars. No tiktoken, good enough for a budget bar."""
        return max(1, len(text) // 4)

    def _total_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(self._estimate_tokens(m.get("content", "")) for m in messages)

    def _emit_tokens(self, messages: list[dict[str, str]]) -> None:
        if not self.ui_queue:
            return
        used = self._total_tokens(messages)
        pct = (used / SETTINGS.context_window_tokens) * 100
        self.ui_queue.put({
            "type": "tokens",
            "used": used,
            "total": SETTINGS.context_window_tokens,
            "pct": min(pct, 100.0),
        })

    def _maybe_compact(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """If token usage exceeds threshold, summarize old history and compact."""
        used = self._total_tokens(messages)
        if used < SETTINGS.context_window_tokens * SETTINGS.compact_threshold:
            return messages
        LOGGER.warning("Contexto al %d%% (%d tokens), compactando...", int(used / SETTINGS.context_window_tokens * 100), used)
        if self.ui_queue:
            self.ui_queue.put({"type": "log", "message": "📦 Compactando contexto..."})
        # Summarize all but the system prompt and last 4 messages.
        to_summarize = [m for m in messages[1:-4]] if len(messages) > 5 else []
        if not to_summarize:
            return messages
        joined = "\n".join(f"[{m['role']}]: {m['content'][:300]}" for m in to_summarize)
        try:
            summary = self._raw_chat([
                {"role": "system", "content": "Resumí la siguiente conversación en bullets concisos, en español rioplatense. Mantener hechos clave y decisiones."},
                {"role": "user", "content": joined},
            ]) or "Conversacion previa resumida."
        except Exception:
            summary = "Conversacion previa resumida."
        # Persist the compacted history.
        self.memory.compact_history(summary)
        # Rebuild messages: system + summary + last 4.
        new_messages: list[dict[str, str]] = [messages[0]]
        new_messages.append({"role": "system", "content": f"Resumen de conversacion previa:\n{summary}"})
        new_messages.extend(messages[-4:])
        if self.ui_queue:
            self.ui_queue.put({"type": "log", "message": f"📦 Contexto compactado: {len(to_summarize)} mensajes -> 1 resumen."})
        return new_messages

    def chat(self, user_message: str) -> str:
        system = self._build_system_prompt(user_message)
        history = self.memory.recent_messages()
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        self.memory.add_message("user", user_message)

        messages = self._maybe_compact(messages)
        self._emit_tokens(messages)

        for _iter in range(SETTINGS.max_tool_iterations):
            reply = self._raw_chat(messages, stream=_iter == 0)
            if reply is None:
                return SETTINGS.ollama_fallback_message

            tool_calls = _TOOL_RE.findall(reply)
            if not tool_calls:
                self.memory.add_message("assistant", reply)
                self._emit_tokens(messages + [{"role": "assistant", "content": reply}])
                return reply

            messages.append({"role": "assistant", "content": reply})
            self.memory.add_message("assistant", reply)

            for tc in tool_calls:
                try:
                    call = json.loads(tc)
                except json.JSONDecodeError:
                    LOGGER.warning("Tool call JSON invalida: %s", tc)
                    continue
                name = call.get("name", "")
                args = call.get("args", {}) or call.get("arguments", {})
                result = call_tool(name, args)
                self.memory.add_message("tool", result, tool_name=name, tool_args=args)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps({"name": name, "result": result}, ensure_ascii=False),
                    }
                )

            self._emit_tokens(messages)

        messages.append(
            {
                "role": "user",
                "content": "Alcanzaste el limite de tools. Respondé en texto plano sin mas tools.",
            }
        )
        reply = self._raw_chat(messages)
        if reply is None:
            return SETTINGS.ollama_fallback_message
        self.memory.add_message("assistant", reply)
        self._emit_tokens(messages + [{"role": "assistant", "content": reply}])
        return reply

    def chat_raw(self, messages: list[dict[str, str]]) -> str:
        return self._raw_chat(messages) or ""

    def extract_memories(self) -> int:
        """Extract long-term memories from the current session and store them as RAG chunks.
        Returns the number of memories extracted.
        # ponytail: one LLM call that outputs bullet lines; each bullet becomes a chunk.
        """
        messages = self.memory.all_session_messages()
        if len(messages) < 4:
            return 0
        # Only consider user + assistant messages, skip tool noise.
        convo = "\n".join(
            f"[{m['role']}]: {m['content'][:300]}"
            for m in messages
            if m["role"] in ("user", "assistant")
        )
        if not convo.strip():
            return 0
        prompt = (
            "Extraé de la siguiente conversación los hechos, preferencias, decisiones "
            "y datos clave que valga la pena recordar a largo plazo sobre el usuario "
            "o sobre vos misma. Una memoria por linea, prefijada con '- '. "
            "Solo hechos concretos, no saludos ni relleno. Si no hay nada memorable, "
            "respondé una linea vacia."
        )
        try:
            raw = self._raw_chat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": convo},
            ]) or ""
        except Exception:
            LOGGER.warning("Memory extraction LLM call failed")
            return 0
        if not raw.strip():
            return 0
        bullets = [line.strip().lstrip("- ").strip() for line in raw.splitlines() if line.strip().startswith("-")]
        if not bullets:
            # Model didn't use bullets; treat each non-empty line as a memory.
            bullets = [line.strip() for line in raw.splitlines() if line.strip()]
        count = 0
        for b in bullets:
            if not b or len(b) < 5:
                continue
            try:
                self.memory.add_chunk("memory", "episode", b)
                count += 1
            except Exception:
                LOGGER.warning("Failed to store a memory chunk")
        if count:
            LOGGER.info("Extracted %d long-term memories from session", count)
        return count

    def _build_system_prompt(self, user_message: str) -> str:
        parts = [self._soul] if self._soul else []
        # RAG: retrieve soul + reflection chunks relevant to the user message.
        try:
            chunks = self.memory.retrieve(user_message, top_k=SETTINGS.rag_top_k)
            if chunks:
                parts.append("\n\n## Memoria relevante (RAG)")
                for c in chunks:
                    parts.append(f"[{c['source']}/{c['kind']}]: {c['text']}")
        except Exception:
            LOGGER.debug("RAG retrieve failed (embed model?)")

        try:
            raptor_chunks = self.memory.retrieve_raptor(user_message, top_k=3)
            if raptor_chunks:
                parts.append("\n\n## Contexto agregado (RAPTOR)")
                for c in raptor_chunks:
                    parts.append(f"[L{c['level']}]: {c['text']}")
        except Exception:
            LOGGER.debug("RAPTOR retrieve failed (tree empty or embed model?)")

        parts.append("\n\n## Tools disponibles")
        parts.append(tools_system_doc())
        parts.append(TOOL_FORMAT_HINT)
        return "\n".join(parts)

    def _raw_chat(self, messages: list[dict[str, str]], stream: bool = False) -> str | None:
        url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": SETTINGS.ollama_model,
            "stream": stream,
            "keep_alive": SETTINGS.ollama_keep_alive,
            "messages": messages,
        }
        data = json.dumps(payload).encode()
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                resp = urllib.request.urlopen(req, timeout=SETTINGS.ollama_timeout_seconds)
                if not stream:
                    body: dict[str, Any] = json.loads(resp.read())
                    text = body.get("message", {}).get("content", "").strip()
                    return text or None
                full: list[str] = []
                buf = ""
                for line in resp:
                    line = line.decode().strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    full.append(token)
                    buf += token
                    if len(buf) >= 20 or chunk.get("done"):
                        if self.ui_queue:
                            self.ui_queue.put({"type": "token", "content": buf})
                        buf = ""
                    if chunk.get("done"):
                        break
                text = "".join(full).strip()
                if text:
                    return text
                LOGGER.warning("Ollama returned empty content (attempt %d/%d)", attempt + 1, self.retries + 1)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                LOGGER.error("Ollama error: %s", exc)
            except Exception as exc:
                LOGGER.exception("Unexpected Ollama error: %s", exc)

            if attempt < self.retries:
                time.sleep(self.retry_delay_seconds)
        return None