"""Ollama client with memory, RAG soul retrieval, and a tool-use loop.

Flow per user turn:
  1. Build system prompt: SOUL.md + RAG soul chunks + tools doc + NDJSON format.
  2. Append history (window) + new user message.
  3. Ask the model. Parse NDJSON lines (thinking/feeling/tool_call/response/memory).
  4. If tool_call: execute tool, append result, loop.
  5. After max iterations or a normal reply, persist everything to memory.
"""

from __future__ import annotations

import functools
import html
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.config import SETTINGS
from src.memory import Memory
from src.tools import TOOLS_DOC, call_tool

LOGGER = logging.getLogger(__name__)

SOUL_PATH = Path(__file__).resolve().parent.parent / "SOUL.md"

NDJSON_FORMAT_HINT = (
    "Respondé en líneas NDJSON. Una línea = un JSON con al menos \"type\".\n"
    '  {"type":"thinking","content":"..."} — tu razonamiento interno\n'
    '  {"type":"response","content":"..."} — tu respuesta al usuario\n'
    '  {"type":"feeling","feeling":"curious"} — expresión emocional (opcional)\n'
    '  {"type":"tool_call","name":"read","args":{"path":"..."}} — invocar tool\n'
    '  {"type":"memory","content":"..."} — persistir un hecho a largo plazo (opcional)\n'
    "Regla: response es para el usuario. tool_call ejecuta la tool. Sin tools, solo response."
)


def parse_model_line(line: str) -> dict | None:
    """Parse a single NDJSON line from the model. Returns normalized dict or None (ignore).
    Tolerante: si falta 'type' o es desconocido pero hay 'content', trata como response.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return {"type": "response", "content": stripped}

    if not isinstance(obj, dict):
        return {"type": "response", "content": stripped}

    t = obj.get("type", "")
    if t == "thinking":
        content = obj.get("content", "")
        return {"type": "thinking", "content": content} if content else None
    if t == "feeling":
        feeling = obj.get("feeling", "")
        return {"type": "feeling", "feeling": feeling} if feeling else None
    if t == "tool_call":
        name = obj.get("name", "")
        args = obj.get("args", {}) or {}
        return {"type": "tool_call", "name": name, "args": args} if name else None
    if t == "response":
        content = obj.get("content", "")
        return {"type": "response", "content": content}
    if t == "memory":
        content = obj.get("content", "")
        return {"type": "memory", "content": content} if content else None
    # type faltante o desconocido: si hay content, tratar como response
    content = obj.get("content", "")
    if content:
        return {"type": "response", "content": content}
    return None


@functools.lru_cache(maxsize=1)
def load_soul_text() -> str:
    try:
        return SOUL_PATH.read_text(encoding="utf-8")
    except Exception:
        LOGGER.warning("No se pudo cargar SOUL.md")
        return ""


class AbortedError(Exception):
    pass


class OllamaClient:
    """Ollama /api/chat client with memory + RAG + tools + context compaction."""

    def __init__(
        self,
        memory: Memory,
        ui_queue: queue.Queue[dict[str, Any]] | None = None,
        retries: int = 2,
        retry_delay_seconds: float = 0.8,
        abort_event: threading.Event | None = None,
    ) -> None:
        self.memory = memory
        self.ui_queue = ui_queue
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._abort_event = abort_event
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

    def _emit_with_reply(self, messages: list[dict[str, str]], reply: str) -> None:
        self._emit_tokens(messages + [{"role": "assistant", "content": reply}])

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
            summary, _ = self._raw_chat([
                {"role": "system", "content": "Resumí la siguiente conversación en bullets concisos, en español rioplatense. Mantener hechos clave y decisiones."},
                {"role": "user", "content": joined},
            ])
            summary = summary or "Conversacion previa resumida."
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

        # ponytail: tool_calls local por iteracion. evita race con soul/sleep que
        # comparten la misma instancia de OllamaClient.
        for _iter in range(SETTINGS.max_tool_iterations):
            if self._abort_event and self._abort_event.is_set():
                return SETTINGS.ollama_fallback_message
            tool_calls: list[dict[str, Any]] = []
            reply, tool_calls = self._raw_chat(messages, stream=_iter == 0)
            if reply is None:
                return SETTINGS.ollama_fallback_message

            if not tool_calls:
                self.memory.add_message("assistant", reply)
                self._emit_with_reply(messages, reply)
                return reply

            messages.append({"role": "assistant", "content": reply})
            self.memory.add_message("assistant", reply)

            for tc in tool_calls:
                name = tc["name"]
                args = tc["args"]
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
        reply, _ = self._raw_chat(messages)
        if reply is None:
            return SETTINGS.ollama_fallback_message
        self.memory.add_message("assistant", reply)
        self._emit_with_reply(messages, reply)
        return reply

    def chat_raw(self, messages: list[dict[str, str]]) -> str:
        try:
            reply, _ = self._raw_chat(messages)
        except AbortedError:
            return ""
        return reply or ""

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
            raw, _ = self._raw_chat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": convo},
            ])
            raw = raw or ""
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
                # ponytail: html.escape previene XSS si el chunk llega al UI como HTML.
                self.memory.add_chunk("memory", "episode", html.escape(b))
                count += 1
            except Exception:
                LOGGER.warning("Failed to store a memory chunk")
        if count:
            LOGGER.info("Extracted %d long-term memories from session", count)
            self.memory.prune_chunks()
            self.memory.checkpoint()
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
        parts.append(TOOLS_DOC)
        parts.append(NDJSON_FORMAT_HINT)
        return "\n".join(parts)

    def _emit_ndjson_event(self, parsed: dict, *, partial: bool) -> None:
        if not self.ui_queue:
            return
        t = parsed["type"]
        if t == "thinking":
            if partial:
                self.ui_queue.put({"type": "thinking_chunk", "content": parsed["content"]})
            else:
                self.ui_queue.put({"type": "thinking", "content": parsed["content"]})
        elif t == "feeling":
            self.ui_queue.put({"type": "feeling", "feeling": parsed["feeling"]})
        elif t == "tool_call":
            self.ui_queue.put({"type": "tool_call", "name": parsed["name"], "args": parsed["args"]})
        elif t == "response":
            if partial:
                self.ui_queue.put({"type": "response_chunk", "content": parsed["content"]})
            else:
                self.ui_queue.put({"type": "response", "content": parsed["content"]})
        elif t == "memory":
            self.ui_queue.put({"type": "log", "message": f"🧠 {parsed['content'][:60]}"})

    def _process_ndjson_line(self, parsed: dict, tool_calls: list[dict[str, Any]], *, emit: bool) -> str | None:
        """Return response content if type==response, None otherwise. Side-effects: tool_calls, memory."""
        # ponytail: tool_calls por param (no atributo de instancia) evita race con soul/sleep.
        if parsed["type"] == "response":
            return parsed["content"]
        if parsed["type"] == "tool_call":
            tool_calls.append(parsed)
            return None
        if parsed["type"] == "memory":
            try:
                # ponytail: html.escape previene XSS si el chunk llega al UI como HTML.
                self.memory.add_chunk("memory", "episode", html.escape(parsed["content"]))
            except Exception:
                LOGGER.warning("Failed to store inline memory")
            return None
        if emit:
            self._emit_ndjson_event(parsed, partial=True)
        return None

    def _raw_chat(self, messages: list[dict[str, str]], stream: bool = False) -> tuple[str | None, list[dict[str, Any]]]:
        url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": SETTINGS.ollama_model,
            "stream": stream,
            "keep_alive": SETTINGS.ollama_keep_alive,
            "messages": messages,
        }
        data = json.dumps(payload).encode()
        for attempt in range(self.retries + 1):
            # ponytail: chequea abort entre retries. cancela el LLM call si el user pidió stop.
            if self._abort_event and self._abort_event.is_set():
                raise AbortedError
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                resp = urllib.request.urlopen(req, timeout=SETTINGS.ollama_timeout_seconds)
                if not stream:
                    body: dict[str, Any] = json.loads(resp.read())
                    text = body.get("message", {}).get("content", "").strip()
                    text, tool_calls = self._parse_ndjson_text(text)
                else:
                    text, tool_calls = self._stream_ndjson(resp)
                if text is not None:
                    return text, tool_calls
                LOGGER.warning("Ollama returned empty content (attempt %d/%d)", attempt + 1, self.retries + 1)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                LOGGER.error("Ollama error: %s", exc)
            except AbortedError:
                raise
            except Exception as exc:
                LOGGER.exception("Unexpected Ollama error: %s", exc)

            if attempt < self.retries:
                time.sleep(self.retry_delay_seconds)
        return None, []

    def _parse_ndjson_text(self, text: str) -> tuple[str | None, list[dict[str, Any]]]:
        """Parse a complete response as NDJSON. Returns (concatenated text, tool_calls)."""
        parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            parsed = parse_model_line(raw_line)
            if parsed is None:
                continue
            out = self._process_ndjson_line(parsed, tool_calls, emit=False)
            if out is not None:
                parts.append(out)
        joined = "\n".join(parts).strip()
        return (joined or None), tool_calls

    def _stream_ndjson(self, resp) -> tuple[str | None, list[dict[str, Any]]]:
        """Stream Ollama: accumulate tokens into complete NDJSON lines, parse each, emit to UI.
        # ponytail: accumulate until newline, then json.loads the full line. No partial-JSON guessing.
        """
        parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        line_buf = ""
        for ollama_line in resp:
            raw = ollama_line.decode().strip()
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if not token:
                if chunk.get("done"):
                    break
                continue
            line_buf += token
            # Each NDJSON line is delimited by \n. Process complete lines as they arrive.
            while "\n" in line_buf:
                complete, line_buf = line_buf.split("\n", 1)
                parsed = parse_model_line(complete)
                if parsed is None:
                    continue
                out = self._process_ndjson_line(parsed, tool_calls, emit=True)
                if out is not None:
                    parts.append(out)
        # Remaining buffer as response fallback
        leftover = line_buf.strip()
        if leftover:
            parsed = parse_model_line(leftover)
            if parsed and parsed["type"] == "response":
                parts.append(parsed["content"])
                self._emit_ndjson_event(parsed, partial=False)
            elif parsed and parsed["type"] == "tool_call":
                tool_calls.append(parsed)
                self._emit_ndjson_event(parsed, partial=False)
            else:
                parts.append(leftover)
                if self.ui_queue:
                    self.ui_queue.put({"type": "response", "content": leftover, "partial": False})
        joined = "\n".join(parts).strip()
        return (joined or None), tool_calls