"""SQLite + WAL persistence: sessions, messages, embeddings, RAG.

No external deps: sqlite3 (stdlib) + urllib (stdlib) to call Ollama /api/embeddings.
Vector search is brute-force cosine over stored embeddings; fine for a personal agent's scale.
# ponytail: O(n) cosine scan, fine until ~10k chunks; switch to sqlite-vec or FAISS if it grows.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from src.config import SETTINGS

LOGGER = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,              -- system|user|assistant|tool
    content TEXT NOT NULL,
    tool_name TEXT,                  -- nullable, only for role='tool'
    tool_args TEXT,                  -- JSON
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,            -- 'soul'|'reflection'|'memory'|'file'
    kind TEXT NOT NULL,              -- 'soul'|'yo'|'superyo'|'ello'|'episode'|'note'
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,         -- JSON list of floats
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS raptor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level INTEGER NOT NULL,          -- 0=leaves(=chunk ids), 1..N=summaries
    cluster_id INTEGER NOT NULL,     -- cluster index within level
    parent_id INTEGER,              -- id of parent node at level-1 (nullable for leaves)
    text TEXT NOT NULL,              -- summary text (or original chunk text at level 0)
    embedding TEXT NOT NULL,         -- JSON list of floats
    member_ids TEXT NOT NULL,        -- JSON list of raptor.id or chunk.id at level 0
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_raptor_level ON raptor(level);
"""


class Memory:
    """Thread-safe SQLite persistence + RAG over Ollama embeddings."""

    def __init__(self) -> None:
        SETTINGS.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(SETTINGS.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(_SCHEMA)
        self._session_id: int | None = None

    # ---- sessions ----

    def start_session(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions(started_at) VALUES (?)", (time.time(),)
            )
            self._session_id = cur.lastrowid
            assert cur.lastrowid is not None
            return cur.lastrowid

    def end_session(self) -> None:
        with self._lock:
            if self._session_id is not None:
                self._conn.execute(
                    "UPDATE sessions SET ended_at=? WHERE id=?",
                    (time.time(), self._session_id),
                )
                self._conn.commit()

    @property
    def session_id(self) -> int | None:
        return self._session_id

    # ---- messages ----

    def add_message(
        self,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages(session_id, role, content, tool_name, tool_args, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    self._session_id,
                    role,
                    content,
                    tool_name,
                    json.dumps(tool_args) if tool_args else None,
                    time.time(),
                ),
            )
            self._conn.commit()
            assert cur.lastrowid is not None
            return cur.lastrowid

    def recent_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        limit = limit or SETTINGS.history_window
        with self._lock:
            summary_rows = self._conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id=? AND role='system' ORDER BY id",
                (self._session_id,),
            ).fetchall()
            rows = self._conn.execute(
                "SELECT role, content, tool_name, tool_args FROM messages "
                "WHERE session_id=? AND role!='system' ORDER BY id DESC LIMIT ?",
                (self._session_id, limit),
            ).fetchall()
        out = [{"role": r, "content": c} for r, c in summary_rows]
        for role, content, tool_name, tool_args in reversed(rows):
            msg: dict[str, Any] = {"role": role, "content": content}
            if tool_name:
                msg["role"] = "tool"
                msg["name"] = tool_name
                if tool_args:
                    try:
                        msg["args"] = json.loads(tool_args)
                    except json.JSONDecodeError:
                        msg["args"] = {}
            out.append(msg)
        return out

    def session_message_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=?",
                (self._session_id,),
            ).fetchone()
        return row[0] if row else 0

    def all_session_messages(self) -> list[dict[str, Any]]:
        """All messages in the current session (for memory extraction)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                (self._session_id,),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def compact_history(self, summary: str) -> int:
        """Replace all messages older than the last few with a single 'system' summary.
        Keeps the most recent `keep_recent` messages for continuity.
        Returns the number of messages deleted.
        # ponytail: keep last 4 messages for immediate context, summarize the rest.
        """
        keep_recent = 4
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (self._session_id, keep_recent),
            ).fetchall()
            keep_ids = [r[0] for r in rows]
            if keep_ids:
                placeholders = ",".join("?" * len(keep_ids))
                cur = self._conn.execute(
                    f"DELETE FROM messages WHERE session_id=? AND id NOT IN ({placeholders})",
                    (self._session_id, *keep_ids),
                )
            else:
                cur = self._conn.execute(
                    "DELETE FROM messages WHERE session_id=?",
                    (self._session_id,),
                )
            deleted = cur.rowcount
            # Insert summary as the oldest message.
            self._conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) "
                "VALUES (?,?,?,?)",
                (self._session_id, "system", summary, time.time()),
            )
            self._conn.commit()
            return deleted

    # ---- embeddings + RAG ----

    def embed(self, text: str) -> list[float]:
        return _embed_ollama(text)

    def add_chunk(self, source: str, kind: str, text: str) -> None:
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM chunks WHERE source=? AND kind=? AND text=? LIMIT 1",
                (source, kind, text),
            ).fetchone()
            if existing:
                return
        emb = self.embed(text)
        with self._lock:
            self._conn.execute(
                "INSERT INTO chunks(source, kind, text, embedding, created_at) VALUES (?,?,?,?,?)",
                (source, kind, text, json.dumps(emb), time.time()),
            )
            self._conn.commit()

    def seed_soul(self, soul_text: str) -> None:
        """Chunk SOUL.md by section (## headers) and embed each. Idempotent: clears old 'soul' chunks first."""
        chunks = _split_sections(soul_text)
        embedded = [(heading, body, self.embed(body)) for heading, body in chunks]
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE source='soul'")
            for heading, body, emb in embedded:
                self._conn.execute(
                    "INSERT INTO chunks(source, kind, text, embedding, created_at) VALUES (?,?,?,?,?)",
                    ("soul", "soul", f"{heading}\n{body}", json.dumps(emb), time.time()),
                )
            self._conn.commit()

    def retrieve(self, query: str, top_k: int | None = None, source: str | None = None) -> list[dict[str, Any]]:
        top_k = top_k or SETTINGS.rag_top_k
        q_emb = self.embed(query)
        with self._lock:
            if source:
                rows = self._conn.execute(
                    "SELECT id, source, kind, text, embedding FROM chunks WHERE source=?",
                    (source,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, source, kind, text, embedding FROM chunks"
                ).fetchall()
        scored = []
        for _id, src, kind, text, emb_json in rows:
            emb = json.loads(emb_json)
            score = _cosine(q_emb, emb)
            scored.append((score, {"id": _id, "source": src, "kind": kind, "text": text}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k]]

    # ---- meta helpers ----

    def _get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else default

    def _set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value)
            )
            self._conn.commit()

    # ---- RAPTOR: hierarchical clustering + summarization ----

    def clear_raptor(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM raptor")
            self._conn.commit()

    def _raptor_insert(self, level: int, cluster_id: int, parent_id: int | None,
                       text: str, emb: list[float], member_ids: list[int]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO raptor(level, cluster_id, parent_id, text, embedding, member_ids, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (level, cluster_id, parent_id, text, json.dumps(emb),
                 json.dumps(member_ids), time.time()),
            )
            self._conn.commit()
            assert cur.lastrowid is not None
            return cur.lastrowid

    def raptor_stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT level, COUNT(*) FROM raptor GROUP BY level ORDER BY level"
            ).fetchall()
        return {lvl: cnt for lvl, cnt in rows}

    def build_raptor(self, summarize_fn) -> list[str]:
        """Build the RAPTOR tree from existing chunks.

        summarize_fn(texts: list[str]) -> str  : calls the LLM to summarize a cluster.
        Returns a log of human-readable steps (for the REM display).
        # ponytail: k-means with stdlib random init + cosine; no numpy. Fine for a
        personal agent's scale (~hundreds of chunks). Switch to sklearn if it grows.
        """
        log: list[str] = []
        # Incremental: skip if no new chunks since last build.
        with self._lock:
            cur_max = self._conn.execute("SELECT COALESCE(MAX(id),0) FROM chunks").fetchone()[0]
        last_max = int(self._get_meta("raptor_last_chunk_id", "0"))
        if cur_max == last_max:
            log.append("Sin chunks nuevos desde el ultimo RAPTOR; saltando.")
            return log

        self.clear_raptor()
        # Level 0: all chunks become leaves.
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, embedding FROM chunks ORDER BY id"
            ).fetchall()
        if not rows:
            log.append("No hay chunks en el RAG; nada que indexar.")
            return log
        leaves: list[dict[str, Any]] = []
        for cid, text, emb_json in rows:
            leaves.append({"id": cid, "text": text, "embedding": json.loads(emb_json)})
        log.append(f"Nivel 0: {len(leaves)} hojas (chunks existentes).")

        # Insert leaves into raptor table.
        for i, leaf in enumerate(leaves):
            self._raptor_insert(0, i, None, leaf["text"], leaf["embedding"], [leaf["id"]])

        current = leaves
        level = 1
        next_cluster_id = 0
        while len(current) > 1 and level <= SETTINGS.raptor_max_levels:
            k = min(SETTINGS.raptor_cluster_k, len(current) - 1)
            if k < 1:
                break
            clusters = _kmeans_cosine(current, k)
            parents: list[dict[str, Any]] = []
            for _cluster_idx, members in enumerate(clusters):
                if not members:
                    continue
                joined = "\n".join(f"- {m['text'][:200]}" for m in members)
                try:
                    summary = summarize_fn(joined).strip()
                except Exception:
                    summary = joined[: SETTINGS.raptor_summary_max_chars]
                if not summary:
                    summary = joined[: SETTINGS.raptor_summary_max_chars]
                summary = summary[: SETTINGS.raptor_summary_max_chars]
                emb = self.embed(summary)
                member_ids = [m["id"] for m in members]
                pid = self._raptor_insert(level, next_cluster_id, None, summary, emb, member_ids)
                parents.append({"id": pid, "text": summary, "embedding": emb, "member_ids": member_ids})
                next_cluster_id += 1
            log.append(f"Nivel {level}: {len(parents)} clusters resumidos.")
            current = parents
            level += 1
        log.append(f"RAPTOR construido: {level - 1} niveles.")
        self._set_meta("raptor_last_chunk_id", str(cur_max))
        return log

    def retrieve_raptor(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or SETTINGS.rag_top_k
        q_emb = self.embed(query)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, level, text, embedding FROM raptor ORDER BY id"
            ).fetchall()
        scored = []
        for rid, lvl, text, emb_json in rows:
            emb = json.loads(emb_json)
            score = _cosine(q_emb, emb)
            scored.append((score, {"id": rid, "level": lvl, "text": text}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k]]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@functools.lru_cache(maxsize=256)
# ponytail: RAM cache de 256 embeddings (~1MB). Si hay miles de chunks únicos, mover a SQLite embeddings_cache(text_hash, embedding) y borrar el lru_cache.
def _embed_ollama(text: str) -> list[float]:
    url = f"{SETTINGS.ollama_base_url.rstrip('/')}/api/embeddings"
    data = json.dumps({
        "model": SETTINGS.ollama_embed_model,
        "prompt": text,
        "keep_alive": SETTINGS.ollama_keep_alive,
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=SETTINGS.ollama_timeout_seconds)
    return json.loads(resp.read())["embedding"]


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown by '## ' headers. Returns [(heading, body), ...]."""
    sections: list[tuple[str, str]] = []
    current_h = "Preambulo"
    current_body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_body or sections:
                sections.append((current_h, "\n".join(current_body).strip()))
            current_h = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_h, "\n".join(current_body).strip()))
    return [(h, b) for h, b in sections if b]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))  # noqa: B905
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _kmeans_cosine(items: list[dict[str, Any]], k: int, iters: int = 10) -> list[list[dict[str, Any]]]:
    """Stdlib k-means with cosine distance. items must have 'embedding' (list[float]).
    # ponytail: random init, fixed iters, no convergence check; good enough for RAPTOR clustering.
    """
    if k <= 0 or not items:
        return [items] if items else []
    n_items = len(items)
    k = min(k, n_items)
    dim = len(items[0]["embedding"])

    rng = random.Random(SETTINGS.raptor_seed)
    embs = [it["embedding"] for it in items]
    centroids = [rng.choice(embs) for _ in range(k)]
    for _ in range(iters):
        clusters: list[list[int]] = [[] for _ in range(k)]
        for i, e in enumerate(embs):
            best = max(range(k), key=lambda c: _cosine(e, centroids[c]))
            clusters[best].append(i)
        for c in range(k):
            if clusters[c]:
                acc = [0.0] * dim
                for idx in clusters[c]:
                    for d in range(dim):
                        acc[d] += embs[idx][d]
                n = len(clusters[c])
                centroids[c] = [v / n for v in acc]
    result: list[list[dict[str, Any]]] = [[] for _ in range(k)]
    for i, it in enumerate(items):
        best = max(range(k), key=lambda c: _cosine(embs[i], centroids[c]))
        result[best].append(it)
    return [c for c in result if c]


