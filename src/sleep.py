"""REM sleep thread: when idle for a long while, Chokita dreams.

She enters a SLEEPING state (kaomoji changes), reindexes her RAG by building
a RAPTOR tree (hierarchical clustering + summarization), and emits progress
to the UI in a distinct color. Any user input wakes her immediately.
# ponytail: one thread, one job — reindex. No dream narration LLM calls; the
summaries themselves are the dream content shown on screen.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

from src.config import SETTINGS
from src.memory import Memory

LOGGER = logging.getLogger(__name__)


class SleepThread(threading.Thread):
    """Idle REM: builds RAPTOR over the RAG periodically while the agent is idle."""

    def __init__(
        self,
        memory: Memory,
        summarize_fn: Callable[[str], str],
        stop_event: threading.Event,
        activity_fn: Callable[[], float],
        ui_queue: queue.Queue[dict[str, Any]],
    ) -> None:
        super().__init__(daemon=True, name="rem-sleep")
        self.memory = memory
        self.summarize = summarize_fn
        self.stop_event = stop_event
        self._last_activity = activity_fn
        self.ui_queue = ui_queue

    def run(self) -> None:
        while not self.stop_event.is_set():
            if self._last_activity() < SETTINGS.rem_idle_threshold_seconds:
                if self.stop_event.wait(timeout=SETTINGS.rem_idle_threshold_seconds):
                    break
                continue
            if self.stop_event.wait(timeout=SETTINGS.rem_raptor_interval_seconds):
                break
            if self.stop_event.is_set():
                break
            if self._last_activity() < SETTINGS.rem_idle_threshold_seconds:
                continue
            try:
                self._dream_once()
            except Exception:
                LOGGER.exception("REM sleep failed")

    def _dream_once(self) -> None:
        self.ui_queue.put({"type": "state", "state": "SLEEPING", "message": "Zzz... reindexando el alma (RAPTOR)"})
        self.ui_queue.put({"type": "log", "message": "🌙 Chokita se duerme y empieza a soñar con su RAG...", "dream": True})
        pruned = self.memory.prune_chunks()
        if pruned["ttl_deleted"] or pruned["cap_deleted"]:
            self.ui_queue.put({
                "type": "log",
                "message": f"  🧹 GC: {pruned['ttl_deleted']} por antigüedad, {pruned['cap_deleted']} por límite.",
                "dream": True,
            })
        self.memory.checkpoint()
        try:
            log = self.memory.build_raptor(self.summarize)
        except Exception:
            self.ui_queue.put({"type": "state", "state": "IDLE", "message": "No pude dormir bien; vuelvo a idle."})
            return
        for line in log:
            self.ui_queue.put({"type": "log", "message": f"  💫 {line}", "dream": True})
        stats = self.memory.raptor_stats()
        self.ui_queue.put({"type": "log", "message": f"  🧠 RAPTOR: {stats}", "dream": True})
        self.memory.checkpoint()
        self.ui_queue.put({"type": "state", "state": "IDLE", "message": "Despierta. Listo."})
        self.ui_queue.put({"type": "log", "message": "☀️ Chokita despierta.", "dream": True})