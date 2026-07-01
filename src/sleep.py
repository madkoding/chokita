"""REM sleep thread: when idle for a long while, Chokita dreams.

She enters a SLEEPING state (kaomoji changes), reindexes her RAG by building
a RAPTOR tree (hierarchical clustering + summarization), and emits progress
to the UI in a distinct color. Any user input wakes her immediately.
# ponytail: one thread, one job — reindex. No dream narration LLM calls; the
summaries themselves are the dream content shown on screen.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from src.config import SETTINGS
from src.memory import Memory

LOGGER = logging.getLogger(__name__)


def _slice_sleep(stop_event: threading.Event, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if stop_event.is_set():
            return True
        time.sleep(min(1.0, end - time.time()))
    return False


class SleepThread(threading.Thread):
    """Idle REM: builds RAPTOR over the RAG periodically while the agent is idle."""

    def __init__(
        self,
        memory: Memory,
        summarize_fn: Callable[[str], str],
        stop_event: threading.Event,
        activity_fn: Callable[[], float],
        ui_queue: object,
    ) -> None:
        super().__init__(daemon=True, name="rem-sleep")
        self.memory = memory
        self.summarize = summarize_fn
        self.stop_event = stop_event
        self._last_activity = activity_fn
        self.ui_queue = ui_queue

    def run(self) -> None:
        while not self.stop_event.is_set():
            # Wait until idle threshold is reached.
            if self._last_activity() < SETTINGS.rem_idle_threshold_seconds:
                time.sleep(SETTINGS.rem_idle_threshold_seconds)
                continue
            # Sleep the raptor interval in slices (responsive to stop).
            if _slice_sleep(self.stop_event, SETTINGS.rem_raptor_interval_seconds):
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
        self._emit_state("SLEEPING", "Zzz... reindexando el alma (RAPTOR)")
        self._emit_log("🌙 Chokita se duerme y empieza a soñar con su RAG...", dream=True)
        try:
            log = self.memory.build_raptor(self.summarize)
        except Exception:
            self._emit_state("IDLE", "No pude dormir bien; vuelvo a idle.")
            return
        for line in log:
            self._emit_log(f"  💫 {line}", dream=True)
        stats = self.memory.raptor_stats()
        self._emit_log(f"  🧠 RAPTOR: {stats}", dream=True)
        self._emit_state("IDLE", "Despierta. Listo.")
        self._emit_log("☀️ Chokita despierta.", dream=True)

    def _emit_state(self, state: str, message: str) -> None:
        self.ui_queue.put({"type": "state", "state": state, "message": message})

    def _emit_log(self, message: str, dream: bool = False) -> None:
        self.ui_queue.put({"type": "log", "message": message, "dream": dream})