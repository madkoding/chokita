"""Tkinter UI for non-blocking assistant status visualization."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from typing import Any

from src.config import KAOMOJIS, SETTINGS

LOGGER = logging.getLogger(__name__)


class AssistantUI:
    """Main-thread-only UI that reacts to queue messages."""

    def __init__(self, ui_queue: "queue.Queue[dict[str, Any]]") -> None:
        self.ui_queue = ui_queue
        self.root = tk.Tk()
        self.root.title("Chokita Assistant")
        self.root.geometry("500x280")

        self.state_var = tk.StringVar(value="IDLE")
        self.kaomoji_var = tk.StringVar(value=KAOMOJIS["IDLE"])
        self.message_var = tk.StringVar(value="Esperando comando...")

        tk.Label(self.root, textvariable=self.kaomoji_var, font=("Arial", 40)).pack(pady=14)
        tk.Label(self.root, textvariable=self.state_var, font=("Arial", 16, "bold")).pack()
        tk.Label(
            self.root,
            textvariable=self.message_var,
            font=("Arial", 12),
            wraplength=470,
            justify="left",
        ).pack(pady=10)

    def _apply_state(self, state: str, message: str | None = None) -> None:
        if state not in KAOMOJIS:
            state = "ERROR"
            message = message or "Estado inválido detectado"
        self.state_var.set(state)
        self.kaomoji_var.set(KAOMOJIS[state])
        if message is not None:
            self.message_var.set(message)

    def _process_queue(self) -> None:
        try:
            while True:
                event = self.ui_queue.get_nowait()
                event_type = event.get("type")
                if event_type == "state":
                    self._apply_state(event.get("state", "ERROR"), event.get("message"))
                elif event_type == "log":
                    self.message_var.set(str(event.get("message", "")))
                elif event_type == "shutdown":
                    self.root.quit()
                    return
        except queue.Empty:
            pass
        except Exception:
            LOGGER.exception("UI queue processing failed")
            self._apply_state("ERROR", "Fallo de interfaz")
        finally:
            self.root.after(SETTINGS.ui_poll_interval_ms, self._process_queue)

    def run(self) -> None:
        self.root.after(SETTINGS.ui_poll_interval_ms, self._process_queue)
        self.root.mainloop()
