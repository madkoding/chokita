from __future__ import annotations

import logging
import queue
from typing import Any

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Input, RichLog, Static

LOGGER = logging.getLogger(__name__)

# --- Tokyo Night-inspired palette ---
BG = "#1a1b26"
BG_DARK = "#16161e"
BG_PANEL = "#1f2335"
BG_INPUT = "#24283b"
BORDER = "#2d3548"
TEXT_PRIMARY = "#c0caf5"
ACCENT = "#7aa2f7"
ACCENT_DIM = "#3b4261"
GREEN = "#9ece6a"
YELLOW = "#e0af68"
RED = "#f7768e"
MAGENTA = "#bb9af7"
CYAN = "#7dcfff"
VIOLET = "#9d7cd8"

CAT_ART = {
    "IDLE": [
        r" /\_/\ ",
        r"( o.o )",
        r" > ^ < ",
    ],
    "LISTENING": [
        r" /\_/\ *",
        r"( @.@ )~",
        r" > w < ",
    ],
    "THINKING": [
        r" /\_/\ ",
        r"( o.o )?",
        r" > ? < ",
    ],
    "SPEAKING": [
        r" /\_/\ ~",
        r"( >w< )",
        r" > w < ",
    ],
    "RECOGNIZED": [
        r" /\_/\ *",
        r"( ^.^ )b",
        r" > w < ",
    ],
    "ERROR": [
        r" /\_/\ ",
        r"( ;.; )",
        r" > _ < ",
    ],
    "SLEEPING": [
        r" /\_/\ zzz",
        r"( -.- )~",
        r" > ~ < ",
    ],
}

STATE_LABEL = {
    "IDLE": "IDLE",
    "LISTENING": "ESCUCHANDO",
    "THINKING": "PENSANDO",
    "SPEAKING": "HABLANDO",
    "RECOGNIZED": "RECONOCIDO",
    "ERROR": "ERROR",
    "SLEEPING": "DURMIENDO",
}

STATE_COLORS = {
    "IDLE": CYAN,
    "LISTENING": GREEN,
    "THINKING": YELLOW,
    "SPEAKING": MAGENTA,
    "RECOGNIZED": "#7dcfff",
    "ERROR": RED,
    "SLEEPING": VIOLET,
}


class KaomojiFace(Static):
    state: reactive[str] = reactive("IDLE")
    blink: reactive[bool] = reactive(False)
    mouth_open: reactive[bool] = reactive(False)

    def render(self) -> Text:
        lines = CAT_ART.get(self.state, CAT_ART["IDLE"])
        if self.blink:
            lines = [line.replace("o.o", "-.-").replace("@.@", "-.-").replace("^.^", "-.-").replace(";.;", "-.-") for line in lines]
        if self.mouth_open and self.state == "SPEAKING":
            lines = [line.replace(">w<", ">o<") for line in lines]
        color = STATE_COLORS.get(self.state, CYAN)
        return Text("\n".join(lines), style=f"bold {color}")


class StatusPill(Static):
    state: reactive[str] = reactive("IDLE")

    def render(self) -> Text:
        color = STATE_COLORS.get(self.state, CYAN)
        label = STATE_LABEL.get(self.state, "?")
        return Text(f"● {label}", style=f"bold {color}")


class TokenBar(Static):
    used: reactive[int] = reactive(0)
    total: reactive[int] = reactive(1)

    def render(self) -> Text:
        pct = (self.used / self.total * 100) if self.total else 0
        if pct < 50:
            color = GREEN
        elif pct < 80:
            color = YELLOW
        else:
            color = RED
        filled = int(pct / 100 * 16)
        bar = "▰" * filled + "▱" * (16 - filled)
        used_k = self.used / 1000
        total_k = self.total / 1000
        return Text(f"{bar} {used_k:.0f}K/{total_k:.0f}K", style=f"dim {color}")


class ResponseBubble(Static):
    """Burbuja estilo chat para la última respuesta de Chokita."""
    message: reactive[str] = reactive("")
    state: reactive[str] = reactive("IDLE")

    def render(self) -> Text:
        if not self.message:
            return Text("", style="dim")
        color = STATE_COLORS.get(self.state, CYAN)
        prefix = Text("🐱 ", style=f"{color}")
        body = Text(self.message, style=TEXT_PRIMARY)
        return prefix + body


class AudioLevel(Static):
    level: reactive[int] = reactive(0)

    def render(self) -> Text:
        if self.level == 0:
            return Text("🎤 ──────────", style="dim #565f89")
        bars = "▁▂▃▄▅▆▇█"
        idx = min(self.level * 8 // 20, 7)
        bar = bars[idx] * 10
        if self.level < 15:
            color = GREEN
        elif self.level < 18:
            color = YELLOW
        else:
            color = RED
        return Text(f"🎤 {bar}", style=color)


class _PasteAwareInput(Input):
    PASTE_THRESHOLD = 200

    def __init__(self, *args: Any, on_big_paste: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_big_paste = on_big_paste

    def _on_paste(self, event: Any) -> None:
        text = getattr(event, "text", "")
        if not text:
            return
        has_newlines = "\n" in text
        if len(text) <= self.PASTE_THRESHOLD and not has_newlines:
            return
        event.stop()
        if self._on_big_paste:
            self._on_big_paste(text)


class FaceApp(App):
    CSS = f"""
    Screen {{
        layout: horizontal;
        background: {BG};
        color: {TEXT_PRIMARY};
    }}

    #left-panel {{
        width: 28;
        dock: left;
        background: {BG_DARK};
        border-right: solid {BORDER};
        padding: 1 1;
        layout: vertical;
    }}

    #right-panel {{
        width: 1fr;
        layout: vertical;
        background: {BG};
    }}

    #face-box {{
        height: 5;
        content-align: center middle;
        background: {BG_PANEL};
        border: solid {BORDER};
        margin: 0 0 1 0;
    }}

    #face {{
        text-align: center;
    }}

    #status-row {{
        height: 1;
        content-align: center middle;
        margin: 0 0 1 0;
    }}

    #token-bar {{
        height: 1;
        margin: 0 0 1 0;
    }}

    #response-bubble {{
        height: auto;
        max-height: 6;
        padding: 0 1;
        margin: 0 0 1 0;
        background: {BG_PANEL};
        border: solid {BORDER};
    }}

    #audio-level {{
        height: 1;
        margin: 0 0 1 0;
    }}

    #chat-panel {{
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: {BG};
        border-bottom: solid {BORDER};
    }}

    #chat {{
        width: 1fr;
        height: 1fr;
        border: none;
        background: {BG};
        padding: 0;
    }}

    #bottom-bar {{
        height: auto;
        dock: bottom;
        background: {BG_DARK};
        border-top: solid {BORDER};
        padding: 0;
    }}

    #input {{
        dock: bottom;
        height: 3;
        margin: 0;
        border: solid {ACCENT_DIM};
        background: {BG_INPUT};
        padding: 0 1;
    }}

    #input:focus {{
        border: solid {ACCENT};
    }}
    """

    def __init__(
        self,
        text_queue: queue.Queue[str],
        ui_queue: queue.Queue[dict[str, Any]],
    ) -> None:
        self.text_queue = text_queue
        self.ui_queue = ui_queue
        self._pasted_buffer: str | None = None
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="left-panel"):
            with Static(id="face-box"):
                yield KaomojiFace(id="face")
            with Static(id="status-row"):
                yield StatusPill(id="status-pill")
            yield TokenBar(id="token-bar")
            yield AudioLevel(id="audio-level")
            yield ResponseBubble(id="response-bubble")
        with Vertical(id="right-panel"):
            with Static(id="chat-panel"):
                yield RichLog(id="chat", highlight=True, markup=True, wrap=True,
                              auto_scroll=True)
            yield _PasteAwareInput(
                id="input",
                on_big_paste=self._handle_big_paste,
                placeholder="Escribí un mensaje y presioná Enter...",
            )

    def on_mount(self) -> None:
        self.set_interval(0.05, self._drain_ui)
        self.set_interval(3.0, self._do_blink)
        self.set_interval(0.12, self._toggle_mouth)
        self.query_one("#input", Input).focus()

    def _do_blink(self) -> None:
        try:
            face = self.query_one("#face", KaomojiFace)
            if face.state != "ERROR":
                face.blink = True
                self.set_timer(0.12, lambda: setattr(face, "blink", False))
        except Exception:
            LOGGER.exception("Error en _do_blink")

    def _toggle_mouth(self) -> None:
        try:
            face = self.query_one("#face", KaomojiFace)
            if face.state == "SPEAKING":
                face.mouth_open = not face.mouth_open
            elif face.mouth_open:
                face.mouth_open = False
        except Exception:
            LOGGER.exception("Error en _toggle_mouth")

    def _drain_ui(self) -> None:
        try:
            while True:
                event = self.ui_queue.get_nowait()
                event_type = event.get("type")
                if event_type == "state":
                    state = event.get("state", "ERROR")
                    face = self.query_one("#face", KaomojiFace)
                    face.state = state
                    pill = self.query_one("#status-pill", StatusPill)
                    pill.state = state
                    bubble = self.query_one("#response-bubble", ResponseBubble)
                    bubble.state = state
                    msg = event.get("message", state)
                    if state == "SPEAKING":
                        self.query_one("#chat", RichLog).write(
                            Text(f"🐱 Chokita: {msg}", style=f"bold {MAGENTA}")
                        )
                        bubble.message = msg
                    elif state == "ERROR":
                        self.query_one("#chat", RichLog).write(
                            Text(f"⚠ {msg}", style=f"bold {RED}")
                        )
                elif event_type == "tokens":
                    bar = self.query_one("#token-bar", TokenBar)
                    bar.used = event.get("used", 0)
                    bar.total = event.get("total", 1)
                elif event_type == "log":
                    msg = str(event.get("message", ""))
                    if event.get("dream"):
                        self.query_one("#chat", RichLog).write(
                            Text(msg, style=f"dim {VIOLET} italic")
                        )
                    else:
                        self.query_one("#chat", RichLog).write(msg)
                elif event_type == "token":
                    self.query_one("#response-bubble", ResponseBubble).message = event.get("content", "")
                elif event_type == "audio_level":
                    self.query_one("#audio-level", AudioLevel).level = event.get("level", 0)
                elif event_type == "shutdown":
                    self.exit(message="bye")
                    return
        except queue.Empty:
            pass
        except Exception:
            LOGGER.exception("Error en _drain_ui")

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        typed = event.value.strip()
        event.input.value = ""
        if self._pasted_buffer:
            full = self._pasted_buffer
            if typed:
                full = f"{full}\n\n---\n{typed}"
            self._pasted_buffer = None
            self.query_one("#input", Input).placeholder = (
                "Escribí un mensaje y presioná Enter..."
            )
            LOGGER.info("Usuario (pegado, %d chars): enviado", len(full))
            self.query_one("#chat", RichLog).write(
                Text(f"👤 Tú: [📋 texto pegado, {len(full)} caracteres]", style=f"bold {GREEN}")
            )
            self._set_thinking()
            self.text_queue.put(full)
            return
        if typed:
            if len(typed) > _PasteAwareInput.PASTE_THRESHOLD:
                LOGGER.info("Usuario (input largo, %d chars): enviado", len(typed))
                self.query_one("#chat", RichLog).write(
                    Text(f"👤 Tú: [📋 {len(typed)} caracteres]", style=f"bold {GREEN}")
                )
            else:
                LOGGER.info("Usuario: %s", typed)
                self.query_one("#chat", RichLog).write(
                    Text(f"👤 Tú: {typed}", style=f"bold {GREEN}")
                )
            self._set_thinking()
            self.text_queue.put(typed)

    def _set_thinking(self) -> None:
        self.query_one("#face", KaomojiFace).state = "THINKING"
        self.query_one("#status-pill", StatusPill).state = "THINKING"
        self.query_one("#response-bubble", ResponseBubble).message = "..."

    def _handle_big_paste(self, text: str) -> None:
        self._pasted_buffer = text
        n = len(text)
        preview = text[:60].replace("\n", " ↵ ").rstrip()
        try:
            self.query_one("#input", Input).placeholder = (
                f"📋 {n} chars — Enter envía, Esc descarta"
            )
            self.query_one("#chat", RichLog).write(
                Text(f"📋 Pegado: {n} caracteres — {preview}...", style=f"bold {CYAN}")
            )
        except Exception:
            LOGGER.exception("Error mostrando paste chip")

    def on_key(self, event: Any) -> None:
        if getattr(event, "key", "") == "escape" and self._pasted_buffer:
            self._pasted_buffer = None
            try:
                self.query_one("#input", Input).placeholder = (
                    "Escribí un mensaje y presioná Enter..."
                )
                self.query_one("#chat", RichLog).write(
                    Text("📋 Pegado descartado.", style=f"dim {YELLOW}")
                )
            except Exception:
                pass
            event.prevent_default()