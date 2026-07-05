from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any, cast

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Input, RichLog, Static

from src.config import SETTINGS

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
        "  /\\_/\\   ",
        " ( o.o )  ",
        "  > ^ <   ",
    ],
    "LISTENING": [
        "  /\\_/\\ * ",
        " ( @.@ )~ ",
        "  > w <   ",
    ],
    "THINKING": [
        "  /\\_/\\   ",
        " ( o.o )? ",
        "  > ? <   ",
    ],
    "SPEAKING": [
        "  /\\_/\\ ~ ",
        " ( >w< )  ",
        "  > w <   ",
    ],
    "RECOGNIZED": [
        "  /\\_/\\ * ",
        " ( ^.^ )b ",
        "  > w <   ",
    ],
    "ERROR": [
        "  /\\_/\\   ",
        " ( ;.; )  ",
        "  > _ <   ",
    ],
    "SLEEPING": [
        "  /\\_/\\zzz",
        " ( -.- )~ ",
        "  > ~ <   ",
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

FEELING_ART: dict[str, list[str]] = {
    "neutral":     ["  /\\_/\\   ", " ( o.o )  ", "  > ^ <   "],
    "curious":     ["  /\\_/\\   ", " ( @.@ )  ", "  > w <   "],
    "happy":       ["  /\\_/\\   ", " ( ^.^ )  ", "  > w <   "],
    "focused":     ["  /\\_/\\   ", " ( 0.0 )  ", "  > ^ <   "],
    "confused":    ["  /\\_/\\   ", " ( o.o )? ", "  > ~ <   "],
    "surprised":   ["  /\\_/\\   ", " ( O.O )  ", "  > o <   "],
    "mischievous": ["  /\\_/\\   ", " ( >.> )  ", "  > ^ <   "],
    "sleepy":      ["  /\\_/\\   ", " ( -.- )~ ", "  > ~ <   "],
}


class KaomojiFace(Static):
    state: reactive[str] = reactive("IDLE")
    blink: reactive[bool] = reactive(False)
    mouth_open: reactive[bool] = reactive(False)
    feeling: reactive[str] = reactive("")

    def render(self) -> Text:
        if self.feeling and self.feeling in FEELING_ART:
            lines = list(FEELING_ART[self.feeling])
        else:
            lines = list(CAT_ART.get(self.state, CAT_ART["IDLE"]))
        if self.blink:
            for _eye in ("o.o", "@.@", "^.^", ";.;", "0.0", "O.O", ">.>"):
                lines = [line.replace(_eye, "-.-") for line in lines]
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

    def on_paste(self, event: Any) -> None:
        text = getattr(event, "text", "")
        if not text:
            return
        has_newlines = "\n" in text
        if len(text) <= self.PASTE_THRESHOLD and not has_newlines:
            return
        event.stop()
        if self._on_big_paste:
            self._on_big_paste(text)


# ponytail: spinner compartido, sin dependencias
_PRELOAD_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class PreloadScreen(Screen):
    """Pantalla de precarga de modelos dentro del TUI. Se autodescarta al terminar."""

    def __init__(self) -> None:
        super().__init__()
        self._jobs = [
            ("🧠", "Ollama (chat)", str(SETTINGS.ollama_model), self._warm_ollama_chat),
            ("🔢", "Ollama (embeddings)", str(SETTINGS.ollama_embed_model), self._warm_ollama_embed),
            ("🎤", "Vosk (STT)", str(SETTINGS.vosk_model_path), self._load_vosk),
            ("🔊", "Piper (TTS)", str(SETTINGS.piper_model_path), self._load_piper),
        ]
        self._statuses: list[str] = ["cargando"] * len(self._jobs)
        self._results: dict[str, Any] = {}
        self._spin_idx = 0

    def compose(self) -> ComposeResult:
        with Vertical(classes="preload-container"):
            yield Static("", classes="preload-gap")
            for i, (ic, name, path, *_) in enumerate(self._jobs):
                yield Static(f"{ic} {name}", classes="preload-name")
                yield Static(f"   {path}", classes="preload-path")
                yield Static(f"   {_PRELOAD_SPINNER[0]} cargando...", id=f"ps-{i}")

    def on_mount(self) -> None:
        self.set_interval(0.08, self._tick)
        for i, (_, _, _, fn) in enumerate(self._jobs):
            threading.Thread(target=self._run_job, args=(i, fn), daemon=True).start()
        self._deadline = time.time() + SETTINGS.preload_timeout_seconds
        self.set_interval(0.2, self._check_done)

    def _tick(self) -> None:
        self._spin_idx = (self._spin_idx + 1) % len(_PRELOAD_SPINNER)
        for i, s in enumerate(self._statuses):
            if s == "cargando":
                w = self.query_one(f"#ps-{i}", Static)
                if w:
                    w.update(f"   {_PRELOAD_SPINNER[self._spin_idx]} cargando...")

    def _run_job(self, idx: int, fn: Callable[[], Any]) -> None:
        try:
            result = fn()
            self._statuses[idx] = "ok"
            self._results[self._jobs[idx][1]] = result
            self.call_from_thread(self._update_status, idx)  # type: ignore[attr-defined]
        except Exception:
            LOGGER.warning("Precarga de %s fallo", self._jobs[idx][1], exc_info=True)
            self._statuses[idx] = "fallo"
            self.call_from_thread(self._update_status, idx)  # type: ignore[attr-defined]

    def _update_status(self, idx: int) -> None:
        s = self._statuses[idx]
        w = self.query_one(f"#ps-{idx}", Static)
        if not w:
            return
        if s == "ok":
            w.update(Text("   ✓ cargado", style=f"bold {GREEN}"))
        elif s == "fallo":
            w.update(Text("   ✗ falló", style=f"bold {RED}"))
        elif s == "timeout":
            w.update(Text("   ⏱ timeout", style=f"bold {YELLOW}"))

    def _check_done(self) -> None:
        if time.time() > self._deadline:
            for i, s in enumerate(self._statuses):
                if s == "cargando":
                    self._statuses[i] = "timeout"
                    self._update_status(i)
            self._finish()
            return
        if all(s != "cargando" for s in self._statuses):
            self._finish()

    def _finish(self) -> None:
        with contextlib.suppress(Exception):
            app = cast("FaceApp", self.app)
            app._on_preload_complete(self._results)
        with contextlib.suppress(Exception):
            self.app.pop_screen()

    def _warm_ollama_chat(self) -> bool:
        data = json.dumps({
            "model": SETTINGS.ollama_model,
            "messages": [{"role": "user", "content": "hola"}],
            "stream": False,
            "keep_alive": SETTINGS.ollama_keep_alive,
        }).encode()
        req = urllib.request.Request(
            f"{SETTINGS.ollama_base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=SETTINGS.ollama_timeout_seconds).read()
        return True

    def _warm_ollama_embed(self) -> bool:
        data = json.dumps({
            "model": SETTINGS.ollama_embed_model,
            "prompt": "test",
            "keep_alive": SETTINGS.ollama_keep_alive,
        }).encode()
        req = urllib.request.Request(
            f"{SETTINGS.ollama_base_url}/api/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=SETTINGS.ollama_timeout_seconds).read()
        return True

    def _load_vosk(self) -> object:
        from vosk import Model
        with contextlib.redirect_stderr(None):
            return Model(str(SETTINGS.vosk_model_path))

    def _load_piper(self) -> Any:
        from piper import PiperVoice
        return PiperVoice.load(str(SETTINGS.piper_model_path))


PreloadScreen.CSS = f"""
PreloadScreen {{
    align: center middle;
    background: {BG};
}}
.preload-gap {{
    height: 3;
}}
.preload-name {{
    color: {ACCENT};
    text-style: bold;
    margin-top: 1;
}}
.preload-path {{
    color: {TEXT_PRIMARY};
    opacity: 0.8;
}}
.preload-status {{
    margin-bottom: 0;
}}
"""


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
        padding: 1 1 0 1;
        layout: vertical;
    }}

    #right-panel {{
        width: 1fr;
        layout: vertical;
        background: {BG};
        padding: 1 0 0 0;
    }}

    #face-box {{
        height: 5;
        content-align: center middle;
        background: {BG_PANEL};
        border: round {BORDER};
        margin: 0 0 1 0;
    }}

    #face {{
        text-align: center;
    }}

    #status-pill {{
        height: 1;
        content-align: center middle;
        margin: 0 0 1 0;
    }}

    #token-bar {{
        height: 1;
        margin: 1 0 1 0;
        border-top: solid {BORDER};
        padding-top: 1;
    }}

    #response-bubble {{
        height: auto;
        max-height: 6;
        padding: 0 1;
        margin: 0 0 1 0;
        background: {BG_PANEL};
        border: round {BORDER};
    }}

    #audio-level {{
        height: 1;
        margin: 0 0 1 0;
    }}

    #chat {{
        width: 1fr;
        height: 1fr;
        border: none;
        background: {BG};
        padding: 0 1;
        overflow: hidden auto;
    }}

    #input {{
        height: 3;
        margin: 1 1 0 1;
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
        on_preload_done: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.text_queue = text_queue
        self.ui_queue = ui_queue
        self.on_preload_done = on_preload_done
        self._pasted_buffer: str | None = None
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="left-panel"):
            with Static(id="face-box"):
                yield KaomojiFace(id="face")
            yield StatusPill(id="status-pill")
            yield TokenBar(id="token-bar")
            yield AudioLevel(id="audio-level")
            yield ResponseBubble(id="response-bubble")
        with Vertical(id="right-panel"):
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
        self.push_screen(PreloadScreen())

    def _on_preload_complete(self, results: dict[str, Any]) -> None:
        if self.on_preload_done:
            self.on_preload_done(results)
        self.set_timer(0.1, lambda: self.query_one("#input", Input).focus())

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
                elif event_type == "thinking":
                    content = event.get("content", "")
                    if content:
                        self.query_one("#chat", RichLog).write(
                            Text(f"🤔 {content}", style=f"dim {YELLOW} italic")
                        )
                elif event_type == "feeling":
                    f = event.get("feeling", "")
                    if f:
                        face = self.query_one("#face", KaomojiFace)
                        face.feeling = f
                elif event_type == "response":
                    content = event.get("content", "")
                    if content:
                        self.query_one("#response-bubble", ResponseBubble).message = content
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
            self._set_ui_state("THINKING", "...")
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
            self._set_ui_state("THINKING", "...")
            self.text_queue.put(typed)

    def _set_ui_state(self, state: str, bubble_msg: str = "...") -> None:
        self.query_one("#face", KaomojiFace).state = state
        self.query_one("#status-pill", StatusPill).state = state
        self.query_one("#response-bubble", ResponseBubble).message = bubble_msg

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