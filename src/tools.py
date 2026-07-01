"""Tools for Chokita: read, grep, glob, write, bash, list. All sandboxed to CHOKITA_WORKDIR.

Ponytail: stdlib only (os, re, subprocess, pathlib). No new deps.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from src.config import SETTINGS

# Hard ceiling on bytes returned by read/bash to keep context bounded.
MAX_OUTPUT_BYTES = 20_000


def _safe(path: Path) -> Path:
    """Resolve and clamp path under workdir."""
    p = path if path.is_absolute() else (SETTINGS.workdir / path)
    p = p.resolve()
    try:
        p.relative_to(SETTINGS.workdir)
    except ValueError:
        raise PermissionError(f"Fuera del workdir: {p}")
    return p


TOOLS: dict[str, dict[str, Any]] = {}


def tool(name: str, desc: str, schema: dict[str, Any]) -> Callable[..., str]:
    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        TOOLS[name] = {"name": name, "description": desc, "schema": schema, "fn": fn}
        return fn

    return deco


@tool(
    "read",
    "Leer un archivo de texto del workdir.",
    {"path": "str", "offset": "int=0", "limit": "int=200"},
)
def read(path: str, offset: int = 0, limit: int = 200) -> str:
    p = _safe(Path(path))
    if not p.exists() or not p.is_file():
        return f"Error: no existe {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    end = offset + limit
    chunk = lines[offset:end]
    out = "\n".join(f"{i+1}: {l}" for i, l in enumerate(chunk, start=offset))
    if len(lines) > end:
        out += f"\n... ({len(lines) - end} lineas mas)"
    return out[:MAX_OUTPUT_BYTES]


@tool(
    "list",
    "Listar entradas de un directorio del workdir.",
    {"path": "str=."},
)
def list_dir(path: str = ".") -> str:
    p = _safe(Path(path))
    if not p.is_dir():
        return f"Error: no es directorio {path}"
    entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    out = []
    for e in entries:
        kind = "/" if e.is_dir() else ""
        out.append(f"{e.name}{kind}")
    return "\n".join(out) or "(vacio)"


@tool(
    "glob",
    "Buscar archivos por patron glob desde la raiz del workdir.",
    {"pattern": "str"},
)
def glob_files(pattern: str) -> str:
    matches = sorted(SETTINGS.workdir.glob(pattern))
    if not matches:
        return f"Sin resultados para {pattern}"
    out = []
    for m in matches[:500]:
        rel = m.relative_to(SETTINGS.workdir)
        out.append(f"{rel}")
    return "\n".join(out)


@tool(
    "grep",
    "Buscar contenido por regex en archivos del workdir.",
    {"pattern": "str", "include": "str=*"},
)
def grep(pattern: str, include: str = "*") -> str:
    rx = re.compile(pattern)
    hits: list[str] = []
    for root, _dirs, files in os.walk(SETTINGS.workdir):
        # skip common noise dirs
        if any(seg in {".git", "__pycache__", "node_modules", ".venv"} for seg in Path(root).parts):
            continue
        for fname in files:
            if not fnmatch.fnmatch(fname, include):
                continue
            fpath = Path(root) / fname
            try:
                for i, line in enumerate(fpath.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        rel = fpath.relative_to(SETTINGS.workdir)
                        hits.append(f"{rel}:{i}: {line[:200]}")
                        if len(hits) >= 200:
                            return "\n".join(hits) + "\n... (limite 200)"
            except (OSError, UnicodeDecodeError):
                continue
    return "\n".join(hits) if hits else "Sin resultados"


@tool(
    "write",
    "Escribir/crear un archivo en el workdir.",
    {"path": "str", "content": "str"},
)
def write(path: str, content: str) -> str:
    p = _safe(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: {len(content)} bytes -> {path}"


@tool(
    "bash",
    "Ejecutar un comando shell en el workdir (timeout 30s).",
    {"command": "str"},
)
def bash(command: str) -> str:
    # ponytail: subprocess with cwd clamp + timeout; no shell injection hardening beyond cwd.
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=str(SETTINGS.workdir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return out[:MAX_OUTPUT_BYTES] if out else "(sin salida)"
    except subprocess.TimeoutExpired:
        return "Error: timeout 30s"


def call_tool(name: str, args: dict[str, Any]) -> str:
    t = TOOLS.get(name)
    if not t:
        return f"Error: tool desconocida '{name}'"
    try:
        return t["fn"](**args)
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:  # noqa: BLE001
        return f"Error en tool {name}: {e}"


def tools_system_doc() -> str:
    """Compact description of all tools for the LLM system prompt."""
    lines = []
    for name, t in TOOLS.items():
        params = ", ".join(f"{k}: {v}" for k, v in t["schema"].items())
        lines.append(f"- {name}({params}): {t['description']}")
    return "\n".join(lines)