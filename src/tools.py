"""Tools for Chokita: read, grep, glob, write, bash, list. All sandboxed to CHOKITA_WORKDIR.

Ponytail: stdlib only (os, re, subprocess, pathlib). No new deps.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from src.config import SETTINGS

MAX_OUTPUT_BYTES = 20_000


def _safe(path: Path) -> Path:
    p = path if path.is_absolute() else (SETTINGS.workdir / path)
    p = p.resolve()
    try:
        p.relative_to(SETTINGS.workdir)
    except ValueError:
        raise PermissionError(f"Fuera del workdir: {p}") from None
    return p


def _read(path: str, offset: int = 0, limit: int = 200) -> str:
    p = _safe(Path(path))
    if not p.exists() or not p.is_file():
        return f"Error: no existe {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    end = offset + limit
    chunk = lines[offset:end]
    out = "\n".join(f"{i+1}: {line}" for i, line in enumerate(chunk, start=offset))
    if len(lines) > end:
        out += f"\n... ({len(lines) - end} lineas mas)"
    return out[:MAX_OUTPUT_BYTES]


def _list_dir(path: str = ".") -> str:
    p = _safe(Path(path))
    if not p.is_dir():
        return f"Error: no es directorio {path}"
    entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    out = []
    for e in entries:
        kind = "/" if e.is_dir() else ""
        out.append(f"{e.name}{kind}")
    return "\n".join(out) or "(vacio)"


def _glob_files(pattern: str) -> str:
    matches = sorted(SETTINGS.workdir.glob(pattern))
    if not matches:
        return f"Sin resultados para {pattern}"
    out = []
    for m in matches[:500]:
        rel = m.relative_to(SETTINGS.workdir)
        out.append(str(rel))
    return "\n".join(out)


def _grep(pattern: str, include: str = "*") -> str:
    rx = re.compile(pattern)
    hits: list[str] = []
    for root, _dirs, files in os.walk(SETTINGS.workdir):
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


def _write(path: str, content: str) -> str:
    p = _safe(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: {len(content)} bytes -> {path}"


def _bash(command: str) -> str:
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


_TOOLS: dict[str, dict[str, Any]] = {
    "read": {
        "fn": _read,
        "params": {"path": "str", "offset": "int=0", "limit": "int=200"},
        "desc": "Leer un archivo de texto del workdir.",
    },
    "list": {
        "fn": _list_dir,
        "params": {"path": "str=."},
        "desc": "Listar entradas de un directorio del workdir.",
    },
    "glob": {
        "fn": _glob_files,
        "params": {"pattern": "str"},
        "desc": "Buscar archivos por patron glob desde la raiz del workdir.",
    },
    "grep": {
        "fn": _grep,
        "params": {"pattern": "str", "include": "str=*"},
        "desc": "Buscar contenido por regex en archivos del workdir.",
    },
    "write": {
        "fn": _write,
        "params": {"path": "str", "content": "str"},
        "desc": "Escribir/crear un archivo en el workdir.",
    },
    "bash": {
        "fn": _bash,
        "params": {"command": "str"},
        "desc": "Ejecutar un comando shell en el workdir (timeout 30s).",
    },
}


def call_tool(name: str, args: dict[str, Any]) -> str:
    t = _TOOLS.get(name)
    if not t:
        return f"Error: tool desconocida '{name}'"
    try:
        return t["fn"](**args)
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error en tool {name}: {e}"


def tools_system_doc() -> str:
    lines = []
    for name, t in _TOOLS.items():
        params = ", ".join(f"{k}: {v}" for k, v in t["params"].items())
        lines.append(f"- {name}({params}): {t['desc']}")
    return "\n".join(lines)