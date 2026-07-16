"""Tools for Chokita: read, grep, glob, write, bash, list. All sandboxed to CHOKITA_WORKDIR.

Ponytail: stdlib only (os, re, shlex, subprocess, pathlib). No new deps.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from src.config import SETTINGS

MAX_OUTPUT_BYTES = 20_000

_DENY_NAMES: frozenset[str] = frozenset(
    {
        "SOUL.md",
        ".env",
        "Dockerfile",
        "docker-compose.yml",
        "AGENTS.md",
        "pyproject.toml",
    }
)
_WRITE_DENY_DIRS: frozenset[str] = frozenset({"src", "tests", "scripts", ".git"})
_BASH_WHITELIST: frozenset[str] = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "echo",
        "pwd",
        "test",
        "true",
        "false",
        "sort",
        "uniq",
        "tr",
        "cut",
        "tee",
        "diff",
        "stat",
        "file",
        "which",
    }
)


def _safe(path: Path) -> Path:
    # ponytail: _safe = sand box. reject ~, .., deny names; resolve to absolute.
    if any(part == "~" for part in path.parts):
        raise PermissionError(f"path con ~ no permitido: {path}")
    p = path if path.is_absolute() else (SETTINGS.workdir / path)
    p = p.resolve()
    try:
        p.relative_to(SETTINGS.workdir)
    except ValueError:
        raise PermissionError(f"Fuera del workdir: {p}") from None
    return p


def _safe_pattern(pattern: str) -> str:
    # ponytail: glob/grep patterns no pueden escapar del workdir.
    if pattern.startswith(("/", "~")) or ".." in Path(pattern).parts:
        raise PermissionError(f"patron fuera del workdir: {pattern}")
    return pattern


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
    _safe_pattern(pattern)
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
    if p.name in _DENY_NAMES:
        return f"Error: '{p.name}' es read-only"
    if any(seg in _WRITE_DENY_DIRS for seg in p.relative_to(SETTINGS.workdir).parts[:-1]):
        return f"Error: no se puede escribir en {p.parent.relative_to(SETTINGS.workdir)}/"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: {len(content)} bytes -> {path}"


def _bash(command: str) -> str:
    # ponytail: shell=False + whitelist. sin esto el LLM rompe el sandbox.
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return f"Error: parseo de comando: {e}"
    if not tokens:
        return "Error: comando vacio"
    binary = Path(tokens[0]).name
    if binary not in _BASH_WHITELIST:
        return f"Error: binario '{binary}' no permitido. whitelist: {sorted(_BASH_WHITELIST)}"
    try:
        r = subprocess.run(
            tokens,
            shell=False,
            cwd=str(SETTINGS.workdir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return out[:MAX_OUTPUT_BYTES] if out else f"(rc={r.returncode}, sin salida)"
    except subprocess.TimeoutExpired:
        return "Error: timeout 30s"


_TOOLS: dict[str, Any] = {
    "read": _read,
    "list": _list_dir,
    "glob": _glob_files,
    "grep": _grep,
    "write": _write,
    "bash": _bash,
}

TOOLS_DOC = """\
- read(path: str, offset: int=0, limit: int=200): Leer un archivo de texto del workdir.
- list(path: str=.): Listar entradas de un directorio del workdir.
- glob(pattern: str): Buscar archivos por patron glob desde la raiz del workdir.
- grep(pattern: str, include: str=*): Buscar contenido por regex en archivos del workdir.
- write(path: str, content: str): Escribir/crear un archivo (deny: SOUL.md, .env, src/, tests/).
- bash(command: str): Ejecutar comando (whitelist: ls, cat, head, tail, grep, find, etc.)."""


def call_tool(name: str, args: dict[str, Any]) -> str:
    fn = _TOOLS.get(name)
    if not fn:
        return f"Error: tool desconocida '{name}'"
    try:
        return fn(**args)
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error en tool {name}: {e}"
