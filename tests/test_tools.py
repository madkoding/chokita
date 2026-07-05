import subprocess
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("CHOKITA_WORKDIR", str(tmp_path))
    import importlib

    import src.config
    importlib.reload(src.config)
    import src.tools as tools_mod
    importlib.reload(tools_mod)
    yield tmp_path


def test_write_read_list(workdir):
    from src.tools import _TOOLS, call_tool
    assert "read" in _TOOLS and "write" in _TOOLS and "list" in _TOOLS
    out = call_tool("write", {"path": "foo.txt", "content": "hola mundo\nlinea 2"})
    assert "OK" in out
    out = call_tool("read", {"path": "foo.txt"})
    assert "hola mundo" in out
    out = call_tool("list", {"path": "."})
    assert "foo.txt" in out


def test_grep(workdir):
    from src.tools import call_tool
    call_tool("write", {"path": "a.py", "content": "def foo():\n  return 1\n"})
    out = call_tool("grep", {"pattern": "foo", "include": "*.py"})
    assert "a.py" in out and "foo" in out


def test_glob(workdir):
    from src.tools import call_tool
    call_tool("write", {"path": "x.txt", "content": ""})
    call_tool("write", {"path": "y.md", "content": ""})
    out = call_tool("glob", {"pattern": "*.md"})
    assert "y.md" in out and "x.txt" not in out


def test_bash(workdir):
    from src.tools import call_tool
    out = call_tool("bash", {"command": "echo hola"})
    assert "hola" in out


def test_read_escape_workdir(workdir):
    from src.tools import call_tool
    out = call_tool("read", {"path": "../../../etc/passwd"})
    assert "Error" in out


def test_read_no_such_file(workdir):
    from src.tools import call_tool
    out = call_tool("read", {"path": "noexiste.txt"})
    assert "Error" in out


def test_read_truncated(workdir):
    from src.tools import call_tool
    content = "\n".join(f"line {i}" for i in range(300))
    call_tool("write", {"path": "big.txt", "content": content})
    out = call_tool("read", {"path": "big.txt", "limit": 10})
    assert "... (290 lineas mas)" in out


def test_list_not_a_directory(workdir):
    from src.tools import call_tool
    call_tool("write", {"path": "file.txt", "content": "hola"})
    out = call_tool("list", {"path": "file.txt"})
    assert "Error" in out


def test_glob_no_matches(workdir):
    from src.tools import call_tool
    out = call_tool("glob", {"pattern": "*.nonexistent"})
    assert "Sin resultados" in out


def test_grep_skip_git_dir(workdir):
    from src.tools import call_tool
    git_dir = workdir / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("ref = HEAD", encoding="utf-8")
    out = call_tool("grep", {"pattern": "HEAD"})
    assert "Sin resultados" in out


def test_grep_include_filter(workdir):
    from src.tools import call_tool
    call_tool("write", {"path": "data.txt", "content": "secret"})
    call_tool("write", {"path": "data.py", "content": "secret"})
    out = call_tool("grep", {"pattern": "secret", "include": "*.py"})
    assert "data.py" in out


def test_grep_hit_limit(workdir):
    from src.tools import call_tool
    for i in range(5):
        call_tool("write", {"path": f"f{i}.py", "content": "# match\n" * 50})
    out = call_tool("grep", {"pattern": "match", "include": "*.py"})
    assert "limite 200" in out


def test_bash_timeout(workdir):
    from src.tools import call_tool
    with patch("src.tools.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
        out = call_tool("bash", {"command": "sleep 100"})
        assert "timeout" in out


def test_call_tool_unknown(workdir):
    from src.tools import call_tool
    out = call_tool("nonexistent", {})
    assert "desconocida" in out


def test_call_tool_exception(workdir):
    from src.tools import call_tool
    out = call_tool("read", {"offset": "not_an_int"})
    assert "Error en tool" in out


def test_grep_binary_file(workdir):
    from src.tools import call_tool
    (workdir / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")
    (workdir / "text.py").write_text("match found", encoding="utf-8")
    out = call_tool("grep", {"pattern": "match"})
    assert "text.py" in out