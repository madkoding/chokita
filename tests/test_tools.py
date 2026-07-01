
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