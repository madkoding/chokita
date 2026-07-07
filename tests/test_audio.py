from src.stt_subprocess import is_stop_command, parse_wake


def test_wake_exact():
    assert parse_wake("chokita abre vscode") == (True, "abre vscode")


def test_wake_alias_choquita():
    assert parse_wake("choquita que hora es") == (True, "que hora es")


def test_wake_alias_chiqui():
    assert parse_wake("chiqui decime algo") == (True, "decime algo")


def test_wake_no_remainder():
    assert parse_wake("chokita") == (True, "")


def test_wake_none():
    assert parse_wake("hola como andas") == (False, "")


def test_wake_middle():
    assert parse_wake("eh chokita para un segundo") == (True, "para un segundo")


def test_wake_uppercase():
    assert parse_wake("CHOKITA abre") == (True, "abre")


def test_wake_accented_alias():
    assert parse_wake("chiquitita ayuda") == (True, "ayuda")


def test_stop_para_accent():
    assert is_stop_command("pará") is True


def test_stop_para_no_accent():
    assert is_stop_command("para") is True


def test_stop_callate():
    assert is_stop_command("cállate") is True


def test_stop_silencio():
    assert is_stop_command("silencio") is True


def test_stop_none():
    assert is_stop_command("abre vscode") is False
