import threading
from unittest.mock import Mock, patch

from src.soul import SoulThread


def _make_soul(chat_fn=None, memory=None, stop_event=None):
    if memory is None:
        memory = Mock()
        memory.retrieve.return_value = []
        memory.add_chunk = Mock()
    if chat_fn is None:
        chat_fn = Mock(return_value="texto de voz")
    if stop_event is None:
        stop_event = Mock()
        stop_event.is_set.return_value = False
    activity_fn = Mock(return_value=999.0)
    soul = SoulThread(memory=memory, chat_fn=chat_fn, stop_event=stop_event, activity_fn=activity_fn)
    return soul, memory, chat_fn


_VOICES_REPLY = "[YO]: texto del yo\n[SUPERYO]: texto del superyo\n[ELLO]: texto del ello"


@patch("random.choice", return_value="¿Que parte de mi personalidad siento mas autentica hoy?")
def test_reflect_once_happy(_mock_choice):
    chat_fn = Mock(side_effect=[_VOICES_REPLY, "nota de sintesis"])
    soul, memory, _chat_fn = _make_soul(chat_fn=chat_fn)
    soul._reflect_once()
    assert memory.add_chunk.call_count == 4
    calls = [c.args for c in memory.add_chunk.call_args_list]
    kinds = [c[1] for c in calls]
    assert "yo" in kinds
    assert "superyo" in kinds
    assert "ello" in kinds
    assert "note" in kinds


@patch("random.choice", return_value="¿Que parte de mi personalidad siento mas autentica hoy?")
def test_reflect_once_empty_voice_aborts(_mock_choice):
    chat_fn = Mock(return_value="")
    soul, memory, _chat_fn = _make_soul(chat_fn=chat_fn)
    soul._reflect_once()
    assert memory.add_chunk.call_count == 0


@patch("random.choice", return_value="¿Que parte de mi personalidad siento mas autentica hoy?")
def test_reflect_once_chat_raises_aborts(_mock_choice):
    chat_fn = Mock(side_effect=RuntimeError("ollama down"))
    soul, memory, _chat_fn = _make_soul(chat_fn=chat_fn)
    soul._reflect_once()
    assert memory.add_chunk.call_count == 0


@patch("random.choice", return_value="¿Que parte de mi personalidad siento mas autentica hoy?")
def test_reflect_once_empty_synth_stores_voices(_mock_choice):
    chat_fn = Mock(side_effect=[_VOICES_REPLY, ""])
    soul, memory, _chat_fn = _make_soul(chat_fn=chat_fn)
    soul._reflect_once()
    assert memory.add_chunk.call_count == 3
    calls = [c.args for c in memory.add_chunk.call_args_list]
    kinds = [c[1] for c in calls]
    assert "note" not in kinds


def test_build_context_with_chunks():
    memory = Mock()
    memory.retrieve.side_effect = [
        [{"source": "soul", "kind": "soul", "text": "Soy Chokita, una agente neko."}],
        [{"source": "reflection", "kind": "yo", "text": "Quiero jugar."}],
    ]
    soul, _memory, _chat_fn = _make_soul(memory=memory)
    ctx = soul._build_context("¿Que parte de mi personalidad siento mas autentica hoy?")
    assert "Nucleo" in ctx
    assert "Reflexiones" in ctx
    assert "Semilla" in ctx
    assert "Soy Chokita" in ctx
    assert "Quiero jugar" in ctx


def test_build_context_empty():
    memory = Mock()
    memory.retrieve.return_value = []
    soul, _memory, _chat_fn = _make_soul(memory=memory)
    ctx = soul._build_context("¿Que parte de mi personalidad siento mas autentica hoy?")
    assert "Nucleo" not in ctx
    assert "Reflexiones" not in ctx
    assert "Semilla" in ctx


@patch("random.choice", return_value="¿Que parte de mi personalidad siento mas autentica hoy?")
def test_reflect_once_less_than_3_voices(_mock_choice):
    chat_fn = Mock(return_value="[YO]: solo yo\n[SUPERYO]: solo superyo")
    soul, memory, _ = _make_soul(chat_fn=chat_fn)
    soul._reflect_once()
    assert memory.add_chunk.call_count == 0


@patch("random.choice", return_value="¿Que parte de mi personalidad siento mas autentica hoy?")
def test_reflect_once_synth_raises(_mock_choice):
    chat_fn = Mock(side_effect=[
        "[YO]: yo\n[SUPERYO]: superyo\n[ELLO]: ello",
        RuntimeError("synth crash"),
    ])
    soul, memory, _ = _make_soul(chat_fn=chat_fn)
    soul._reflect_once()
    assert memory.add_chunk.call_count == 3
    calls = [c.args[1] for c in memory.add_chunk.call_args_list]
    assert "note" not in calls


def test_run_stops_immediately():
    stop = threading.Event()
    stop.set()
    soul, _, _ = _make_soul(stop_event=stop)
    soul.run()

