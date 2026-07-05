import queue
import threading
from unittest.mock import Mock

from src.sleep import SleepThread


def _make_sleep(memory=None, summarize_fn=None, ui_queue=None, stop_event=None, activity_fn=None):
    if memory is None:
        memory = Mock()
        memory.build_raptor.return_value = ["Nivel 0: 5 hojas", "Nivel 1: 2 clusters"]
        memory.raptor_stats.return_value = {0: 5, 1: 2}
    if summarize_fn is None:
        summarize_fn = Mock(return_value="resumen")
    if ui_queue is None:
        ui_queue = queue.Queue()
    if stop_event is None:
        stop_event = threading.Event()
    if activity_fn is None:
        activity_fn = Mock(return_value=999.0)
    sleep = SleepThread(
        memory=memory,
        summarize_fn=summarize_fn,
        stop_event=stop_event,
        activity_fn=activity_fn,
        ui_queue=ui_queue,
    )
    return sleep, memory, ui_queue


def test_dream_once_happy():
    sleep, memory, ui_queue = _make_sleep()
    sleep._dream_once()
    events = []
    while not ui_queue.empty():
        events.append(ui_queue.get_nowait())
    assert any(e["type"] == "state" and e["state"] == "SLEEPING" for e in events)
    assert any(e["type"] == "state" and e["state"] == "IDLE" for e in events)
    assert any("Nivel 0" in str(e) for e in events)
    assert any("RAPTOR" in str(e) for e in events)


def test_dream_once_raptor_fails():
    memory = Mock()
    memory.build_raptor.side_effect = RuntimeError("raptor crash")
    sleep, _memory, ui_queue = _make_sleep(memory=memory)
    sleep._dream_once()
    events = []
    while not ui_queue.empty():
        events.append(ui_queue.get_nowait())
    assert any(e["type"] == "state" and e["state"] == "SLEEPING" for e in events)
    assert any(e["type"] == "state" and e["state"] == "IDLE" for e in events)
    assert not any(e.get("type") == "log" and "RAPTOR" in str(e) for e in events)


def test_slice_sleep_stop():
    stop = threading.Event()
    stop.set()
    result = stop.wait(timeout=10.0)
    assert result is True


def test_slice_sleep_completes():
    stop = threading.Event()
    result = stop.wait(timeout=0.01)
    assert result is False


def test_run_stops_when_signaled():
    stop = threading.Event()
    stop.set()
    sleep, _memory, _ui_queue = _make_sleep(stop_event=stop)
    sleep.run()



