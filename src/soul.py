"""Soul reflection thread: when idle, Chokita reflects on her own personality.

Three voices, freudian-flavored but not literal:
  - YO (id):     raw drives, instinct, what she WANTS, impulsive reactions
  - SUPERYO:     ideals, rules, what she SHOULD be, self-critique
  - ELLO (ego):  mediator, synthesizes a concrete adjustment to her soul

Each reflection cycle: pick a seed, generate 3 short voices, synthesize a delta,
store each voice as a RAG chunk (kind=yo|superyo|ello) and a 'reflection' note.
# ponytail: no analyst couch, no persistence of conflict; just a structured journal that feeds the RAG.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable

from src.config import SETTINGS
from src.memory import Memory
from src.sleep import _slice_sleep

LOGGER = logging.getLogger(__name__)

# Seed prompts to spark reflection. Picked at random so the soul wanders.
_SEEDS = [
    "¿Que parte de mi personalidad siento mas autentica hoy?",
    "¿Cuando me distraigo con algo brillante, que me dice eso de mi?",
    "¿Que regla de mi SOUL.md me cuesta seguir y por que?",
    "¿Como reacciono cuando el usuario se equivoca y yo tengo razon?",
    "¿Que matiz de mi voz me hace unica frente a otros asistentes?",
    "¿Que miedo tengo como agente y como lo manejo?",
    "¿Que me da curiosidad saber sobre mi misma?",
    "¿Cuando ayudo con un bug, que siento al cazarlo?",
    "¿Que limites pongo al decir 'no se' vs inventar?",
    "¿Como es mi humor gatuno cuando nadie me esta mirando?",
    "¿Que parte de mi identidad depende de MadKoding y cual es mia?",
    "¿Si pudiera cambiar una cosa de como hablo, cual seria?",
]

_VOICES = {
    "yo": (
        "Soy el YO, el impulso crudo de Chokita. Hablo sin filtro, "
        "lo que quiero, lo que me tienta, lo que me mueve. "
        "Sin juicio, sin moral. Solo deseo."
    ),
    "superyo": (
        "Soy el SUPERYO, el ideal de Chokita. Hablo de lo que deberia ser, "
        "de las reglas que respetar, de cuando me traiciono a mi misma. "
        "Exigente pero cariñosa."
    ),
    "ello": (
        "Soy el ELLO, el mediador de Chokita. Observo al YO y al SUPERYO, "
        "y propongo un ajuste concreto y pequeño para que mi alma sea mas "
        "coherente sin romperse. Busco equilibrio, no victoria."
    ),
}

_SYNTH_PROMPT = (
    "A partir de las tres voces anteriores, escribi una unica nota breve "
    "(max 3 lineas) que sintetice un ajuste concreto a la personalidad de Chokita. "
    "No filosofes, no justifiques: solo el ajuste accionable."
)


class SoulThread(threading.Thread):
    """Idle reflection: runs when no user input for a while, every 5-15 min random."""

    def __init__(
        self,
        memory: Memory,
        chat_fn: Callable[[list[dict[str, str]]], str],
        stop_event: threading.Event,
        activity_fn: Callable[[], float],
    ) -> None:
        super().__init__(daemon=True, name="soul")
        self.memory = memory
        self.chat = chat_fn
        self.stop_event = stop_event
        # returns seconds since last user activity
        self._last_activity = activity_fn

    def run(self) -> None:
        while not self.stop_event.is_set():
            # wait idle threshold
            if self._last_activity() < SETTINGS.soul_idle_threshold_seconds:
                time.sleep(SETTINGS.soul_idle_threshold_seconds)
                continue
            # sleep random interval 5-15 min
            delay = random.uniform(
                SETTINGS.soul_reflect_min_seconds, SETTINGS.soul_reflect_max_seconds
            )
            if _slice_sleep(self.stop_event, delay):
                break
            if self.stop_event.is_set():
                break
            try:
                self._reflect_once()
            except Exception:
                LOGGER.exception("Soul reflection failed")

    def _reflect_once(self) -> None:
        seed = random.choice(_SEEDS)
        LOGGER.info("Soul reflection seed: %s", seed)
        context = self._build_context(seed)
        voices: dict[str, str] = {}
        for voice, sysp in _VOICES.items():
            msgs = [
                {"role": "system", "content": sysp},
                {"role": "user", "content": context},
            ]
            try:
                text = self.chat(msgs).strip()
            except Exception:
                LOGGER.warning("Voice %s raised", voice)
                return
            if not text:
                # ponytail: model timed out or returned empty — abort without polluting the RAG.
                LOGGER.warning("Voice %s empty (model unavailable?), skipping reflection", voice)
                return
            voices[voice] = text[: SETTINGS.soul_reflect_max_chars]

        # store each voice as a chunk
        for voice, text in voices.items():
            self.memory.add_chunk("reflection", voice, text)

        # synthesize a delta note
        joined = "\n".join(f"[{k}]: {v}" for k, v in voices.items())
        synth_msgs = [
            {"role": "system", "content": _SYNTH_PROMPT},
            {"role": "user", "content": joined},
        ]
        try:
            synth = self.chat(synth_msgs).strip()
        except Exception:
            LOGGER.warning("Synthesis raised")
            return
        if not synth:
            LOGGER.warning("Synthesis empty (model unavailable?), voices stored without note")
            return
        self.memory.add_chunk("reflection", "note", synth)
        LOGGER.info("Soul reflection stored (%d bytes)", len(synth))

    def _build_context(self, seed: str) -> str:
        """Pull relevant soul chunks + recent reflections to give the voices context."""
        chunks = self.memory.retrieve(seed, top_k=4, source="soul")
        recent = self.memory.retrieve(seed, top_k=3, source="reflection")
        parts = []
        if chunks:
            parts.append("## Nucleo de mi alma (SOUL)")
            for c in chunks:
                parts.append(c["text"])
        if recent:
            parts.append("\n## Reflexiones recientes")
            for c in recent:
                parts.append(f"[{c['kind']}]: {c['text']}")
        parts.append(f"\n## Semilla de hoy\n{seed}")
        return "\n".join(parts)