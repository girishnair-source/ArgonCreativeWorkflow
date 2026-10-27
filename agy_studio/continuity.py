"""Shared continuity helpers used by the linter, decomposer, working-set builder and Continuity Supervisor.

* which entity-states are active at a given clip (mid-scene transitions; linter L4, conditioning L6),
* which characters speak in a clip (L15/L16),
* word counting that works for Devanagari (L18 speech-duration estimate),
* entity-token extraction (L3).

``Scene.entity_states_active`` lists the states at the START of the scene; ``mid_scene_transitions`` switch an
entity from ``from_state`` to ``to_state`` from (``at_shot_id``, ``at_clip_index``) onward.
"""

from __future__ import annotations

import re
from typing import Iterable

from .models import GenClip, Scene

# Devanagari block minus the danda punctuation marks (U+0964, U+0965) so "है।" counts as one word.
_WORD_RE = re.compile(r"[\w\u0900-\u0963\u0966-\u097F']+", re.UNICODE)
ENTITY_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]+\b")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def estimate_speech_s(texts: Iterable[str], wps: float) -> float:
    if wps <= 0:
        raise ValueError("speech pace must be > 0 words/second")
    return sum(count_words(t) for t in texts) / wps


def entity_tokens(text: str) -> list[str]:
    """All ``[A-Z][A-Z0-9_]+`` tokens in order of appearance (duplicates removed)."""
    out: list[str] = []
    for tok in ENTITY_TOKEN_RE.findall(text or ""):
        if tok not in out:
            out.append(tok)
    return out


def entity_id_of_state(state_id: str, entity_ids: Iterable[str]) -> str | None:
    """Owning entity of a state id (longest ``<ENTITY>_`` prefix wins)."""
    best: str | None = None
    for eid in entity_ids:
        if state_id.startswith(eid + "_") and (best is None or len(eid) > len(best)):
            best = eid
    return best


# ---------------------------------------------------------------------------------------------
# Entity-state activity
# ---------------------------------------------------------------------------------------------
def all_scene_states(scene: Scene) -> list[str]:
    """Every state that is active somewhere in the scene (start states + transition targets)."""
    out = list(dict.fromkeys(scene.entity_states_active))
    for tr in scene.mid_scene_transitions:
        if tr.to_state not in out:
            out.append(tr.to_state)
    return out


def active_states_at(scene: Scene, shot_order: list[str], shot_id: str, clip_index: int) -> list[str]:
    """States active at clip ``clip_index`` (1-based) of ``shot_id``, applying mid-scene transitions in order."""
    pos = {sid: i for i, sid in enumerate(shot_order)}
    unknown = len(shot_order) + 1
    current = (pos.get(shot_id, unknown), clip_index)
    states = list(dict.fromkeys(scene.entity_states_active))
    ordered = sorted(scene.mid_scene_transitions, key=lambda t: (pos.get(t.at_shot_id, unknown), t.at_clip_index))
    for tr in ordered:
        if (pos.get(tr.at_shot_id, unknown), tr.at_clip_index) <= current:
            states = [s for s in states if s != tr.from_state]
            if tr.to_state not in states:
                states.append(tr.to_state)
    return states


def active_state_for_entity(scene: Scene, shot_order: list[str], shot_id: str, clip_index: int,
                            entity_id: str) -> str | None:
    for st in active_states_at(scene, shot_order, shot_id, clip_index):
        if st.startswith(entity_id + "_"):
            return st
    return None


# ---------------------------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------------------------
def on_screen_speakers(clip: GenClip) -> list[str]:
    return list(dict.fromkeys(ln.character_id for ln in clip.spec.audio_spec.dialogue))


def off_screen_speakers(clip: GenClip) -> list[str]:
    return list(dict.fromkeys(ln.character_id for ln in clip.spec.audio_spec.off_screen))


def speakers_in_clip(clip: GenClip) -> list[str]:
    """Distinct characters with a line in the clip (on-screen and off-screen), in order of first line."""
    return list(dict.fromkeys(ln.character_id for ln in clip.spec.audio_spec.all_lines()))


def clip_line_texts(clip: GenClip) -> list[str]:
    return [ln.text for ln in clip.spec.audio_spec.all_lines()]
