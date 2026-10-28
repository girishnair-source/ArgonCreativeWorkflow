"""Structured LLM adapter for creative/production agents (BUILD_SPEC_v3 §5 ``agent_llm``, §6 agent roles).

In fake mode (``fake=True``, unit tests only), produces deterministic, schema-valid proposal envelopes
tailored to the invoked role and scene working set so agent tests and full scene loops run 100% offline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .. import fsio
from ..registry import ModelSpec
from .base import BaseAdapter, adapter_kwargs, check_kind, gemini_generate_json

_ROLE_RE = re.compile(r"\(([a-z0-9_]+);\s*scope=(scene|global);\s*output_kind=([a-z0-9_]+)\)")
_VARIANTS_RE = re.compile(r"One JSON object with (\d+) proposal variants")
_WS_HEADER_RE = re.compile(r"# SCENE WORKING SET (SC\d{2,4})[^\n]*\n(\{.*?\})\n\n# PREFERENCES", re.DOTALL)


class LlmAdapter(BaseAdapter):
    """Structured JSON LLM adapter for §6 roles (registry kind ``llm``)."""

    agent_name = "agy.adapter.llm"

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.4,
    ) -> dict[str, Any]:
        if callable(self.client):
            out = self.client(system, user, schema=schema, temperature=temperature)
            return dict(out) if isinstance(out, dict) else {"items": out}

        if self.fake:
            out = self._fake_envelope(system, user)
            self.last_result = {"fake": True, "provider_model": self.provider_model()}
            return out

        out = gemini_generate_json(
            self.model_chain(),
            self.spec.location,
            system,
            [user],
            schema=schema,
            temperature=temperature,
            client=self.client,
        )
        self.last_result = {"fake": False, "provider_model": self.provider_model()}
        return out

    def _fake_envelope(self, system: str, user: str) -> dict[str, Any]:
        m_role = _ROLE_RE.search(user)
        role = m_role.group(1) if m_role else "director"
        output_kind = m_role.group(3) if m_role else "proposal"
        m_var = _VARIANTS_RE.search(user)
        n_variants = max(2, min(4, int(m_var.group(1)) if m_var else 3))

        ws_data: dict[str, Any] = {}
        scene_id = "SC01"
        m_ws = _WS_HEADER_RE.search(user)
        if m_ws:
            scene_id = m_ws.group(1)
            try:
                ws_data = json.loads(m_ws.group(2))
            except Exception:  # noqa: BLE001
                ws_data = {}

        proposals: list[dict[str, Any]] = []
        for idx in range(1, n_variants + 1):
            label = chr(64 + idx)
            content = self._fake_content_for(role, output_kind, scene_id, ws_data, idx=idx, label=label, user=user)
            prov = [f"working_set:{scene_id}", f"role:{role}", f"variant:V{idx}"]
            proposals.append(
                {
                    "id": f"V{idx}",
                    "content": content,
                    "rationale": f"Variant V{idx} ({label}) balances dramatic pacing and continuity for {role}.",
                    "provenance": prov,
                }
            )
        digest = fsio.stable_hash([role, scene_id, n_variants], 8)
        return {
            "proposals": proposals,
            "rationale": f"Generated {n_variants} distinct {output_kind} variants for {role} [{digest}].",
            "questions_for_human": [],
        }

    def _fake_content_for(
        self,
        role: str,
        output_kind: str,
        scene_id: str,
        ws_data: dict[str, Any],
        *,
        idx: int,
        label: str,
        user: str,
    ) -> dict[str, Any]:
        scene_obj = ws_data.get("scene") if isinstance(ws_data.get("scene"), dict) else {}
        dialogue = list(scene_obj.get("dialogue") or ws_data.get("dialogue") or [])
        beats = list(scene_obj.get("beats") or ws_data.get("beats") or [])
        states = [p.get("id", "") for p in (ws_data.get("entity_states") or []) if isinstance(p, dict)]
        chars = [s.split("_STATE_")[0] for s in states if "_STATE_" in s] or ["CHAR_RISHI"]
        primary_char = chars[0]
        lens = ("35mm anamorphic, 9:16 vertical", "50mm prime, 9:16 vertical", "28mm wide prime, 9:16 vertical",
                "85mm portrait prime, 9:16 vertical")[(idx - 1) % 4]
        coverage = ("medium_two_shot", "ots_a", "close_up_a", "wide_master")[(idx - 1) % 4]

        if role in ("director", "cinematographer", "storyboard_artist", "blocking_choreographer") or "shot" in output_kind:
            alt_match = re.search(r"alternative versions of shot (SC\d+_SH\d+)", user)
            if alt_match:
                target_shot_id = alt_match.group(1)
                return {
                    "label": f"A{idx}",
                    "summary": f"Alternative {label} using {lens}",
                    "shots": [
                        self._make_fake_shot(
                            scene_id,
                            shot_id=target_shot_id,
                            order=1,
                            primary_char=primary_char,
                            chars=chars,
                            lines=[ln.get("line_id") for ln in dialogue[:1] if isinstance(ln, dict) and ln.get("line_id")],
                            beats=[b.get("beat_id") or b.get("id") for b in beats[:1] if isinstance(b, dict) and (b.get("beat_id") or b.get("id"))],
                            lens=lens,
                            coverage=coverage,
                        )
                    ],
                }
            shots: list[dict[str, Any]] = []
            if dialogue:
                for k, ln in enumerate(dialogue, start=1):
                    lid = ln.get("line_id") or f"{scene_id}_L{k:03d}"
                    spk = ln.get("character_id") or primary_char
                    beat_ids = [b.get("beat_id") or b.get("id") for b in beats if isinstance(b, dict) and (b.get("beat_id") or b.get("id"))][:1]
                    shots.append(
                        self._make_fake_shot(
                            scene_id,
                            shot_id=f"{scene_id}_SH{k:02d}",
                            order=k,
                            primary_char=spk,
                            chars=chars,
                            lines=[lid],
                            beats=beat_ids,
                            lens=lens,
                            coverage=coverage if k == 1 else "ots_a",
                        )
                    )
            else:
                shots.append(
                    self._make_fake_shot(
                        scene_id,
                        shot_id=f"{scene_id}_SH01",
                        order=1,
                        primary_char=primary_char,
                        chars=chars,
                        lines=[],
                        beats=[b.get("beat_id") or b.get("id") for b in beats[:1] if isinstance(b, dict) and (b.get("beat_id") or b.get("id"))],
                        lens=lens,
                        coverage=coverage,
                    )
                )
            return {
                "label": label,
                "summary": f"Strategy {label}: {len(shots)} shot(s) with {lens}",
                "shots": shots,
            }

        if role in ("voice_director", "sound_designer", "composer") or "audio" in output_kind:
            deliveries = {
                str(ln.get("line_id")): f"tense, natural pace (variant {label})"
                for ln in dialogue
                if isinstance(ln, dict) and ln.get("line_id")
            }
            cues = [
                {
                    "cue_id": f"{scene_id}_CUE01",
                    "start_s": 0.0,
                    "duration_s": max(6.0, 6.0 * max(1, len(dialogue))),
                    "prompt": f"Subtle atmospheric thriller pulse for {scene_id} ({label}), instrumental only.",
                }
            ]
            return {"deliveries": deliveries, "cues": cues}

        return {
            "label": label,
            "role": role,
            "output_kind": output_kind,
            "scene_id": scene_id,
            "summary": f"Proposed {output_kind} option {label} for {scene_id}",
        }

    @staticmethod
    def _make_fake_shot(
        scene_id: str,
        *,
        shot_id: str,
        order: int,
        primary_char: str,
        chars: list[str],
        lines: list[str],
        beats: list[str],
        lens: str,
        coverage: str,
    ) -> dict[str, Any]:
        others = [c for c in chars if c != primary_char]
        comp_extra = f" with {', '.join(others)} silent in foreground" if others else ""
        return {
            "id": shot_id,
            "scene_id": scene_id,
            "order": order,
            "intention": f"Capture {primary_char} beat in {coverage}",
            "coverage_type": coverage,
            "duration_s": 6.0,
            "environment_angle": "wide_establishing",
            "camera_side": "south",
            "beat_ids": beats,
            "actions": [f"{primary_char} holds gaze steadily"],
            "lines": lines,
            "line_action_index": {lid: 0 for lid in lines},
            "continuous": True,
            "spec": {
                "visual": {
                    "subject_and_action": f"{primary_char} holds gaze steadily{comp_extra}",
                    "camera": "steady locked-off medium framing",
                    "lens_and_aspect": lens,
                    "lighting": "practical sodium vapour streetlamp key from camera left",
                    "palette": "deep slate blue, warm amber highlights, muted charcoal shadows",
                    "composition": f"{primary_char} centered on upper third{comp_extra}",
                    "atmosphere": "humid night air with faint rooftop haze",
                    "mood": "tense, watchful suspense",
                    "attribution": "Directed for vertical cinema, 24fps",
                },
                "audio_spec": {
                    "dialogue": [],
                    "off_screen": [],
                    "ambience": "quiet night rooftop wind and distant city hum",
                    "diegetic_sfx": ["cloth rustle"],
                    "music": "none",
                },
            },
        }


def create_adapter(spec: ModelSpec, **kw: Any) -> LlmAdapter:
    check_kind(spec, "llm")
    return LlmAdapter(spec, **adapter_kwargs(kw))
