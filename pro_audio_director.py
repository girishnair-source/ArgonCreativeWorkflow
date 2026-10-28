"""Pro Audio Stem Generator (`pro_audio_director.py`) — Vertex AI Lyria-002 Music & Omni 1.1 360p Foley/SFX.

Generates real AI background scores (via Google Cloud Vertex AI `lyria-002`) and
realistic environmental Foley/SFX stems (via Gemini Omni 1.1 Flash `360p` blank/dark
video generation + audio extraction, with high-fidelity multi-layered acoustic fallback).
"""
from __future__ import annotations

import base64
import math
import os
import random
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any

import certifi
import httpx
from google import genai
from google.genai import interactions

from schemas import ASSET_DIR

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())



def _vertex_project() -> str:
    """Vertex project from the env or the active gcloud config — never a hidden default.

    A missing project raises ``agy_studio.adapters.auth.AuthError`` with the one command that sets it.
    """
    from agy_studio.adapters import auth  # noqa: PLC0415 - stdlib-only module shared with the AGY adapters

    return auth.vertex_project()

VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
OMNI_LOCATION = os.environ.get("OMNI_LOCATION", "global")
OMNI_MODEL_ID = "gemini-omni-1.1-flash-preview"


def _gcloud_token() -> str:
    """Fresh gcloud access token; raises ``agy_studio.adapters.auth.AuthError`` with the real gcloud error + fix."""
    from agy_studio.adapters import auth  # noqa: PLC0415 - stdlib-only module shared with the AGY adapters

    return auth.gcloud_access_token(auth.preferred_account())


def generate_lyria_music_stem(
    prompt: str,
    target_wav: Path,
    duration_seconds: float = 75.0,
    negative_prompt: str = "vocals, singing, speech, spoken words, cheering, clapping",
    force: bool = False,
) -> Path:
    """Generate a real AI background music stem via Vertex AI `lyria-002` and loop/crossfade to `duration_seconds`."""
    marker_file = target_wav.with_suffix(".lyria.ok")
    if not force and target_wav.exists() and marker_file.exists() and target_wav.stat().st_size > 500_000:
        return target_wav

    target_wav.parent.mkdir(parents=True, exist_ok=True)
    raw_lyria_wav = target_wav.parent / f".raw_lyria_{target_wav.name}"

    try:
        project = _vertex_project()
        token = _gcloud_token()
        url = (
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/"
            f"{project}/locations/{VERTEX_LOCATION}/publishers/google/models/lyria-002:predict"
        )
        body: dict[str, Any] = {
            "instances": [{"prompt": prompt, "negative_prompt": negative_prompt}],
            "parameters": {"sample_count": 1},
        }
        print(f"[Lyria-002] Generating AI music score -> {target_wav.name} ...")
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "x-goog-user-project": project,
            },
            json=body,
            timeout=120.0,
        )
        resp.raise_for_status()
        preds = resp.json().get("predictions", [])
        if preds and preds[0].get("bytesBase64Encoded"):
            raw_bytes = base64.b64decode(preds[0]["bytesBase64Encoded"])
            raw_lyria_wav.write_bytes(raw_bytes)
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin:
                fade_out_start = max(1.0, duration_seconds - 2.5)
                subprocess.run(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-stream_loop",
                        "-1",  # loop the ~32.8 s Lyria clip as often as needed; -t below trims to the scene length
                        "-i",
                        str(raw_lyria_wav),
                        "-t",
                        f"{duration_seconds:.2f}",
                        "-af",
                        f"afade=t=in:st=0:d=1.2,afade=t=out:st={fade_out_start:.2f}:d=2.5,volume=0.85",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "48000",
                        "-ac",
                        "2",
                        str(target_wav),
                    ],
                    check=True,
                    capture_output=True,
                )
                raw_lyria_wav.unlink(missing_ok=True)
                marker_file.write_text("lyria-002\n", encoding="utf-8")
                print(f"[Lyria-002] Saved full-episode score: {target_wav.name} ({target_wav.stat().st_size / 1048576:.2f} MB)")
                _ledger_log("music_generation", target_wav, model="lyria-002", name=target_wav.name, status="completed",
                            api_calls=1, audio_seconds_generated=32.8, stem_seconds=round(float(duration_seconds), 2))
                return target_wav
    except Exception as exc:
        print(f"[Lyria-002] Warning: {exc}")
        _ledger_log("music_generation", target_wav, model="lyria-002", name=target_wav.name, status="failed",
                    api_calls=1, error=str(exc)[:200])
    finally:
        if raw_lyria_wav.exists():
            raw_lyria_wav.unlink(missing_ok=True)

    return synthesize_cinematic_sfx_wav(target_wav, duration_seconds=duration_seconds, sfx_prompt=prompt)


def _ledger_log(event: str, target: Path, **fields: Any) -> None:
    """Best-effort entry in the project's production ledger (06_production_logs); never raises."""
    try:
        import production_ledger  # noqa: PLC0415 - optional, lives at the workspace root

        project = production_ledger.project_from_path(target)
        if project:
            production_ledger.safe_log(event, project=project, **fields)
    except Exception as exc:  # noqa: BLE001
        print(f"[ledger] skipped ({exc})")


def generate_omni_360p_sfx_batch(
    jobs: list[tuple[str, Path, int]],
    *,
    force: bool = False,
    poll_timeout_seconds: int = 180,
) -> dict[str, Path]:
    """Submit multiple blank 360p Omni 1.1 Flash SFX jobs concurrently (`background=True`), poll, and extract WAVs."""
    import time
    from google.oauth2.credentials import Credentials

    results: dict[str, Path] = {}
    pending: list[tuple[str, str, Path, int]] = []  # (interaction_id, prompt, target_wav, dur)

    project = _vertex_project()
    creds = Credentials(token=_gcloud_token(), quota_project_id=project)
    from agy_studio.adapters.base import genai_http_options  # noqa: PLC0415 - hard request timeout

    client = genai.Client(
        enterprise=True,
        project=project,
        location=OMNI_LOCATION,
        credentials=creds,
        http_options=genai_http_options("video"),
    )

    for sfx_prompt, target_wav, dur_sec in jobs:
        marker_file = target_wav.with_suffix(".omni_sfx.ok")
        if not force and target_wav.exists() and marker_file.exists() and target_wav.stat().st_size > 150_000:
            results[target_wav.name] = target_wav
            continue
        target_wav.parent.mkdir(parents=True, exist_ok=True)
        dur = max(4, min(10, int(round(dur_sec))))
        prompt = (
            "Minimal dark abstract studio soundstage card, pitch-black atmospheric background with subtle dust motes. "
            f"Pure foley sound effects and environmental audio only, zero speech, zero vocals: {sfx_prompt}"
        )
        try:
            inter = client.interactions.create(
                model=OMNI_MODEL_ID,
                input=prompt,
                generation_config=interactions.GenerationConfig(
                    video_config=interactions.VideoConfig(task="text_to_video")
                ),
                response_format=interactions.VideoResponseFormat(
                    aspect_ratio="9:16",
                    duration=f"{dur}s",
                    resolution="360p",
                ),
                background=True,
            )
            print(f"[Omni 1.1 360p SFX] Submitted {target_wav.name} ({dur}s) -> {inter.id}")
            pending.append((str(inter.id), sfx_prompt, target_wav, dur))
        except Exception as exc:
            print(f"[Omni 1.1 360p SFX] Submit fallback for {target_wav.name}: {str(exc)[:120]}")
            results[target_wav.name] = synthesize_cinematic_sfx_wav(target_wav, duration_seconds=float(dur_sec), sfx_prompt=sfx_prompt)

    started = time.time()
    while pending and (time.time() - started) < poll_timeout_seconds:
        time.sleep(12.0)
        still: list[tuple[str, str, Path, int]] = []
        for inter_id, sfx_prompt, target_wav, dur in pending:
            try:
                inter = client.interactions.get(id=inter_id)
            except Exception:
                still.append((inter_id, sfx_prompt, target_wav, dur))
                continue
            status_str = str(getattr(inter, "status", "") or "").lower()
            if status_str in ("in_progress", "running", "pending", "queued", "processing", ""):
                still.append((inter_id, sfx_prompt, target_wav, dur))
                continue
            contents: list[Any] = []
            for step in getattr(inter, "steps", []) or []:
                if getattr(step, "type", "") == "model_output":
                    contents.extend(getattr(step, "content", []) or [])
            tmp_mp4 = target_wav.with_suffix(".omni360p_sfx.mp4")
            try:
                if status_str == "completed" and contents and getattr(contents[0], "data", None):
                    data_val = contents[0].data
                    raw_bytes = base64.b64decode(data_val) if isinstance(data_val, str) else bytes(data_val)
                    tmp_mp4.write_bytes(raw_bytes)
                    ffmpeg_bin = shutil.which("ffmpeg")
                    if ffmpeg_bin:
                        subprocess.run(
                            [
                                ffmpeg_bin,
                                "-y",
                                "-i",
                                str(tmp_mp4),
                                "-vn",
                                "-acodec",
                                "pcm_s16le",
                                "-ar",
                                "48000",
                                "-ac",
                                "2",
                                str(target_wav),
                            ],
                            check=True,
                            capture_output=True,
                        )
                        target_wav.with_suffix(".omni_sfx.ok").write_text("omni-360p\n", encoding="utf-8")
                        print(f"[Omni 1.1 360p SFX] Extracted WAV -> {target_wav.name} ({target_wav.stat().st_size // 1024} KB)")
                        results[target_wav.name] = target_wav
                        _ledger_log_sfx(target_wav, inter_id, dur, inter, "completed")
                        continue
            except Exception as exc:
                print(f"[Omni 1.1 360p SFX] Extract error for {target_wav.name}: {exc}")
            finally:
                if tmp_mp4.exists():
                    tmp_mp4.unlink(missing_ok=True)
            _ledger_log_sfx(target_wav, inter_id, dur, inter, "filtered" if status_str == "completed" else "failed")
            results[target_wav.name] = synthesize_cinematic_sfx_wav(target_wav, duration_seconds=float(dur), sfx_prompt=sfx_prompt)
        pending = still

    for _inter_id, sfx_prompt, target_wav, dur in pending:
        _ledger_log_sfx(target_wav, _inter_id, dur, None, "failed")
        results[target_wav.name] = synthesize_cinematic_sfx_wav(target_wav, duration_seconds=float(dur), sfx_prompt=sfx_prompt)
    return results


def _ledger_log_sfx(target_wav: Path, inter_id: str, dur: int, inter: Any, status: str) -> None:
    """Production-ledger entry for one 360p Omni SFX video, with the API's real token usage when available."""
    usage = None
    try:
        import production_ledger  # noqa: PLC0415

        usage = production_ledger.usage_dict(inter) if inter is not None else None
    except Exception:  # noqa: BLE001
        usage = None
    fields: dict[str, Any] = {
        "model": "omni-1.1-flash-360p-sfx", "provider_model": OMNI_MODEL_ID, "purpose": "sfx_360p",
        "name": target_wav.name, "operation": inter_id, "status": status, "resolution": "360p",
        "duration_s": float(dur), "duration_source": "requested", "reshoot": False,
    }
    if usage:
        fields.update(input_tokens=int(usage.get("total_input_tokens", 0)), output_tokens=int(usage.get("total_output_tokens", 0)),
                      thought_tokens=int(usage.get("total_thought_tokens", 0)), tokens_source="api")
        fields["total_tokens"] = fields["input_tokens"] + fields["output_tokens"] + fields["thought_tokens"]
    _ledger_log("video_generation", target_wav, **fields)


def synthesize_cinematic_sfx_wav(
    target_path: Path,
    duration_seconds: float = 8.0,
    sfx_prompt: str = "",
) -> Path:
    """Synthesize a rich multi-layered stereo foley/SFX stem (wind, heartbeat, footsteps, scooty, phone ring)."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 48000
    total_samples = int(sample_rate * duration_seconds)
    p_low = sfx_prompt.lower()
    rng = random.Random(hash(sfx_prompt) & 0xFFFFFFFF)

    has_wind = any(k in p_low for k in ("wind", "terrace", "rooftop", "breeze", "night"))
    has_heartbeat = any(k in p_low for k in ("heart", "pulse", "thump", "bass", "dread"))
    has_footsteps = any(k in p_low for k in ("footstep", "boot", "sprint", "run", "sneaker", "concrete"))
    has_scooty = any(k in p_low for k in ("scooty", "engine", "traffic", "street", "exhaust"))
    has_ring = any(k in p_low for k in ("ring", "buzz", "phone", "call", "chime", "hum", "glow", "drone"))

    with wave.open(str(target_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        lp_l = 0.0
        lp_r = 0.0
        for i in range(total_samples):
            t = i / sample_rate
            val_l = 0.0
            val_r = 0.0

            # 1. Filtered pink/brown wind ambience
            if has_wind or not (has_heartbeat or has_footsteps or has_scooty or has_ring):
                n_l = rng.uniform(-1.0, 1.0)
                n_r = rng.uniform(-1.0, 1.0)
                gust = 0.55 + 0.45 * math.sin(2.0 * math.pi * 0.35 * t)
                lp_l = 0.985 * lp_l + 0.015 * n_l
                lp_r = 0.985 * lp_r + 0.015 * n_r
                val_l += lp_l * 9500.0 * gust
                val_r += lp_r * 9500.0 * gust

            # 2. Sub-bass lub-dub heartbeat thump
            if has_heartbeat:
                beat_phase = (t * 1.15) % 1.0
                if beat_phase < 0.12:
                    env = math.sin(math.pi * (beat_phase / 0.12))
                    thump = math.sin(2.0 * math.pi * 52.0 * t) * env * 7500.0
                    val_l += thump
                    val_r += thump
                elif 0.18 <= beat_phase < 0.28:
                    env = math.sin(math.pi * ((beat_phase - 0.18) / 0.10))
                    thump = math.sin(2.0 * math.pi * 46.0 * t) * env * 5500.0
                    val_l += thump
                    val_r += thump

            # 3. Concrete footsteps / sprint impacts
            if has_footsteps:
                step_rate = 3.2 if "sprint" in p_low or "run" in p_low else 1.8
                step_phase = (t * step_rate) % 1.0
                if step_phase < 0.06:
                    impact_env = math.exp(-step_phase * 45.0)
                    step_sig = (rng.uniform(-0.8, 0.8) + math.sin(2.0 * math.pi * 95.0 * t)) * impact_env * 6500.0
                    val_l += step_sig * 1.05
                    val_r += step_sig * 0.95

            # 4. Scooty engine idle & street pass-by
            if has_scooty:
                rpm_freq = 68.0 + 14.0 * math.sin(2.0 * math.pi * 0.5 * t)
                eng = (
                    0.6 * math.sin(2.0 * math.pi * rpm_freq * t)
                    + 0.3 * math.sin(2.0 * math.pi * rpm_freq * 2.0 * t)
                ) * 4800.0
                val_l += eng
                val_r += eng

            # 5. Sinister smartphone vibration / red-ring pulse
            if has_ring:
                ring_phase = (t * 1.2) % 1.0
                if ring_phase < 0.55:
                    mod = math.sin(2.0 * math.pi * 16.0 * t)
                    tone = (
                        0.6 * math.sin(2.0 * math.pi * 440.0 * t)
                        + 0.4 * math.sin(2.0 * math.pi * 554.37 * t)
                    ) * (0.6 + 0.4 * mod) * 4200.0
                    val_l += tone
                    val_r += tone

            s_l = max(-32000, min(32000, int(val_l)))
            s_r = max(-32000, min(32000, int(val_r)))
            frames.extend(struct.pack("<hh", s_l, s_r))
        wf.writeframes(bytes(frames))
    return target_path


if __name__ == "__main__":
    ep_dir = ASSET_DIR / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    generate_lyria_music_stem(
        prompt=(
            "Dark cinematic Mumbai night thriller score, tense pulsing sub-bass, "
            "urgent pizzicato strings, high-stakes rooftop suspense, modern Indian neo-noir soundtrack, 112 bpm"
        ),
        target_wav=ep_dir / "ep1_lyria_score_a2.wav",
        duration_seconds=74.0,
    )
    generate_lyria_music_stem(
        prompt=(
            "Eerie sci-fi mystery thriller underscore, deep ominous synth drone, "
            "subtle metallic clockwork pulse, suspenseful apartment tension, no vocals, 100 bpm"
        ),
        target_wav=ep_dir / "ep2_lyria_score_a2.wav",
        duration_seconds=72.0,
    )
