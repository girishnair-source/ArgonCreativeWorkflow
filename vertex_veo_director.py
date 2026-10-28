"""Generic Vertex AI Veo 3.1 & Gemini Omni 1.1 Video Synthesis Backend (`vertex_veo_director.py`).

Project-agnostic and episode-agnostic video generation engine supporting:
- 2D/3D Cartoons, 20-Minute Multi-Scene Episodes, Feature Movies, and Vertical Microdramas.
- Keyframe (frame 0) + Filmmaker-framework motion prompt:
    -> {veo-model}:predictLongRunning  (16:9 or 9:16, 4-10s, 1080p/4K, native audio, GCS output)
    -> {veo-model}:fetchPredictOperation polling
    -> RAI re-roll / fast-model fallback
    -> GCS download -> generated_assets/episodes/epN_shot_0X.mp4

Auth: the gcloud user credential (``gcloud auth print-access-token``) — no extra Python deps.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import certifi
import httpx
from google import genai
from google.genai import interactions

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from production_bible import (
    I2VShotDirection,
    PromptLocks,
    _file_fingerprint,
    compile_i2v_prompt,
    compile_negative_prompt,
)
from schemas import ASSET_DIR

# No hidden default project / bucket: AGY resolves the project through agy_studio.adapters.auth.vertex_project()
# (VERTEX_PROJECT_ID / GOOGLE_CLOUD_PROJECT -> active gcloud config) and delivers inline when no bucket is set.
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
OMNI_LOCATION = os.environ.get("OMNI_LOCATION", "global")
OMNI_MODEL_ID = "gemini-omni-1.1-flash-preview"
GCS_BUCKET_ROOT = os.environ.get("VERTEX_GCS_BUCKET_ROOT", "").rstrip("/")
VEO_MODEL_CHAIN: list[str] = [
    OMNI_MODEL_ID,
    "veo-3.1-generate-001",
    "veo-3.1-fast-generate-001",
]
TARGET_RESOLUTION = "1080p"
TARGET_DURATION_SECONDS = 8
OMNI_DEFAULT_DURATION_SECONDS = 10
MAX_RAI_RETRIES_PER_MODEL = 2
TERMINAL_OK = {"completed"}
TERMINAL_BAD = {"failed", "filtered"}


def is_omni_model(model: str) -> bool:
    return "omni" in model.lower()


# ---------------------------------------------------------------------------------------------
# gcloud access-token provider (cached, refreshed on 401 / age) — FAIL FAST, never silent
# ---------------------------------------------------------------------------------------------
class GcloudToken:
    """Cached ``gcloud auth print-access-token``.

    Discovery, stderr capture and error wording live in :mod:`agy_studio.adapters.auth`: when gcloud cannot mint a
    token (expired login, Context-Aware Access block, missing binary) :meth:`get` raises
    :class:`agy_studio.adapters.auth.AuthError` whose message contains gcloud's own error text plus the exact
    ``gcloud auth login`` / ``context_aware/use_client_certificate`` commands. Nothing downstream may catch that and
    continue with an offline stand-in.
    """

    def __init__(self, account: str | None = None) -> None:
        self._token: str | None = None
        self._fetched_at = 0.0
        self._account = account

    def _resolve_account(self) -> str | None:
        """Explicit account -> ``AGY_GCLOUD_ACCOUNT`` / ``CLOUDSDK_CORE_ACCOUNT`` -> gcloud's active account."""
        from agy_studio.adapters import auth  # noqa: PLC0415 - stdlib-only module, shared with the AGY adapters

        if self._account:
            return self._account
        return auth.preferred_account()

    def get(self, force: bool = False) -> str:
        if force or not self._token or (time.time() - self._fetched_at) > 45 * 60:
            from agy_studio.adapters import auth  # noqa: PLC0415

            self._token = auth.gcloud_access_token(self._resolve_account())
            self._fetched_at = time.time()
        return self._token


class VertexError(RuntimeError):
    def __init__(self, status: int, message: str, reason: str = "") -> None:
        super().__init__(f"HTTP {status} {reason}: {message}")
        self.status = status
        self.message = message
        self.reason = reason


# ---------------------------------------------------------------------------------------------
# Gemini Omni 1.1 Flash client (Agent Platform Interactions API via google-genai)
# ---------------------------------------------------------------------------------------------
class GeminiOmniClient:
    """Client for `gemini-omni-1.1-flash-preview` via `genai.Client(enterprise=True).interactions`."""

    def __init__(self, project: str, token: GcloudToken, location: str = OMNI_LOCATION) -> None:
        self.project = project
        self.location = location
        self.token = token
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        from google.oauth2.credentials import Credentials

        from agy_studio.adapters.base import genai_http_options  # noqa: PLC0415 - hard request timeout

        creds = Credentials(token=self.token.get(), quota_project_id=self.project)
        self._client = genai.Client(
            enterprise=True,
            project=self.project,
            location=self.location,
            credentials=creds,
            http_options=genai_http_options("video"),
        )
        return self._client

    @staticmethod
    def _encode_file(path: Path, media_type: str) -> dict[str, str]:
        suffix = path.suffix.lower()
        if media_type == "image":
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            raw_bytes = path.read_bytes()
        else:
            mime = "video/mp4"
            # Strip audio from reference videos so Omni 1.1 never triggers 'unable to process speech edits'
            muted_path = path.parent / f".omni_muted_{path.name}"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(path), "-c:v", "copy", "-an", str(muted_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                raw_bytes = muted_path.read_bytes()
            except Exception:
                raw_bytes = path.read_bytes()
            finally:
                if muted_path.exists():
                    muted_path.unlink(missing_ok=True)
        return {
            "type": media_type,
            "mime_type": mime,
            "data": base64.b64encode(raw_bytes).decode("utf-8"),
        }

    def start_generation(
        self,
        model: str,
        prompt: str,
        image_path: Path,
        *,
        reference_image_paths: list[Path] | None = None,
        reference_video_paths: list[Path] | None = None,
        end_image_path: Path | None = None,
        task: str | None = None,
        aspect_ratio: str = "9:16",
        duration_seconds: int = OMNI_DEFAULT_DURATION_SECONDS,
        resolution: str = TARGET_RESOLUTION,
        gcs_output_prefix: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Submit an async (background=True) Omni 1.1 video interaction."""
        model_id = OMNI_MODEL_ID if model == "gemini-omni-1.1-flash" else model
        inputs: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        refs_img = [p for p in (reference_image_paths or []) if p.exists()]
        refs_vid = [p for p in (reference_video_paths or []) if p.exists()]

        if task is None:
            if refs_img or refs_vid:
                task = "reference_to_video"
            else:
                task = "image_to_video"

        # Primary keyframe is always included first
        inputs.append(self._encode_file(image_path, "image"))
        if task == "image_to_video" and end_image_path and end_image_path.exists():
            inputs.append(self._encode_file(end_image_path, "image"))
        elif task == "reference_to_video":
            for r_img in refs_img[:9]:
                if r_img.resolve() != image_path.resolve():
                    inputs.append(self._encode_file(r_img, "image"))
            for r_vid in refs_vid[:3]:
                inputs.append(self._encode_file(r_vid, "video"))

        duration_str = f"{max(3, min(10, int(duration_seconds)))}s"
        resp_kwargs: dict[str, Any] = {
            "aspect_ratio": aspect_ratio,
            "duration": duration_str,
            "resolution": resolution,
        }
        if gcs_output_prefix:
            resp_kwargs["delivery"] = "uri"
            resp_kwargs["gcs_uri"] = gcs_output_prefix.rstrip("/")

        interaction = self._get_client().interactions.create(
            model=model_id,
            input=inputs,
            generation_config=interactions.GenerationConfig(
                video_config=interactions.VideoConfig(task=task)
            ),
            response_format=interactions.VideoResponseFormat(**resp_kwargs),
            background=True,
        )
        op_id = str(interaction.id)
        params_used = {
            "task": task,
            "aspect_ratio": aspect_ratio,
            "duration": duration_str,
            "resolution": resolution,
            "reference_images": [str(image_path), *[str(p) for p in refs_img if p.resolve() != image_path.resolve()]],
            "reference_videos": [str(p) for p in refs_vid],
        }
        if gcs_output_prefix:
            params_used["gcs_uri"] = gcs_output_prefix.rstrip("/")
        return op_id, params_used

    def fetch_interaction(self, interaction_id: str) -> Any:
        return self._get_client().interactions.get(id=interaction_id)


# ---------------------------------------------------------------------------------------------
# Vertex AI Veo client
# ---------------------------------------------------------------------------------------------
class VertexVeoClient:
    def __init__(self, project: str, location: str, token: GcloudToken) -> None:
        self.project = project
        self.location = location
        self.token = token
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        self.models_base = f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token.get()}",
            "Content-Type": "application/json",
            "x-goog-user-project": self.project,
        }

    async def _post(self, url: str, body: dict[str, Any], retries: int = 6) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    resp = await client.post(url, headers=self._headers(), json=body)
            except httpx.TransportError as exc:
                last_err = exc
                await asyncio.sleep(5.0 * (attempt + 1))
                continue
            if resp.status_code == 401:
                self.token.get(force=True)
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                wait_s = min(90.0, 10.0 * (attempt + 1))
                print(f"[Vertex] {resp.status_code} on {url.rsplit(':', 1)[-1]}; retry in {wait_s:.0f}s")
                last_err = VertexError(resp.status_code, resp.text[:300])
                await asyncio.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                try:
                    err = resp.json().get("error", {})
                except Exception:
                    err = {"message": resp.text[:500]}
                reason = ""
                for det in err.get("details", []) or []:
                    if isinstance(det, dict) and det.get("reason"):
                        reason = str(det["reason"])
                raise VertexError(resp.status_code, str(err.get("message", "")), reason)
            return dict(resp.json())
        raise RuntimeError(f"Vertex request failed after retries: {last_err}")

    async def start_image_to_video(
        self,
        model: str,
        prompt: str,
        negative_prompt: str,
        image_path: Path,
        gcs_output_prefix: str | None,
        resolution: str = TARGET_RESOLUTION,
        duration_seconds: int = TARGET_DURATION_SECONDS,
        seed: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Submit a Veo image-to-video job. Returns (operation_name, parameters_used)."""
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        instance: dict[str, Any] = {
            "prompt": prompt,
            "image": {"bytesBase64Encoded": base64.b64encode(image_path.read_bytes()).decode("ascii"), "mimeType": mime},
        }
        params: dict[str, Any] = {
            "aspectRatio": "9:16",
            "durationSeconds": duration_seconds,
            "sampleCount": 1,
            "generateAudio": True,
            "personGeneration": "allow_adult",
            "negativePrompt": negative_prompt,
        }
        if resolution:
            params["resolution"] = resolution
        if gcs_output_prefix:
            params["storageUri"] = gcs_output_prefix.rstrip("/") + "/"
        if seed is not None:
            params["seed"] = int(seed)
        res = await self._post(f"{self.models_base}/{model}:predictLongRunning", {"instances": [instance], "parameters": params})
        op_name = str(res.get("name", ""))
        if not op_name:
            raise RuntimeError(f"predictLongRunning returned no operation name: {res}")
        return op_name, {k: v for k, v in params.items()}

    async def fetch_operation(self, model: str, operation_name: str) -> dict[str, Any]:
        return await self._post(f"{self.models_base}/{model}:fetchPredictOperation", {"operationName": operation_name})


async def download_gcs_object(gcs_uri: str, dest: Path, token: GcloudToken) -> Path:
    assert gcs_uri.startswith("gs://"), gcs_uri
    bucket, _, obj = gcs_uri[5:].partition("/")
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(obj, safe='')}?alt=media"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {token.get()}"})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return dest
    except Exception as exc:
        print(f"[GCS] JSON API download failed ({str(exc)[:120]}); falling back to gcloud storage cp")
        env = dict(os.environ)
        prev = env.get("CLOUDSDK_METRICS_ENVIRONMENT", "")
        env["CLOUDSDK_METRICS_ENVIRONMENT"] = f"{prev} datacloud.antigravity".strip()
        subprocess.run(["gcloud", "storage", "cp", gcs_uri, str(dest)], check=True, env=env)
        return dest


def _extract_videos(op: dict[str, Any]) -> tuple[list[dict[str, Any]], int, list[str]]:
    resp = op.get("response", {}) or {}
    videos = resp.get("videos") or resp.get("generatedSamples") or []
    filtered = int(resp.get("raiMediaFilteredCount", 0) or 0)
    reasons = [str(r) for r in (resp.get("raiMediaFilteredReasons") or [])]
    return list(videos), filtered, reasons


# ---------------------------------------------------------------------------------------------
# Episode registry
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeRun:
    episode: int
    title: str
    directions: dict[str, I2VShotDirection]
    locks: PromptLocks
    payload_builder: Callable[[], list[dict[str, Any]]]
    state_file: Path
    plan_file: Path
    gcs_output_prefix: str
    archive_dir: Path  # where any pre-existing local clip is moved (never deleted)


def _load_default_episodes() -> dict[int, EpisodeRun]:
    """Load episodes from `projects/like_share_kill_or_die` if present; otherwise return empty dict."""
    try:
        from projects.like_share_kill_or_die.episode1_directions import (
            EP1_PROMPT_LOCKS,
            EP1_SHOT_DIRECTIONS,
            build_episode1_payloads,
        )
        from projects.like_share_kill_or_die.episode2_directions import (
            EP2_PROMPT_LOCKS,
            EP2_SHOT_DIRECTIONS,
            build_episode2_payloads,
        )

        return {
            1: EpisodeRun(
                episode=1,
                title="EP01_Daring",
                directions=EP1_SHOT_DIRECTIONS,
                locks=EP1_PROMPT_LOCKS,
                payload_builder=build_episode1_payloads,
                state_file=ASSET_DIR / "vertex_veo_ep01_state.json",
                plan_file=ASSET_DIR / "vertex_veo_ep01_plan.json",
                gcs_output_prefix=os.environ.get(
                    "VERTEX_GCS_OUTPUT_PREFIX", f"{GCS_BUCKET_ROOT}/EP01/v2_vertex_veo"
                ),
                archive_dir=ASSET_DIR / "episodes" / "v1_archive_ep01",
            ),
            2: EpisodeRun(
                episode=2,
                title="EP02_TheParcel",
                directions=EP2_SHOT_DIRECTIONS,
                locks=EP2_PROMPT_LOCKS,
                payload_builder=build_episode2_payloads,
                state_file=ASSET_DIR / "vertex_veo_ep02_state.json",
                plan_file=ASSET_DIR / "vertex_veo_ep02_plan.json",
                gcs_output_prefix=f"{GCS_BUCKET_ROOT}/EP02/v1_vertex_veo",
                archive_dir=ASSET_DIR / "episodes" / "v0_mock_ep2",
            ),
        }
    except ImportError:
        return {}


EPISODES: dict[int, EpisodeRun] = _load_default_episodes()



# ---------------------------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------------------------
def load_state(ep: EpisodeRun) -> dict[str, Any]:
    if ep.state_file.exists():
        try:
            return dict(json.loads(ep.state_file.read_text(encoding="utf-8")))
        except Exception:
            pass
    return {
        "backend": "vertex-veo",
        "episode": ep.episode,
        "project": VERTEX_PROJECT_ID,
        "location": VERTEX_LOCATION,
        "gcs_output_prefix": ep.gcs_output_prefix,
        "generations": {},
    }


def save_state(ep: EpisodeRun, state: dict[str, Any]) -> None:
    ep.state_file.parent.mkdir(parents=True, exist_ok=True)
    ep.state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _archive_existing_clip(ep: EpisodeRun, shot_id: str) -> None:
    """Move any existing local clip for this shot aside — never delete (data-loss prevention)."""
    ep.archive_dir.mkdir(parents=True, exist_ok=True)
    ep_dir = ASSET_DIR / "episodes"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for candidate in (ep_dir / f"{shot_id.lower()}.mp4", ep_dir / "raw_720p" / f"{shot_id.lower()}.mp4"):
        if not candidate.exists():
            continue
        suffix = "_raw720p" if candidate.parent.name == "raw_720p" else ""
        target = ep.archive_dir / f"{shot_id.lower()}{suffix}.mp4"
        if target.exists():
            target = ep.archive_dir / f"{shot_id.lower()}{suffix}_{stamp}.mp4"
        shutil.move(str(candidate), str(target))
        print(f"[Archive] {candidate.name} -> {target.relative_to(ASSET_DIR)}")


def _plan(ep: EpisodeRun, target_shots: list[str] | None) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for b in ep.payload_builder():
        if target_shots and b["shot_id"] not in target_shots:
            continue
        d = ep.directions[b["shot_id"]]
        plan.append(
            {
                "shot_id": b["shot_id"],
                "keyframe": b["keyframe_image_path"],
                "keyframe_exists": Path(b["keyframe_image_path"]).exists(),
                "model_chain": VEO_MODEL_CHAIN,
                "prompt": compile_i2v_prompt(d, mode="image-to-video", locks=ep.locks),
                "negativePrompt": compile_negative_prompt(d, locks=ep.locks),
            }
        )
    return plan


# ---------------------------------------------------------------------------------------------
# Main run loop (same semantics as vertex_veo_episode1_director.run_episode1_on_vertex)
# ---------------------------------------------------------------------------------------------
async def run_episode_on_vertex(
    ep: EpisodeRun,
    *,
    target_shots: list[str] | None,
    force: bool,
    poll_until_done: bool,
    model_preference: str | None,
    inline_delivery: bool,
    poll_budget_minutes: int = 60,
    duration_override: int | None = None,
    omni_task_override: str | None = None,
    extra_reference_images: list[Path] | None = None,
    extra_reference_videos: list[Path] | None = None,
) -> dict[str, Any]:
    global VERTEX_PROJECT_ID  # noqa: PLW0603 - legacy CLI entry point; AGY adapters pass the project explicitly
    if not VERTEX_PROJECT_ID:
        from agy_studio.adapters import auth  # noqa: PLC0415

        VERTEX_PROJECT_ID = auth.vertex_project()
    token = GcloudToken()
    veo_client = VertexVeoClient(VERTEX_PROJECT_ID, VERTEX_LOCATION, token)
    omni_client = GeminiOmniClient(VERTEX_PROJECT_ID, token, OMNI_LOCATION)
    state = load_state(ep)
    state["project"], state["location"] = VERTEX_PROJECT_ID, VERTEX_LOCATION
    state["omni_location"] = OMNI_LOCATION
    state["gcs_output_prefix"] = None if inline_delivery else ep.gcs_output_prefix

    chain = list(VEO_MODEL_CHAIN)
    if model_preference:
        norm_pref = OMNI_MODEL_ID if model_preference == "gemini-omni-1.1-flash" else model_preference
        chain = [norm_pref, *[m for m in chain if m != norm_pref]]
    print(f"[Vertex] {ep.title}: project={VERTEX_PROJECT_ID} veo_loc={VERTEX_LOCATION} omni_loc={OMNI_LOCATION} "
          f"chain={chain} output={'inline' if inline_delivery else ep.gcs_output_prefix}")

    payloads = ep.payload_builder()
    beats = [b for b in payloads if not target_shots or b["shot_id"] in target_shots]
    beats_by_id = {b["shot_id"]: b for b in beats}
    missing_kf = [s for s, b in beats_by_id.items() if not Path(b["keyframe_image_path"]).exists()]
    if missing_kf:
        raise FileNotFoundError(f"Keyframes not staged for {missing_kf}; stage generated_assets/episodes/*.jpg first.")
    gens: dict[str, Any] = state.setdefault("generations", {})

    async def submit(shot_id: str, model: str, chain_idx: int, rai_attempt: int, prev: dict[str, Any] | None) -> dict[str, Any]:
        beat = beats_by_id[shot_id]
        direction = ep.directions[shot_id]
        prompt = compile_i2v_prompt(direction, mode="image-to-video", locks=ep.locks)
        negative = compile_negative_prompt(direction, locks=ep.locks)
        kf = Path(beat["keyframe_image_path"])
        gcs_prefix = None if inline_delivery else f"{ep.gcs_output_prefix}/{shot_id.lower()}"
        resolution = TARGET_RESOLUTION
        salt = random.randrange(1, 100_000) if force else 0
        seed = (int(_file_fingerprint(kf)[:6], 16) + rai_attempt * 7919 + salt) % 4_294_967_295

        if is_omni_model(model):
            dur = duration_override or int(beat.get("duration_seconds") or OMNI_DEFAULT_DURATION_SECONDS)
            dur = max(3, min(10, dur))
            ref_imgs: list[Path] = [Path(p) for p in (beat.get("reference_image_paths") or []) if Path(p).exists()]
            if extra_reference_images:
                ref_imgs.extend([p for p in extra_reference_images if p.exists()])
            ref_vids: list[Path] = [Path(p) for p in (beat.get("reference_video_paths") or []) if Path(p).exists()]
            if extra_reference_videos:
                ref_vids.extend([p for p in extra_reference_videos if p.exists()])

            omni_prompt = f"{prompt}\n\nSTRICT NEGATIVE CONSTRAINTS (DO NOT INCLUDE): {negative}"
            print(
                f"[Omni 1.1] {shot_id}: interactions.create model={model} task={omni_task_override or ('reference_to_video' if (ref_imgs or ref_vids) else 'image_to_video')} "
                f"res={resolution} {dur}s frame0={kf.name} refs_img={len(ref_imgs)} refs_vid={len(ref_vids)}"
            )
            op_id, params = await asyncio.to_thread(
                omni_client.start_generation,
                model,
                omni_prompt,
                kf,
                reference_image_paths=ref_imgs,
                reference_video_paths=ref_vids,
                task=omni_task_override,
                aspect_ratio=str(beat.get("aspect_ratio") or "9:16"),
                duration_seconds=dur,
                resolution=resolution,
                gcs_output_prefix=None if inline_delivery else gcs_prefix,
            )
            print(f"[Omni 1.1] {shot_id} -> interaction_id {op_id}")
            return {
                "backend": "gemini-omni-1.1",
                "model": model,
                "mode": params.get("task", "image_to_video"),
                "operation_name": op_id,
                "status": "running",
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
                "keyframe": str(kf),
                "keyframe_fingerprint": _file_fingerprint(kf),
                "prompt": omni_prompt,
                "negative_prompt": negative,
                "parameters": dict(params),
                "model_chain_index": chain_idx,
                "rai_attempt": rai_attempt,
                "previous_attempts": (prev or {}).get("previous_attempts", []),
            }

        last_exc: Exception | None = None
        veo_dur = duration_override if duration_override in (4, 6, 8) else TARGET_DURATION_SECONDS
        for _attempt in range(3):
            try:
                print(f"[Vertex] {shot_id}: predictLongRunning model={model} res={resolution or 'default'} "
                      f"{veo_dur}s seed={seed} frame0={kf.name} out={'inline' if not gcs_prefix else gcs_prefix}")
                op_name, params = await veo_client.start_image_to_video(
                    model, prompt, negative, kf, gcs_prefix, resolution=resolution, duration_seconds=veo_dur, seed=seed
                )
                print(f"[Vertex] {shot_id} -> operation {op_name.rsplit('/', 1)[-1]}")
                return {
                    "backend": "vertex-veo",
                    "model": model,
                    "mode": "image-to-video",
                    "operation_name": op_name,
                    "status": "running",
                    "submitted_at": datetime.now().isoformat(timespec="seconds"),
                    "keyframe": str(kf),
                    "keyframe_fingerprint": _file_fingerprint(kf),
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "parameters": dict(params),
                    "model_chain_index": chain_idx,
                    "rai_attempt": rai_attempt,
                    "previous_attempts": (prev or {}).get("previous_attempts", []),
                }
            except VertexError as exc:
                last_exc = exc
                msg = exc.message.lower()
                if resolution and ("resolution" in msg or "1080" in msg):
                    print(f"[Vertex] {shot_id}: {exc.message[:140]} -> retrying at 720p")
                    resolution = "720p"
                    continue
                if gcs_prefix and ("storage" in msg or "bucket" in msg or "gcs" in msg):
                    print(f"[Vertex] {shot_id}: {exc.message[:140]} -> retrying with inline delivery")
                    gcs_prefix = None
                    continue
                raise
        raise RuntimeError(f"{shot_id}: could not submit: {last_exc}")

    # 1. Submit
    for shot_id in beats_by_id:
        existing = gens.get(shot_id, {})
        kf_fp = _file_fingerprint(Path(beats_by_id[shot_id]["keyframe_image_path"]))
        keyframe_changed = existing.get("keyframe_fingerprint") not in (None, kf_fp)
        local = existing.get("local_video_path")
        if existing.get("status") == "completed" and local and Path(local).exists() and not force and not keyframe_changed:
            print(f"[Vertex] {shot_id}: already completed -> {Path(local).name}")
            continue
        if not force and existing.get("status") == "running" and existing.get("operation_name"):
            print(f"[Vertex] {shot_id}: already running ({existing['operation_name'].rsplit('/', 1)[-1]})")
            continue
        if (ASSET_DIR / "episodes" / f"{shot_id.lower()}.mp4").exists():
            _archive_existing_clip(ep, shot_id)
        try:
            gens[shot_id] = await submit(shot_id, chain[0], 0, 0, None)
        except Exception as sub_exc:
            print(f"[Submit] {shot_id} failed on {chain[0]} ({str(sub_exc)[:200]}); falling back to next model in chain")
            if len(chain) > 1:
                gens[shot_id] = await submit(shot_id, chain[1], 1, 0, {"previous_attempts": [{"model": chain[0], "error": str(sub_exc)}]})
            else:
                raise
        save_state(ep, state)
        await asyncio.sleep(1.5)

    if not poll_until_done:
        return {"status": "submitted", "state_file": str(ep.state_file)}

    # 2. Poll / download / fallback
    pending = [s for s in beats_by_id if gens.get(s, {}).get("status") not in TERMINAL_OK | TERMINAL_BAD]
    poll_idx = 0
    max_polls = max(1, int(poll_budget_minutes * 60 / 15))
    while pending and poll_idx < max_polls:
        poll_idx += 1
        await asyncio.sleep(15.0)
        still: list[str] = []
        for shot_id in pending:
            g = gens[shot_id]
            model_name = str(g.get("model", ""))
            if is_omni_model(model_name):
                try:
                    inter = await asyncio.to_thread(omni_client.fetch_interaction, g["operation_name"])
                except Exception as exc:
                    print(f"[Poll #{poll_idx}] {shot_id} (Omni 1.1): fetch failed ({str(exc)[:140]}); retrying")
                    still.append(shot_id)
                    continue
                status_str = str(getattr(inter, "status", "") or "").lower()
                if status_str in ("in_progress", "running", "pending", "queued", "processing", ""):
                    print(f"[Poll #{poll_idx}] {shot_id} ({model_name}): {status_str or 'running'}")
                    still.append(shot_id)
                    continue

                contents: list[Any] = []
                for step in getattr(inter, "steps", []) or []:
                    if getattr(step, "type", "") == "model_output":
                        contents.extend(getattr(step, "content", []) or [])
                if not contents and getattr(inter, "outputs", None):
                    contents.extend(getattr(inter, "outputs", []) or [])

                if status_str == "completed" and contents:
                    c0 = contents[0]
                    dest = ASSET_DIR / "episodes" / f"{shot_id.lower()}.mp4"
                    raw720 = ASSET_DIR / "episodes" / "raw_720p" / dest.name
                    if raw720.exists():
                        raw720.unlink()
                    uri_val = getattr(c0, "uri", None)
                    data_val = getattr(c0, "data", None)
                    if uri_val and str(uri_val).startswith("gs://"):
                        print(f"[Omni 1.1] {shot_id}: done -> {uri_val}")
                        await download_gcs_object(str(uri_val), dest, token)
                        g["gcs_uri"] = str(uri_val)
                    elif data_val:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        raw_bytes = base64.b64decode(data_val) if isinstance(data_val, str) else bytes(data_val)
                        dest.write_bytes(raw_bytes)
                    else:
                        raise RuntimeError(f"{shot_id}: Omni interaction completed but no uri/data in {c0}")
                    g["status"] = "completed"
                    g["local_video_path"] = str(dest)
                    g["completed_at"] = datetime.now().isoformat(timespec="seconds")
                    g["rai_filtered_count"] = 0
                    print(f"[Omni 1.1] Downloaded {shot_id} -> {dest.name} ({dest.stat().st_size / 1048576:.2f} MB)")
                    save_state(ep, state)
                    continue

                err = {"status": status_str, "detail": str(inter)[:400]}
                videos, filtered, reasons = [], 0, [status_str]
            else:
                try:
                    op = await veo_client.fetch_operation(g["model"], g["operation_name"])
                except Exception as exc:
                    print(f"[Poll #{poll_idx}] {shot_id}: fetch failed ({str(exc)[:120]}); retrying")
                    still.append(shot_id)
                    continue
                if not op.get("done"):
                    print(f"[Poll #{poll_idx}] {shot_id} ({g['model']}): running")
                    still.append(shot_id)
                    continue

                err = op.get("error")
                videos, filtered, reasons = _extract_videos(op)
                if videos and not err:
                    v0 = videos[0]
                    dest = ASSET_DIR / "episodes" / f"{shot_id.lower()}.mp4"
                    raw720 = ASSET_DIR / "episodes" / "raw_720p" / dest.name
                    if raw720.exists():
                        raw720.unlink()  # stale raw must not be re-normalized over the new clip
                    if v0.get("gcsUri"):
                        print(f"[Vertex] {shot_id}: done -> {v0['gcsUri']}")
                        await download_gcs_object(str(v0["gcsUri"]), dest, token)
                        g["gcs_uri"] = v0["gcsUri"]
                    elif v0.get("bytesBase64Encoded"):
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(base64.b64decode(v0["bytesBase64Encoded"]))
                    else:
                        raise RuntimeError(f"{shot_id}: operation done but no gcsUri/bytes in {json.dumps(v0)[:300]}")
                    g["status"] = "completed"
                    g["local_video_path"] = str(dest)
                    g["completed_at"] = datetime.now().isoformat(timespec="seconds")
                    g["rai_filtered_count"] = filtered
                    print(f"[Vertex] Downloaded {shot_id} -> {dest.name} ({dest.stat().st_size / 1048576:.2f} MB)")
                    save_state(ep, state)
                    continue

            failure = {
                "model": g["model"],
                "operation_name": g["operation_name"],
                "error": err,
                "rai_filtered_count": filtered,
                "rai_reasons": reasons,
                "rai_attempt": g.get("rai_attempt", 0),
            }
            print(f"[Vertex] {shot_id} FAILED on {g['model']}: error={json.dumps(err)[:200] if err else None} "
                  f"filtered={filtered} reasons={reasons}")
            prev_attempts = [*g.get("previous_attempts", []), failure]
            chain_idx = int(g.get("model_chain_index", 0))
            rai_attempt = int(g.get("rai_attempt", 0))
            is_rai = filtered > 0 and not err
            if is_rai and rai_attempt + 1 < MAX_RAI_RETRIES_PER_MODEL:
                next_model, next_idx, next_rai = g["model"], chain_idx, rai_attempt + 1
            elif chain_idx + 1 < len(chain):
                next_model, next_idx, next_rai = chain[chain_idx + 1], chain_idx + 1, 0
            else:
                g["status"] = "filtered" if is_rai else "failed"
                g["previous_attempts"] = prev_attempts
                g["error"] = err or {"rai_reasons": reasons}
                save_state(ep, state)
                continue
            print(f"[Vertex] {shot_id}: re-submitting on {next_model} (rai_attempt={next_rai})")
            try:
                new_g = await submit(shot_id, next_model, next_idx, next_rai, {"previous_attempts": prev_attempts})
            except Exception as exc:
                print(f"[Vertex] {shot_id}: re-submit failed ({str(exc)[:160]}); will retry next poll")
                still.append(shot_id)
                continue
            gens[shot_id] = new_g
            save_state(ep, state)
            still.append(shot_id)
        pending = still

    summary = {s: gens.get(s, {}).get("status") for s in beats_by_id}
    return {"status": "done" if all(v == "completed" for v in summary.values()) else "partial",
            "episode": ep.title, "shots": summary, "state_file": str(ep.state_file)}


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gemini Omni 1.1 Flash & Vertex AI Veo 3.1 direct backend (episode-agnostic)")
    ap.add_argument("--episode", type=int, choices=sorted(EPISODES), required=True)
    ap.add_argument("--shots", nargs="*", default=None, help="Subset of shot IDs, e.g. EP1_SHOT_04")
    ap.add_argument("--force", action="store_true", help="Archive existing clips and regenerate")
    ap.add_argument("--dry-run", action="store_true", help="Compile prompts only; no API calls")
    ap.add_argument("--no-poll", action="store_true", help="Submit only; re-run without it to poll")
    ap.add_argument("--model", default=None, help="Primary model override (default: gemini-omni-1.1-flash-preview)")
    ap.add_argument("--duration", type=int, default=None, help="Duration in seconds (3-10s for Omni 1.1, 4/6/8s for Veo)")
    ap.add_argument("--task", default=None, help="Omni 1.1 task override: image_to_video, reference_to_video, extend, edit")
    ap.add_argument("--ref-images", nargs="*", default=None, help="Additional reference image paths for Omni 1.1 reference_to_video")
    ap.add_argument("--ref-videos", nargs="*", default=None, help="Additional reference video paths for Omni 1.1 reference_to_video")
    ap.add_argument("--inline", action="store_true", help="Inline base64 delivery instead of GCS")
    ap.add_argument("--poll-minutes", type=int, default=60, help="Polling budget in minutes")
    args = ap.parse_args()
    ep = EPISODES[args.episode]

    if args.dry_run:
        plan = _plan(ep, args.shots)
        ep.plan_file.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        for p in plan:
            print(f"{p['shot_id']}: keyframe={'ok' if p['keyframe_exists'] else 'MISSING'} "
                  f"prompt {len(p['prompt'])} chars, neg {len(p['negativePrompt'])} chars")
        print(f"Plan written to {ep.plan_file}")
        raise SystemExit(0)

    result = asyncio.run(
        run_episode_on_vertex(
            ep,
            target_shots=args.shots,
            force=args.force,
            poll_until_done=not args.no_poll,
            model_preference=args.model,
            inline_delivery=args.inline,
            poll_budget_minutes=args.poll_minutes,
            duration_override=args.duration,
            omni_task_override=args.task,
            extra_reference_images=[Path(p) for p in args.ref_images] if args.ref_images else None,
            extra_reference_videos=[Path(p) for p in args.ref_videos] if args.ref_videos else None,
        )
    )
    print(json.dumps(result, indent=2))

