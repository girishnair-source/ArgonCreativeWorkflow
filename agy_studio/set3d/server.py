"""set3d viewer server — serves the three.js viewer and a small JSON API over one project's sets.

    GET  /                                  viewer (``?env=ENV_ID`` preselects a set)
    GET  /api/sets                          [{env_id, title, cameras, has_still}]
    GET  /api/set/{env}                     SetLayout JSON
    PUT  /api/set/{env}                     replace the layout (primitives/lights/mannequins edited in the viewer)
    GET  /api/set/{env}/still               the locked geometry still (image) when there is one
    GET  /api/set/{env}/cameras             [SetCamera]
    POST /api/set/{env}/cameras             upsert one SetCamera
    DELETE /api/set/{env}/cameras/{name}    remove an unlocked camera
    POST /api/set/{env}/render/{name}       {blockout_png_b64, depth_png_b64, width, height, visible_mannequins}
                                            -> writes renders/<name>/<ts>/{blockout.png, depth.png, camera.json}
    GET  /api/set/{env}/render/{name}       latest render manifest + image urls
    GET  /files/{rel}                       read-only access to files under the project root

Binds to 127.0.0.1 by default; set ``--host 0.0.0.0`` only when the viewer must be reached from another machine.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..fsio import PathSecurityError, atomic_write_bytes, atomic_write_json, now_iso
from ..project import Project
from .schema import SetCamera, SetLayout
from .store import (
    SetNotFoundError,
    latest_render_dir,
    list_sets,
    load_set,
    new_render_dir,
    render_manifest,
    save_set,
)

VIEWER_DIR = Path(__file__).with_name("viewer")
MAX_RENDER_BYTES = 40 * 1024 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def _png_bytes(b64: Any, what: str) -> bytes:
    if not isinstance(b64, str) or not b64:
        raise ValueError(f"{what} missing")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64, validate=True)
    if len(data) > MAX_RENDER_BYTES:
        raise ValueError(f"{what} larger than {MAX_RENDER_BYTES // (1024 * 1024)} MB")
    if not data.startswith(PNG_MAGIC):
        raise ValueError(f"{what} is not a PNG")
    return data


def build_app(project: Project) -> Starlette:
    # -- helpers ------------------------------------------------------------------------------------
    def env_of(request: Request) -> str:
        return str(request.path_params["env"]).upper()

    def layout_of(request: Request) -> SetLayout:
        return load_set(project, env_of(request))

    # -- routes -------------------------------------------------------------------------------------
    async def index(_: Request) -> Response:
        return FileResponse(VIEWER_DIR / "index.html")

    async def sets(_: Request) -> Response:
        rows = []
        for env_id in list_sets(project):
            lay = load_set(project, env_id)
            env = project.load_environment(env_id)
            rows.append({"env_id": env_id, "title": lay.title or env_id, "cameras": len(lay.cameras),
                         "has_still": bool(env.geometry_still and project.abs(env.geometry_still).is_file()),
                         "version": lay.version})
        return JSONResponse({"project_id": project.project_id, "aspect_ratio": project.config.settings.aspect_ratio,
                             "sets": rows})

    async def get_set(request: Request) -> Response:
        try:
            return JSONResponse(layout_of(request).model_dump(mode="json"))
        except SetNotFoundError as exc:
            return _err(404, str(exc))

    async def put_set(request: Request) -> Response:
        try:
            current = layout_of(request)
            data = await request.json()
            if not isinstance(data, dict):
                return _err(400, "layout must be a JSON object")
            data["environment_id"] = current.environment_id
            data["cameras"] = [c.model_dump(mode="json") for c in current.cameras]  # cameras have their own API
            data["generated_by"] = current.generated_by
            data["source_still"] = current.source_still
            data.pop("version", None)
            new = SetLayout.model_validate(data)
            new.version = current.version
            save_set(project, new)
            return JSONResponse({"ok": True, "version": new.version})
        except SetNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _err(400, str(exc))

    async def still(request: Request) -> Response:
        env = project.load_environment(env_of(request))
        if not env.geometry_still:
            return _err(404, "no geometry still")
        p = project.abs(env.geometry_still)
        if not p.is_file():
            return _err(404, "geometry still missing on disk")
        return FileResponse(p)

    async def cameras(request: Request) -> Response:
        try:
            return JSONResponse([c.model_dump(mode="json") for c in layout_of(request).cameras])
        except SetNotFoundError as exc:
            return _err(404, str(exc))

    async def post_camera(request: Request) -> Response:
        try:
            lay = layout_of(request)
            cam = SetCamera.model_validate(await request.json())
            prev = lay.camera(cam.name)
            if prev is not None:
                cam.last_render = prev.last_render
                if prev.locked and not cam.locked:
                    cam.locked = True  # unlocking is a chat decision, not a viewer click
            lay.upsert_camera(cam)
            save_set(project, lay, bump=False)
            return JSONResponse(cam.model_dump(mode="json"))
        except SetNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _err(400, str(exc))

    async def delete_camera(request: Request) -> Response:
        try:
            lay = layout_of(request)
            if not lay.remove_camera(str(request.path_params["name"])):
                return _err(404, "no such camera")
            save_set(project, lay, bump=False)
            return JSONResponse({"ok": True})
        except SetNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _err(400, str(exc))

    async def post_render(request: Request) -> Response:
        try:
            lay = layout_of(request)
            name = str(request.path_params["name"])
            cam = lay.camera(name)
            if cam is None:
                return _err(404, f"save camera {name!r} before rendering it")
            body = await request.json()
            blockout = _png_bytes(body.get("blockout_png_b64"), "blockout")
            depth = _png_bytes(body.get("depth_png_b64"), "depth") if body.get("depth_png_b64") else None
            rdir = new_render_dir(project, lay.environment_id, name)
            atomic_write_bytes(rdir / "blockout.png", blockout)
            if depth:
                atomic_write_bytes(rdir / "depth.png", depth)
            manifest = {
                "ts": now_iso(), "environment_id": lay.environment_id, "camera": cam.model_dump(mode="json"),
                "width": int(body.get("width") or 0), "height": int(body.get("height") or 0),
                "visible_mannequins": [str(x) for x in (body.get("visible_mannequins") or [])],
                "set_version": lay.version, "renderer": str(body.get("renderer") or "three.js"),
                "lens": cam.describe(),
            }
            atomic_write_json(rdir / "camera.json", manifest)
            cam.last_render = project.rel(rdir)
            lay.upsert_camera(cam)
            save_set(project, lay, bump=False)
            return JSONResponse({"ok": True, "render_dir": project.rel(rdir),
                                 "blockout": f"/files/{project.rel(rdir / 'blockout.png')}",
                                 "depth": f"/files/{project.rel(rdir / 'depth.png')}" if depth else None})
        except SetNotFoundError as exc:
            return _err(404, str(exc))
        except Exception as exc:  # noqa: BLE001
            return _err(400, str(exc))

    async def get_render(request: Request) -> Response:
        try:
            lay = layout_of(request)
            name = str(request.path_params["name"])
            rdir = latest_render_dir(project, lay.environment_id, name)
            if rdir is None:
                return _err(404, "not rendered yet")
            m = render_manifest(rdir)
            m["blockout"] = f"/files/{project.rel(rdir / 'blockout.png')}"
            m["depth"] = f"/files/{project.rel(rdir / 'depth.png')}" if (rdir / "depth.png").is_file() else None
            return JSONResponse(m)
        except SetNotFoundError as exc:
            return _err(404, str(exc))

    async def files(request: Request) -> Response:
        rel = str(request.path_params["rel"])
        try:
            p = project.abs(rel)
        except (PathSecurityError, ValueError) as exc:
            return _err(400, str(exc))
        if not p.is_file():
            return _err(404, "no such file")
        mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if not (mt.startswith("image/") or mt in ("application/json", "text/plain")):
            return _err(403, "only images and json are served")
        return FileResponse(p, media_type=mt)

    routes = [
        Route("/", index),
        Route("/api/sets", sets),
        Route("/api/set/{env}", get_set, methods=["GET"]),
        Route("/api/set/{env}", put_set, methods=["PUT"]),
        Route("/api/set/{env}/still", still),
        Route("/api/set/{env}/cameras", cameras, methods=["GET"]),
        Route("/api/set/{env}/cameras", post_camera, methods=["POST"]),
        Route("/api/set/{env}/cameras/{name}", delete_camera, methods=["DELETE"]),
        Route("/api/set/{env}/render/{name}", post_render, methods=["POST"]),
        Route("/api/set/{env}/render/{name}", get_render, methods=["GET"]),
        Route("/files/{rel:path}", files),
        Mount("/static", app=StaticFiles(directory=str(VIEWER_DIR)), name="static"),
    ]
    app = Starlette(routes=routes)
    app.state.project = project
    return app


def serve(project: Project, *, host: str = "127.0.0.1", port: int = 8765, env_id: str | None = None,
          open_browser: bool = False) -> None:
    import uvicorn  # noqa: PLC0415

    url = f"http://{host}:{port}/" + (f"?env={env_id}" if env_id else "")
    print(f"set3d viewer for {project.project_id}: {url}  (Ctrl-C to stop)")
    if open_browser:
        import webbrowser  # noqa: PLC0415

        webbrowser.open(url)
    uvicorn.run(build_app(project), host=host, port=port, log_level="warning")
