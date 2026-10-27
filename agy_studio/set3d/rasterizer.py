"""Server-side 3D perspective blockout & depth renderer for set3d cameras.

Produces the exact ``blockout.png`` + ``depth.png`` + ``camera.json`` bundle under
``bible/sets/<ENV>/renders/<CAM>/<ts>/`` that the browser Three.js viewer saves, so AGY can render and
style-pass saved cameras directly from CLI/Python or present 3D blockout reference sheets in chat.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..fsio import atomic_write_bytes, atomic_write_text, now_iso
from ..project import Project
from .schema import Mannequin, Primitive, SetCamera, SetLayout
from .store import load_set, new_render_dir, save_set


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if l < 1e-9:
        return (0.0, 0.0, -1.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    s = h.lstrip("#")
    if len(s) != 6:
        return (150, 150, 150)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return (
        max(0, min(255, int(round(rgb[0] * factor)))),
        max(0, min(255, int(round(rgb[1] * factor)))),
        max(0, min(255, int(round(rgb[2] * factor)))),
    )


def _box_faces(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
    color: tuple[int, int, int],
    label: str = "",
) -> list[dict[str, Any]]:
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(rad), math.sin(rad)

    local_Wait = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    verts = []
    for lx, ly, lz in local_Wait:
        wx = cx + lx * cos_y + lz * sin_y
        wy = cy + ly
        wz = cz - lx * sin_y + lz * cos_y
        verts.append((wx, wy, wz))

    # 6 faces with shading factors
    face_defs = [
        ((0, 1, 2, 3), 0.78),  # back (-Z)
        ((5, 4, 7, 6), 0.92),  # front (+Z)
        ((4, 0, 3, 7), 0.72),  # left (-X)
        ((1, 5, 6, 2), 0.85),  # right (+X)
        ((3, 2, 6, 7), 1.05),  # top (+Y)
        ((4, 5, 1, 0), 0.58),  # bottom (-Y)
    ]
    faces = []
    for idx_tuple, factor in face_defs:
        pts = [verts[i] for i in idx_tuple]
        faces.append({
            "pts": pts,
            "color": _shade(color, factor),
            "edge": _shade(color, 0.45),
            "label": label if factor == 1.05 else "",
            "center": (cx, cy + hy, cz),
        })
    return faces


def _mannequin_boxes(m: Mannequin) -> list[dict[str, Any]]:
    x, y, z = m.position
    h = m.height_m
    w_rgb = _hex_to_rgb(m.wardrobe_color)
    s_rgb = _hex_to_rgb(m.skin_color)
    faces: list[dict[str, Any]] = []
    if m.pose == "sitting":
        # Seated torso + legs + head
        faces.extend(_box_faces((x, y + 0.25 * h, z), (0.36, 0.45 * h, 0.36), m.facing_deg, _shade(w_rgb, 0.75)))
        faces.extend(_box_faces((x, y + 0.56 * h, z), (0.40, 0.32 * h, 0.24), m.facing_deg, w_rgb, label=m.label or m.character_id))
        faces.extend(_box_faces((x, y + 0.79 * h, z), (0.20, 0.14 * h, 0.20), m.facing_deg, s_rgb))
    else:
        # Standing legs + torso + head
        faces.extend(_box_faces((x, y + 0.24 * h, z), (0.30, 0.48 * h, 0.22), m.facing_deg, _shade(w_rgb, 0.75)))
        faces.extend(_box_faces((x, y + 0.64 * h, z), (0.42, 0.34 * h, 0.24), m.facing_deg, w_rgb, label=m.label or m.character_id))
        faces.extend(_box_faces((x, y + 0.90 * h, z), (0.21, 0.14 * h, 0.21), m.facing_deg, s_rgb))
    return faces


def render_camera_to_images(
    layout: SetLayout,
    cam: SetCamera,
    *,
    width: int = 720,
    height: int = 1280,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Render (clean_blockout_img, hud_blockout_img, depth_img) for ``cam`` against ``layout``."""
    if cam.aspect_ratio == "16:9":
        width, height = 1280, 720
    elif cam.aspect_ratio == "1:1":
        width, height = 1024, 1024

    sky_rgb = _hex_to_rgb(layout.sky_color)
    img = Image.new("RGB", (width, height), sky_rgb)
    depth_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    depth_draw = ImageDraw.Draw(depth_img)

    cam_pos = cam.position
    forward = _normalize(_sub(cam.look_at, cam_pos))
    world_up = (0.0, 1.0, 0.0)
    right = _cross(forward, world_up)
    if _dot(right, right) < 1e-6:
        right = (1.0, 0.0, 0.0)
    else:
        right = _normalize(right)
    up = _normalize(_cross(right, forward))

    vfov_rad = math.radians(cam.vertical_fov_deg())
    focal_px = (height / 2.0) / math.tan(vfov_rad / 2.0)

    def to_cam(pt: tuple[float, float, float]) -> tuple[float, float, float]:
        rel = _sub(pt, cam_pos)
        return (_dot(rel, right), _dot(rel, up), _dot(rel, forward))

    def project_cam(cpt: tuple[float, float, float]) -> tuple[float, float]:
        cx, cy, cz = cpt
        z = max(cz, 0.12)
        sx = (width / 2.0) + (cx / z) * focal_px
        sy = (height / 2.0) - (cy / z) * focal_px
        return (sx, sy)

    def clip_near(cpts: list[tuple[float, float, float]], near: float = 0.25) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        n = len(cpts)
        for i in range(n):
            a = cpts[i]
            b = cpts[(i + 1) % n]
            a_in = a[2] >= near
            b_in = b[2] >= near
            if a_in and b_in:
                out.append(b)
            elif a_in and not b_in:
                t = (near - a[2]) / (b[2] - a[2])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), near))
            elif not a_in and b_in:
                t = (near - a[2]) / (b[2] - a[2])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), near))
                out.append(b)
        return out

    arch_faces: list[dict[str, Any]] = []
    # Floor grid slab
    bw, bh, bd = layout.bounds
    arch_faces.extend(_box_faces((0.0, -0.05, 0.0), (bw, 0.1, bd), 0.0, _hex_to_rgb(layout.floor_color)))

    for p in layout.primitives:
        # Skip ceiling or wall slabs that sit directly on top of the camera or behind camera blocking view
        c_cam = to_cam(p.position)
        if p.structural and p.position[1] > bh * 0.82 and p.size[1] < 0.4:
            continue
        if p.structural and c_cam[2] < 0.35 and (p.size[0] > bw * 0.6 or p.size[2] > bd * 0.6):
            continue
        arch_faces.extend(_box_faces(p.position, p.size, p.rotation_y_deg, _hex_to_rgb(p.color), label=p.label or p.id))

    mq_faces: list[dict[str, Any]] = []
    for m in layout.mannequins:
        if m.visible:
            mq_faces.extend(_mannequin_boxes(m))

    def _draw_faces(faces_list: list[dict[str, Any]], target_draw: ImageDraw.ImageDraw, record_depth: bool = False) -> list[tuple[float, float, str]]:
        drawable: list[tuple[float, list[tuple[float, float]], dict[str, Any]]] = []
        for f in faces_list:
            cpts = [to_cam(pt) for pt in f["pts"]]
            clipped = clip_near(cpts, 0.25)
            if len(clipped) < 3:
                continue
            avg_z = sum(pt[2] for pt in clipped) / len(clipped)
            poly2d = [project_cam(pt) for pt in clipped]
            drawable.append((avg_z, poly2d, f))
        drawable.sort(key=lambda item: item[0], reverse=True)
        max_depth = max(bw, bd) * 1.4
        lbls: list[tuple[float, float, str]] = []
        for avg_z, poly2d, f in drawable:
            target_draw.polygon(poly2d, fill=f["color"], outline=f["edge"])
            if record_depth:
                d_val = int(max(15, min(255, round(255.0 * (1.0 - min(avg_z / max_depth, 1.0))))))
                depth_draw.polygon(poly2d, fill=d_val)
            if f.get("label"):
                lc = to_cam(f["center"])
                if lc[2] > 0.4:
                    sx, sy = project_cam(lc)
                    if 20 <= sx <= width - 20 and 20 <= sy <= height - 20:
                        lbls.append((sx, sy, f["label"]))
        return lbls

    # clean_img has ONLY room architecture + furniture (no mannequin boxes) for clean style-pass
    _draw_faces(arch_faces, draw, record_depth=False)
    clean_img = img.copy()

    # hud_img has room architecture + furniture + posed mannequins + 180° axis + HUD labels
    hud_img = Image.new("RGB", (width, height), sky_rgb)
    hud_draw = ImageDraw.Draw(hud_img)
    labels_to_draw = _draw_faces(arch_faces + mq_faces, hud_draw, record_depth=True)

    # Draw 180-degree Axis of Action line on floor on HUD image
    if layout.axis_of_action:
        a0, a1 = layout.axis_of_action
        c0, c1 = to_cam((a0[0], 0.03, a0[2])), to_cam((a1[0], 0.03, a1[2]))
        seg = clip_near([c0, c1, c1], 0.25)
        if len(seg) >= 2:
            p0, p1 = project_cam(seg[0]), project_cam(seg[1])
            hud_draw.line([p0, p1], fill=(235, 80, 60), width=3)

    # Draw HUD header & landmark labels on HUD image
    font = ImageFont.load_default()
    seen_labels: set[str] = set()
    for sx, sy, text in reversed(labels_to_draw):
        if text in seen_labels:
            continue
        seen_labels.add(text)
        short = text[:22]
        hud_draw.rectangle([sx - 3, sy - 10, sx + len(short) * 6 + 4, sy + 4], fill=(20, 24, 32))
        hud_draw.text((sx, sy - 9), short, fill=(235, 240, 248), font=font)

    hud = f"SET3D {layout.environment_id} | CAM: {cam.name} | {int(cam.focal_mm)}mm ({cam.aspect_ratio})"
    hud_draw.rectangle([0, 0, width, 28], fill=(16, 20, 28))
    hud_draw.text((10, 8), hud, fill=(240, 244, 250), font=font)

    return clean_img, hud_img, depth_img


def render_and_save_camera(project: Project, env_id: str, cam_name: str) -> dict[str, Any]:
    """Render ``cam_name`` of ``env_id`` and persist ``blockout.png`` + ``blockout_hud.png`` + ``depth.png``."""
    import io
    import json

    layout = load_set(project, env_id)
    cam = layout.camera(cam_name)
    if cam is None:
        raise ValueError(f"{env_id} has no camera {cam_name!r}")
    clean_img, hud_img, depth_img = render_camera_to_images(layout, cam)
    b_buf = io.BytesIO()
    clean_img.save(b_buf, format="PNG")
    h_buf = io.BytesIO()
    hud_img.save(h_buf, format="PNG")
    d_buf = io.BytesIO()
    depth_img.save(d_buf, format="PNG")
    visible_mq = [m.id for m in layout.mannequins if m.visible]

    rdir = new_render_dir(project, env_id, cam_name)
    atomic_write_bytes(rdir / "blockout.png", b_buf.getvalue())
    atomic_write_bytes(rdir / "blockout_hud.png", h_buf.getvalue())
    atomic_write_bytes(rdir / "depth.png", d_buf.getvalue())
    manifest = {
        "ts": now_iso(),
        "environment_id": env_id,
        "camera": cam.model_dump(mode="json"),
        "width": clean_img.width,
        "height": clean_img.height,
        "visible_mannequins": visible_mq,
        "set_version": layout.version,
        "renderer": "agy.set3d.rasterizer",
        "lens": cam.describe(),
    }
    atomic_write_text(rdir / "camera.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    cam.last_render = project.rel(rdir)
    layout.upsert_camera(cam)
    save_set(project, layout, bump=False)
    return {
        "render_dir": project.rel(rdir),
        "blockout": project.rel(rdir / "blockout.png"),
        "blockout_hud": project.rel(rdir / "blockout_hud.png"),
        "depth": project.rel(rdir / "depth.png"),
    }
