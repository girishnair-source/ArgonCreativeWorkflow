"""On-Demand 3D / Blender Scene Exporter & Screenshot-to-Keyframe Re-Anchoring (Gate 3 Control Loop).

Implements the Gate 3 precision camera & character blocking workflow from the AGY Studio v4 blueprint:
1. :func:`export_blender_scene` exports any ``SetLayout`` (``bible/sets/<ENV_ID>/set.json``) into
   ``bible/sets/<ENV_ID>/blender/`` as:
   - ``scene.obj`` + ``scene.mtl`` (standard 3D mesh + materials for furniture, walls & character mannequins)
   - ``scene.gltf`` (glTF 2.0 node graph with cameras, lights, blocks & mannequins)
   - ``open_in_blender.py`` (native Blender Python script setting up metric room blocks, lights, mannequins & all
     saved ``SetCamera`` objects at the exact focal length and aspect ratio: ``blender --python open_in_blender.py``).
2. :func:`reanchor_shot_from_screenshot` accepts a screenshot captured by the user from **Blender** or the **3D Viewer**,
   resolves the shot's character and location reference sheets via the strict 3-tier Bible resolver
   (``SceneBible -> GlobalContextBible -> AskUser``), and regenerates the shot's storyboard keyframe (``board_frame``)
   to match the user's exact 3D composition.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from ..bible_hierarchy import build_global_context_bible, resolve_reference
from ..decisions import DecisionLog
from ..fsio import generated_by, next_version_path, save_versioned
from ..models import ClipStatus, DecisionAction, DecisionTier
from ..project import Project, parse_artifact_id
from .schema import Mannequin, Primitive, SetLayout
from .store import load_set

AGENT = "agy.set3d.blender"


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return (0.6, 0.6, 0.6)
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _box_vertices_faces(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    rot_y_deg: float,
    v_offset: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    rad = math.radians(rot_y_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    local_corners = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    verts: list[tuple[float, float, float]] = []
    for lx, ly, lz in local_corners:
        rx = lx * cos_a + lz * sin_a
        rz = -lx * sin_a + lz * cos_a
        verts.append((cx + rx, cy + ly, cz + rz))

    o = v_offset
    faces = [
        (o + 1, o + 2, o + 3, o + 4),
        (o + 5, o + 8, o + 7, o + 6),
        (o + 1, o + 5, o + 6, o + 2),
        (o + 2, o + 6, o + 7, o + 3),
        (o + 3, o + 7, o + 8, o + 4),
        (o + 5, o + 1, o + 4, o + 8),
    ]
    return verts, faces


def _mannequin_boxes(m: Mannequin) -> list[Primitive]:
    """Represent a posed mannequin as torso + head bounding primitives for OBJ export."""
    x, y, z = m.position
    h = m.height_m if m.pose != "sitting" else m.height_m * 0.78
    torso_h = h * 0.72
    head_s = max(0.20, h * 0.16)
    return [
        Primitive(
            id=f"{m.id}_body",
            kind="box",
            label=f"{m.character_id} ({m.pose} body)",
            position=(x, y + torso_h / 2.0, z),
            size=(0.42, torso_h, 0.26),
            rotation_y_deg=m.facing_deg,
            color=m.wardrobe_color,
        ),
        Primitive(
            id=f"{m.id}_head",
            kind="box",
            label=f"{m.character_id} (head)",
            position=(x, y + torso_h + head_s / 2.0 + 0.03, z),
            size=(head_s, head_s, head_s),
            rotation_y_deg=m.facing_deg,
            color=m.skin_color,
        ),
    ]


def export_blender_scene(project: Project, env_id: str) -> dict[str, str]:
    """Export ``SetLayout`` to ``bible/sets/<ENV_ID>/blender/`` (.obj, .mtl, .gltf, open_in_blender.py)."""
    layout: SetLayout = load_set(project, env_id.upper())
    out_dir = project.abs(f"bible/sets/{layout.environment_id}/blender")
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_path = out_dir / "scene.obj"
    mtl_path = out_dir / "scene.mtl"
    gltf_path = out_dir / "scene.gltf"
    py_path = out_dir / "open_in_blender.py"

    all_blocks: list[Primitive] = list(layout.primitives)
    for m in layout.mannequins:
        if m.visible:
            all_blocks.extend(_mannequin_boxes(m))

    obj_lines = [
        f"# AGY Studio 3D Virtual Set Export: {layout.environment_id}",
        f"mtllib {mtl_path.name}",
        "",
    ]
    mtl_lines = [f"# AGY Studio Materials for {layout.environment_id}", ""]

    v_offset = 0
    seen_mats: set[str] = set()
    for blk in all_blocks:
        mat_name = f"mat_{blk.id}"
        if mat_name not in seen_mats:
            seen_mats.add(mat_name)
            r, g, b = _hex_to_rgb01(blk.color)
            mtl_lines += [
                f"newmtl {mat_name}",
                f"Kd {r:.4f} {g:.4f} {b:.4f}",
                "Ka 0.1000 0.1000 0.1000",
                "Ks 0.0500 0.0500 0.0500",
                "d 1.0",
                "",
            ]
        verts, faces = _box_vertices_faces(blk.position, blk.size, blk.rotation_y_deg, v_offset)
        v_offset += len(verts)
        obj_lines.append(f"o {blk.id}")
        obj_lines.append(f"usemtl {mat_name}")
        for vx, vy, vz in verts:
            obj_lines.append(f"v {vx:.4f} {vy:.4f} {vz:.4f}")
        for f1, f2, f3, f4 in faces:
            obj_lines.append(f"f {f1} {f2} {f3} {f4}")
        obj_lines.append("")

    obj_path.write_text("\n".join(obj_lines), encoding="utf-8")
    mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")

    # Write glTF 2.0 scene hierarchy metadata (nodes for primitives, mannequins, lights, and cameras)
    nodes: list[dict[str, Any]] = []
    cameras_gltf: list[dict[str, Any]] = []
    for blk in layout.primitives:
        nodes.append(
            {
                "name": blk.id,
                "translation": list(blk.position),
                "scale": list(blk.size),
                "extras": {"kind": blk.kind, "label": blk.label, "color": blk.color, "material_note": blk.material_note},
            }
        )
    for m in layout.mannequins:
        nodes.append(
            {
                "name": f"MANNEQUIN_{m.character_id}",
                "translation": list(m.position),
                "extras": {
                    "character_id": m.character_id,
                    "pose": m.pose,
                    "height_m": m.height_m,
                    "facing_deg": m.facing_deg,
                    "wardrobe_color": m.wardrobe_color,
                },
            }
        )
    for idx, cam in enumerate(layout.cameras):
        cameras_gltf.append(
            {
                "name": cam.name,
                "type": "perspective",
                "perspective": {
                    "aspectRatio": cam.aspect_value(),
                    "yfov": math.radians(cam.vertical_fov_deg()),
                    "znear": 0.05,
                    "zfar": 100.0,
                },
            }
        )
        nodes.append(
            {
                "name": f"CAM_{cam.name}",
                "camera": idx,
                "translation": list(cam.position),
                "extras": {"look_at": list(cam.look_at), "focal_mm": cam.focal_mm, "aspect_ratio": cam.aspect_ratio},
            }
        )

    gltf_doc = {
        "asset": {"version": "2.0", "generator": "AGY Studio v4 set3d Blender Exporter"},
        "scene": 0,
        "scenes": [{"name": layout.environment_id, "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "cameras": cameras_gltf,
    }
    gltf_path.write_text(json.dumps(gltf_doc, indent=2), encoding="utf-8")

    # Write executable Blender Python script (`blender --python open_in_blender.py`)
    blend_path = out_dir / f"{layout.environment_id}.blend"
    render_preview_path = out_dir / f"{layout.environment_id}_WIDE_MASTER_blender.png"
    wall_photo_path = project.abs("library/environments/ENV_RISHIS_HOUSE/v2/wall_family_photo.png")
    mother_ref_path = project.abs("library/packs/CHAR_MOTHER_STATE_DEFAULT/v2/front.png")
    room_ref_path = project.abs("library/environments/ENV_RISHIS_HOUSE/v2/dining_room_master.jpg")

    layout_json = layout.model_dump_json()
    blender_py = f'''"""Auto-generated Blender 4.x/5.x 3D Set & Articulated Cast Builder for {layout.environment_id}.

Run interactively in Blender:
    /Applications/Blender.app/Contents/MacOS/Blender --python {py_path}

Or build & save .blend + preview render headlessly:
    /Applications/Blender.app/Contents/MacOS/Blender -b --python {py_path} -- --save-and-render
"""
import json
import math
import os
import sys

LAYOUT = json.loads({layout_json!r})
BLEND_OUT = {str(blend_path)!r}
RENDER_OUT = {str(render_preview_path)!r}
WALL_PHOTO_IMG = {str(wall_photo_path)!r}
MOTHER_REF_IMG = {str(mother_ref_path)!r}
ROOM_REF_IMG = {str(room_ref_path)!r}

try:
    import bpy
    import mathutils

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 100
    scene.unit_settings.system = "METRIC"

    # Choose fast EEVEE engine for instant viewport & render previews
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue

    def hex_to_rgba(h, alpha=1.0):
        h = (h or "#999999").lstrip("#")
        if len(h) != 6:
            return (0.6, 0.6, 0.6, alpha)
        # Convert sRGB hex to approximate linear RGB for Blender shaders
        r, g, b = [((int(h[i:i+2], 16) / 255.0) ** 2.2) for i in (0, 2, 4)]
        return (r, g, b, alpha)

    _MAT_CACHE = {{}}

    def make_mat(name, hex_color, roughness=0.65, metallic=0.0, emission_strength=0.0, image_path=None):
        key = (name, hex_color, roughness, metallic, emission_strength, image_path)
        if key in _MAT_CACHE:
            return _MAT_CACHE[key]
        mat = bpy.data.materials.new(name=name)
        rgba = hex_to_rgba(hex_color, 1.0)
        mat.diffuse_color = rgba
        mat.roughness = roughness
        mat.metallic = metallic
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = roughness
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = metallic
            if emission_strength > 0:
                for em_key in ("Emission Color", "Emission"):
                    if em_key in bsdf.inputs:
                        bsdf.inputs[em_key].default_value = rgba
                        break
                if "Emission Strength" in bsdf.inputs:
                    bsdf.inputs["Emission Strength"].default_value = emission_strength
            if image_path and os.path.isfile(image_path):
                try:
                    tex = nt.nodes.new("ShaderNodeTexImage")
                    tex.image = bpy.data.images.load(image_path)
                    nt.links.new(bsdf.inputs["Base Color"], tex.outputs["Color"])
                except Exception:
                    pass
        _MAT_CACHE[key] = mat
        return mat

    def ensure_collection(name):
        if name in bpy.data.collections:
            return bpy.data.collections[name]
        col = bpy.data.collections.new(name)
        scene.collection.children.link(col)
        return col

    col_arch = ensure_collection("01_Room_Architecture")
    col_props = ensure_collection("02_Furniture_And_Tableware")
    col_chars = ensure_collection("03_Characters_Articulated_3D")
    col_refs = ensure_collection("04_Reference_Image_Plates")
    col_refs.hide_render = True
    col_cams = ensure_collection("05_Cameras_And_Lighting")

    def link_only(obj, target_col):
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        target_col.objects.link(obj)

    def add_box(name, loc, dims, mat, col, rot_euler=(0.0, 0.0, 0.0), parent=None):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot_euler)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = (dims[0], dims[1], dims[2])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        link_only(obj, col)
        if parent is not None:
            obj.parent = parent
        return obj

    def add_cyl(name, loc, radius, depth, mat, col, rot_euler=(0.0, 0.0, 0.0), parent=None):
        bpy.ops.mesh.primitive_cylinder_add(vertices=28, radius=radius, depth=depth, location=loc, rotation=rot_euler)
        obj = bpy.context.active_object
        obj.name = name
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        link_only(obj, col)
        if parent is not None:
            obj.parent = parent
        return obj

    def add_sphere(name, loc, radius, mat, col, scale=(1.0, 1.0, 1.0), parent=None):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=radius, location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        for poly in obj.data.polygons:
            poly.use_smooth = True
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        link_only(obj, col)
        if parent is not None:
            obj.parent = parent
        return obj

    # 1. Build Room Architecture & Furniture from LAYOUT["primitives"]
    for p in LAYOUT.get("primitives", []):
        pid = p["id"]
        kind = p.get("kind", "box")
        px, py, pz = p["position"]
        sx, sy, sz = p["size"]
        # Three.js (x, y, z) -> Blender (x, -z, y)
        bloc = (px, -pz, py)
        rot_z = math.radians(float(p.get("rotation_y_deg", 0.0)))
        col_hex = p.get("color", "#999999")
        is_struct = bool(p.get("structural", False))
        target_col = col_arch if is_struct or any(k in pid for k in ("door", "window", "clock", "photo", "switch", "fan")) else col_props

        # Choose material properties based on object role
        is_metal = any(k in pid for k in ("plate", "tumbler", "roti_bowl", "latch"))
        is_wood = any(k in pid for k in ("table", "chair", "door", "frame"))
        is_window = "window_glass" in pid
        mat = make_mat(
            f"MAT_{{pid}}",
            col_hex,
            roughness=0.22 if is_metal else (0.42 if is_wood else 0.78),
            metallic=0.85 if is_metal else 0.0,
            emission_strength=2.5 if is_window else 0.0,
        )

        # Build open slatted chair backrests (2 vertical posts + 3 horizontal rails with open gaps between them)
        if pid.endswith("_back") and "chair_" in pid:
            slat_mat = make_mat("MAT_chair_slat", "#5c381c", roughness=0.4)
            if sx >= sz:
                # Backrest runs along X (North/South chairs)
                for p_idx, x_off in enumerate((-sx * 0.46, sx * 0.46)):
                    add_box(f"{{pid}}_post_{{p_idx}}", (bloc[0] + x_off, bloc[1], bloc[2]), (0.04, sz, sy), slat_mat, col_props)
                for s_idx, z_off in enumerate((-0.14, 0.02, 0.18)):
                    add_box(f"{{pid}}_slat_{{s_idx}}", (bloc[0], bloc[1], bloc[2] + z_off), (sx * 0.94, sz * 0.9, 0.055), slat_mat, col_props)
            else:
                # Backrest runs along Y (East/West chairs)
                for p_idx, y_off in enumerate((-sz * 0.46, sz * 0.46)):
                    add_box(f"{{pid}}_post_{{p_idx}}", (bloc[0], bloc[1] + y_off, bloc[2]), (sx, 0.04, sy), slat_mat, col_props)
                for s_idx, z_off in enumerate((-0.14, 0.02, 0.18)):
                    add_box(f"{{pid}}_slat_{{s_idx}}", (bloc[0], bloc[1], bloc[2] + z_off), (sx * 0.9, sz * 0.94, 0.055), slat_mat, col_props)
            continue

        if kind == "cylinder":
            if "clock" in pid:
                # Wall clock sits flat against the back wall (axis along Y)
                add_cyl(pid, bloc, sx * 0.5, sy, mat, target_col, rot_euler=(math.radians(90.0), 0.0, 0.0))
            else:
                add_cyl(pid, bloc, sx * 0.5, sy, mat, target_col, rot_euler=(0.0, 0.0, rot_z))
        elif kind == "sphere":
            add_sphere(pid, bloc, sx * 0.5, mat, target_col)
        else:
            add_box(pid, bloc, (sx, sz, sy), mat, target_col, rot_euler=(0.0, 0.0, rot_z))

        # Add textured canvas inside the framed family photo on the back wall
        if pid == "family_photo_frame":
            photo_mat = make_mat("MAT_family_photo_art", "#d8c3a5", roughness=0.4, image_path=WALL_PHOTO_IMG)
            add_box("family_photo_canvas", (bloc[0], bloc[1] - 0.018, bloc[2]), (sx * 0.86, 0.005, sy * 0.84), photo_mat, col_arch)

    # 2. Build Articulated 3D Characters for all Mannequins in the Room
    hair_mat = make_mat("MAT_Hair_Dark", "#161413", roughness=0.55)
    dark_pants_mat = make_mat("MAT_Dark_Trousers", "#1f232b", roughness=0.7)
    phone_mat = make_mat("MAT_Smartphone", "#121317", roughness=0.2, metallic=0.4)
    steel_mat = make_mat("MAT_Steel_Cup", "#cfd6dd", roughness=0.2, metallic=0.88)
    paper_mat = make_mat("MAT_Newspaper", "#e8e4da", roughness=0.8)
    earphone_mat = make_mat("MAT_Earphones_White", "#f5f5f5", roughness=0.3)

    for mq in LAYOUT.get("mannequins", []):
        if not mq.get("visible", True):
            continue
        cid = mq["character_id"]
        label = mq.get("label") or cid
        h = float(mq.get("height_m", 1.72))
        pose = mq.get("pose", "standing")
        mx, my, mz = mq["position"]
        facing_deg = float(mq.get("facing_deg", 0.0))

        # Root Empty positioned on floor at (mx, -mz, my) and rotated so local +Y is forward
        root = bpy.data.objects.new(f"CHAR_{{cid}}", None)
        root.empty_display_type = "ARROWS"
        root.empty_display_size = 0.25
        root.location = (mx, -mz, my)
        root.rotation_euler = (0.0, 0.0, -math.radians(facing_deg))
        col_chars.objects.link(root)

        skin_mat = make_mat(f"MAT_Skin_{{cid}}", mq.get("skin_color", "#b88766"), roughness=0.55)
        top_mat = make_mat(f"MAT_Wardrobe_{{cid}}", mq.get("wardrobe_color", "#3b5998"), roughness=0.68)

        if pose == "sitting":
            seat_z = 0.46
            torso_z = seat_z + 0.27
            shoulder_z = seat_z + 0.46
            head_z = seat_z + 0.63

            # Pelvis / Hips on chair seat
            add_box(f"{{cid}}_Pelvis", (0.0, -0.02, seat_z + 0.04), (0.32, 0.24, 0.10), dark_pants_mat, col_chars, parent=root)
            # Torso
            add_box(f"{{cid}}_Torso", (0.0, -0.02, torso_z), (0.34, 0.20, 0.40), top_mat, col_chars, parent=root)
            # Neck & Head
            add_cyl(f"{{cid}}_Neck", (0.0, 0.0, shoulder_z + 0.05), 0.048, 0.09, skin_mat, col_chars, parent=root)
            add_sphere(f"{{cid}}_Head", (0.0, 0.02, head_z), 0.105, skin_mat, col_chars, scale=(0.92, 1.0, 1.12), parent=root)
            # Hair cap
            add_sphere(f"{{cid}}_Hair", (0.0, 0.0, head_z + 0.03), 0.108, hair_mat, col_chars, scale=(0.95, 1.02, 0.95), parent=root)
            # Eyes (looking forward along local +Y)
            for side_x in (-0.036, 0.036):
                add_sphere(f"{{cid}}_Eye_{{side_x}}", (side_x, 0.098, head_z + 0.01), 0.012, hair_mat, col_chars, parent=root)

            # Seated Thighs (extending forward along +Y) & Calves (dropping down to floor)
            for side_lbl, sx in (("L", -0.10), ("R", 0.10)):
                add_box(f"{{cid}}_Thigh_{{side_lbl}}", (sx, 0.16, seat_z + 0.03), (0.11, 0.34, 0.10), dark_pants_mat, col_chars, parent=root)
                add_cyl(f"{{cid}}_Calf_{{side_lbl}}", (sx, 0.31, 0.23), 0.048, 0.42, dark_pants_mat, col_chars, parent=root)
                add_box(f"{{cid}}_Foot_{{side_lbl}}", (sx, 0.36, 0.03), (0.09, 0.18, 0.05), skin_mat, col_chars, parent=root)

            # Character-specific seated arm poses & script props
            if cid == "CHAR_FATHER":
                # Father: eyeglasses, right hand sipping chai tumbler at mouth, left hand holding newspaper
                glass_frame_mat = make_mat("MAT_Father_Glasses", "#222222", roughness=0.2, metallic=0.8)
                add_box("CHAR_FATHER_Glasses", (0.0, 0.108, head_z + 0.012), (0.11, 0.02, 0.025), glass_frame_mat, col_chars, parent=root)
                # Left arm extended forward holding newspaper
                add_cyl("CHAR_FATHER_Arm_L", (-0.20, 0.10, shoulder_z - 0.09), 0.042, 0.26, top_mat, col_chars, rot_euler=(math.radians(-45), 0, 0), parent=root)
                add_cyl("CHAR_FATHER_Forearm_L", (-0.18, 0.28, shoulder_z - 0.14), 0.036, 0.24, skin_mat, col_chars, rot_euler=(math.radians(-70), 0, math.radians(15)), parent=root)
                add_box("PROP_Father_Newspaper", (-0.06, 0.38, shoulder_z - 0.06), (0.28, 0.015, 0.32), paper_mat, col_chars, rot_euler=(math.radians(25), 0, math.radians(-15)), parent=root)
                # Right arm raised with chai tumbler near lips
                add_cyl("CHAR_FATHER_Arm_R", (0.20, 0.10, shoulder_z - 0.06), 0.042, 0.25, top_mat, col_chars, rot_euler=(math.radians(-55), 0, 0), parent=root)
                add_cyl("CHAR_FATHER_Forearm_R", (0.12, 0.22, shoulder_z + 0.04), 0.036, 0.24, skin_mat, col_chars, rot_euler=(math.radians(-35), 0, math.radians(-25)), parent=root)
                add_cyl("PROP_Father_Chai_Glass", (0.04, 0.18, head_z - 0.05), 0.032, 0.095, steel_mat, col_chars, parent=root)

            elif cid == "CHAR_RISHI":
                # Rishi: white wired earphones in ears, looking down at smartphone in left hand, eating with right hand
                add_cyl("CHAR_RISHI_Arm_L", (-0.20, 0.10, shoulder_z - 0.11), 0.042, 0.26, top_mat, col_chars, rot_euler=(math.radians(-40), 0, 0), parent=root)
                add_cyl("CHAR_RISHI_Forearm_L", (-0.12, 0.28, 0.83), 0.036, 0.24, skin_mat, col_chars, rot_euler=(math.radians(-80), 0, math.radians(20)), parent=root)
                add_box("PROP_Rishi_Phone", (-0.08, 0.36, 0.85), (0.075, 0.14, 0.012), phone_mat, col_chars, rot_euler=(math.radians(35), 0, 0), parent=root)
                add_cyl("CHAR_RISHI_Arm_R", (0.20, 0.10, shoulder_z - 0.11), 0.042, 0.26, top_mat, col_chars, rot_euler=(math.radians(-45), 0, 0), parent=root)
                add_cyl("CHAR_RISHI_Forearm_R", (0.10, 0.28, 0.84), 0.036, 0.24, skin_mat, col_chars, rot_euler=(math.radians(-75), 0, math.radians(-20)), parent=root)
                # Wired earphones from ears down to phone
                for ex in (-0.095, 0.095):
                    add_cyl(f"PROP_Rishi_Earphone_Wire_{{ex}}", (ex * 0.6, 0.16, 0.96), 0.005, 0.28, earphone_mat, col_chars, rot_euler=(math.radians(-25), 0, 0), parent=root)

            elif cid == "CHAR_RIA":
                # Ria: ponytail at back of head, right hand over poha plate, left hand holding phone secretly under table!
                add_sphere("CHAR_RIA_Ponytail", (0.0, -0.12, head_z - 0.02), 0.065, hair_mat, col_chars, scale=(0.7, 1.3, 1.6), parent=root)
                add_cyl("CHAR_RIA_Arm_R", (0.19, 0.10, shoulder_z - 0.11), 0.038, 0.25, top_mat, col_chars, rot_euler=(math.radians(-45), 0, 0), parent=root)
                add_cyl("CHAR_RIA_Forearm_R", (0.12, 0.28, 0.82), 0.033, 0.22, skin_mat, col_chars, rot_euler=(math.radians(-80), 0, math.radians(-18)), parent=root)
                # Left hand secretly holding phone below the table edge (z = 0.64m < table 0.75m)
                add_cyl("CHAR_RIA_Arm_L", (-0.19, 0.06, shoulder_z - 0.13), 0.038, 0.25, top_mat, col_chars, rot_euler=(math.radians(-20), 0, 0), parent=root)
                add_cyl("CHAR_RIA_Forearm_L", (-0.14, 0.22, 0.66), 0.033, 0.22, skin_mat, col_chars, rot_euler=(math.radians(-65), 0, math.radians(15)), parent=root)
                add_box("PROP_Ria_Hidden_Phone", (-0.10, 0.30, 0.64), (0.075, 0.14, 0.012), phone_mat, col_chars, rot_euler=(math.radians(20), 0, 0), parent=root)

        else:
            # Standing pose (CHAR_MOTHER entering from kitchen / CHAR_DELIVERY_MAN outside door)
            salwar_mat = make_mat(f"MAT_Salwar_{{cid}}", "#4a1818" if cid == "CHAR_MOTHER" else "#1f2937", roughness=0.7)
            dupatta_mat = make_mat(f"MAT_Dupatta_{{cid}}", "#7a542e", roughness=0.65)
            head_z = h - 0.12
            shoulder_z = h - 0.28
            torso_z = h - 0.52

            # Legs
            for side_lbl, sx in (("L", -0.09), ("R", 0.09)):
                add_cyl(f"{{cid}}_Leg_{{side_lbl}}", (sx, 0.0, 0.42), 0.055, 0.78, salwar_mat, col_chars, parent=root)
                add_box(f"{{cid}}_Foot_{{side_lbl}}", (sx, 0.06, 0.03), (0.09, 0.20, 0.05), skin_mat, col_chars, parent=root)

            # Long Kameez / Tunic or Shirt
            tunic_len = 0.68 if cid == "CHAR_MOTHER" else 0.48
            tunic_z = h - 0.60 if cid == "CHAR_MOTHER" else torso_z
            add_box(f"{{cid}}_Torso", (0.0, 0.0, tunic_z), (0.33, 0.21, tunic_len), top_mat, col_chars, parent=root)
            add_cyl(f"{{cid}}_Neck", (0.0, 0.0, shoulder_z + 0.05), 0.045, 0.09, skin_mat, col_chars, parent=root)
            add_sphere(f"{{cid}}_Head", (0.0, 0.02, head_z), 0.102, skin_mat, col_chars, scale=(0.92, 1.0, 1.12), parent=root)
            add_sphere(f"{{cid}}_Hair", (0.0, -0.01, head_z + 0.02), 0.105, hair_mat, col_chars, scale=(0.95, 1.05, 0.95), parent=root)

            if cid == "CHAR_MOTHER":
                # Draped brown dupatta across chest & shoulders + low hair bun + kitchen cloth in hands
                add_sphere("CHAR_MOTHER_Bun", (0.0, -0.11, head_z - 0.02), 0.055, hair_mat, col_chars, parent=root)
                add_box("CHAR_MOTHER_Dupatta_Drape", (0.0, 0.06, shoulder_z - 0.12), (0.36, 0.12, 0.30), dupatta_mat, col_chars, parent=root)
                for side_lbl, sx, s_sign in (("L", -0.20, 1), ("R", 0.20, -1)):
                    add_cyl(f"CHAR_MOTHER_Arm_{{side_lbl}}", (sx, 0.04, shoulder_z - 0.12), 0.038, 0.25, top_mat, col_chars, rot_euler=(math.radians(-20), 0, 0), parent=root)
                    add_cyl(f"CHAR_MOTHER_Forearm_{{side_lbl}}", (sx * 0.65, 0.15, shoulder_z - 0.24), 0.032, 0.22, skin_mat, col_chars, rot_euler=(math.radians(-65), 0, math.radians(s_sign * 25)), parent=root)
                cloth_mat = make_mat("MAT_Mother_Cloth", "#e8e2d5", roughness=0.8)
                add_box("PROP_Mother_Cloth", (0.0, 0.22, shoulder_z - 0.25), (0.12, 0.08, 0.06), cloth_mat, col_chars, parent=root)

                # Also add a textured 3D Reference Image Card outside the left wall in 04_Reference_Image_Plates
                if os.path.isfile(MOTHER_REF_IMG):
                    m_card_mat = make_mat("MAT_Mother_Ref_Card", "#ffffff", roughness=0.5, image_path=MOTHER_REF_IMG)
                    add_box("REF_PLATE_CHAR_MOTHER", (-2.75, 1.4, 0.85), (0.55, 0.02, 1.65), m_card_mat, col_refs)
            else:
                # Delivery man holding the brown parcel box
                for side_lbl, sx in (("L", -0.20), ("R", 0.20)):
                    add_cyl(f"{{cid}}_Arm_{{side_lbl}}", (sx, 0.08, shoulder_z - 0.12), 0.040, 0.26, top_mat, col_chars, rot_euler=(math.radians(-35), 0, 0), parent=root)
                parcel_mat = make_mat("MAT_Parcel_Cardboard", "#b08252", roughness=0.75)
                add_box("PROP_Delivery_Parcel_Box", (0.0, 0.24, shoulder_z - 0.22), (0.28, 0.24, 0.18), parcel_mat, col_chars, parent=root)

        # Floating 3D text label above each character so the Director can identify everyone immediately
        try:
            bpy.ops.object.text_add(location=(0.0, 0.0, (0.46 + 0.82) if pose == "sitting" else (h + 0.12)), rotation=(math.radians(90.0), 0.0, math.radians(180.0)))
            txt_obj = bpy.context.active_object
            txt_obj.name = f"LABEL_{{cid}}"
            txt_obj.data.body = label.split("(")[0].strip()
            txt_obj.data.size = 0.09
            txt_obj.data.align_x = "CENTER"
            txt_obj.data.extrude = 0.008
            lbl_mat = make_mat(f"MAT_Label_{{cid}}", "#ffffff", roughness=0.3, emission_strength=1.5)
            txt_obj.data.materials.append(lbl_mat)
            link_only(txt_obj, col_chars)
            txt_obj.parent = root
        except Exception:
            pass

    # Add the master room reference plate in 04_Reference_Image_Plates (placed to the left outside the room for side-by-side comparison)
    if os.path.isfile(ROOM_REF_IMG):
        r_card_mat = make_mat("MAT_Room_Master_Ref", "#ffffff", roughness=0.5, image_path=ROOM_REF_IMG)
        add_box("REF_PLATE_MASTER_ROOM", (-2.75, 0.0, 1.35), (1.35, 0.02, 2.40), r_card_mat, col_refs)

    # 3. Build Scene Lighting from LAYOUT["lights"]
    for lspec in LAYOUT.get("lights", []):
        lid = lspec["id"]
        lkind = lspec.get("kind", "point")
        lx, ly, lz = lspec["position"]
        bloc = (lx, -lz, ly)
        btype = "SUN" if lkind == "sun" else ("AREA" if lkind == "area" else ("SPOT" if lkind == "spot" else "POINT"))
        ldata = bpy.data.lights.new(name=lid, type=btype)
        rgba = hex_to_rgba(lspec.get("color", "#fff5e1"))
        ldata.color = rgba[:3]
        raw_int = float(lspec.get("intensity", 60.0))
        ldata.energy = 3.5 if btype == "SUN" else raw_int * 2.5
        lobj = bpy.data.objects.new(name=f"LIGHT_{{lid}}", object_data=ldata)
        lobj.location = bloc
        if lspec.get("target"):
            tx, ty, tz = lspec["target"]
            t_vec = mathutils.Vector((tx - lx, -(tz - lz), ty - ly))
            if t_vec.length > 1e-4:
                lobj.rotation_euler = t_vec.to_track_quat("-Z", "Y").to_euler()
        col_cams.objects.link(lobj)

    # 4. Build Saved Cameras with exact focal length, 9:16 sensor framing, and look_at orientation
    active_cam = None
    for cam_spec in LAYOUT.get("cameras", []):
        cname = cam_spec["name"]
        cam_data = bpy.data.cameras.new(name=cname)
        cam_data.lens = float(cam_spec.get("focal_mm", 35.0))
        cam_data.sensor_width = 36.0
        cam_obj = bpy.data.objects.new(f"CAM_{{cname}}", cam_data)
        col_cams.objects.link(cam_obj)
        px, py, pz = cam_spec["position"]
        lx, ly, lz = cam_spec["look_at"]
        cam_obj.location = (px, -pz, py)
        direction = mathutils.Vector((lx - px, -(lz - pz), ly - py))
        if direction.length > 1e-4:
            cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        if cname == "WIDE_MASTER" or active_cam is None:
            active_cam = cam_obj

    if active_cam is not None:
        scene.camera = active_cam

    # Set all 3D Viewports to Material Preview + Camera Perspective when opened in GUI mode
    if not bpy.app.background:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.shading.type = "MATERIAL"
                            if space.region_3d:
                                space.region_3d.view_perspective = "CAMERA"

    # Always save the .blend file so it can be opened directly in Blender.app
    os.makedirs(os.path.dirname(BLEND_OUT), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT)
    print("Saved Blender file:", BLEND_OUT)

    if "--save-and-render" in sys.argv:
        scene.render.engine = "CYCLES"
        if hasattr(scene, "cycles"):
            scene.cycles.device = "CPU"
            scene.cycles.samples = 8
            if hasattr(scene.cycles, "use_denoising"):
                scene.cycles.use_denoising = False
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = RENDER_OUT
        bpy.ops.render.render(write_still=True)
        print("Rendered preview image:", RENDER_OUT)

        cam_dir = os.path.join(os.path.dirname(BLEND_OUT), "cam_renders_9_16")
        os.makedirs(cam_dir, exist_ok=True)
        for obj in list(bpy.data.objects):
            if obj.type == "CAMERA" and obj.name.startswith("CAM_"):
                short_name = obj.name[4:]
                scene.camera = obj
                out_cam_png = os.path.join(cam_dir, f"{{short_name}}_3d_clean.png")
                scene.render.filepath = out_cam_png
                bpy.ops.render.render(write_still=True)
                print("Rendered camera blockout:", out_cam_png)

except ImportError:
    print("Loaded layout outside Blender:", LAYOUT["environment_id"])
'''
    py_path.write_text(blender_py, encoding="utf-8")
    fpv_html_path = export_standalone_fpv_html(project, layout)

    # Refresh Global Context Bible with the blender_dir_uri
    build_global_context_bible(project, save=True)

    return {
        "environment_id": layout.environment_id,
        "blender_dir": project.rel(out_dir),
        "obj": project.rel(obj_path),
        "mtl": project.rel(mtl_path),
        "gltf": project.rel(gltf_path),
        "blender_script": project.rel(py_path),
        "blend_file": project.rel(blend_path),
        "fpv_html": project.rel(fpv_html_path),
    }


def export_standalone_fpv_html(project: Project, layout: SetLayout) -> Path:
    """Build a 100% standalone, zero-Blender single-file HTML 3D FPV & Orbit Director viewer
    (Three.js + Custom GLSL Shaders at 60 FPS + articulated characters) that opens directly in any browser.
    """
    import base64

    viewer_dir = Path(__file__).with_name("viewer")
    index_html = (viewer_dir / "index.html").read_text(encoding="utf-8")
    viewer_css = (viewer_dir / "viewer.css").read_text(encoding="utf-8")
    viewer_js = (viewer_dir / "viewer.js").read_text(encoding="utf-8")
    three_js_bytes = (viewer_dir / "vendor" / "three.module.min.js").read_bytes()
    orbit_js_bytes = (viewer_dir / "vendor" / "addons" / "controls" / "OrbitControls.js").read_bytes()

    three_b64 = base64.b64encode(three_js_bytes).decode("ascii")
    orbit_b64 = base64.b64encode(orbit_js_bytes).decode("ascii")

    importmap_json = json.dumps(
        {
            "imports": {
                "three": f"data:text/javascript;base64,{three_b64}",
                "three/addons/controls/OrbitControls.js": f"data:text/javascript;base64,{orbit_b64}",
            }
        }
    )
    layout_json = json.dumps(layout.model_dump(mode="json"))

    standalone = index_html.replace(
        '<link rel="stylesheet" href="/static/viewer.css">',
        f"<style>\n{viewer_css}\n</style>",
    )
    standalone = standalone.replace(
        '{ "imports": { "three": "/static/vendor/three.module.min.js", "three/addons/": "/static/vendor/addons/" } }',
        importmap_json,
    )
    standalone = standalone.replace(
        '<script type="module" src="/static/viewer.js"></script>',
        f"<script>window.__INLINE_SET_LAYOUT__ = {layout_json};</script>\n"
        f'<script type="module">\n{viewer_js}\n</script>',
    )

    out_dir = project.abs(f"bible/sets/{layout.environment_id}/blender")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{layout.environment_id}_FPV_Director.html"
    html_path.write_text(standalone, encoding="utf-8")
    return html_path


def compute_view_space_blocking_summary(layout: SetLayout, camera_name: str | None = None) -> str:
    """Project 3D character positions into the active camera's view space (left/center/right, foreground/midground/background)
    so off-axis and Over-The-Shoulder (OTS) shots never suffer from wide-master camera-angle bleed.
    """
    if not layout.characters:
        return ""
    cam = None
    if camera_name:
        clean_name = camera_name.replace("cam:", "").strip()
        cam = next((c for c in layout.cameras if c.name == clean_name), None)
    if cam is None and layout.cameras:
        cam = layout.cameras[-1]
    if cam is None:
        return ""

    cx, cy, cz = cam.position
    tx, ty, tz = cam.look_at
    vx, vz = tx - cx, tz - cz
    v_len = math.hypot(vx, vz) or 1.0
    vx /= v_len
    vz /= v_len
    # Right vector in (x, z) plane when Up is +Y: Forward x Up = (vx, 0, vz) x (0, 1, 0) = (-vz, 0, vx)
    rx, rz = -vz, vx

    projected: list[tuple[float, float, str, str]] = []
    for ch in layout.characters:
        px, _py, pz = ch.position
        dx, dz = px - cx, pz - cz
        depth = dx * vx + dz * vz
        if depth <= 0.15 or depth > 3.2:
            continue
        horiz = dx * rx + dz * rz
        norm_x = horiz / max(depth, 0.35)
        h_zone = "LEFT" if norm_x < -0.15 else ("RIGHT" if norm_x > 0.15 else "CENTER")
        d_zone = "FOREGROUND (OTS)" if depth < 1.15 else ("MIDGROUND" if depth < 2.2 else "BACKGROUND")
        projected.append((norm_x, depth, f"{ch.label} [{d_zone} {h_zone}]", ch.character_id))

    if not projected:
        return ""
    projected.sort(key=lambda item: item[0])
    order_str = " -> ".join(item[2] for item in projected)
    return (
        f"VIEW-SPACE 3D FRUSTUM ORDER (Left-to-Right in camera {cam.name}): {order_str}. "
        "CRITICAL: Do NOT copy the frontal seating layout or background wall landmarks from any wide master sheet; "
        "preserve this exact 3D camera rotation and left-to-right character placement."
    )


def reanchor_shot_from_screenshot(
    project: Project,
    shot_id: str,
    screenshot_path: str | Path,
    *,
    image_adapter: Any,
    steer: str = "",
    user: str = "director",
) -> dict[str, Any]:
    """Use a user's Blender / 3D Viewer screenshot + resolved Bible Character & Location sheets
    (1. SceneBible -> 2. GlobalContextBible -> 3. AskUser) to recreate the shot's keyframe and mark downstream clips
    for re-generation.
    """
    ids = parse_artifact_id(shot_id)
    scene_id = ids["scene"]
    if not scene_id or not ids["shot"]:
        raise ValueError(f"Expected a shot id like SC01_SH01, got {shot_id!r}")

    src_screen = Path(screenshot_path).expanduser().resolve()
    if not src_screen.is_file():
        raise FileNotFoundError(f"3D/Blender screenshot not found at {screenshot_path}")

    scene = project.load_scene(scene_id)
    shot = project.load_shot(shot_id)

    # Copy user screenshot into scenes/<SC>/boards/screenshots/
    screen_dir = project.scene_dir(scene_id) / "boards" / "screenshots"
    screen_dir.mkdir(parents=True, exist_ok=True)
    staged_screen = next_version_path(screen_dir, f"{shot.id}_3d_screenshot", src_screen.suffix or ".png")
    shutil.copy2(src_screen, staged_screen)
    staged_screen_rel = project.rel(staged_screen)

    # Strict 3-tier reference lookup (SceneBible -> GlobalContextBible -> AskUser)
    reference_Paths: list[str] = [str(staged_screen)]
    reference_rel_uris: list[str] = [staged_screen_rel]
    reference_labels: list[str] = ["[Image1] 3D/Blender Camera & Blocking Screenshot (match exact angle & positions)"]
    unresolved_questions: list[str] = []

    if scene.environment_id:
        env_ref = resolve_reference(project, scene_id, scene.environment_id, ask_on_missing=True)
        if env_ref.sheet_uri and project.abs(env_ref.sheet_uri).is_file():
            idx = len(reference_Paths) + 1
            reference_Paths.append(str(project.abs(env_ref.sheet_uri)))
            reference_rel_uris.append(env_ref.sheet_uri)
            reference_labels.append(
                f"[Image{idx}] Location Sheet ({env_ref.resolved_id} from {env_ref.source_tier})"
            )
        elif env_ref.question_id:
            unresolved_questions.append(env_ref.question_id)

    char_targets = list(getattr(shot, "characters_in_frame", None) or scene.entity_states_active)
    for cid in char_targets:
        cref = resolve_reference(project, scene_id, cid, ask_on_missing=True)
        if cref.sheet_uri and project.abs(cref.sheet_uri).is_file():
            idx = len(reference_Paths) + 1
            reference_Paths.append(str(project.abs(cref.sheet_uri)))
            reference_rel_uris.append(cref.sheet_uri)
            reference_labels.append(
                f"[Image{idx}] Character/Prop Sheet ({cref.resolved_id} from {cref.source_tier}: {cref.description})"
            )
        elif cref.question_id:
            unresolved_questions.append(cref.question_id)

    view_space_note = ""
    if scene.environment_id:
        try:
            from .store import SetStore
            s_store = SetStore(project)
            if s_store.exists(scene.environment_id):
                s_layout = s_store.load(scene.environment_id)
                view_space_note = compute_view_space_blocking_summary(s_layout, getattr(shot, "spatial_anchor", None))
        except Exception:
            view_space_note = ""

    shot_desc = getattr(shot, "intention", "") or getattr(shot, "description", "") or shot.spec.visual.subject_and_action
    prompt_parts = [
        f"Recreate the storyboard keyframe for shot {shot.id} ({shot_desc}).",
        "CRITICAL SPATIAL LOCK: Follow [Image1] (user-supplied 3D/Blender camera & mannequin blocking screenshot) "
        "for the exact camera perspective, horizon line, lens framing, and character positions in the room.",
        "Replace any 3D grey-box surfaces and mannequins with the photorealistic location architecture and character "
        "likenesses/wardrobe from the Bible reference sheets: " + "; ".join(reference_labels) + ".",
    ]
    if view_space_note:
        prompt_parts.append(view_space_note)
    if steer.strip():
        prompt_parts.append(f"DIRECTOR STEER: {steer.strip()}")
    prompt = " ".join(prompt_parts)

    boards_dir = project.scene_dir(scene_id) / "boards"
    boards_dir.mkdir(parents=True, exist_ok=True)
    new_board_path = next_version_path(boards_dir, shot.id, ".png")
    aspect = project.config.settings.aspect_ratio
    if image_adapter is not None:
        image_adapter.generate(prompt, new_board_path, reference_images=reference_Paths, aspect_ratio=aspect)
    else:
        shutil.copy2(staged_screen, new_board_path)

    new_board_rel = project.rel(new_board_path)
    shot.board_frame = new_board_rel
    shot.version += 1
    shot.generated_by = generated_by(AGENT, notes=f"reanchored from {staged_screen_rel}")
    project.save_shot(shot)

    # Mark clips of this shot as stale / needs_relint so the scene re-generates from the new keyframe
    stale_clips: list[str] = []
    for clip in project.list_clips(shot.id):
        clip.conditioning.spatial_anchor = new_board_rel
        clip.status = ClipStatus.STALE
        clip.version += 1
        project.save_clip(clip)
        stale_clips.append(clip.id)

    dlog = DecisionLog(project)
    dec = dlog.record(
        tier=DecisionTier.CRAFT,
        action=DecisionAction.STEER,
        target_artifact_id=shot.id,
        gate="G3",
        user=user,
        payload={
            "scene_id": scene_id,
            "screenshot_uri": staged_screen_rel,
            "new_board_frame": new_board_rel,
            "reference_images_used": reference_rel_uris,
            "stale_clips": stale_clips,
        },
        rationale=steer or f"Re-anchored {shot.id} keyframe from 3D/Blender screenshot {src_screen.name}",
    )

    return {
        "shot_id": shot.id,
        "scene_id": scene_id,
        "screenshot": staged_screen_rel,
        "screenshot_uri": staged_screen_rel,
        "board_frame": new_board_rel,
        "reference_images": reference_rel_uris,
        "reference_images_used": reference_rel_uris,
        "unresolved_questions": unresolved_questions,
        "stale_clips": stale_clips,
        "decision_id": dec.id,
    }
