"""3D virtual set schema (set3d) — a to-scale grey-box layout of one Environment.

Units are metres, Y is up, the floor is the plane y=0. A layout is authored once per Environment (by the Set
Designer agent, or by AGY in manual mode), edited in the browser viewer, and every saved ``SetCamera`` can be
rendered into a blockout + depth reference and style-passed into ``Environment.derived_angles.cameras``.

The schema is intentionally small (boxes, planes, cylinders, spheres, lights, mannequins) so that an LLM can
author it reliably and the browser can rebuild it deterministically.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vec3 = tuple[float, float, float]
PrimitiveKind = Literal["box", "plane", "cylinder", "sphere"]
MannequinPose = Literal["standing", "sitting", "walking", "lying", "kneeling"]
LightKind = Literal["sun", "point", "spot", "area", "ambient"]

CAMERA_NAME_RX = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{0,31}")
HEX_RX = re.compile(r"#[0-9a-fA-F]{6}")
FULL_FRAME_SENSOR_MM = 36.0
ASPECT_VALUES: dict[str, float] = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "2.39:1": 2.39, "4:3": 4 / 3}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _finite(v: Any, what: str) -> Vec3:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"{what} must be [x, y, z]")
    out = []
    for x in v:
        f = float(x)
        if not math.isfinite(f):
            raise ValueError(f"{what} must be finite")
        out.append(f)
    return (out[0], out[1], out[2])


class Primitive(_Model):
    """One grey-box building block. ``size`` is full extent (w, h, d) for box/plane, (diameter, height, _) for a
    cylinder and (diameter, _, _) for a sphere. ``position`` is the centre; ``rotation_y_deg`` turns it about Y."""

    id: str
    kind: PrimitiveKind = "box"
    label: str = ""  # human description ("parapet wall", "water tank", "sofa")
    position: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3 = (1.0, 1.0, 1.0)
    rotation_y_deg: float = 0.0
    color: str = "#9a9a9a"  # flat shade used in the blockout (the style pass repaints it)
    material_note: str = ""  # free text for the style pass ("rough concrete", "wet tiles")
    structural: bool = False  # walls / floors / stairs: never auto-removed when the layout is simplified

    @field_validator("id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{0,47}", v):
            raise ValueError(f"primitive id {v!r} must be alphanumeric/_/- and start with a letter")
        return v

    @field_validator("position", "size", mode="before")
    @classmethod
    def _v_vec(cls, v: Any) -> Vec3:
        return _finite(v, "vector")

    @field_validator("size")
    @classmethod
    def _v_size(cls, v: Vec3) -> Vec3:
        if any(x <= 0 for x in v):
            raise ValueError("size components must be > 0")
        if any(x > 500 for x in v):
            raise ValueError("size components must be <= 500 m")
        return v

    @field_validator("color")
    @classmethod
    def _v_color(cls, v: str) -> str:
        if not HEX_RX.fullmatch(v):
            raise ValueError(f"color {v!r} must be #rrggbb")
        return v.lower()


class SetLight(_Model):
    id: str
    kind: LightKind = "point"
    position: Vec3 = (0.0, 3.0, 0.0)
    target: Vec3 | None = None  # for sun / spot
    color: str = "#ffffff"
    intensity: float = 1.0
    label: str = ""  # "sodium street lamp", "golden hour sun through the stairwell door"

    @field_validator("position", "target", mode="before")
    @classmethod
    def _v_vec(cls, v: Any) -> Any:
        return None if v is None else _finite(v, "light vector")

    @field_validator("color")
    @classmethod
    def _v_color(cls, v: str) -> str:
        if not HEX_RX.fullmatch(v):
            raise ValueError(f"color {v!r} must be #rrggbb")
        return v.lower()

    @field_validator("intensity")
    @classmethod
    def _v_int(cls, v: float) -> float:
        if not (0 <= v <= 1000):
            raise ValueError("intensity must be within 0..1000")
        return v


class Mannequin(_Model):
    """Simple posed stand-in for one character (blocking, eyelines, scale). Never a likeness."""

    id: str
    character_id: str  # Entity id (CHAR_*) or PROP_* for a hand prop holder
    label: str = ""
    height_m: float = 1.70
    pose: MannequinPose = "standing"
    position: Vec3 = (0.0, 0.0, 0.0)  # feet / contact point on the floor
    facing_deg: float = 0.0  # yaw, 0 = looking down -Z
    wardrobe_color: str = "#4a6fa5"
    skin_color: str = "#c8a27a"
    visible: bool = True

    @field_validator("position", mode="before")
    @classmethod
    def _v_vec(cls, v: Any) -> Vec3:
        return _finite(v, "mannequin position")

    @field_validator("height_m")
    @classmethod
    def _v_h(cls, v: float) -> float:
        if not (0.4 <= v <= 2.6):
            raise ValueError("height_m must be within 0.4..2.6")
        return v

    @field_validator("wardrobe_color", "skin_color")
    @classmethod
    def _v_color(cls, v: str) -> str:
        if not HEX_RX.fullmatch(v):
            raise ValueError(f"color {v!r} must be #rrggbb")
        return v.lower()


class SetCamera(_Model):
    """A saved camera. ``focal_mm`` is a full-frame-equivalent focal length (36 mm sensor width)."""

    name: str  # usually the shot id suffix: SH01, SH03_reverse
    position: Vec3 = (0.0, 1.6, 4.0)
    look_at: Vec3 = (0.0, 1.2, 0.0)
    focal_mm: float = 35.0
    roll_deg: float = 0.0
    aspect_ratio: Literal["16:9", "9:16", "1:1", "2.39:1", "4:3"] = "9:16"
    note: str = ""  # "over Rishi's shoulder onto Mohnish"
    shot_id: str | None = None  # CreativeShot id this camera is intended for
    locked: bool = False
    last_render: str | None = None  # project-relative dir of the latest blockout render

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        if not CAMERA_NAME_RX.fullmatch(v):
            raise ValueError(f"camera name {v!r}: letters, digits, _ or -, max 32, starts with a letter")
        return v

    @field_validator("position", "look_at", mode="before")
    @classmethod
    def _v_vec(cls, v: Any) -> Vec3:
        return _finite(v, "camera vector")

    @field_validator("focal_mm")
    @classmethod
    def _v_focal(cls, v: float) -> float:
        if not (8 <= v <= 400):
            raise ValueError("focal_mm must be within 8..400")
        return v

    @model_validator(mode="after")
    def _v_distinct(self) -> SetCamera:
        if all(abs(a - b) < 1e-6 for a, b in zip(self.position, self.look_at)):
            raise ValueError("camera position and look_at must differ")
        return self

    def horizontal_fov_deg(self) -> float:
        return math.degrees(2 * math.atan(FULL_FRAME_SENSOR_MM / (2 * self.focal_mm)))

    def aspect_value(self) -> float:
        return ASPECT_VALUES[self.aspect_ratio]

    def vertical_fov_deg(self) -> float:
        half_w = math.tan(math.radians(self.horizontal_fov_deg() / 2))
        return math.degrees(2 * math.atan(half_w / self.aspect_value()))

    def describe(self) -> str:
        """Lens language for prompts: 'low 24 mm wide from 3.9 m, looking slightly up'."""
        dx, dy, dz = (b - a for a, b in zip(self.position, self.look_at))
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        pitch = math.degrees(math.atan2(dy, math.sqrt(dx * dx + dz * dz)))
        h = self.position[1]
        height = "ground-level" if h < 0.6 else "low" if h < 1.2 else "eye-level" if h < 1.9 else \
            "high" if h < 3.5 else "overhead"
        f = self.focal_mm
        lens = "ultra-wide" if f < 20 else "wide" if f < 32 else "normal" if f < 55 else "portrait" if f < 100 \
            else "telephoto"
        tilt = "looking level" if abs(pitch) < 5 else ("looking up" if pitch > 0 else "looking down") + \
            (" steeply" if abs(pitch) > 30 else " slightly" if abs(pitch) < 15 else "")
        return f"{height} {int(round(f))} mm {lens} from {dist:.1f} m, {tilt}"


class SetLayout(_Model):
    environment_id: str
    title: str = ""
    units: Literal["m"] = "m"
    bounds: Vec3 = (12.0, 4.0, 12.0)  # overall w, h, d of the playable space, centred at origin
    floor_color: str = "#6e6e6e"
    sky_color: str = "#1c2230"
    axis_of_action: tuple[Vec3, Vec3] | None = None  # two floor points: the 180° line (L13)
    primitives: list[Primitive] = Field(default_factory=list, max_length=400)
    lights: list[SetLight] = Field(default_factory=list, max_length=32)
    mannequins: list[Mannequin] = Field(default_factory=list, max_length=24)
    cameras: list[SetCamera] = Field(default_factory=list, max_length=200)
    notes: str = ""  # designer's rationale: what the key landmarks are and where they sit
    source_still: str | None = None  # geometry still the layout was derived from
    version: int = 1
    generated_by: dict[str, Any] | None = None

    @field_validator("environment_id")
    @classmethod
    def _v_env(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", v):
            raise ValueError(f"environment_id {v!r} must be UPPER_SNAKE")
        return v

    @field_validator("bounds", mode="before")
    @classmethod
    def _v_bounds(cls, v: Any) -> Vec3:
        b = _finite(v, "bounds")
        if any(x <= 0 or x > 1000 for x in b):
            raise ValueError("bounds must be within 0..1000 m")
        return b

    @field_validator("axis_of_action", mode="before")
    @classmethod
    def _v_axis(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError("axis_of_action must be two [x, y, z] floor points")
        return (_finite(v[0], "axis point"), _finite(v[1], "axis point"))

    @field_validator("floor_color", "sky_color")
    @classmethod
    def _v_color(cls, v: str) -> str:
        if not HEX_RX.fullmatch(v):
            raise ValueError(f"color {v!r} must be #rrggbb")
        return v.lower()

    @model_validator(mode="after")
    def _v_unique(self) -> SetLayout:
        for what, items in (("primitive", self.primitives), ("light", self.lights), ("mannequin", self.mannequins)):
            ids = [getattr(i, "id") for i in items]
            dup = {i for i in ids if ids.count(i) > 1}
            if dup:
                raise ValueError(f"duplicate {what} id(s): {sorted(dup)}")
        names = [c.name for c in self.cameras]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate camera name(s): {sorted(dup)}")
        return self

    # -- cameras ----------------------------------------------------------------------------------
    def camera(self, name: str) -> SetCamera | None:
        return next((c for c in self.cameras if c.name == name), None)

    def upsert_camera(self, cam: SetCamera) -> SetCamera:
        existing = self.camera(cam.name)
        if existing is not None and existing.locked and existing.model_dump() != cam.model_dump():
            raise ValueError(f"camera {cam.name} is locked; unlock it before moving it")
        self.cameras = [c for c in self.cameras if c.name != cam.name] + [cam]
        return cam

    def remove_camera(self, name: str) -> bool:
        before = len(self.cameras)
        cam = self.camera(name)
        if cam is not None and cam.locked:
            raise ValueError(f"camera {name} is locked")
        self.cameras = [c for c in self.cameras if c.name != name]
        return len(self.cameras) != before

    def summary(self) -> str:
        parts = [f"{self.environment_id}: {len(self.primitives)} block(s), {len(self.lights)} light(s), "
                 f"{len(self.mannequins)} mannequin(s), {len(self.cameras)} camera(s)"]
        for c in sorted(self.cameras, key=lambda c: c.name):
            tag = " [locked]" if c.locked else ""
            rendered = " · rendered" if c.last_render else ""
            parts.append(f"  {c.name}{tag}: {c.describe()}{rendered}" + (f" — {c.note}" if c.note else ""))
        return "\n".join(parts)


def layout_json_schema() -> dict[str, Any]:
    """JSON schema handed to the LLM (cameras excluded: the editor authors those in the viewer)."""
    schema = SetLayout.model_json_schema()
    props = schema.get("properties", {})
    for drop in ("cameras", "generated_by", "version", "source_still", "units"):
        props.pop(drop, None)
    schema["required"] = [r for r in schema.get("required", []) if r in props]
    return schema
