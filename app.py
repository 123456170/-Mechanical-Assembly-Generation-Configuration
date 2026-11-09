import os
import sys
import io
import json
import math
import base64
import tempfile
import unittest
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SELF_TEST = "--self-test" in sys.argv

if not _SELF_TEST:
    import streamlit as st
else:
    st = None


# -----------------------------------------------------------------------------
# M1 constants
# -----------------------------------------------------------------------------

ALIGN_TOL = 0.05
AXIAL_TOL = 0.05
ROT_TOL = 0.25
COLLISION_TOL = 0.01
FIT_TOL = 0.02

DEFAULT_PARAMS: Dict[str, float] = {
    "shaft_diameter": 10.0,
    "shaft_length": 60.0,
    "shoulder_diameter": 15.0,
    "shoulder_width": 5.0,
    "key_length": 12.0,
    "key_height": 2.0,
    "key_width": 4.0,
    "bearing_bore": 10.0,
    "bearing_outer": 26.0,
    "bearing_width": 8.0,
}

SHAFT_VARIANTS: Dict[str, Dict[str, float]] = {
    "Shaft: 8 mm shoulder/keyed": {
        "shaft_diameter": 8.0,
        "shaft_length": 55.0,
        "shoulder_diameter": 12.0,
        "shoulder_width": 4.0,
        "key_length": 10.0,
        "key_height": 1.8,
        "key_width": 3.5,
    },
    "Shaft: 10 mm shoulder/keyed": {
        "shaft_diameter": 10.0,
        "shaft_length": 60.0,
        "shoulder_diameter": 15.0,
        "shoulder_width": 5.0,
        "key_length": 12.0,
        "key_height": 2.0,
        "key_width": 4.0,
    },
    "Shaft: 12 mm shoulder/keyed": {
        "shaft_diameter": 12.0,
        "shaft_length": 65.0,
        "shoulder_diameter": 18.0,
        "shoulder_width": 5.0,
        "key_length": 14.0,
        "key_height": 2.2,
        "key_width": 4.5,
    },
}

BEARING_VARIANTS: Dict[str, Dict[str, float]] = {
    "Bearing: 8 mm bore simplified": {
        "bearing_bore": 8.0,
        "bearing_outer": 22.0,
        "bearing_width": 7.0,
    },
    "Bearing: 10 mm bore simplified": {
        "bearing_bore": 10.0,
        "bearing_outer": 26.0,
        "bearing_width": 8.0,
    },
    "Bearing: 12 mm bore simplified": {
        "bearing_bore": 12.0,
        "bearing_outer": 28.0,
        "bearing_width": 8.0,
    },
}

DEFAULT_SHAFT_VARIANT = "Shaft: 10 mm shoulder/keyed"
DEFAULT_BEARING_VARIANT = "Bearing: 10 mm bore simplified"

CONFIGURATIONS = [
    "Valid fixed assembly",
    "Exploded view",
    "Manual placement",
    "Invalid: bore too small",
    "Invalid: radial misalignment",
    "Invalid: shoulder interference",
    "Invalid: incompatible shoulder",
]


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class Placement:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0


@dataclass
class MeshPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: str


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def rotation_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def transform_vertices(vertices: np.ndarray, placement: Placement) -> np.ndarray:
    r = rotation_x(placement.roll)
    t = np.array([placement.x, placement.y, placement.z], dtype=float)
    return (r @ vertices.T).T + t


def _theta_array(sections: int) -> np.ndarray:
    sections = int(max(8, sections))
    return np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)


def cylinder_mesh(radius: float, x0: float, x1: float, sections: int = 28) -> Tuple[np.ndarray, np.ndarray]:
    radius = float(max(radius, 1e-6))
    sections = int(max(8, sections))
    theta = _theta_array(sections)

    ring0 = np.column_stack(
        (
            np.full(sections, float(x0), dtype=float),
            radius * np.cos(theta),
            radius * np.sin(theta),
        )
    )
    ring1 = np.column_stack(
        (
            np.full(sections, float(x1), dtype=float),
            radius * np.cos(theta),
            radius * np.sin(theta),
        )
    )

    vertices = np.vstack([ring0, ring1]).astype(float)
    faces: List[List[int]] = []

    for i in range(sections):
        j = (i + 1) % sections
        a = i
        b = j
        c = sections + i
        d = sections + j
        faces.append([a, b, d])
        faces.append([a, d, c])

    c0 = vertices.shape[0]
    vertices = np.vstack([vertices, [[float(x0), 0.0, 0.0]]])
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([c0, j, i])

    c1 = vertices.shape[0]
    vertices = np.vstack([vertices, [[float(x1), 0.0, 0.0]]])
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([c1, sections + i, sections + j])

    return vertices, np.array(faces, dtype=np.int32)


def tube_mesh(
    outer_radius: float,
    inner_radius: float,
    x0: float,
    x1: float,
    sections: int = 28,
) -> Tuple[np.ndarray, np.ndarray]:
    outer_radius = float(max(outer_radius, 0.001))
    inner_radius = float(max(inner_radius, 0.001))

    if inner_radius >= outer_radius:
        inner_radius = max(outer_radius * 0.5, outer_radius - 0.001)

    sections = int(max(8, sections))
    theta = _theta_array(sections)

    outer0 = np.column_stack(
        (
            np.full(sections, float(x0), dtype=float),
            outer_radius * np.cos(theta),
            outer_radius * np.sin(theta),
        )
    )
    outer1 = np.column_stack(
        (
            np.full(sections, float(x1), dtype=float),
            outer_radius * np.cos(theta),
            outer_radius * np.sin(theta),
        )
    )
    inner0 = np.column_stack(
        (
            np.full(sections, float(x0), dtype=float),
            inner_radius * np.cos(theta),
            inner_radius * np.sin(theta),
        )
    )
    inner1 = np.column_stack(
        (
            np.full(sections, float(x1), dtype=float),
            inner_radius * np.cos(theta),
            inner_radius * np.sin(theta),
        )
    )

    vertices = np.vstack([outer0, outer1, inner0, inner1]).astype(float)
    s = sections
    faces: List[List[int]] = []

    for i in range(s):
        j = (i + 1) % s

        # Outer cylindrical wall
        faces.append([i, j, s + j])
        faces.append([i, s + j, s + i])

        # Inner cylindrical wall
        faces.append([2 * s + i, 3 * s + i, 3 * s + j])
        faces.append([2 * s + i, 3 * s + j, 2 * s + j])

        # Front annular face
        faces.append([i, j, 2 * s + j])
        faces.append([i, 2 * s + j, 2 * s + i])

        # Rear annular face
        faces.append([s + i, 3 * s + i, 3 * s + j])
        faces.append([s + i, 3 * s + j, s + j])

    return vertices, np.array(faces, dtype=np.int32)


def box_mesh(center: np.ndarray, size: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    hx, hy, hz = size / 2.0
    x, y, z = center

    vertices = np.array(
        [
            [x - hx, y - hy, z - hz],
            [x + hx, y - hy, z - hz],
            [x + hx, y + hy, z - hz],
            [x - hx, y + hy, z - hz],
            [x - hx, y - hy, z + hz],
            [x + hx, y - hy, z + hz],
            [x + hx, y + hy, z + hz],
            [x - hx, y + hy, z + hz],
        ],
        dtype=float,
    )

    faces = np.array(
        [
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
            [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6],
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
        ],
        dtype=np.int32,
    )

    return vertices, faces


def sphere_mesh(
    center: np.ndarray,
    radius: float,
    stacks: int = 8,
    segments: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float)
    radius = float(max(radius, 1e-4))
    stacks = int(max(4, stacks))
    segments = int(max(6, segments))

    verts: List[List[float]] = []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(segments):
            theta = 2.0 * math.pi * j / segments
            verts.append(
                [
                    center[0] + radius * math.sin(phi) * math.cos(theta),
                    center[1] + radius * math.sin(phi) * math.sin(theta),
                    center[2] + radius * math.cos(phi),
                ]
            )

    vertices = np.array(verts, dtype=float)
    faces: List[List[int]] = []

    for i in range(stacks):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + ((j + 1) % segments)
            c = (i + 1) * segments + j
            d = (i + 1) * segments + ((j + 1) % segments)

            if i != 0:
                faces.append([a, c, b])
            if i != stacks - 1:
                faces.append([b, c, d])

    if not faces:
        return vertices, np.empty((0, 3), dtype=np.int32)

    return vertices, np.array(faces, dtype=np.int32)


# -----------------------------------------------------------------------------
# Component mesh builders
# -----------------------------------------------------------------------------

def build_shaft_parts(
    params: Dict[str, float],
    placement: Placement,
    sections: int = 28,
) -> List[MeshPart]:
    parts: List[MeshPart] = []

    shaft_r = float(params["shaft_diameter"]) / 2.0
    shoulder_r = float(params["shoulder_diameter"]) / 2.0
    shoulder_w = float(params["shoulder_width"])
    length = float(params["shaft_length"])

    if shoulder_w > 0.0:
        v, f = cylinder_mesh(shoulder_r, -shoulder_w, 0.0, sections=sections)
        parts.append(
            MeshPart(
                name="Shaft shoulder",
                vertices=transform_vertices(v, placement),
                faces=f,
                color="#9AA0A6",
            )
        )

    if length > 0.0:
        v, f = cylinder_mesh(shaft_r, 0.0, length, sections=sections)
        parts.append(
            MeshPart(
                name="Shaft main journal",
                vertices=transform_vertices(v, placement),
                faces=f,
                color="#D4B483",
            )
        )

    kl = float(params["key_length"])
    kh = float(params["key_height"])
    kw = float(params["key_width"])

    if kl > 0.0 and kh > 0.0 and kw > 0.0 and length > kl + 2.0:
        min_x = kl / 2.0 + 1.0
        max_x = length - kl / 2.0 - 1.0
        if max_x > min_x:
            kx = min(max(length * 0.65, min_x), max_x)
            center = np.array([kx, shaft_r + kh / 2.0, 0.0], dtype=float)
            v, f = box_mesh(center, np.array([kl, kh, kw], dtype=float))
            parts.append(
                MeshPart(
                    name="Shaft key",
                    vertices=transform_vertices(v, placement),
                    faces=f,
                    color="#E84855",
                )
            )

    return parts


def build_bearing_parts(
    params: Dict[str, float],
    placement: Optional[Placement] = None,
    sections: int = 28,
    include_balls: bool = True,
    ball_count: int = 8,
) -> List[MeshPart]:
    if placement is None:
        placement = Placement()

    parts: List[MeshPart] = []

    inner_r = float(params["bearing_bore"]) / 2.0
    outer_r = float(params["bearing_outer"]) / 2.0
    width = float(params["bearing_width"])

    if width > 0.0:
        v, f = tube_mesh(outer_r, inner_r, 0.0, width, sections=sections)
        parts.append(
            MeshPart(
                name="Bearing race simplified tube",
                vertices=transform_vertices(v, placement),
                faces=f,
                color="#4C78A8",
            )
        )

    if include_balls and outer_r > inner_r + 0.25 and width > 0.0:
        track_r = (outer_r + inner_r) / 2.0
        ball_r = max(0.25, (outer_r - inner_r) * 0.27)
        n = int(max(4, ball_count))

        for k in range(n):
            ang = 2.0 * math.pi * k / n
            center = np.array(
                [
                    width / 2.0,
                    track_r * math.cos(ang),
                    track_r * math.sin(ang),
                ],
                dtype=float,
            )
            v, f = sphere_mesh(center, ball_r, stacks=6, segments=10)
            parts.append(
                MeshPart(
                    name=f"Bearing ball {k + 1}",
                    vertices=transform_vertices(v, placement),
                    faces=f,
                    color="#D9D9D9",
                )
            )

    return parts


# -----------------------------------------------------------------------------
# Constraint, collision, and validation logic
# -----------------------------------------------------------------------------

def angle_difference_deg(a: float, b: float = 0.0) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def effective_params(base: Dict[str, float], configuration: str) -> Dict[str, float]:
    p = base.copy()

    if configuration == "Invalid: bore too small":
        p["bearing_bore"] = max(3.0, float(p["shaft_diameter"]) - 1.0)

    elif configuration == "Invalid: incompatible shoulder":
        p["shoulder_diameter"] = max(3.0, float(p["bearing_bore"]) - 0.5)

    return p


def resolve_placement(
    params: Dict[str, float],
    configuration: str,
    controls: Dict[str, float],
) -> Placement:
    if configuration in {
        "Valid fixed assembly",
        "Invalid: bore too small",
        "Invalid: incompatible shoulder",
    }:
        return Placement(0.0, 0.0, 0.0, 0.0)

    if configuration == "Exploded view":
        # Move the shaft fully clear of the bearing front face,
        # then add the user-selected extra explosion gap.
        explode_x = -float(params["shaft_length"]) - float(controls.get("explode_offset", 35.0))
        return Placement(explode_x, 0.0, 0.0, 0.0)

    if configuration == "Manual placement":
        return Placement(
            float(controls.get("manual_x", 0.0)),
            float(controls.get("manual_y", 0.0)),
            float(controls.get("manual_z", 0.0)),
            float(controls.get("roll", 0.0)),
        )

    if configuration == "Invalid: radial misalignment":
        clearance = float(params["bearing_bore"]) / 2.0 - float(params["shaft_diameter"]) / 2.0
        offset = max(0.25, clearance + 0.75)
        return Placement(0.0, offset, 0.0, 0.0)

    if configuration == "Invalid: shoulder interference":
        return Placement(2.0, 0.0, 0.0, 0.0)

    return Placement(0.0, 0.0, 0.0, 0.0)


def target_placement(
    params: Dict[str, float],
    configuration: str,
    controls: Dict[str, float],
) -> Placement:
    if configuration == "Exploded view":
        explode_x = -float(params["shaft_length"]) - float(controls.get("explode_offset", 35.0))
        return Placement(explode_x, 0.0, 0.0, 0.0)

    if configuration == "Manual placement":
        return Placement(
            float(controls.get("manual_x", 0.0)),
            float(controls.get("manual_y", 0.0)),
            float(controls.get("manual_z", 0.0)),
            float(controls.get("roll", 0.0)),
        )

    return Placement(0.0, 0.0, 0.0, 0.0)


def evaluate_constraints(params: Dict[str, float], placement: Placement) -> List[Dict[str, Any]]:
    radial = math.hypot(placement.y, placement.z)
    axial = abs(placement.x)
    rot = angle_difference_deg(placement.roll, 0.0)

    constraints = [
        {
            "name": "Coaxial axis alignment",
            "type": "alignment constraint",
            "target": 0.0,
            "actual": radial,
            "tolerance": ALIGN_TOL,
            "unit": "mm",
        },
        {
            "name": "Axial shoulder seating",
            "type": "mating constraint",
            "target": 0.0,
            "actual": axial,
            "tolerance": AXIAL_TOL,
            "unit": "mm",
        },
        {
            "name": "Fixed rotational lock",
            "type": "fixed joint constraint",
            "target": 0.0,
            "actual": rot,
            "tolerance": ROT_TOL,
            "unit": "deg",
        },
    ]

    for c in constraints:
        c["satisfied"] = bool(float(c["actual"]) <= float(c["tolerance"]))

    return constraints


def detect_collisions(params: Dict[str, float], placement: Placement) -> List[Dict[str, Any]]:
    collisions: List[Dict[str, Any]] = []

    shaft_r = float(params["shaft_diameter"]) / 2.0
    bore_r = float(params["bearing_bore"]) / 2.0
    shoulder_r = float(params["shoulder_diameter"]) / 2.0
    shoulder_w = float(params["shoulder_width"])

    radial = math.hypot(placement.y, placement.z)

    if shaft_r + radial > bore_r + COLLISION_TOL:
        penetration = shaft_r + radial - bore_r
        collisions.append(
            {
                "pair": "Shaft OD vs bearing bore",
                "type": "radial interference",
                "penetration_mm": float(max(0.0, penetration)),
            }
        )

    if (
        placement.x > COLLISION_TOL
        and shoulder_w > 0.0
        and shoulder_r > bore_r - COLLISION_TOL
    ):
        collisions.append(
            {
                "pair": "Shaft shoulder vs bearing front face/bore",
                "type": "axial interference",
                "penetration_mm": float(max(0.0, placement.x)),
            }
        )

    return collisions


def validate_compatibility(params: Dict[str, float]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    shaft_r = float(params["shaft_diameter"]) / 2.0
    bore_r = float(params["bearing_bore"]) / 2.0
    shoulder_r = float(params["shoulder_diameter"]) / 2.0
    shoulder_w = float(params["shoulder_width"])
    outer_r = float(params["bearing_outer"]) / 2.0

    clearance = bore_r - shaft_r

    if float(params["shaft_diameter"]) <= 0.0:
        errors.append("Shaft diameter must be positive.")
    if float(params["shaft_length"]) <= 0.0:
        errors.append("Shaft length must be positive.")
    if float(params["bearing_bore"]) <= 0.0:
        errors.append("Bearing bore must be positive.")
    if float(params["bearing_outer"]) <= 0.0:
        errors.append("Bearing outer diameter must be positive.")
    if float(params["bearing_width"]) <= 0.0:
        errors.append("Bearing width must be positive.")

    if shaft_r > bore_r + FIT_TOL:
        errors.append(
            f"Shaft diameter {2 * shaft_r:.2f} mm is larger than bearing bore {2 * bore_r:.2f} mm."
        )
    elif abs(clearance) <= FIT_TOL:
        warnings.append("Nominal zero-clearance fit. Contact is expected but not penetration.")

    if shoulder_w > 0.0 and shoulder_r <= bore_r + FIT_TOL:
        errors.append(
            "Shoulder diameter must be larger than bearing bore to provide axial location."
        )

    if shoulder_w > 0.0 and shoulder_r < shaft_r - FIT_TOL:
        errors.append(
            "Shoulder diameter must not be smaller than shaft diameter."
        )

    if outer_r <= bore_r:
        errors.append("Bearing outer diameter must be larger than bearing bore.")

    return errors, warnings


def summarize_status(
    compatibility_errors: List[str],
    collisions: List[Dict[str, Any]],
    constraints: List[Dict[str, Any]],
) -> Tuple[str, str, List[str]]:
    unsatisfied = [c["name"] for c in constraints if not c["satisfied"]]

    if compatibility_errors:
        return "INVALID COMPONENT COMPATIBILITY", "error", unsatisfied

    if collisions:
        return "COLLISION DETECTED", "error", unsatisfied

    if unsatisfied:
        return "CONSTRAINTS NOT SATISFIED", "warning", unsatisfied

    return "VALID FIXED ASSEMBLY", "success", unsatisfied


def placement_error(actual: Placement, target: Placement) -> Tuple[float, float]:
    linear = math.sqrt(
        (actual.x - target.x) ** 2
        + (actual.y - target.y) ** 2
        + (actual.z - target.z) ** 2
    )
    angular = angle_difference_deg(actual.roll, target.roll)
    return linear, angular


def build_project_json(
    project_name: str,
    params: Dict[str, float],
    placement: Placement,
    constraints: List[Dict[str, Any]],
    collisions: List[Dict[str, Any]],
    compatibility_errors: List[str],
    compatibility_warnings: List[str],
) -> str:
    payload = {
        "project_name": project_name,
        "milestone": "M1",
        "description": "Fixed connection between a parametric shaft and a simplified bearing.",
        "parameters": params,
        "placement": asdict(placement),
        "constraints": constraints,
        "collisions": collisions,
        "compatibility_errors": compatibility_errors,
        "compatibility_warnings": compatibility_warnings,
    }
    return json.dumps(payload, indent=2)


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def make_plotly_figure(parts: List[MeshPart]):
    import plotly.graph_objects as go

    fig = go.Figure()
    all_vertices: List[np.ndarray] = []

    for part in parts:
        if part.vertices.size == 0 or part.faces.size == 0:
            continue

        fig.add_trace(
            go.Mesh3d(
                x=part.vertices[:, 0],
                y=part.vertices[:, 1],
                z=part.vertices[:, 2],
                i=part.faces[:, 0],
                j=part.faces[:, 1],
                k=part.faces[:, 2],
                name=part.name,
                color=part.color,
                opacity=0.92,
                flatshading=True,
            )
        )
        all_vertices.append(part.vertices)

    if all_vertices:
        v = np.vstack(all_vertices)
        mins = v.min(axis=0)
        maxs = v.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = max(float(np.max(maxs - mins)), 1.0) * 1.18

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[center[0] - extent / 2.0, center[0] + extent / 2.0],
                    title="X [mm]",
                ),
                yaxis=dict(
                    range=[center[1] - extent / 2.0, center[1] + extent / 2.0],
                    title="Y [mm]",
                ),
                zaxis=dict(
                    range=[center[2] - extent / 2.0, center[2] + extent / 2.0],
                    title="Z [mm]",
                ),
                aspectmode="cube",
            )
        )

    fig.update_layout(
        title="M1 Interactive Shaft-Bearing Assembly",
        height=680,
        margin=dict(l=0, r=0, b=0, t=40),
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.02),
    )

    return fig


def _render_matplotlib_frame(
    parts: List[MeshPart],
    elev: float = 18.0,
    azim: float = 35.0,
    figsize: Tuple[float, float] = (6.4, 3.84),
    dpi: int = 100,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    all_vertices: List[np.ndarray] = []

    for part in parts:
        if part.vertices.size == 0 or part.faces.size == 0:
            continue

        polys = part.vertices[part.faces]
        ax.add_collection3d(
            Poly3DCollection(
                polys,
                facecolors=part.color,
                edgecolors=(0.0, 0.0, 0.0, 0.0),
                alpha=0.92,
            )
        )
        all_vertices.append(part.vertices)

    if all_vertices:
        v = np.vstack(all_vertices)
        mins = v.min(axis=0)
        maxs = v.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = max(float(np.max(maxs - mins)), 1.0) * 1.18

        ax.set_xlim(center[0] - extent / 2.0, center[0] + extent / 2.0)
        ax.set_ylim(center[1] - extent / 2.0, center[1] + extent / 2.0)
        ax.set_zlim(center[2] - extent / 2.0, center[2] + extent / 2.0)
        ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()

    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    return img


def generate_demo_video_bytes() -> Dict[str, Any]:
    """
    Generates a short live MP4 demo automatically:
    1. Shaft starts fully exploded away from the bearing.
    2. Shaft translates into the bearing.
    3. Fixed connection is reached.
    4. Camera orbits to show the assembled state.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    params = DEFAULT_PARAMS.copy()
    frames: List[np.ndarray] = []

    n_frames = 48
    fps = 24

    start_x = -(float(params["shaft_length"]) + 35.0)

    for idx in range(n_frames):
        t = idx / float(n_frames - 1)

        if t < 0.58:
            u = t / 0.58
            ease = 3.0 * u * u - 2.0 * u * u * u
            x = start_x * (1.0 - ease)
            azim = 25.0
        else:
            x = 0.0
            u = (t - 0.58) / 0.42
            azim = 25.0 + 80.0 * u

        shaft_placement = Placement(x=x, y=0.0, z=0.0, roll=0.0)
        bearing_placement = Placement(0.0, 0.0, 0.0, 0.0)

        parts = build_shaft_parts(params, shaft_placement, sections=16)
        parts += build_bearing_parts(
            params,
            bearing_placement,
            sections=16,
            include_balls=True,
            ball_count=6,
        )

        frame = _render_matplotlib_frame(parts, elev=18.0, azim=azim)
        frames.append(frame)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        mp4_path = tf.name

    try:
        try:
            imageio.mimsave(
                mp4_path,
                frames,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
                pixelformat="yuv420p",
            )
        except Exception:
            imageio.mimsave(mp4_path, frames, fps=fps)

        with open(mp4_path, "rb") as f:
            data = f.read()

        if not data:
            raise RuntimeError("Generated MP4 file is empty.")

        try:
            os.remove(mp4_path)
        except Exception:
            pass

        return {"bytes": data, "mime": "video/mp4", "error": ""}

    except Exception as exc:
        gif_path = os.path.splitext(mp4_path)[0] + ".gif"
        png_path = os.path.splitext(mp4_path)[0] + ".png"

        try:
            imageio.mimsave(gif_path, frames, duration=50, loop=0)
            with open(gif_path, "rb") as f:
                data = f.read()
            try:
                os.remove(gif_path)
            except Exception:
                pass
            if data:
                return {"bytes": data, "mime": "image/gif", "error": str(exc)}
        except Exception:
            pass

        try:
            plt.imsave(png_path, frames[-1])
            with open(png_path, "rb") as f:
                data = f.read()
            try:
                os.remove(png_path)
            except Exception:
                pass
            if data:
                return {"bytes": data, "mime": "image/png", "error": str(exc)}
        except Exception:
            pass

        return {"bytes": b"", "mime": "video/mp4", "error": str(exc)}


if st is not None:
    cached_generate_demo_video_bytes = st.cache_data(
        show_spinner="Generating live M1 assembly demo video..."
    )(generate_demo_video_bytes)
else:
    cached_generate_demo_video_bytes = generate_demo_video_bytes


# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------

def export_stl_bytes(parts: List[MeshPart]) -> bytes:
    try:
        import trimesh
    except Exception:
        return b""

    meshes = []
    for part in parts:
        if part.vertices.size == 0 or part.faces.size == 0:
            continue
        try:
            meshes.append(
                trimesh.Trimesh(
                    vertices=part.vertices,
                    faces=part.faces,
                    process=False,
                )
            )
        except Exception:
            pass

    if not meshes:
        return b""

    try:
        scene = trimesh.Scene(meshes)
        data = scene.export(file_type="stl")
    except Exception:
        try:
            combined = trimesh.util.concatenate(meshes)
            data = combined.export(file_type="stl")
        except Exception:
            return b""

    if isinstance(data, str):
        data = data.encode("utf-8")

    return data


def try_export_step(
    params: Dict[str, float],
    placement: Placement,
) -> Tuple[Optional[bytes], str]:
    """
    Attempts native STEP export using OpenCASCADE through OCP/CadQuery.
    If unavailable, returns None and a message.
    """
    try:
        from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.BRep import BRep_BBuilder
        from OCP.STEPControl import STEPControl_Writer
    except Exception as exc:
        return None, f"OpenCASCADE/OCP not available: {exc}. Install CadQuery optionally for native STEP."

    try:
        shaft_r = float(params["shaft_diameter"]) / 2.0
        shoulder_r = float(params["shoulder_diameter"]) / 2.0
        shoulder_w = float(params["shoulder_width"])
        shaft_l = float(params["shaft_length"])

        bore_r = float(params["bearing_bore"]) / 2.0
        outer_r = float(params["bearing_outer"]) / 2.0
        bearing_w = float(params["bearing_width"])

        dir_x = gp_Dir(1, 0, 0)
        shapes = []

        if shoulder_w > 0.0:
            shoulder = BRepPrimAPI_MakeCylinder(
                gp_Ax2(
                    gp_Pnt(placement.x - shoulder_w, placement.y, placement.z),
                    dir_x,
                ),
                shoulder_r,
                shoulder_w,
            ).Shape()
            shapes.append(shoulder)

        if shaft_l > 0.0:
            main = BRepPrimAPI_MakeCylinder(
                gp_Ax2(
                    gp_Pnt(placement.x, placement.y, placement.z),
                    dir_x,
                ),
                shaft_r,
                shaft_l,
            ).Shape()
            shapes.append(main)

        if bearing_w > 0.0:
            outer = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), dir_x),
                outer_r,
                bearing_w,
            ).Shape()

            try:
                bore = BRepPrimAPI_MakeCylinder(
                    gp_Ax2(gp_Pnt(-1.0, 0.0, 0.0), dir_x),
                    bore_r,
                    bearing_w + 2.0,
                ).Shape()

                cut_op = BRepAlgoAPI_Cut(outer, bore)
                cut_op.Build()
                bearing = cut_op.Shape()

                if bearing.IsNull():
                    bearing = outer
            except Exception:
                bearing = outer

            shapes.append(bearing)

        if not shapes:
            return None, "No valid solids to export."

        builder = BRep_BBuilder()
        builder.Init()
        for shape in shapes:
            builder.Add(shape)
        compound = builder.Shape()

        if compound.IsNull():
            return None, "OpenCASCADE compound is null."

        try:
            from OCP.STEPControl import STEPControl_AsIs
            transfer_type = STEPControl_AsIs
        except Exception:
            try:
                from OCP.STEPControl import STEPControl_StepModelType
                transfer_type = STEPControl_StepModelType.STEPControl_AsIs
            except Exception:
                transfer_type = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            step_path = os.path.join(tmpdir, "m1_assembly.step").replace("\\", "/")
            writer = STEPControl_Writer()
            writer.Transfer(compound, transfer_type)
            writer.Write(step_path)

            if not os.path.exists(step_path) or os.path.getsize(step_path) == 0:
                return None, "STEP writer produced an empty file."

            with open(step_path, "rb") as f:
                data = f.read()

        return data, "Native OpenCASCADE STEP exported successfully."

    except Exception as exc:
        return None, f"STEP export failed: {exc}"


# -----------------------------------------------------------------------------
# Automated tests
# -----------------------------------------------------------------------------

class M1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = DEFAULT_PARAMS.copy()

    def test_valid_fixed_demo(self) -> None:
        placement = Placement(0.0, 0.0, 0.0, 0.0)
        errors, _ = validate_compatibility(self.params)
        constraints = evaluate_constraints(self.params, placement)
        collisions = detect_collisions(self.params, placement)
        status, severity, _ = summarize_status(errors, collisions, constraints)

        self.assertEqual(errors, [])
        self.assertTrue(all(c["satisfied"] for c in constraints))
        self.assertEqual(collisions, [])
        self.assertEqual(status, "VALID FIXED ASSEMBLY")
        self.assertEqual(severity, "success")

    def test_invalid_bore_too_small(self) -> None:
        p = effective_params(self.params, "Invalid: bore too small")
        errors, _ = validate_compatibility(p)
        collisions = detect_collisions(p, Placement(0.0, 0.0, 0.0, 0.0))

        self.assertTrue(len(errors) > 0)
        self.assertTrue(len(collisions) > 0)

    def test_radial_misalignment_collision(self) -> None:
        clearance = self.params["bearing_bore"] / 2.0 - self.params["shaft_diameter"] / 2.0
        placement = Placement(0.0, clearance + 0.5, 0.0, 0.0)

        collisions = detect_collisions(self.params, placement)
        constraints = evaluate_constraints(self.params, placement)

        self.assertTrue(any(c["type"] == "radial interference" for c in collisions))
        self.assertFalse(constraints[0]["satisfied"])

    def test_should_interference(self) -> None:
        placement = Placement(1.5, 0.0, 0.0, 0.0)
        collisions = detect_collisions(self.params, placement)
        constraints = evaluate_constraints(self.params, placement)

        self.assertTrue(any(c["type"] == "axial interference" for c in collisions))
        self.assertFalse(constraints[1]["satisfied"])

    def test_exploded_constraints_unsat(self) -> None:
        placement = Placement(-20.0, 0.0, 0.0, 0.0)
        constraints = evaluate_constraints(self.params, placement)

        self.assertFalse(constraints[1]["satisfied"])

    def test_mesh_generation(self) -> None:
        parts = build_shaft_parts(self.params, Placement(), sections=12)
        parts += build_bearing_parts(
            self.params,
            Placement(),
            sections=12,
            include_balls=False,
        )

        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertGreater(part.vertices.shape[0], 0)
            self.assertGreater(part.faces.shape[0], 0)

    def test_project_json(self) -> None:
        placement = Placement(0.0, 0.0, 0.0, 0.0)
        constraints = evaluate_constraints(self.params, placement)
        collisions = detect_collisions(self.params, placement)
        errors, warnings = validate_compatibility(self.params)

        js = build_project_json(
            "Unit Test Project",
            self.params,
            placement,
            constraints,
            collisions,
            errors,
            warnings,
        )
        data = json.loads(js)

        self.assertEqual(data["milestone"], "M1")
        self.assertEqual(data["project_name"], "Unit Test Project")
        self.assertIn("parameters", data)
        self.assertIn("placement", data)

    def test_stl_export(self) -> None:
        parts = build_shaft_parts(self.params, Placement(), sections=12)
        parts += build_bearing_parts(
            self.params,
            Placement(),
            sections=12,
            include_balls=False,
        )
        data = export_stl_bytes(parts)

        if not data:
            self.skipTest("trimesh STL export unavailable.")

        self.assertGreater(len(data), 100)

    def test_step_export_optional(self) -> None:
        data, msg = try_export_step(self.params, Placement())

        if data is None:
            self.skipTest(f"Optional STEP kernel unavailable: {msg}")

        self.assertGreater(len(data), 100)

    @unittest.skipUnless(
        os.getenv("RUN_SLOW_TESTS"),
        "Set RUN_SLOW_TESTS=1 to render the live demo video during tests.",
    )
    def test_demo_video_generation_slow(self) -> None:
        result = generate_demo_video_bytes()
        self.assertGreater(len(result["bytes"]), 1000)


def run_self_test() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(M1Tests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def _run_tests_in_app() -> Tuple[str, bool]:
    buf = io.StringIO()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(M1Tests)
    runner = unittest.TextTestRunner(stream=buf, verbosity=2)
    result = runner.run(suite)
    return buf.getvalue(), result.wasSuccessful()


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def _init_session_state() -> None:
    for k, v in DEFAULT_PARAMS.items():
        st.session_state.setdefault(k, v)

    st.session_state.setdefault("project_name", "M1 Shaft-Bearing Fixed Assembly")
    st.session_state.setdefault("shaft_variant_select", DEFAULT_SHAFT_VARIANT)
    st.session_state.setdefault("bearing_variant_select", DEFAULT_BEARING_VARIANT)
    st.session_state.setdefault("configuration", CONFIGURATIONS[0])
    st.session_state.setdefault("explode_offset", 35.0)
    st.session_state.setdefault("manual_x", 0.0)
    st.session_state.setdefault("manual_y", 0.0)
    st.session_state.setdefault("manual_z", 0.0)
    st.session_state.setdefault("roll", 0.0)


def _apply_shaft_variant() -> None:
    selected = st.session_state.get("shaft_variant_select")
    if selected in SHAFT_VARIANTS:
        st.session_state.update(SHAFT_VARIANTS[selected])


def _apply_bearing_variant() -> None:
    selected = st.session_state.get("bearing_variant_select")
    if selected in BEARING_VARIANTS:
        st.session_state.update(BEARING_VARIANTS[selected])


def _reset_demo() -> None:
    for k, v in DEFAULT_PARAMS.items():
        st.session_state[k] = v

    st.session_state["project_name"] = "M1 Shaft-Bearing Fixed Assembly"
    st.session_state["shaft_variant_select"] = DEFAULT_SHAFT_VARIANT
    st.session_state["bearing_variant_select"] = DEFAULT_BEARING_VARIANT
    st.session_state["configuration"] = CONFIGURATIONS[0]
    st.session_state["explode_offset"] = 35.0
    st.session_state["manual_x"] = 0.0
    st.session_state["manual_y"] = 0.0
    st.session_state["manual_z"] = 0.0
    st.session_state["roll"] = 0.0

    for key in ["step_bytes", "step_msg", "stl_bytes"]:
        st.session_state.pop(key, None)

    if hasattr(cached_generate_demo_video_bytes, "clear"):
        try:
            cached_generate_demo_video_bytes.clear()
        except Exception:
            pass


def main() -> None:
    st.set_page_config(
        page_title="Mechanical Assembly Generator — M1",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    st.title("⚙️ Mechanical Assembly Generation & Configuration — M1")
    st.caption(
        "Milestone 1: assemble two parametric mechanical components using a fixed connection. "
        "The app auto-loads a realistic shaft-bearing demo, generates a live simulation video, "
        "and supports interactive editing, collision checks, validation, and export."
    )

    # ---------------------------------------------------------------------
    # Sidebar controls
    # ---------------------------------------------------------------------
    st.sidebar.header("M1 Controls")

    project_name = st.sidebar.text_input("Project name", key="project_name")

    st.sidebar.subheader("Component library / replacement")
    st.sidebar.selectbox(
        "Shaft component",
        options=list(SHAFT_VARIANTS.keys()),
        key="shaft_variant_select",
        on_change=_apply_shaft_variant,
    )
    st.sidebar.selectbox(
        "Bearing component",
        options=list(BEARING_VARIANTS.keys()),
        key="bearing_variant_select",
        on_change=_apply_bearing_variant,
    )

    st.sidebar.subheader("Parametric dimensions")
    st.sidebar.number_input(
        "Shaft diameter (mm)",
        min_value=3.0,
        max_value=40.0,
        step=0.1,
        key="shaft_diameter",
    )
    st.sidebar.number_input(
        "Shaft length (mm)",
        min_value=10.0,
        max_value=160.0,
        step=0.5,
        key="shaft_length",
    )
    st.sidebar.number_input(
        "Shoulder diameter (mm)",
        min_value=4.0,
        max_value=70.0,
        step=0.1,
        key="shoulder_diameter",
    )
    st.sidebar.number_input(
        "Shoulder width (mm)",
        min_value=0.0,
        max_value=25.0,
        step=0.1,
        key="shoulder_width",
    )
    st.sidebar.number_input(
        "Key length (mm)",
        min_value=0.0,
        max_value=40.0,
        step=0.1,
        key="key_length",
    )
    st.sidebar.number_input(
        "Key height (mm)",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        key="key_height",
    )
    st.sidebar.number_input(
        "Key width (mm)",
        min_value=0.0,
        max_value=15.0,
        step=0.1,
        key="key_width",
    )

    st.sidebar.number_input(
        "Bearing bore (mm)",
        min_value=3.0,
        max_value=60.0,
        step=0.1,
        key="bearing_bore",
    )
    st.sidebar.number_input(
        "Bearing outer diameter (mm)",
        min_value=6.0,
        max_value=100.0,
        step=0.1,
        key="bearing_outer",
    )
    st.sidebar.number_input(
        "Bearing width (mm)",
        min_value=1.0,
        max_value=40.0,
        step=0.1,
        key="bearing_width",
    )

    st.sidebar.subheader("Assembly configuration")
    configuration = st.sidebar.selectbox(
        "Configuration preset",
        options=CONFIGURATIONS,
        key="configuration",
    )

    st.sidebar.slider(
        "Exploded extra gap (mm)",
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        key="explode_offset",
        disabled=(configuration != "Exploded view"),
    )

    st.sidebar.slider(
        "Manual axial offset (mm)",
        min_value=-200.0,
        max_value=100.0,
        step=0.1,
        key="manual_x",
        disabled=(configuration != "Manual placement"),
    )
    st.sidebar.slider(
        "Manual radial Y offset (mm)",
        min_value=-8.0,
        max_value=8.0,
        step=0.05,
        key="manual_y",
        disabled=(configuration != "Manual placement"),
    )
    st.sidebar.slider(
        "Manual radial Z offset (mm)",
        min_value=-8.0,
        max_value=8.0,
        step=0.05,
        key="manual_z",
        disabled=(configuration != "Manual placement"),
    )
    st.sidebar.slider(
        "Shaft roll about X (deg)",
        min_value=0.0,
        max_value=360.0,
        step=1.0,
        key="roll",
        disabled=(configuration != "Manual placement"),
    )

    sidebar_col1, sidebar_col2 = st.sidebar.columns(2)

    if sidebar_col1.button("Reset demo"):
        _reset_demo()
        st.rerun()

    if sidebar_col2.button("Run M1 tests"):
        with st.spinner("Running built-in automated tests..."):
            output, ok = _run_tests_in_app()
        st.code(output, language="text")
        if ok:
            st.success("Quick M1 tests passed.")
        else:
            st.error("Some M1 tests failed.")

    st.sidebar.subheader("M1 scope")
    st.sidebar.markdown(
        "- Two parametric components: shaft and bearing.\n"
        "- Fixed mating/alignment connection.\n"
        "- Collision detection.\n"
        "- Assembly validation.\n"
        "- Exploded view.\n"
        "- Live auto-generated demo video.\n"
        "- JSON, STL, and optional STEP export.\n"
        "- Built-in automated tests."
    )

    # ---------------------------------------------------------------------
    # Compute current assembly state
    # ---------------------------------------------------------------------
    base_params = {k: float(st.session_state[k]) for k in DEFAULT_PARAMS}
    params = effective_params(base_params, configuration)

    controls = {
        "explode_offset": float(st.session_state["explode_offset"]),
        "manual_x": float(st.session_state["manual_x"]),
        "manual_y": float(st.session_state["manual_y"]),
        "manual_z": float(st.session_state["manual_z"]),
        "roll": float(st.session_state["roll"]),
    }

    placement = resolve_placement(params, configuration, controls)
    target = target_placement(params, configuration, controls)

    constraints = evaluate_constraints(params, placement)
    collisions = detect_collisions(params, placement)
    compatibility_errors, compatibility_warnings = validate_compatibility(params)
    status, severity, unsatisfied = summarize_status(
        compatibility_errors,
        collisions,
        constraints,
    )
    linear_error, angular_error = placement_error(placement, target)

    shaft_parts = build_shaft_parts(params, placement, sections=28)
    bearing_parts = build_bearing_parts(params, Placement(), sections=28, include_balls=True, ball_count=8)
    parts = shaft_parts + bearing_parts

    satisfied_count = sum(1 for c in constraints if c["satisfied"])
    total_count = len(constraints)

    # ---------------------------------------------------------------------
    # Status banner
    # ---------------------------------------------------------------------
    if severity == "success":
        st.success(f"**Assembly status:** {status}")
    elif severity == "error":
        st.error(f"**Assembly status:** {status}")
    else:
        st.warning(f"**Assembly status:** {status}")

    if unsatisfied:
        st.warning("Unsatisfied constraints: " + ", ".join(unsatisfied))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Constraint satisfaction", f"{satisfied_count}/{total_count}")
    m2.metric("Collisions", len(collisions))
    m3.metric("Placement error vs target", f"{linear_error:.3f} mm")
    m4.metric("Angular error vs target", f"{angular_error:.2f} deg")

    # ---------------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------------
    tab_video, tab_3d, tab_library, tab_validation, tab_tree, tab_export = st.tabs(
        [
            "Live demo video",
            "Interactive 3D",
            "Component library",
            "Constraints & validation",
            "Assembly tree",
            "Export",
        ]
    )

    # ---------------------------------------------------------------------
    # Live demo video tab
    # ---------------------------------------------------------------------
    with tab_video:
        st.subheader("Automatic live simulation")
        st.markdown(
            "This video is generated locally on first launch. It shows the shaft starting fully exploded, "
            "translating into the bearing, reaching the fixed seated position, and then orbiting for inspection."
        )

        if st.button("Regenerate live demo video"):
            if hasattr(cached_generate_demo_video_bytes, "clear"):
                try:
                    cached_generate_demo_video_bytes.clear()
                except Exception:
                    pass
            st.rerun()

        demo = cached_generate_demo_video_bytes()
        data = demo.get("bytes", b"")
        mime = demo.get("mime", "video/mp4")
        err = demo.get("error", "")

        if data:
            if mime == "video/mp4":
                b64 = base64.b64encode(data).decode("ascii")
                html = (
                    "<div style=\"text-align:center;\">"
                    f"<video controls autoplay muted loop playsinline "
                    f"style=\"width:100%; max-width:920px; border:1px solid #dddddd; border-radius:10px; background:#ffffff;\" "
                    f"src=\"data:{mime};base64,{b64}\"></video>"
                    "</div>"
                )
                try:
                    st.components.v1.html(html, height=560)
                except Exception:
                    st.warning("Embedded player unavailable. Use the download button below.")
            else:
                st.image(data, use_container_width=True)

            ext = "mp4"
            if mime == "image/gif":
                ext = "gif"
            elif mime == "image/png":
                ext = "png"

            st.download_button(
                f"Download live demo ({ext.upper()})",
                data,
                f"m1_shaft_bearing_demo.{ext}",
                mime,
            )

            if err:
                st.warning(f"Video renderer fallback used: {err}")
        else:
            st.error("Demo video generation failed. The interactive 3D view below still works.")

        st.divider()
        st.markdown(
            "**Interaction model:** the video is the automatic working demo. "
            "Use the sidebar sliders and configuration presets to edit the live assembly in the Interactive 3D tab."
        )

    # ---------------------------------------------------------------------
    # Interactive 3D tab
    # ---------------------------------------------------------------------
    with tab_3d:
        st.subheader("Editable assembly preview")
        fig = make_plotly_figure(parts)
        st.plotly_chart(fig, use_container_width=True)

        st.write(
            f"**Current placement:** x={placement.x:.3f} mm, "
            f"y={placement.y:.3f} mm, z={placement.z:.3f} mm, roll={placement.roll:.2f} deg"
        )
        st.write(f"**Configuration:** {configuration}")

    # ---------------------------------------------------------------------
    # Component library tab
    # ---------------------------------------------------------------------
    with tab_library:
        st.subheader("Reusable component library")

        library_rows = []
        for label, d in SHAFT_VARIANTS.items():
            library_rows.append(
                {
                    "Category": "Shaft",
                    "Variant": label,
                    "Diameter (mm)": d["shaft_diameter"],
                    "Length (mm)": d["shaft_length"],
                    "Shoulder diameter (mm)": d["shoulder_diameter"],
                    "Shoulder width (mm)": d["shoulder_width"],
                }
            )

        for label, d in BEARING_VARIANTS.items():
            library_rows.append(
                {
                    "Category": "Bearing",
                    "Variant": label,
                    "Diameter (mm)": d["bearing_bore"],
                    "Length (mm)": d["bearing_width"],
                    "Shoulder diameter (mm)": d["bearing_outer"],
                    "Shoulder width (mm)": 0.0,
                }
            )

        st.dataframe(library_rows, use_container_width=True)

        st.subheader("Current parametric metadata")
        param_rows = [{"Parameter": k, "Value (mm)": v} for k, v in params.items()]
        st.dataframe(param_rows, use_container_width=True)

        st.subheader("Connection interfaces")
        interface_rows = [
            {
                "Component": "Shaft",
                "Interface": "Main journal OD",
                "Type": "Cylindrical shaft interface",
                "Radius (mm)": params["shaft_diameter"] / 2.0,
                "Axis": "X",
                "Local span": f"x = 0 to {params['shaft_length']:.2f}",
            },
            {
                "Component": "Shaft",
                "Interface": "Shoulder face",
                "Type": "Planar axial location interface",
                "Radius (mm)": params["shoulder_diameter"] / 2.0,
                "Axis": "X",
                "Local span": f"x = -{params['shoulder_width']:.2f} to 0",
            },
            {
                "Component": "Bearing",
                "Interface": "Inner bore",
                "Type": "Cylindrical bore interface",
                "Radius (mm)": params["bearing_bore"] / 2.0,
                "Axis": "X",
                "Local span": f"x = 0 to {params['bearing_width']:.2f}",
            },
            {
                "Component": "Bearing",
                "Interface": "Front face",
                "Type": "Planar mating interface",
                "Radius (mm)": params["bearing_outer"] / 2.0,
                "Axis": "X",
                "Local span": "x = 0",
            },
        ]
        st.dataframe(interface_rows, use_container_width=True)

    # ---------------------------------------------------------------------
    # Constraints & validation tab
    # ---------------------------------------------------------------------
    with tab_validation:
        st.subheader("Constraint solver results")

        for c in constraints:
            icon = "✅" if c["satisfied"] else "❌"
            st.markdown(
                f"{icon} **{c['name']}** — {c['type']}  \n"
                f"Actual: `{float(c['actual']):.4f} {c['unit']}`  \n"
                f"Tolerance: `{float(c['tolerance']):.4f} {c['unit']}`"
            )

        st.subheader("Collision detection")
        if collisions:
            for col in collisions:
                st.error(
                    f"**{col['pair']}** — {col['type']} — "
                    f"penetration `{float(col['penetration_mm']):.4f} mm`"
                )
        else:
            st.success("No penetrations detected.")

        st.subheader("Component compatibility")
        if compatibility_errors:
            for e in compatibility_errors:
                st.error(e)
        else:
            st.success("Component dimensions are compatible for the M1 fixed connection.")

        if compatibility_warnings:
            for w in compatibility_warnings:
                st.warning(w)

        st.subheader("Known invalid configuration support")
        st.markdown(
            "Use the sidebar configuration presets to deliberately generate invalid assemblies:\n"
            "- bore too small\n"
            "- radial misalignment\n"
            "- shoulder interference\n"
            "- incompatible shoulder\n\n"
            "The validator and collision detector should flag these cases."
        )

    # ---------------------------------------------------------------------
    # Assembly tree tab
    # ---------------------------------------------------------------------
    with tab_tree:
        st.subheader("Assembly tree")

        tree = f"""Project: {project_name}
Milestone: M1
Assembly: Shaft-Bearing Fixed Connection
├─ Static component: Bearing
│  ├─ Race tube
│  │  ├─ Bore diameter: {params['bearing_bore']:.2f} mm
│  │  ├─ Outer diameter: {params['bearing_outer']:.2f} mm
│  │  └─ Width: {params['bearing_width']:.2f} mm
│  └─ Balls: visual simplification
└─ Placed component: Shaft
   ├─ Shoulder
   │  ├─ Diameter: {params['shoulder_diameter']:.2f} mm
   │  └─ Width: {params['shoulder_width']:.2f} mm
   ├─ Main journal
   │  ├─ Diameter: {params['shaft_diameter']:.2f} mm
   │  └─ Length: {params['shaft_length']:.2f} mm
   └─ Key
      ├─ Length: {params['key_length']:.2f} mm
      ├─ Height: {params['key_height']:.2f} mm
      └─ Width: {params['key_width']:.2f} mm

Constraints:
"""

        for c in constraints:
            tree += f"   - {c['name']}: {'satisfied' if c['satisfied'] else 'unsatisfied'}\n"

        tree += "\nCollisions:\n"
        if collisions:
            for col in collisions:
                tree += f"   - {col['pair']}: {col['type']}, penetration {col['penetration_mm']:.4f} mm\n"
        else:
            tree += "   - none\n"

        st.code(tree, language="text")

    # ---------------------------------------------------------------------
    # Export tab
    # ---------------------------------------------------------------------
    with tab_export:
        st.subheader("Save and export")

        project_json = build_project_json(
            project_name=project_name,
            params=params,
            placement=placement,
            constraints=constraints,
            collisions=collisions,
            compatibility_errors=compatibility_errors,
            compatibility_warnings=compatibility_warnings,
        )

        st.download_button(
            "Download project JSON",
            project_json,
            "m1_assembly.json",
            "application/json",
        )

        st.markdown("---")
        st.subheader("Mesh and CAD export")

        if st.button("Generate STL mesh"):
            with st.spinner("Building STL mesh..."):
                st.session_state["stl_bytes"] = export_stl_bytes(parts)

        if st.session_state.get("stl_bytes"):
            st.download_button(
                "Download STL",
                st.session_state["stl_bytes"],
                "m1_assembly.stl",
                "model/stl",
            )
        else:
            st.info("Click Generate STL mesh to create a downloadable mesh file.")

        if st.button("Generate STEP via OpenCASCADE/CadQuery"):
            with st.spinner("Attempting native STEP export..."):
                step_bytes, step_msg = try_export_step(params, placement)
                st.session_state["step_bytes"] = step_bytes
                st.session_state["step_msg"] = step_msg

        if st.session_state.get("step_bytes"):
            st.download_button(
                "Download STEP",
                st.session_state["step_bytes"],
                "m1_assembly.step",
                "application/step",
            )
        elif st.session_state.get("step_msg"):
            st.info(st.session_state["step_msg"])
        else:
            st.info(
                "Native STEP export is optional. Install CadQuery on Windows to enable OpenCASCADE-backed STEP output. "
                "Without it, JSON and STL fallbacks remain available."
            )

        st.markdown("---")
        st.subheader("M1 evaluation checklist")
        st.markdown(
            "- Constraint satisfaction: visible in Constraints & validation.\n"
            "- Collision detection accuracy: valid, exploded, radial, axial, and bore-invalid presets are included.\n"
            "- Placement accuracy: displayed as linear and angular error versus configured target.\n"
            "- Assembly regeneration: every parameter/control change rebuilds geometry and validation.\n"
            "- Component compatibility: shaft/bearing replacement and invalid presets are checked.\n"
            "- Export correctness: JSON always available; STL mesh fallback; optional native STEP."
        )


if _SELF_TEST:
    sys.exit(run_self_test())
else:
    main()