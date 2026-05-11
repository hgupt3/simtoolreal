"""Step 3 of the one_leg SDF + primitive hybrid pipeline.

This variant keeps detailed OBJ contact geometry where FurnitureBench needs
it most:

  * Receptive: a local patch cut from the original square_table_top canonical
    OBJ around the active leg hole is loaded as an SDF collision mesh. The
    rest of the top is represented by coarse boxes that stop at the patch
    boundary.
  * Inserter: the threaded / tenon end of square_table_leg4 is loaded as an
    SDF collision mesh. The remaining cuboidal leg body is represented by a
    single box primitive. Visual geometry intentionally mirrors collision
    geometry so ``visualize_problems`` shows the shapes Isaac will use.

Run after step2:
    .venv/bin/python -m peg_in_hole_dynamic.furniture_bench.one_leg_problem_setup.step3_generate_sdf_hybrid
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
from xml.dom import minidom

import numpy as np
import trimesh

from peg_in_hole_dynamic.furniture_bench.one_leg_problem_setup.step2_generate_assets import (
    ASSETS_FB,
    HOLE_PADDING_M,
    REPO_ROOT,
)
from peg_in_hole_dynamic.sdf_hybrid_utils import (
    add_box_collision_visual,
    add_mesh_visual,
    add_sdf_collision,
)


SDF_RES = 256
THREAD_LENGTH_M = 0.025
DETAIL_PATCH_PAD_M = 0.016
# Bulk fixture is modeled as an open-top tray: a thin base plate at the
# table's bottom face + 4 thin perimeter walls running from the base plate
# top up to the table top. The SDF hole patch sits inside the cavity at
# the table-top height where the leg threads engage.
BASE_PLATE_THICKNESS_M = 0.005
WALL_THICKNESS_M       = 0.005


def _open_top_tray_boxes(
    plate_xmin: float, plate_xmax: float,
    plate_ymin: float, plate_ymax: float,
    plate_zmin: float, plate_zmax: float,
    base_t: float = BASE_PLATE_THICKNESS_M,
    wall_t: float = WALL_THICKNESS_M,
) -> List[Tuple[float, float, float, float, float, float]]:
    """Tile the table footprint as an open-top tray: a single base plate at
    the table's bottom face + 4 thin perimeter walls running from the base
    plate top up to the table's top face.

    Returns a list of (cx, cy, cz, sx, sy, sz) tuples — 5 boxes total. The
    cavity in the middle (between the walls and above the base plate) is
    open and is filled in collision by the SDF hole-patch mesh.
    """
    sx_full = plate_xmax - plate_xmin
    sy_full = plate_ymax - plate_ymin
    cx_mid  = (plate_xmin + plate_xmax) / 2
    cy_mid  = (plate_ymin + plate_ymax) / 2

    base_top = plate_zmin + base_t
    wall_z_lo = base_top
    wall_z_hi = plate_zmax
    wall_z_ext = wall_z_hi - wall_z_lo
    wall_z_mid = (wall_z_lo + wall_z_hi) / 2

    out: List[Tuple[float, float, float, float, float, float]] = []
    # Base plate (full footprint at the bottom of the table)
    out.append((cx_mid, cy_mid, (plate_zmin + base_top) / 2,
                sx_full, sy_full, base_t))
    # North/South walls span the full X (overlap the corners)
    out.append((cx_mid, plate_ymax - wall_t / 2, wall_z_mid,
                sx_full, wall_t, wall_z_ext))
    out.append((cx_mid, plate_ymin + wall_t / 2, wall_z_mid,
                sx_full, wall_t, wall_z_ext))
    # East/West walls span the inner Y range (between N/S walls), so the
    # corner regions are covered by the N/S walls only.
    inner_sy = max(0.0, sy_full - 2 * wall_t)
    out.append((plate_xmax - wall_t / 2, cy_mid, wall_z_mid,
                wall_t, inner_sy, wall_z_ext))
    out.append((plate_xmin + wall_t / 2, cy_mid, wall_z_mid,
                wall_t, inner_sy, wall_z_ext))
    return out


def _write_pretty(root: ET.Element, out_path: Path) -> None:
    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(
        indent="  ", encoding="utf-8",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pretty)


def _slice_by_axis_plane(
    mesh: trimesh.Trimesh,
    axis: int,
    split: float,
    keep: str,
) -> trimesh.Trimesh:
    """Clip a mesh against an axis-aligned plane without dropping
    triangles that cross the split.

    The leg OBJ has a few large triangles spanning the shoulder where the
    tenon meets the body. Face filtering by "all vertices are in range"
    drops those triangles entirely, creating a visible missing chunk.
    ``slice_plane`` introduces split-plane vertices and preserves both
    surface area partitions.
    """
    if keep not in {"lower", "upper"}:
        raise ValueError(f"keep must be 'lower' or 'upper', got {keep!r}")
    origin = np.zeros(3)
    origin[axis] = split
    normal = np.zeros(3)
    normal[axis] = -1.0 if keep == "lower" else 1.0
    out = mesh.slice_plane(plane_origin=origin, plane_normal=normal, cap=False)
    if out is None or len(out.faces) == 0:
        raise ValueError("axis plane slice produced no faces")
    return out


def _extract_receiver_detail_patch(
    canonical_top: trimesh.Trimesh,
    hole_xy: Tuple[float, float],
    hole_xy_size: Tuple[float, float],
) -> Tuple[trimesh.Trimesh, Tuple[float, float, float, float]]:
    hx, hy = hole_xy
    sx, sy = hole_xy_size
    x_lo = hx - sx / 2 - DETAIL_PATCH_PAD_M
    x_hi = hx + sx / 2 + DETAIL_PATCH_PAD_M
    y_lo = hy - sy / 2 - DETAIL_PATCH_PAD_M
    y_hi = hy + sy / 2 + DETAIL_PATCH_PAD_M
    patch_extent = np.array([x_hi - x_lo, y_hi - y_lo])

    selected = []
    for comp in canonical_top.split(only_watertight=False):
        bb = comp.bounds
        center = (bb[0] + bb[1]) / 2
        ext_xy = bb[1, :2] - bb[0, :2]
        center_in_patch = x_lo <= center[0] <= x_hi and y_lo <= center[1] <= y_hi
        overlaps_patch = not (
            bb[1, 0] < x_lo or bb[0, 0] > x_hi
            or bb[1, 1] < y_lo or bb[0, 1] > y_hi
        )
        # Drop whole-table planes/edges that merely overlap the local crop.
        local_sized = bool(np.all(ext_xy <= patch_extent + 0.006))
        if center_in_patch and overlaps_patch and local_sized:
            selected.append(comp)

    if not selected:
        raise ValueError("no original OBJ components found around one_leg hole")
    patch = trimesh.util.concatenate(selected)
    return patch, (x_lo, x_hi, y_lo, y_hi)


def _write_inserter_hybrid_urdf(
    out_urdf: Path,
    robot_name: str,
    thread_obj_rel: str,
    body_box_center: Tuple[float, float, float],
    body_box_size: Tuple[float, float, float],
) -> None:
    robot = ET.Element("robot", attrib={"name": robot_name})

    base = ET.SubElement(robot, "link", attrib={"name": "base_link"})
    iner = ET.SubElement(base, "inertial")
    ET.SubElement(iner, "mass", attrib={"value": "0.05"})
    ET.SubElement(iner, "inertia", attrib={
        "ixx": "1e-5", "ixy": "0", "ixz": "0",
        "iyy": "1e-5", "iyz": "0", "izz": "1e-5",
    })

    mat_thread = ET.SubElement(robot, "material", attrib={"name": "thread"})
    ET.SubElement(mat_thread, "color", attrib={"rgba": "0.95 0.55 0.10 1.0"})
    mat_body = ET.SubElement(robot, "material", attrib={"name": "body"})
    ET.SubElement(mat_body, "color", attrib={"rgba": "0.50 0.50 0.50 1.0"})

    thread_link = ET.SubElement(robot, "link", attrib={"name": "thread_link"})
    add_mesh_visual(thread_link, thread_obj_rel, material_name="thread")
    add_sdf_collision(thread_link, thread_obj_rel, resolution=SDF_RES)
    j = ET.SubElement(robot, "joint", attrib={"name": "base_to_thread", "type": "fixed"})
    ET.SubElement(j, "parent", attrib={"link": "base_link"})
    ET.SubElement(j, "child", attrib={"link": "thread_link"})
    ET.SubElement(j, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})

    body_link = ET.SubElement(robot, "link", attrib={"name": "body_link"})
    add_box_collision_visual(
        body_link, body_box_size, body_box_center, material_name="body",
    )
    j = ET.SubElement(robot, "joint", attrib={"name": "base_to_body", "type": "fixed"})
    ET.SubElement(j, "parent", attrib={"link": "base_link"})
    ET.SubElement(j, "child", attrib={"link": "body_link"})
    ET.SubElement(j, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})

    _write_pretty(robot, out_urdf)


def _write_receptive_hybrid_urdf(
    out_urdf: Path,
    robot_name: str,
    bulk_boxes: List[Tuple[float, float, float, float, float, float]],
    detail_patch_rel: str,
) -> None:
    robot = ET.Element("robot", attrib={"name": robot_name})
    link = ET.SubElement(robot, "link", attrib={"name": "receptive"})

    mat_bulk = ET.SubElement(robot, "material", attrib={"name": "bulk"})
    ET.SubElement(mat_bulk, "color", attrib={"rgba": "0.78 0.55 0.30 1.0"})
    mat_hole = ET.SubElement(robot, "material", attrib={"name": "hole_detail"})
    ET.SubElement(mat_hole, "color", attrib={"rgba": "0.30 0.65 0.85 1.0"})

    for cx, cy, cz, sx, sy, sz in bulk_boxes:
        add_box_collision_visual(
            link, (sx, sy, sz), (cx, cy, cz), material_name="bulk",
        )
    add_mesh_visual(link, detail_patch_rel, material_name="hole_detail")
    add_sdf_collision(link, detail_patch_rel, resolution=SDF_RES)

    _write_pretty(robot, out_urdf)


def main() -> None:
    setup_path = ASSETS_FB / "square_table" / "one_leg_setup.json"
    if not setup_path.is_file():
        raise SystemExit(f"missing {setup_path}; run step2 first.")
    setup = json.loads(setup_path.read_text())

    parent_name = setup["parent_part"]
    child_name = setup["child_part"]
    active_hole_xy = tuple(float(v) for v in setup["hole_xy_canonical"])
    hole_xy_size = tuple(float(v) for v in setup["hole_xy_size_canonical"])

    print("\n=== one_leg receptive SDF-hybrid ===")
    top_obj = ASSETS_FB / "square_table" / parent_name / f"{parent_name}_canonical.obj"
    canonical_top = trimesh.load(str(top_obj), force="mesh", process=False)
    detail_patch, detail_bbox_xy = _extract_receiver_detail_patch(
        canonical_top, active_hole_xy, hole_xy_size,
    )

    hybrid_patch_dir = ASSETS_FB / "square_table" / "hole_patches" / "sdf_hybrid"
    if hybrid_patch_dir.exists():
        shutil.rmtree(hybrid_patch_dir)
    hybrid_patch_dir.mkdir(parents=True, exist_ok=True)
    detail_patch_path = hybrid_patch_dir / "one_leg_hole_detail.obj"
    detail_patch.export(str(detail_patch_path))
    print(
        f"  detail patch: {len(detail_patch.vertices)}v / {len(detail_patch.faces)}f "
        f"watertight={detail_patch.is_watertight}"
    )
    print(f"  detail OBJ → {detail_patch_path.relative_to(REPO_ROOT)}")

    bb = canonical_top.bounds
    # Open-top tray: base plate + 4 perimeter walls. The cavity in the
    # middle is left open so the SDF hole patch is the only thing the leg
    # collides with at the active hole.
    bulk_boxes = _open_top_tray_boxes(
        plate_xmin=float(bb[0, 0]), plate_xmax=float(bb[1, 0]),
        plate_ymin=float(bb[0, 1]), plate_ymax=float(bb[1, 1]),
        plate_zmin=float(bb[0, 2]), plate_zmax=float(bb[1, 2]),
    )

    fixtures_dir = ASSETS_FB / "square_table" / "insertion_fixtures"
    receptive_urdf = fixtures_dir / "one_leg_sdf_hybrid.urdf"
    _write_receptive_hybrid_urdf(
        out_urdf=receptive_urdf,
        robot_name="one_leg_receptive_sdf_hybrid",
        bulk_boxes=bulk_boxes,
        detail_patch_rel="../hole_patches/sdf_hybrid/one_leg_hole_detail.obj",
    )
    print(f"  bulk frame: {len(bulk_boxes)} boxes")
    print(f"  receptive URDF → {receptive_urdf.relative_to(REPO_ROOT)}")

    print("\n=== one_leg inserter SDF-hybrid ===")
    leg_dir = ASSETS_FB / "square_table" / child_name
    leg_obj = leg_dir / f"{child_name}_canonical.obj"
    canonical_leg = trimesh.load(str(leg_obj), force="mesh", process=False)

    axis = 0
    x_min, x_max = canonical_leg.bounds[:, axis]
    thread_hi = float(x_min + THREAD_LENGTH_M)
    thread_mesh = _slice_by_axis_plane(canonical_leg, axis, thread_hi, keep="lower")
    body_mesh = _slice_by_axis_plane(canonical_leg, axis, thread_hi, keep="upper")

    sdf_dir = leg_dir / "sdf_hybrid"
    if sdf_dir.exists():
        shutil.rmtree(sdf_dir)
    sdf_dir.mkdir(parents=True, exist_ok=True)
    thread_obj = sdf_dir / f"{child_name}_thread.obj"
    body_obj = sdf_dir / f"{child_name}_body.obj"
    thread_mesh.export(str(thread_obj))
    body_mesh.export(str(body_obj))
    print(
        f"  thread mesh: {len(thread_mesh.vertices)}v / {len(thread_mesh.faces)}f "
        f"x=[{x_min:.5f}, {thread_hi:.5f}]"
    )
    print(f"  thread OBJ → {thread_obj.relative_to(REPO_ROOT)}")

    body_box_min, body_box_max = body_mesh.bounds
    body_box_center = tuple(float(v) for v in ((body_box_min + body_box_max) / 2))
    body_box_size = tuple(float(v) for v in (body_box_max - body_box_min))

    inserter_urdf = sdf_dir / f"{child_name}_sdf_hybrid.urdf"
    _write_inserter_hybrid_urdf(
        out_urdf=inserter_urdf,
        robot_name=f"{child_name}_sdf_hybrid",
        thread_obj_rel=thread_obj.name,
        body_box_center=body_box_center,
        body_box_size=body_box_size,
    )
    print(
        "  body collision: 1 box "
        f"center={np.round(body_box_center, 6).tolist()} "
        f"size={np.round(body_box_size, 6).tolist()}"
    )
    print(f"  inserter URDF → {inserter_urdf.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
