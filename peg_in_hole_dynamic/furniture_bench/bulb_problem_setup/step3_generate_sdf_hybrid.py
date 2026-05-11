"""Step 3 of the bulb-into-base problem-setup pipeline.

Mirrors one_leg_problem_setup/step3_generate_sdf_hybrid.py:
  * Receptive: SDF mesh patch sliced from the lamp_base canonical OBJ
    around the bulb socket, surrounded by an open-top tray bulk (1 base
    plate + 4 perimeter walls). The tray helper is shared with one_leg
    so any future tweak lands in both fixtures.
  * Inserter: thread tip SDF (canonical-X min end of the bulb, since the
    threaded screw base is at storage -Y → canonical -X) + body box.

Run after step2:
    .venv/bin/python -m peg_in_hole_dynamic.furniture_bench.bulb_problem_setup.step3_generate_sdf_hybrid
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

from peg_in_hole_dynamic.fmb.run_coacd import run_coacd_python
from peg_in_hole_dynamic.furniture_bench.one_leg_problem_setup.step3_generate_sdf_hybrid import (
    _slice_by_axis_plane,
)
from peg_in_hole_dynamic.sdf_hybrid_utils import (
    add_box_collision_visual,
    add_mesh_collision,
    add_mesh_visual,
    add_sdf_collision,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_FB = REPO_ROOT / "assets" / "urdf" / "furniture_bench"

SDF_RES = 256
# The bulb's "lower part" (everything from the threaded tip up through the
# narrow neck before the glass globe widens) goes into the SDF. The neck
# region transitions from ~18 mm at the tip to ~45 mm at the glass shoulder
# around canonical X = -0.0205 from the centered-canonical bottom (-0.0605),
# so 40 mm covers the full screw cap + neck. The wide glass globe above
# stays in the CoACD body hulls.
THREAD_LENGTH_M = 0.040
# Receptive split: anything above ``BASE_HEIGHT_M`` from the canonical
# lamp_base bottom is treated as the active socket region and goes into
# the SDF hole mesh. Anything below is a single primitive box approximating
# the wide flat base. 12 mm is the thickness of the flat bottom slab
# before the lamp_base geometry starts narrowing into the socket neck.
BASE_HEIGHT_M = 0.012
# The bulb body is a glass globe, which a single AABB approximates poorly.
# CoACD it with a cap on hull count — without the cap the upstream bulb
# OBJ (~25k faces, complex glass surface) generates 165 hulls, which is
# slow at sim startup and at every physics step.
BODY_COACD_KW = dict(threshold=0.05, max_convex_hull=24, seed=0)


def _write_pretty(root: ET.Element, out_path: Path) -> None:
    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(
        indent="  ", encoding="utf-8",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pretty)


def _extract_receiver_socket_region(
    canonical_base: trimesh.Trimesh,
    base_top_z: float,
) -> trimesh.Trimesh:
    """Return the portion of the canonical lamp_base above ``base_top_z``.

    Used as the SDF mesh for the active socket region. The split is a
    horizontal Z plane: everything above goes into the SDF, everything
    below is approximated by a primitive base box. trimesh's plane slicer
    handles non-watertight meshes (the lamp_base mesh is split across
    ~22k disconnected components from upstream apriltag/edge OBJs)."""
    out = canonical_base.slice_plane(
        plane_origin=(0.0, 0.0, base_top_z),
        plane_normal=(0.0, 0.0, 1.0),
        cap=False,
    )
    if out is None or len(out.faces) == 0:
        raise ValueError(
            f"slicing canonical lamp_base above z={base_top_z} produced no faces"
        )
    return out


def _write_inserter_hybrid_urdf(
    out_urdf: Path,
    robot_name: str,
    thread_obj_rel: str,
    body_hull_filenames: List[str],
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
    # One <visual>/<collision> per CoACD hull so the body collision tracks
    # the glass-globe shape and visualize_problems renders each hull in a
    # distinct palette color.
    for hull_filename in body_hull_filenames:
        add_mesh_visual(body_link, hull_filename, material_name="body")
        add_mesh_collision(body_link, hull_filename)
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
    setup_path = ASSETS_FB / "lamp" / "bulb_screw_setup.json"
    if not setup_path.is_file():
        raise SystemExit(f"missing {setup_path}; run step2 first.")
    setup = json.loads(setup_path.read_text())

    parent_name = setup["parent_part"]
    child_name  = setup["child_part"]
    active_hole_xy = tuple(float(v) for v in setup["hole_xy_canonical"])
    hole_xy_size   = tuple(float(v) for v in setup["hole_xy_size_canonical"])

    print("\n=== bulb_screw receptive SDF-hybrid ===")
    base_obj = ASSETS_FB / "lamp" / parent_name / f"{parent_name}_canonical.obj"
    canonical_base = trimesh.load(str(base_obj), force="mesh", process=False)
    bb = canonical_base.bounds
    plate_zmin = float(bb[0, 2])
    base_top_z = plate_zmin + BASE_HEIGHT_M

    # Everything above ``base_top_z`` is the SDF socket region.
    detail_patch = _extract_receiver_socket_region(canonical_base, base_top_z)
    hybrid_patch_dir = ASSETS_FB / "lamp" / "hole_patches" / "sdf_hybrid"
    if hybrid_patch_dir.exists():
        shutil.rmtree(hybrid_patch_dir)
    hybrid_patch_dir.mkdir(parents=True, exist_ok=True)
    detail_patch_path = hybrid_patch_dir / "bulb_screw_hole_detail.obj"
    detail_patch.export(str(detail_patch_path))
    print(
        f"  socket SDF region: {len(detail_patch.vertices)}v / {len(detail_patch.faces)}f "
        f"(z >= {base_top_z:+.4f})"
    )
    print(f"  socket OBJ → {detail_patch_path.relative_to(REPO_ROOT)}")

    # Bulk = a single primitive box covering the wide flat base of the
    # lamp_base from canonical bottom up to ``base_top_z``. Full canonical
    # XY footprint.
    box_sx = float(bb[1, 0] - bb[0, 0])
    box_sy = float(bb[1, 1] - bb[0, 1])
    box_sz = base_top_z - plate_zmin
    box_cx = float((bb[0, 0] + bb[1, 0]) / 2)
    box_cy = float((bb[0, 1] + bb[1, 1]) / 2)
    box_cz = (plate_zmin + base_top_z) / 2
    bulk_boxes = [(box_cx, box_cy, box_cz, box_sx, box_sy, box_sz)]

    fixtures_dir = ASSETS_FB / "lamp" / "insertion_fixtures"
    receptive_urdf = fixtures_dir / "bulb_screw_sdf_hybrid.urdf"
    _write_receptive_hybrid_urdf(
        out_urdf=receptive_urdf,
        robot_name="bulb_screw_receptive_sdf_hybrid",
        bulk_boxes=bulk_boxes,
        detail_patch_rel="../hole_patches/sdf_hybrid/bulb_screw_hole_detail.obj",
    )
    print(f"  bulk frame: {len(bulk_boxes)} boxes")
    print(f"  receptive URDF → {receptive_urdf.relative_to(REPO_ROOT)}")

    print("\n=== bulb_screw inserter SDF-hybrid ===")
    bulb_dir = ASSETS_FB / "lamp" / child_name
    bulb_obj = bulb_dir / f"{child_name}_canonical.obj"
    canonical_bulb = trimesh.load(str(bulb_obj), force="mesh", process=False)

    # Threaded screw base is at canonical -X (storage -Y → canonical -X under
    # the inserter rotation). Slice the bottom 25 mm off.
    axis = 0
    x_min, x_max = canonical_bulb.bounds[:, axis]
    thread_hi = float(x_min + THREAD_LENGTH_M)
    thread_mesh = _slice_by_axis_plane(canonical_bulb, axis, thread_hi, keep="lower")
    body_mesh = _slice_by_axis_plane(canonical_bulb, axis, thread_hi, keep="upper")

    sdf_dir = bulb_dir / "sdf_hybrid"
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

    # CoACD the body so the glass-globe shape is approximated by a
    # collection of convex hulls instead of a single AABB.
    body_coacd_dir = sdf_dir / "coacd"
    run_coacd_python(body_obj, body_coacd_dir, **BODY_COACD_KW)
    body_hull_filenames = [
        f"coacd/{p.name}" for p in sorted(body_coacd_dir.glob("decomp_*.obj"))
    ]
    if not body_hull_filenames:
        print("  WARN: CoACD on body produced 0 hulls; falling back to body.obj as a single mesh.")
        body_hull_filenames = [body_obj.name]

    inserter_urdf = sdf_dir / f"{child_name}_sdf_hybrid.urdf"
    _write_inserter_hybrid_urdf(
        out_urdf=inserter_urdf,
        robot_name=f"{child_name}_sdf_hybrid",
        thread_obj_rel=thread_obj.name,
        body_hull_filenames=body_hull_filenames,
    )
    print(f"  body collision: {len(body_hull_filenames)} CoACD hulls")
    print(f"  inserter URDF → {inserter_urdf.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
