"""Step 3 of the SDF + CoACD hybrid pipeline (fmb.peg_board_1 branch).

Mirrors ``fabrica/beam_2x_problem_setup/step2_generate_sdf_hybrid.py`` for
the FMB peg-board pipeline. Per (peg, hole) pair from
``peg_board_1_assemblies.json``:

  * Inserter (peg): split canonical mesh into the bottom 25 mm "tip" along
    its long axis (canonical Z for upright peg meshes), CoACD the body,
    write a 3-link URDF — base_link → tip_link (SDF) + body_link (CoACD).

  * Receptive (peg_board_<N>): reuse the existing ``hole_patches/<hole>.obj``
    sliced by step2 — wrap THAT mesh as an SDF in a new receptive URDF.
    The same 4-box bulk frame around the active hole is reused; other
    holes on the same board are still ignored.

The pose convention for the inserter is the same compose used by
``problems.py:_register_peg_board_problems``:

    q = R_x180 ∘ R_yaw_saved ∘ R_canonical_inv

where R_canonical_inv = R_z(-90°) iff the canonical mesh was rotated 90°
about Z when generated (recorded in ``canonical_meta.json``).

Run:
    .venv/bin/python -m peg_in_hole_dynamic.fmb.peg_board_problem_setup.step3_generate_sdf_hybrid
    .venv/bin/python -m peg_in_hole_dynamic.fmb.peg_board_problem_setup.step3_generate_sdf_hybrid --hole hole_4
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from xml.dom import minidom

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from peg_in_hole_dynamic.fmb.run_coacd import run_coacd_python
from peg_in_hole_dynamic.sdf_hybrid_utils import (
    add_box_collision_visual,
    add_mesh_collision,
    add_mesh_visual,
    add_sdf_collision,
    slice_mesh_by_axis_range,
)
from peg_in_hole_dynamic.fmb.peg_board_problem_setup.step2_generate_assets import (
    HOLE_PADDING,
    _board_to_A_frame,
    _decompose_plate_around_one_hole,
)


REPO_ROOT      = Path(__file__).resolve().parents[3]
ASSETS_DIR     = REPO_ROOT / "assets" / "urdf" / "fmb"
JSON_PATH      = Path(__file__).with_name("peg_board_1_assemblies.json")

BOARD_NAME     = "peg_board_1"
BOARD_OBJ      = ASSETS_DIR / "boards" / BOARD_NAME / f"{BOARD_NAME}.obj"
PEG_DIR        = ASSETS_DIR / "pegs"

TIP_HEIGHT_M   = 0.025
SDF_RES        = 256
BODY_COACD_KW  = dict(threshold=0.03, max_convex_hull=-1, seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# Pose helpers (must match fmb/problems.py:_register_peg_board_problems)
# ─────────────────────────────────────────────────────────────────────────────

def _peg_assembled_quat(yaw_saved_deg: float, rotated_z90: bool) -> R:
    """The receiver-A-frame rotation applied to the peg's canonical mesh."""
    R_x180 = R.from_euler("x", 180, degrees=True)
    R_yaw  = R.from_euler("z", float(yaw_saved_deg), degrees=True)
    R_can_inv = R.from_euler("z", -90, degrees=True) if rotated_z90 else R.identity()
    return R_x180 * R_yaw * R_can_inv


def _long_axis_index(mesh: trimesh.Trimesh) -> int:
    return int(np.argmax(mesh.extents))


def _tip_side_for_long_axis(canonical: trimesh.Trimesh, axis: int,
                             rotation: R) -> str:
    """Pick canonical "min" or "max" end as the tip — the end whose canonical
    +axis aligns with the world insertion direction (0, 0, -1) after the
    URDF's pose-applied rotation. Same rule as the fabrica step.
    """
    R_can_to_A = rotation.as_matrix()
    tip_dir_can = R_can_to_A.T @ np.array([0.0, 0.0, -1.0])
    pos_dot = float(tip_dir_can[axis])
    return "max" if pos_dot > 0 else "min"


def _slice_canonical_into_tip_and_body(canonical: trimesh.Trimesh,
                                       axis: int, side: str
                                       ) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    bb_min, bb_max = canonical.bounds.copy()
    long_extent = bb_max[axis] - bb_min[axis]
    if long_extent <= TIP_HEIGHT_M + 1e-4:
        raise ValueError(
            f"Canonical mesh long-axis extent ({long_extent*1000:.1f} mm) is "
            f"not larger than the requested tip height ({TIP_HEIGHT_M*1000:.1f} mm)."
        )
    if side == "min":
        tip = slice_mesh_by_axis_range(canonical, axis, bb_min[axis], bb_min[axis] + TIP_HEIGHT_M)
        body = slice_mesh_by_axis_range(canonical, axis, bb_min[axis] + TIP_HEIGHT_M, bb_max[axis])
    else:
        tip = slice_mesh_by_axis_range(canonical, axis, bb_max[axis] - TIP_HEIGHT_M, bb_max[axis])
        body = slice_mesh_by_axis_range(canonical, axis, bb_min[axis], bb_max[axis] - TIP_HEIGHT_M)
    return tip, body


# ─────────────────────────────────────────────────────────────────────────────
# URDF writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_pretty(root: ET.Element, out_path: Path) -> None:
    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ", encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pretty)


def _write_inserter_hybrid_urdf(out_urdf: Path, robot_name: str,
                                 tip_obj_rel: str, body_hull_filenames: List[str]) -> None:
    """3-link URDF: base_link → tip_link (SDF) + body_link (CoACD).
    Per-hull visuals on body_link so the visualizer renders each piece in
    its own palette colour (matches existing fabrica inserter convention)."""
    robot = ET.Element("robot", attrib={"name": robot_name})

    base = ET.SubElement(robot, "link", attrib={"name": "base_link"})
    iner = ET.SubElement(base, "inertial")
    ET.SubElement(iner, "mass",   attrib={"value": "0.05"})
    ET.SubElement(iner, "inertia",
                  attrib={"ixx": "1e-5", "ixy": "0", "ixz": "0",
                          "iyy": "1e-5", "iyz": "0", "izz": "1e-5"})

    mat_tip  = ET.SubElement(robot, "material", attrib={"name": "tip"})
    ET.SubElement(mat_tip, "color", attrib={"rgba": "0.95 0.55 0.10 1.0"})
    mat_body = ET.SubElement(robot, "material", attrib={"name": "body"})
    ET.SubElement(mat_body, "color", attrib={"rgba": "0.50 0.50 0.50 1.0"})

    tip_link = ET.SubElement(robot, "link", attrib={"name": "tip_link"})
    add_mesh_visual(tip_link, tip_obj_rel, material_name="tip")
    add_sdf_collision(tip_link, tip_obj_rel, resolution=SDF_RES)
    j = ET.SubElement(robot, "joint", attrib={"name": "base_to_tip", "type": "fixed"})
    ET.SubElement(j, "parent", attrib={"link": "base_link"})
    ET.SubElement(j, "child",  attrib={"link": "tip_link"})
    ET.SubElement(j, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})

    body_link = ET.SubElement(robot, "link", attrib={"name": "body_link"})
    for hull_filename in body_hull_filenames:
        add_mesh_visual(body_link, hull_filename, material_name="body")
        add_mesh_collision(body_link, hull_filename)
    j = ET.SubElement(robot, "joint", attrib={"name": "base_to_body", "type": "fixed"})
    ET.SubElement(j, "parent", attrib={"link": "base_link"})
    ET.SubElement(j, "child",  attrib={"link": "body_link"})
    ET.SubElement(j, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})

    _write_pretty(robot, out_urdf)


def _write_receptive_hybrid_urdf(
    out_urdf: Path, robot_name: str,
    bulk_boxes: List[Tuple[float, float, float, float, float, float]],
    hole_patch_mesh_rel: str,
) -> None:
    """Per-(board, peg) receptive URDF. Single <link "plate"> with:
        * one <box> per bulk frame box (in A-frame coords),
        * one <visual>+<collision> SDF mesh = the hole patch (in A-frame
          coords already, since fmb hole-patches are sliced after the
          board has been transformed into A-frame).
    """
    robot = ET.Element("robot", attrib={"name": robot_name})
    link  = ET.SubElement(robot, "link", attrib={"name": "plate"})

    mat_bulk = ET.SubElement(robot, "material", attrib={"name": "bulk"})
    ET.SubElement(mat_bulk, "color", attrib={"rgba": "0.78 0.55 0.30 1.0"})
    mat_hole = ET.SubElement(robot, "material", attrib={"name": "hole"})
    ET.SubElement(mat_hole, "color", attrib={"rgba": "0.30 0.65 0.85 1.0"})

    for cx, cy, cz, sx, sy, sz in bulk_boxes:
        add_box_collision_visual(link, (sx, sy, sz), (cx, cy, cz),
                                 material_name="bulk")

    add_mesh_visual(link, hole_patch_mesh_rel, material_name="hole")
    add_sdf_collision(link, hole_patch_mesh_rel, resolution=SDF_RES)

    _write_pretty(robot, out_urdf)


# ─────────────────────────────────────────────────────────────────────────────
# Per-(peg, hole) driver
# ─────────────────────────────────────────────────────────────────────────────

def _process_pair(hole_id: str, info: dict, board_A: trimesh.Trimesh) -> None:
    peg_name = info["peg"]
    print(f"\n=== fmb.{BOARD_NAME}.{peg_name}_sdf_hybrid (hole={hole_id}) ===")

    # ── Inserter side ──────────────────────────────────────────────
    peg_dir       = PEG_DIR / peg_name
    canonical_obj = peg_dir / f"{peg_name}_canonical.obj"
    meta_path     = peg_dir / "canonical_meta.json"
    if not canonical_obj.is_file() or not meta_path.is_file():
        print(f"  WARN: missing canonical assets for {peg_name}; skip")
        return

    canonical = trimesh.load_mesh(str(canonical_obj), process=False)
    rotated_z90 = bool(json.loads(meta_path.read_text())["canonical_rotated_z90"])

    rotation = _peg_assembled_quat(info["yaw_deg"], rotated_z90)
    axis     = _long_axis_index(canonical)
    side     = _tip_side_for_long_axis(canonical, axis, rotation)
    print(f"  peg long axis = {'XYZ'[axis]} ({side}-end is tip)")

    tip_mesh, body_mesh = _slice_canonical_into_tip_and_body(canonical, axis, side)
    print(f"  tip {len(tip_mesh.vertices)}v / body {len(body_mesh.vertices)}v")

    sdf_dir  = peg_dir / "sdf_hybrid"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    tip_obj  = sdf_dir / f"{peg_name}_tip.obj"
    body_obj = sdf_dir / f"{peg_name}_body.obj"
    tip_mesh.export(str(tip_obj))
    body_mesh.export(str(body_obj))

    body_coacd_dir = sdf_dir / "coacd"
    run_coacd_python(body_obj, body_coacd_dir, **BODY_COACD_KW)
    body_hull_filenames = [
        f"coacd/{p.name}" for p in sorted(body_coacd_dir.glob("decomp_*.obj"))
    ]
    if not body_hull_filenames:
        print("  WARN: CoACD on body produced 0 hulls; falling back to body mesh as a single shape.")
        body_hull_filenames = [body_obj.name]

    inserter_robot_name = f"fmb_{peg_name}_sdf_hybrid"
    inserter_urdf = sdf_dir / f"{peg_name}_sdf_hybrid.urdf"
    _write_inserter_hybrid_urdf(
        out_urdf=inserter_urdf, robot_name=inserter_robot_name,
        tip_obj_rel=tip_obj.name, body_hull_filenames=body_hull_filenames,
    )
    print(f"  inserter URDF → {inserter_urdf.relative_to(REPO_ROOT)}  "
          f"(tip SDF + {len(body_hull_filenames)} body hulls)")

    # ── Receptive side ────────────────────────────────────────────
    # Reuse the existing hole-patch obj sliced by step2.
    patch_dir   = ASSETS_DIR / "boards" / BOARD_NAME / "hole_patches"
    hole_patch  = patch_dir / f"{hole_id}.obj"
    if not hole_patch.is_file():
        print(f"  WARN: missing {hole_patch}; run step2 first. skip.")
        return

    plate_xmin, plate_xmax = float(board_A.bounds[0, 0]), float(board_A.bounds[1, 0])
    plate_ymin, plate_ymax = float(board_A.bounds[0, 1]), float(board_A.bounds[1, 1])
    plate_zmin, plate_zmax = float(board_A.bounds[0, 2]), float(board_A.bounds[1, 2])
    cx, cy = info["hole_xy_A"]
    dx, dy = info["hole_bbox"]
    hole_bbox_xy = (
        cx - dx / 2 - HOLE_PADDING, cx + dx / 2 + HOLE_PADDING,
        cy - dy / 2 - HOLE_PADDING, cy + dy / 2 + HOLE_PADDING,
    )
    bulk_boxes = _decompose_plate_around_one_hole(
        plate_xmin, plate_xmax, plate_ymin, plate_ymax, plate_zmin, plate_zmax,
        hole_bbox_xy,
    )
    print(f"  bulk frame: {len(bulk_boxes)} boxes")

    fixtures_dir = ASSETS_DIR / "boards" / BOARD_NAME / "insertion_fixtures"
    receptive_urdf = fixtures_dir / f"{BOARD_NAME}_{peg_name}_sdf_hybrid.urdf"
    receptive_robot_name = f"fixture_{BOARD_NAME}_{peg_name}_sdf_hybrid"
    hole_patch_mesh_rel = f"../hole_patches/{hole_patch.name}"
    _write_receptive_hybrid_urdf(
        out_urdf=receptive_urdf, robot_name=receptive_robot_name,
        bulk_boxes=bulk_boxes, hole_patch_mesh_rel=hole_patch_mesh_rel,
    )
    print(f"  receptive URDF → {receptive_urdf.relative_to(REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hole", default=None,
                        help="hole_id (e.g. hole_4) to process; default = all entries.")
    args = parser.parse_args()

    if not JSON_PATH.is_file():
        raise SystemExit(f"Missing {JSON_PATH}; run step1 first.")
    data: dict = json.loads(JSON_PATH.read_text())

    board_A = _board_to_A_frame(trimesh.load_mesh(str(BOARD_OBJ), process=False))

    if args.hole is not None:
        if args.hole not in data:
            raise SystemExit(f"hole {args.hole!r} not in {JSON_PATH.name}")
        _process_pair(args.hole, data[args.hole], board_A)
    else:
        for hole_id in sorted(data.keys()):
            _process_pair(hole_id, data[hole_id], board_A)


if __name__ == "__main__":
    main()
