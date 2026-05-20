"""Step 2 of the SDF + CoACD hybrid pipeline (fabrica.beam_2x branch).

Mirrors ``step1_generate_assets.py`` but for the ``_sdf_hybrid`` Problem
variant only. Produces, for one inserter at a time:

  * ``<inserter>/sdf_hybrid/<assembly>_<inserter>_sdf_hybrid.urdf`` —
    a 3-link inserter URDF combining
        - tip_link  : SDF mesh of the bottom 25 mm of the canonical mesh,
        - body_link : CoACD hulls of the rest (re-CoACD'd, so no hull
                      overlaps the tip region).
  * ``insertion_fixtures/part_<inserter>_sdf_hybrid.urdf`` — a per-problem
    receptive URDF combining
        - the same 4-box bulk frame around the hole as step1,
        - one A-frame-positioned SDF mesh covering just the receiver's
          hole patch (sliced from the receiver canonical mesh by the
          inserter's bbox + 5 mm padding, transformed into the receiver's
          canonical frame).

Other-fixture-part bboxes (non-receiver fixture parts) are unchanged
from step1.

Initial scope (per the plan): only ``inserter_id == "2"``. Loop is
straightforward to extend once we've validated this one end-to-end.

Run:
    .venv/bin/python -m peg_in_hole_dynamic.fabrica.beam_3x_problem_setup.step2_generate_sdf_hybrid
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
    slice_mesh_by_3d_bbox,
    slice_mesh_by_axis_range,
    slice_mesh_by_xy_bbox,
    subtract_3d_bbox_from_mesh,
)
from peg_in_hole_dynamic.fabrica.beam_3x_problem_setup.step1_generate_assets import (
    NEAR_PADDING,
    _assembled_pose,
    _bbox,
    _bbox_to_box_primitive,
    _decompose_around_hole,
    _load_part_meshes,
    _transform_to_A,
)


REPO_ROOT     = Path(__file__).resolve().parents[3]
ASSEMBLY      = "beam_3x"
ASSEMBLY_DIR  = REPO_ROOT / "assets" / "urdf" / "fabrica" / ASSEMBLY
FIXTURES_DIR  = ASSEMBLY_DIR / "insertion_fixtures"

TIP_HEIGHT_M  = 0.0375     # inserter tip = bottom 37.5 mm (1.5x beam_2x's 25 mm to match the 1.5x larger pillars in beam_3x)
HOLE_PAD      = NEAR_PADDING   # 5 mm — same padding step1 uses for hull overlap
SDF_RES       = 256        # per-shape SDF resolution for tip + receiver hole patch
BODY_COACD_KW = dict(threshold=0.03, max_convex_hull=-1, seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# Inserter splitting: insertion-axis tip + body
# ─────────────────────────────────────────────────────────────────────────────

def _insertion_axis(tip_dir: np.ndarray) -> int:
    """Canonical axis most aligned with the insertion direction.

    The world insertion direction is (0,0,-1); ``tip_dir`` is that vector
    transformed into the inserter's canonical frame. The axis whose
    canonical basis vector projects most strongly onto ``tip_dir`` is the
    one we slice the tip off of.
    """
    return int(np.argmax(np.abs(tip_dir)))


def _a_frame_bbox_in_canonical(
    bb_A: Tuple[np.ndarray, np.ndarray],
    pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project an axis-aligned A-frame bbox into the canonical frame of an
    object whose pose is ``(pos, quat)`` (canonical → A). Returns the AABB
    of the rotated bbox in canonical."""
    pos, quat = pose
    R_can_to_A = R.from_quat(quat).as_matrix()
    R_A_to_can = R_can_to_A.T
    bb_min, bb_max = bb_A
    corners_A = np.array([
        [bb_min[0], bb_min[1], bb_min[2]],
        [bb_min[0], bb_min[1], bb_max[2]],
        [bb_min[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_max[1], bb_max[2]],
        [bb_max[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_max[1], bb_min[2]],
        [bb_max[0], bb_max[1], bb_max[2]],
    ])
    corners_C = (corners_A - np.asarray(pos)) @ R_A_to_can.T
    return corners_C.min(axis=0), corners_C.max(axis=0)


def _slice_inserter_canonical(
    canonical: trimesh.Trimesh,
    tip_dir: np.ndarray,
    receiver_bb_can: Tuple[np.ndarray, np.ndarray] | None = None,
) -> Tuple[int, str, trimesh.Trimesh, trimesh.Trimesh]:
    """Pick the canonical axis aligned with insertion, then cut into
    (tip, body) where ``tip`` is a 3D slab covering only the receiver's
    contact region: ``TIP_HEIGHT_M``-thick along the insertion axis on
    the side that goes "down" in world, and clipped to the receiver's
    footprint (``receiver_bb_can`` ± ``HOLE_PAD``) in the other two axes.

    If ``receiver_bb_can`` is ``None``, the tip spans the canonical's full
    extent in the other two axes (the previous behavior — useful when the
    inserter's lateral extent is fully covered by the receiver, or when no
    receiver bbox is available).

    Returns ``(axis_index, tip_side ∈ {"min", "max"}, tip_mesh, body_mesh)``.
    """
    axis = _insertion_axis(tip_dir)
    bb_min, bb_max = canonical.bounds.copy()
    axis_extent = bb_max[axis] - bb_min[axis]
    # Auto-clamp the tip slab so it never exceeds 1/3 of the inserter's
    # extent along the insertion axis. Pillars (228 mm in beam_3x) keep
    # the full TIP_HEIGHT_M (37.5 mm < 76 mm). Caps (46 mm thick in
    # beam_3x) get clamped to ~15 mm — just the peg stub, not the whole
    # cap body. Without this clamp, TIP_HEIGHT_M was eating ~80% of cap
    # geometry into the SDF region.
    effective_tip = min(TIP_HEIGHT_M, axis_extent / 3.0)
    if axis_extent <= effective_tip + 1e-4:
        raise ValueError(
            f"Canonical mesh extent along insertion axis "
            f"({axis_extent*1000:.1f} mm) is not larger than the requested "
            f"tip height ({effective_tip*1000:.1f} mm)."
        )

    pos_dot = float(tip_dir[axis])
    side = "max" if pos_dot > 0 else "min"

    # Build the 3D tip box in canonical frame.
    # Insertion-axis range:
    #   - If the receiver's projection along this axis fully encloses the
    #     inserter (e.g. a cap is a through-sleeve and the receiver pillar
    #     passes entirely through it), there is no "tip face" — the column
    #     at the hole position spans the inserter's full extent. Use full
    #     extent in that case.
    #   - Otherwise, take the effective_tip-thick slab on the leading side
    #     (classic peg-tip behavior for pillars going into a hole).
    # Cross-section (other two axes): clipped to the receiver's projected
    # footprint when the inserter is significantly wider than the receiver
    # (see loop below), otherwise full inserter extent.
    tip_bb_min = bb_min.copy()
    tip_bb_max = bb_max.copy()
    # "Receiver dwarfs inserter on the insertion axis" check: when the
    # receiver's projected depth along this axis is significantly larger
    # than the inserter's own extent (e.g. a pillar passes through a cap's
    # through-hole), the cap has no real "tip face" — it's a sleeve. Use
    # the inserter's full extent on this axis instead of a slab.
    fully_enclosed_in_axis = False
    if receiver_bb_can is not None:
        ins_axis_extent  = bb_max[axis] - bb_min[axis]
        recv_axis_extent = receiver_bb_can[1][axis] - receiver_bb_can[0][axis]
        fully_enclosed_in_axis = recv_axis_extent > 2.0 * ins_axis_extent
    if not fully_enclosed_in_axis:
        if side == "min":
            tip_bb_max[axis] = bb_min[axis] + effective_tip
        else:
            tip_bb_min[axis] = bb_max[axis] - effective_tip

    if receiver_bb_can is not None:
        recv_min, recv_max = receiver_bb_can
        for other in range(3):
            if other == axis:
                continue
            inserter_w = bb_max[other] - bb_min[other]
            recv_w = recv_max[other] - recv_min[other]
            # Only clip on axes where the inserter is significantly wider
            # (>=2x) than the receiver's projected footprint. Caps (long
            # along one axis, similar in the other) get clipped along the
            # long axis only — leaves the cap's full height/width along
            # the other axis so the SDF tip is the full peg cross-section,
            # not half of it. Pillars (smaller than receiver in both
            # non-axis dimensions) never trigger any clipping → full
            # pillar cross-section is kept.
            if inserter_w > 2.0 * recv_w:
                tip_bb_min[other] = max(bb_min[other], recv_min[other] - HOLE_PAD)
                tip_bb_max[other] = min(bb_max[other], recv_max[other] + HOLE_PAD)
                if tip_bb_max[other] <= tip_bb_min[other]:
                    raise ValueError(
                        f"Receiver footprint along canonical axis {('XYZ'[other])} "
                        f"({recv_min[other]:.4f}, {recv_max[other]:.4f}) does not "
                        f"overlap inserter canonical bbox "
                        f"({bb_min[other]:.4f}, {bb_max[other]:.4f})."
                    )

    tip = slice_mesh_by_3d_bbox(canonical, tip_bb_min, tip_bb_max)
    body = subtract_3d_bbox_from_mesh(canonical, tip_bb_min, tip_bb_max)
    return axis, side, tip, body


# ─────────────────────────────────────────────────────────────────────────────
# Receiver hole-patch slicing
# ─────────────────────────────────────────────────────────────────────────────

def _world_bbox_to_receiver_canonical_bbox(
    inserter_bbox_A: Tuple[np.ndarray, np.ndarray],
    receiver_pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """Project the inserter's A-frame XY bbox into the receiver's
    canonical frame, return (x_lo, x_hi, y_lo, y_hi) for slicing.

    A-frame point ``p_A`` maps to receiver-canonical via
    ``p_C = R_can_to_A^{-1} (p_A − t_A)``. We sample the 8 corners of
    the inserter's A-frame bbox and take their AABB in canonical frame.
    """
    pos, quat = receiver_pose
    R_can_to_A = R.from_quat(quat).as_matrix()        # canonical → A
    R_A_to_can = R_can_to_A.T

    bb_min, bb_max = inserter_bbox_A
    corners_A = np.array([
        [bb_min[0], bb_min[1], bb_min[2]],
        [bb_min[0], bb_min[1], bb_max[2]],
        [bb_min[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_max[1], bb_max[2]],
        [bb_max[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_max[1], bb_min[2]],
        [bb_max[0], bb_max[1], bb_max[2]],
    ])
    corners_C = (corners_A - np.asarray(pos)) @ R_A_to_can.T
    cmin = corners_C.min(axis=0)
    cmax = corners_C.max(axis=0)
    return (float(cmin[0]), float(cmax[0]), float(cmin[1]), float(cmax[1]))


# ─────────────────────────────────────────────────────────────────────────────
# URDF writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_pretty(root: ET.Element, out_path: Path) -> None:
    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ", encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pretty)


def _write_inserter_hybrid_urdf(out_urdf: Path, robot_name: str,
                                 tip_obj_rel: str, body_obj_rel: str,
                                 body_hull_filenames: List[str]) -> None:
    """A 3-link URDF for the hybrid inserter:

        base_link  (inertial only — collapsed away by Isaac Gym at load)
          ├─ fixed → tip_link   (SDF mesh)
          └─ fixed → body_link  (CoACD hull mesh refs)
    """
    robot = ET.Element("robot", attrib={"name": robot_name})

    base = ET.SubElement(robot, "link", attrib={"name": "base_link"})
    iner = ET.SubElement(base, "inertial")
    ET.SubElement(iner, "mass",   attrib={"value": "0.05"})
    ET.SubElement(iner, "inertia",
                  attrib={"ixx": "1e-5", "ixy": "0", "ixz": "0",
                          "iyy": "1e-5", "iyz": "0", "izz": "1e-5"})

    # Materials — bright tip, gray body, distinguishable in viz.
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
    # Emit one <visual> + <collision> per CoACD hull so visualize_problems.py
    # renders each hull in its own palette colour (matches the viz convention
    # for CoACD-decomposed receptives).
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
    other_part_boxes: List[Tuple[float, float, float, float, float, float]],
    hole_patch_mesh_rel: str,
    receiver_pose: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
) -> None:
    """A single-link receptive URDF with:
       * one box per bulk-frame entry (at A-frame coords);
       * one box per non-receiver fixture part bbox (A-frame);
       * one SDF mesh visual+collision pair for the hole patch, with
         `<origin>` carrying the receiver's A-frame pose so the
         canonical-frame patch lands at the right place in A-frame.
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
    for cx, cy, cz, sx, sy, sz in other_part_boxes:
        add_box_collision_visual(link, (sx, sy, sz), (cx, cy, cz),
                                 material_name="bulk")

    pos, quat = receiver_pose
    rpy = R.from_quat(quat).as_euler("xyz")
    add_mesh_visual(link, hole_patch_mesh_rel,
                    origin_xyz=tuple(pos),
                    origin_rpy=tuple(float(v) for v in rpy),
                    material_name="hole")
    add_sdf_collision(link, hole_patch_mesh_rel,
                      resolution=SDF_RES,
                      origin_xyz=tuple(pos),
                      origin_rpy=tuple(float(v) for v in rpy))

    _write_pretty(robot, out_urdf)


# ─────────────────────────────────────────────────────────────────────────────
# Per-problem driver
# ─────────────────────────────────────────────────────────────────────────────

def _process_problem(transforms: dict, steps: List[str],
                     inserter_id: str, receiver_id: str) -> None:
    print(f"\n=== fabrica.{ASSEMBLY}.part_{inserter_id}_sdf_hybrid "
          f"(receiver part_{receiver_id}) ===")
    fixture_pids = steps[: steps.index(inserter_id)]
    if receiver_id not in fixture_pids:
        print(f"  WARN: receiver {receiver_id} not in fixture {fixture_pids}; skip")
        return

    # ── Inserter side ───────────────────────────────────────────────
    inserter_canonical, _ = _load_part_meshes(inserter_id)
    inserter_pose = _assembled_pose(transforms, inserter_id)
    inserter_A    = _transform_to_A(inserter_canonical, inserter_pose)
    inserter_bb_A = _bbox(inserter_A)

    # Localize the SDF tip to just the receiver's contact footprint:
    # project the receiver's A-frame bbox into the inserter's canonical
    # frame so we can clip the tip slab in the two non-insertion axes.
    recv_canonical_for_clip, _ = _load_part_meshes(receiver_id)
    recv_pose_for_clip = _assembled_pose(transforms, receiver_id)
    recv_A_for_clip = _transform_to_A(recv_canonical_for_clip, recv_pose_for_clip)
    recv_bb_A_for_clip = _bbox(recv_A_for_clip)
    receiver_bb_in_inserter_can = _a_frame_bbox_in_canonical(
        recv_bb_A_for_clip, inserter_pose,
    )

    # World insertion direction is (0, 0, -1) in the env's A-frame; project
    # it back into canonical so we can pick which axis end is the tip.
    R_can_to_A = R.from_quat(inserter_pose[1]).as_matrix()
    insertion_dir_can = R_can_to_A.T @ np.array([0.0, 0.0, -1.0])
    # Clip tip to receiver's projected footprint (with HOLE_PAD). For
    # inserters narrower than the receiver (pillars vs base hole), the
    # clip is a no-op — tip = full pillar cross-section. For inserters
    # wider than the receiver (caps vs pillar), this restricts the tip
    # to the peg region where insertion actually happens, instead of
    # uniformly slabbing the cap's whole bottom face.
    axis, side, tip_mesh, body_mesh = _slice_inserter_canonical(
        inserter_canonical, insertion_dir_can,
        receiver_bb_can=receiver_bb_in_inserter_can,
    )
    print(f"  inserter insertion axis = {'XYZ'[axis]} ({side}-end is tip, "
          f"clipped to receiver footprint), "
          f"tip {len(tip_mesh.vertices)}v / body {len(body_mesh.vertices)}v")

    inserter_dir = ASSEMBLY_DIR / inserter_id / "sdf_hybrid"
    inserter_dir.mkdir(parents=True, exist_ok=True)
    tip_obj  = inserter_dir / f"{inserter_id}_tip.obj"
    body_obj = inserter_dir / f"{inserter_id}_body.obj"
    tip_mesh.export(str(tip_obj))
    body_mesh.export(str(body_obj))

    # CoACD the body so the hulls live entirely in the body region —
    # reusing the part's existing CoACD output would double-cover the
    # tip region with mesh shapes and shadow the SDF cooked there.
    body_coacd_dir = inserter_dir / "coacd"
    run_coacd_python(body_obj, body_coacd_dir, **BODY_COACD_KW)
    body_hull_filenames = [
        f"coacd/{p.name}" for p in sorted(body_coacd_dir.glob("decomp_*.obj"))
    ]
    if not body_hull_filenames:
        print("  WARN: CoACD on body produced 0 hulls; falling back to body.obj as a single mesh.")
        body_hull_filenames = [body_obj.name]

    inserter_robot_name = f"{ASSEMBLY}_{inserter_id}_sdf_hybrid"
    inserter_urdf = inserter_dir / f"{inserter_robot_name}.urdf"
    _write_inserter_hybrid_urdf(
        out_urdf=inserter_urdf, robot_name=inserter_robot_name,
        tip_obj_rel=tip_obj.name, body_obj_rel=body_obj.name,
        body_hull_filenames=body_hull_filenames,
    )
    print(f"  inserter URDF → {inserter_urdf.relative_to(REPO_ROOT)}  "
          f"(tip SDF + {len(body_hull_filenames)} body hulls)")

    # ── Receiver side ───────────────────────────────────────────────
    recv_canonical, _ = _load_part_meshes(receiver_id)
    recv_pose = _assembled_pose(transforms, receiver_id)
    recv_A    = _transform_to_A(recv_canonical, recv_pose)
    recv_part_bb = _bbox(recv_A)

    # Slice receiver canonical mesh by the inserter's A-frame bbox
    # projected into the receiver's canonical frame (with 5 mm pad).
    x_lo, x_hi, y_lo, y_hi = _world_bbox_to_receiver_canonical_bbox(
        inserter_bb_A, recv_pose,
    )
    x_lo -= HOLE_PAD; x_hi += HOLE_PAD
    y_lo -= HOLE_PAD; y_hi += HOLE_PAD
    hole_patch = slice_mesh_by_xy_bbox(recv_canonical, x_lo, x_hi, y_lo, y_hi)
    if not hole_patch.is_watertight:
        hole_patch.fill_holes()
        hole_patch.merge_vertices()
    receiver_dir = ASSEMBLY_DIR / receiver_id / "sdf_hybrid"
    receiver_dir.mkdir(parents=True, exist_ok=True)
    # Filename includes inserter_id so two inserters sharing a receiver
    # (e.g. parts 2 and 3 both inserting into part 6) don't overwrite each
    # other's hole patches — each hole gets its own SDF asset.
    hole_patch_obj = receiver_dir / f"{receiver_id}_hole_patch_for_part_{inserter_id}.obj"
    hole_patch.export(str(hole_patch_obj))
    print(f"  receiver hole patch → {hole_patch_obj.relative_to(REPO_ROOT)}  "
          f"({len(hole_patch.vertices)}v {len(hole_patch.faces)}f, "
          f"watertight={hole_patch.is_watertight})")

    bulk_boxes = _decompose_around_hole(recv_part_bb, inserter_bb_A)
    other_boxes = []
    for pid in fixture_pids:
        if pid == receiver_id:
            continue
        cano, _ = _load_part_meshes(pid)
        a_mesh = _transform_to_A(cano, _assembled_pose(transforms, pid))
        other_boxes.append(_bbox_to_box_primitive(_bbox(a_mesh)))
    print(f"  bulk frame: {len(bulk_boxes)} boxes; other-fixture parts: {len(other_boxes)}")

    receptive_urdf = FIXTURES_DIR / f"part_{inserter_id}_sdf_hybrid.urdf"
    receptive_robot_name = f"fixture_{ASSEMBLY}_part_{inserter_id}_sdf_hybrid"
    hole_patch_mesh_rel = f"../{receiver_id}/sdf_hybrid/{hole_patch_obj.name}"
    _write_receptive_hybrid_urdf(
        out_urdf=receptive_urdf, robot_name=receptive_robot_name,
        bulk_boxes=bulk_boxes, other_part_boxes=other_boxes,
        hole_patch_mesh_rel=hole_patch_mesh_rel,
        receiver_pose=recv_pose,
    )
    print(f"  receptive URDF → {receptive_urdf.relative_to(REPO_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inserter", default="2",
                        help="Inserter part_id (default: 2). Repeat the script with "
                             "different values to regenerate other beam_2x problems.")
    args = parser.parse_args()

    transforms: dict = json.loads(
        (ASSEMBLY_DIR / "canonical_transforms.json").read_text()
    )
    order: dict = json.loads(
        (ASSEMBLY_DIR / "assembly_order.json").read_text()
    )
    inserts_into: Dict[str, str] = order["inserts_into"]
    steps: List[str] = order["steps"]

    if args.inserter not in inserts_into:
        raise SystemExit(
            f"Inserter {args.inserter!r} has no entry in inserts_into "
            f"({list(inserts_into)})."
        )
    receiver_id = inserts_into[args.inserter]
    _process_problem(transforms, steps, args.inserter, receiver_id)


if __name__ == "__main__":
    main()
