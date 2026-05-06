"""Step 2 of one_leg problem-setup pipeline.

For each part used by the one_leg task (square_table_top, square_table_leg4):
  * Apply universal y-up → z-up rotation R_x(+90°).
  * Center at bbox centroid.
  * Apply axis permutation P enforcing strict X > Y > Z (proper rotation).
  * Save canonical OBJ + ``canonical_meta.json`` (records
    ``R_storage_to_canonical = P @ R_x(+90°)``).
  * Run CoACD on the canonical mesh; emit per-part visual + CoACD URDFs.

Then build the per-problem receptive URDF for one_leg:
  * Pick the active corner pose from
    ``assembly.json`` (4 are equivalent — we use the first).
  * Decompose the table top into 4 axis-aligned bulk boxes that tile
    everything except a small column around the active hole.
  * Slice the active-hole column out of the canonical top mesh and CoACD it.
  * Emit ``insertion_fixtures/one_leg.urdf`` referencing the bulk boxes
    + the active-hole CoACD hulls.

Run:
    .venv/bin/python -m peg_in_hole_dynamic.furniture_bench.one_leg_problem_setup.step2_generate_assets
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from peg_in_hole_dynamic.fmb.run_coacd import run_coacd_python
from peg_in_hole_dynamic.sdf_hybrid_utils import (
    trimesh_to_manifold,
    manifold_to_trimesh,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_FB = REPO_ROOT / "assets" / "urdf" / "furniture_bench"
ASSEMBLY_JSON = ASSETS_FB / "square_table" / "assembly.json"

# Y-up → Z-up rotation as a 3x3 numpy array. SciPy convention agrees:
# R.from_euler("x", 90, degrees=True).as_matrix() == this.
R_YUP_TO_ZUP = R.from_euler("x", 90, degrees=True).as_matrix()

# Per-role canonical rotation. The strict X>Y>Z permutation we use for fmb
# pegs is *unstable* under tiny floating-point asymmetries in flat parts
# (the square_table_top has X=0.1625 but Z=0.162501 — argsort flips axes
# and the top ends up upside-down in canonical). We instead pick a fixed
# convention per role:
#   "receiver": R_storage_to_canonical = R_x(+90°). Source thin axis (Y, the
#       upstream "vertical assembly axis") lands on canonical Z, top
#       surface at canonical +Z. Loaded with URDF rot=identity, the
#       receiver sits flat with its top surface at world +Z.
#   "inserter": R_storage_to_canonical = R_y(+90°) ∘ R_x(+90°). Source long
#       axis (Y) → canonical X; source -Y end (tenon) → canonical -X.
# Every furniture_bench part in the upstream uses Y as its key axis (long
# for legs, thin for tops/seats), so this pair of conventions covers all
# the parts we'll register.
R_RECEIVER_STORAGE_TO_CANONICAL = R_YUP_TO_ZUP
R_INSERTER_STORAGE_TO_CANONICAL = R.from_euler("y", 90, degrees=True).as_matrix() @ R_YUP_TO_ZUP

# CoACD knobs — same defaults as fmb pegs / hole patches.
PART_COACD_KW  = dict(threshold=0.03, max_convex_hull=-1)
PATCH_COACD_KW = dict(threshold=0.05, max_convex_hull=12)

# Padding around the active hole (XY only) — width of the slice ring of
# slab material around the hole that gets CoACD'd. Matches fmb's
# HOLE_PADDING but bumped to 5 mm so the ring is thick enough for
# CoACD to fit hulls.
HOLE_PADDING_M = 0.005

# Clearance per side between the leg cross-section and the synthesized
# hole opening — gives the leg ~1 mm of total play (0.5 mm per side).
HOLE_CLEARANCE_M = 0.0005


def _generate_canonical_for_part(piece: str, part_name: str, role: str) -> dict:
    """Re-bake one part to canonical and run CoACD.

    role ∈ {"receiver", "inserter"} picks the canonical orientation —
    see ``R_RECEIVER_STORAGE_TO_CANONICAL`` /
    ``R_INSERTER_STORAGE_TO_CANONICAL`` at the top of this module.
    """
    print(f"\n=== Part: {piece}/{part_name} (role={role}) ===")
    part_dir = ASSETS_FB / piece / part_name
    src_obj = part_dir / f"{part_name}.obj"
    if not src_obj.is_file():
        raise SystemExit(f"missing source mesh {src_obj}")

    if role == "receiver":
        R_storage_to_canonical = R_RECEIVER_STORAGE_TO_CANONICAL
    elif role == "inserter":
        R_storage_to_canonical = R_INSERTER_STORAGE_TO_CANONICAL
    else:
        raise ValueError(f"unknown role: {role!r}")

    raw = trimesh.load(str(src_obj), force="mesh", process=False)

    canonical = raw.copy()
    canonical.vertices = (R_storage_to_canonical @ canonical.vertices.T).T
    # Center after rotation so canonical bbox centroid is at origin.
    canonical.vertices = canonical.vertices - (canonical.bounds[0] + canonical.bounds[1]) / 2

    canonical_obj = part_dir / f"{part_name}_canonical.obj"
    canonical.export(str(canonical_obj))
    e = canonical.bounds[1] - canonical.bounds[0]
    print(f"  canonical extent: ({e[0]*1000:.1f}, {e[1]*1000:.1f}, {e[2]*1000:.1f}) mm")
    print(f"  R_storage_to_canonical = {np.round(R_storage_to_canonical, 4).tolist()}")

    meta = {
        "role": role,
        "R_storage_to_canonical": R_storage_to_canonical.tolist(),
    }
    (part_dir / "canonical_meta.json").write_text(json.dumps(meta, indent=2))

    # Visual canonical URDF (always generated).
    _write_part_visual_urdf(
        part_dir / f"{part_name}_canonical.urdf",
        robot_name=part_name,
        mesh_rel=f"{part_name}_canonical.obj",
    )
    print(f"  visual canonical URDF → {(part_dir / f'{part_name}_canonical.urdf').relative_to(REPO_ROOT)}")

    # CoACD only for inserters. The upstream FurnitureBench receiver meshes
    # (e.g. square_table_top) are visual-only — disconnected apriltag quads
    # with no solid table body — so CoACD on them yields garbage. Receivers
    # use synthetic bulk-box decomposition in the receptive URDF instead.
    if role == "inserter":
        coacd_dir = part_dir / "coacd"
        run_coacd_python(canonical_obj, coacd_dir, **PART_COACD_KW)
        hull_filenames = [hp.name for hp in sorted(coacd_dir.glob("decomp_*.obj"))]
        _write_part_coacd_urdf(
            coacd_dir / f"{part_name}_coacd.urdf",
            robot_name=f"{part_name}_coacd",
            hull_filenames=hull_filenames,
        )
        print(f"  CoACD URDF            → {(coacd_dir / f'{part_name}_coacd.urdf').relative_to(REPO_ROOT)}  ({len(hull_filenames)} hulls)")
    return meta


def _write_part_visual_urdf(out_urdf: Path, robot_name: str, mesh_rel: str) -> None:
    out_urdf.write_text(
        f"""<?xml version="1.0"?>
<robot name="{robot_name}">
  <link name="{robot_name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>
    </collision>
    <inertial><density value="800.0"/></inertial>
  </link>
</robot>
"""
    )


def _write_part_coacd_urdf(out_urdf: Path, robot_name: str, hull_filenames: List[str]) -> None:
    visual_blocks = "\n".join(
        f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
        f'<geometry><mesh filename="{fn}" scale="1 1 1"/></geometry></visual>'
        for fn in hull_filenames
    )
    collision_blocks = "\n".join(
        f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
        f'<geometry><mesh filename="{fn}" scale="1 1 1"/></geometry></collision>'
        for fn in hull_filenames
    )
    out_urdf.write_text(
        f"""<?xml version="1.0"?>
<robot name="{robot_name}">
  <link name="{robot_name}">
{visual_blocks}
{collision_blocks}
    <inertial><density value="800.0"/></inertial>
  </link>
</robot>
"""
    )


# ─── Receptive URDF for one_leg ────────────────────────────────────────


def _decompose_plate_around_one_hole(
    plate_xmin: float, plate_xmax: float,
    plate_ymin: float, plate_ymax: float,
    plate_zmin: float, plate_zmax: float,
    hole_bbox_xy: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float, float, float]]:
    """Returns up to 4 axis-aligned (cx, cy, cz, sx, sy, sz) boxes that
    tile the plate except for the hole's column. Mirrors
    ``fmb...peg_board_problem_setup.step2_generate_assets._decompose_plate_around_one_hole``
    — see that docstring for the geometric layout.
    """
    cz = (plate_zmin + plate_zmax) / 2
    sz = plate_zmax - plate_zmin
    hx_lo, hx_hi, hy_lo, hy_hi = hole_bbox_xy
    out: List[Tuple[float, float, float, float, float, float]] = []
    if plate_ymax > hy_hi:
        out.append(((plate_xmin + plate_xmax) / 2, (hy_hi + plate_ymax) / 2, cz,
                    plate_xmax - plate_xmin, plate_ymax - hy_hi, sz))
    if hy_lo > plate_ymin:
        out.append(((plate_xmin + plate_xmax) / 2, (plate_ymin + hy_lo) / 2, cz,
                    plate_xmax - plate_xmin, hy_lo - plate_ymin, sz))
    if hx_lo > plate_xmin:
        out.append(((plate_xmin + hx_lo) / 2, (hy_lo + hy_hi) / 2, cz,
                    hx_lo - plate_xmin, hy_hi - hy_lo, sz))
    if plate_xmax > hx_hi:
        out.append(((hx_hi + plate_xmax) / 2, (hy_lo + hy_hi) / 2, cz,
                    plate_xmax - hx_hi, hy_hi - hy_lo, sz))
    return out


def _slice_around_hole(
    slab: trimesh.Trimesh,
    hole_xy: Tuple[float, float],
    hole_xy_size: Tuple[float, float],
    pad: float,
) -> trimesh.Trimesh:
    """Boolean-intersect a slab mesh with a vertical column around one
    hole xy. Mirrors ``fmb..._slice_hole_column``."""
    cx, cy = hole_xy
    dx, dy = hole_xy_size
    x_lo, x_hi = cx - dx / 2 - pad, cx + dx / 2 + pad
    y_lo, y_hi = cy - dy / 2 - pad, cy + dy / 2 + pad
    z_lo = float(slab.bounds[0, 2]) - 0.001
    z_hi = float(slab.bounds[1, 2]) + 0.001
    box = trimesh.creation.box(
        extents=(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo),
        transform=trimesh.transformations.translation_matrix([
            (x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2,
        ]),
    )
    a = trimesh_to_manifold(slab)
    b = trimesh_to_manifold(box)
    return manifold_to_trimesh(a ^ b)


def _build_synthetic_slab_with_holes(
    bbox: np.ndarray,
    hole_xy_centers: List[Tuple[float, float]],
    hole_xy_size: Tuple[float, float],
) -> trimesh.Trimesh:
    """Watertight slab matching the canonical bbox with rectangular
    holes punched through the Z axis at each (x, y) position.

    The upstream FurnitureBench visual mesh is non-watertight (~22000
    disconnected apriltag quads + a few open polygon shells), so
    ``manifold3d`` boolean slicing fails on it directly. We instead
    synthesize the slab from primitives — all we need from the source
    is its bounding box and the per-corner hole positions.
    """
    sx = float(bbox[1, 0] - bbox[0, 0])
    sy = float(bbox[1, 1] - bbox[0, 1])
    sz = float(bbox[1, 2] - bbox[0, 2])
    cx = float((bbox[0, 0] + bbox[1, 0]) / 2)
    cy = float((bbox[0, 1] + bbox[1, 1]) / 2)
    cz = float((bbox[0, 2] + bbox[1, 2]) / 2)

    slab = trimesh.creation.box(
        extents=(sx, sy, sz),
        transform=trimesh.transformations.translation_matrix([cx, cy, cz]),
    )
    z_thru = sz + 0.002    # epsilon so the hole punches all the way through
    hsx, hsy = hole_xy_size

    slab_m = trimesh_to_manifold(slab)
    for hx, hy in hole_xy_centers:
        hole_box = trimesh.creation.box(
            extents=(hsx, hsy, z_thru),
            transform=trimesh.transformations.translation_matrix([hx, hy, cz]),
        )
        slab_m = slab_m - trimesh_to_manifold(hole_box)
    return manifold_to_trimesh(slab_m)


def _write_one_leg_receptive_urdf(
    out_urdf: Path,
    bulk_boxes: List[Tuple[float, float, float, float, float, float]],
    synthetic_visual_rel: str,
    patch_hull_rels: List[str],
) -> None:
    """Receptive URDF for one_leg.

    Visual: the synthesized watertight slab (so all 4 holes are clearly
    visible). Collision: bulk-boxes around the active hole + CoACD hulls
    of the active-hole patch (mirrors the fmb peg_board pattern).
    """
    lines: List[str] = [
        f'    <visual name="slab_v"><origin xyz="0 0 0" rpy="0 0 0"/>'
        f'<geometry><mesh filename="{synthetic_visual_rel}" scale="1 1 1"/></geometry></visual>',
    ]
    for i, (cx, cy, cz, sx, sy, sz) in enumerate(bulk_boxes):
        lines.append(
            f'    <collision name="bulk_{i}_c"><origin xyz="{cx} {cy} {cz}" rpy="0 0 0"/>'
            f'<geometry><box size="{sx} {sy} {sz}"/></geometry></collision>'
        )
    for rel in patch_hull_rels:
        lines.append(
            f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
            f'<geometry><mesh filename="{rel}" scale="1 1 1"/></geometry></collision>'
        )
    body = "\n".join(lines)
    out_urdf.write_text(
        f"""<?xml version="1.0"?>
<robot name="one_leg_receptive">
  <link name="receptive">
{body}
  </link>
</robot>
"""
    )


def _build_one_leg_receptive(
    canonical_top: trimesh.Trimesh,
    all_hole_xy_centers: List[Tuple[float, float]],
    active_hole_xy: Tuple[float, float],
    hole_xy_size: Tuple[float, float],
) -> Tuple[Path, dict]:
    """Build the synthesized slab + active hole CoACD + bulk boxes."""
    out_dir = ASSETS_FB / "square_table" / "insertion_fixtures"
    patches_dir = ASSETS_FB / "square_table" / "hole_patches"
    if patches_dir.exists():
        shutil.rmtree(patches_dir)
    patches_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize a watertight slab matching the canonical top's bbox
    #    with a rectangular hole punched through at each corner.
    synthetic = _build_synthetic_slab_with_holes(
        canonical_top.bounds, all_hole_xy_centers, hole_xy_size,
    )
    synthetic_path = ASSETS_FB / "square_table" / "square_table_top" / "square_table_top_synthetic.obj"
    synthetic.export(str(synthetic_path))
    print(f"  synthesized slab: V={len(synthetic.vertices)} F={len(synthetic.faces)} watertight={synthetic.is_watertight}")
    print(f"  → {synthetic_path.relative_to(REPO_ROOT)}")

    # 2. Slice the active-hole column out of the synthesized slab → CoACD.
    patch = _slice_around_hole(
        synthetic, active_hole_xy, hole_xy_size, HOLE_PADDING_M,
    )
    patch_path = patches_dir / "one_leg.obj"
    patch.export(str(patch_path))
    patch_coacd_dir = patches_dir / "one_leg_coacd"
    run_coacd_python(patch_path, patch_coacd_dir, **PATCH_COACD_KW)
    patch_hull_rels = [
        f"../hole_patches/one_leg_coacd/{p.name}"
        for p in sorted(patch_coacd_dir.glob("decomp_*.obj"))
    ]
    print(f"  active patch: {len(patch_hull_rels)} CoACD hulls")

    # 3. Bulk-box decomposition around the active hole's slice region.
    bb = canonical_top.bounds
    hx_lo = active_hole_xy[0] - hole_xy_size[0] / 2 - HOLE_PADDING_M
    hx_hi = active_hole_xy[0] + hole_xy_size[0] / 2 + HOLE_PADDING_M
    hy_lo = active_hole_xy[1] - hole_xy_size[1] / 2 - HOLE_PADDING_M
    hy_hi = active_hole_xy[1] + hole_xy_size[1] / 2 + HOLE_PADDING_M
    bulk_boxes = _decompose_plate_around_one_hole(
        plate_xmin=float(bb[0, 0]), plate_xmax=float(bb[1, 0]),
        plate_ymin=float(bb[0, 1]), plate_ymax=float(bb[1, 1]),
        plate_zmin=float(bb[0, 2]), plate_zmax=float(bb[1, 2]),
        hole_bbox_xy=(hx_lo, hx_hi, hy_lo, hy_hi),
    )

    # 4. Receptive URDF: visual = synthesized slab, collision = bulk + patch hulls.
    out_urdf = out_dir / "one_leg.urdf"
    synthetic_visual_rel = "../square_table_top/square_table_top_synthetic.obj"
    _write_one_leg_receptive_urdf(out_urdf, bulk_boxes, synthetic_visual_rel, patch_hull_rels)
    print(f"  bulk frame: {len(bulk_boxes)} boxes around the active hole")
    print(f"  receptive URDF → {out_urdf.relative_to(REPO_ROOT)}")

    return out_urdf, {
        "hole_xy_canonical": list(active_hole_xy),
        "hole_xy_size_canonical": list(hole_xy_size),
    }


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    if not ASSEMBLY_JSON.is_file():
        raise SystemExit(f"missing {ASSEMBLY_JSON} — run step1 first.")
    data = json.loads(ASSEMBLY_JSON.read_text())

    # Determine parts used by the one_leg task.
    one_leg_pairs = data["tasks"]["one_leg"]["should_be_assembled"]
    parts_by_idx = {p["idx"]: p["name"] for p in data["parts"]}
    print(f"one_leg pairs: {one_leg_pairs}")

    # Each (parent, child) pair: parent is the receiver, child is the inserter.
    role_by_part = {}
    for parent_idx, child_idx in one_leg_pairs:
        role_by_part.setdefault(parts_by_idx[parent_idx], "receiver")
        role_by_part.setdefault(parts_by_idx[child_idx], "inserter")

    # Step 2a: per-part canonical assets.
    for part_name, role in sorted(role_by_part.items()):
        _generate_canonical_for_part("square_table", part_name, role)

    # Step 2b: one_leg receptive URDF.
    print("\n=== one_leg receptive URDF ===")
    parent_idx, child_idx = one_leg_pairs[0]
    parent_name = parts_by_idx[parent_idx]
    child_name  = parts_by_idx[child_idx]
    rel_key     = f"{parent_idx}->{child_idx}"
    rel_poses_yup = data["assembled_rel_poses_yup"][rel_key]

    # Pick the first of the (symmetric) corner poses.
    chosen = rel_poses_yup[0]
    pos_yup_obj = np.asarray(chosen["pos"], dtype=float)
    rpy_yup     = np.asarray(chosen["rpy_xyz"], dtype=float)

    # ─── Insertion-depth fix ──────────────────────────────────────────
    # Upstream's pos is the child's *OBJ origin* in the parent's source
    # frame. Our URDF anchors the child at its *canonical centroid*
    # (after centering). The leg's source bbox centroid sits at
    # y = -0.0125 — i.e., the OBJ origin is 12.5 mm above the centroid.
    # Forwarding upstream's pos verbatim lands the leg 12.5 mm too high
    # (not deep enough into the hole). Bake the offset in here.
    leg_src = trimesh.load(
        str(ASSETS_FB / "square_table" / child_name / f"{child_name}.obj"),
        force="mesh", process=False,
    )
    leg_centroid_in_src = (leg_src.bounds[0] + leg_src.bounds[1]) / 2
    R_yup_rel = R.from_euler("xyz", rpy_yup)
    pos_yup_centroid = pos_yup_obj + R_yup_rel.apply(leg_centroid_in_src)
    pos_canonical = R_YUP_TO_ZUP @ pos_yup_centroid
    print(f"  chosen corner pose (yup obj-origin): pos={pos_yup_obj.tolist()}, rpy={rpy_yup.tolist()}")
    print(f"  leg src centroid offset (yup):       {leg_centroid_in_src.tolist()}")
    print(f"  pos_yup centroid                   = {pos_yup_centroid.tolist()}")
    print(f"  → canonical (z-up) pos             = {np.round(pos_canonical, 5).tolist()}")

    # Active hole xy = leg's centroid xy in receiver canonical.
    active_hole_xy = (float(pos_canonical[0]), float(pos_canonical[1]))

    # All 4 holes' xy (one per symmetric corner pose). We synthesize a
    # slab with all 4 holes so the visualizer shows the full table top.
    all_hole_xy_centers: List[Tuple[float, float]] = []
    for p in rel_poses_yup:
        p_yup_centroid = np.asarray(p["pos"], dtype=float) \
            + R.from_euler("xyz", p["rpy_xyz"]).apply(leg_centroid_in_src)
        c_can = R_YUP_TO_ZUP @ p_yup_centroid
        all_hole_xy_centers.append((float(c_can[0]), float(c_can[1])))

    # Hole xy size = leg's narrow tenon-shaft cross-section + clearance.
    # The leg has a 30×30 body but a narrower shaft passing through the
    # hole — sample at y = -0.030 in source (well into the shaft).
    sample_y = -0.030
    pts_at_y = leg_src.vertices[np.abs(leg_src.vertices[:, 1] - sample_y) < 0.001]
    if len(pts_at_y) >= 4:
        shaft_x = float(pts_at_y[:, 0].max() - pts_at_y[:, 0].min())
        shaft_z = float(pts_at_y[:, 2].max() - pts_at_y[:, 2].min())
    else:
        shaft_x = float(leg_src.bounds[1, 0] - leg_src.bounds[0, 0])
        shaft_z = float(leg_src.bounds[1, 2] - leg_src.bounds[0, 2])
    hole_xy_size = (shaft_x + 2 * HOLE_CLEARANCE_M, shaft_z + 2 * HOLE_CLEARANCE_M)
    print(f"  shaft xz at y={sample_y}: ({shaft_x*1000:.1f}, {shaft_z*1000:.1f}) mm")
    print(f"  synthesized hole size (with clearance): "
          f"({hole_xy_size[0]*1000:.1f}, {hole_xy_size[1]*1000:.1f}) mm")
    print(f"  active hole xy center: {active_hole_xy}")

    canonical_top = trimesh.load(
        str(ASSETS_FB / "square_table" / parent_name / f"{parent_name}_canonical.obj"),
        force="mesh", process=False,
    )
    out_urdf, info = _build_one_leg_receptive(
        canonical_top, all_hole_xy_centers, active_hole_xy, hole_xy_size,
    )

    # Persist the corrected pose so problems.py uses it directly.
    info["active_corner_pos_yup_obj"]  = chosen["pos"]
    info["active_corner_rpy_yup"]      = chosen["rpy_xyz"]
    info["leg_centroid_in_src"]        = leg_centroid_in_src.tolist()
    info["pos_canonical_centroid"]     = pos_canonical.tolist()
    info["parent_part"] = parent_name
    info["child_part"]  = child_name
    info["receptive_urdf"] = out_urdf.relative_to(REPO_ROOT / "assets").as_posix()
    (ASSETS_FB / "square_table" / "one_leg_setup.json").write_text(json.dumps(info, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
