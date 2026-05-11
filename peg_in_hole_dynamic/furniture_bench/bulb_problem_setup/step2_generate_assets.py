"""Step 2 of the bulb-into-base problem-setup pipeline.

Bakes ``lamp_base`` (role=receiver) and ``lamp_bulb`` (role=inserter) into
canonical frames and runs CoACD on the bulb. Computes the active screw-hole
center in the base's canonical XY from the upstream y-up assembled pose
(with the same OBJ-origin → bbox-centroid correction one_leg applies),
samples the bulb's thread cross-section to size the hole, and writes
``assets/urdf/furniture_bench/lamp/bulb_screw_setup.json`` for step3 +
problems.py to consume.

Unlike the one_leg pipeline this script does **not** synthesize a
through-hole slab or emit an old-style receptive URDF — the lamp_base
mesh is not a flat plate, so the visual approximation isn't useful. The
``furniture_bench.bulb_screw`` problems are registered only in the
sdf_hybrid form (via step3).

Run:
    .venv/bin/python -m peg_in_hole_dynamic.furniture_bench.bulb_problem_setup.step2_generate_assets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from peg_in_hole_dynamic.fmb.run_coacd import run_coacd_python

# We reuse the one_leg conventions for canonical rotations so all
# furniture_bench parts canonicalize the same way.
from peg_in_hole_dynamic.furniture_bench.one_leg_problem_setup.step2_generate_assets import (
    R_INSERTER_STORAGE_TO_CANONICAL,
    R_YUP_TO_ZUP,
    PART_COACD_KW,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_FB = REPO_ROOT / "assets" / "urdf" / "furniture_bench"
ASSEMBLY_JSON = ASSETS_FB / "lamp" / "assembly.json"
SETUP_JSON = ASSETS_FB / "lamp" / "bulb_screw_setup.json"

# Upstream lamp is a *pendant* assembly: the lamp_base attaches to the
# ceiling and the bulb hangs below it. For sim we want the base sitting on
# the table with the socket facing UP and the bulb approaching from above.
# We achieve that by flipping the receiver upside-down in canonical:
# instead of one_leg's R_x(+90°), use R_x(-90°). This sends yup -Y (the
# upstream socket side) to canonical +Z, so the socket sits on top of the
# canonical receiver and the bulb's relative pose lands above z=0.
R_RECEIVER_STORAGE_TO_CANONICAL = R.from_euler("x", -90, degrees=True).as_matrix()

# Per-side clearance between bulb-shaft cross-section and the synthesized
# hole opening. Same value one_leg uses.
HOLE_CLEARANCE_M = 0.0005


# ─────────────────────────────────────────────────────────────────────────────
# Canonical asset generation (same shape as one_leg.step2)
# ─────────────────────────────────────────────────────────────────────────────

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


def _generate_canonical_for_part(part_name: str, role: str) -> dict:
    print(f"\n=== Part: lamp/{part_name} (role={role}) ===")
    part_dir = ASSETS_FB / "lamp" / part_name
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
    canonical.vertices = canonical.vertices - (canonical.bounds[0] + canonical.bounds[1]) / 2

    canonical_obj = part_dir / f"{part_name}_canonical.obj"
    canonical.export(str(canonical_obj))
    e = canonical.bounds[1] - canonical.bounds[0]
    print(f"  canonical extent: ({e[0]*1000:.1f}, {e[1]*1000:.1f}, {e[2]*1000:.1f}) mm")

    meta = {
        "role": role,
        "R_storage_to_canonical": R_storage_to_canonical.tolist(),
    }
    (part_dir / "canonical_meta.json").write_text(json.dumps(meta, indent=2))

    _write_part_visual_urdf(
        part_dir / f"{part_name}_canonical.urdf",
        robot_name=part_name,
        mesh_rel=f"{part_name}_canonical.obj",
    )
    print(f"  visual canonical URDF → {(part_dir / f'{part_name}_canonical.urdf').relative_to(REPO_ROOT)}")

    if role == "inserter":
        coacd_dir = part_dir / "coacd"
        run_coacd_python(canonical_obj, coacd_dir, **PART_COACD_KW)
        hull_filenames = [hp.name for hp in sorted(coacd_dir.glob("decomp_*.obj"))]
        _write_part_coacd_urdf(
            coacd_dir / f"{part_name}_coacd.urdf",
            robot_name=f"{part_name}_coacd",
            hull_filenames=hull_filenames,
        )
        print(f"  CoACD URDF → {(coacd_dir / f'{part_name}_coacd.urdf').relative_to(REPO_ROOT)} "
              f"({len(hull_filenames)} hulls)")

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Hole xy + size detection
# ─────────────────────────────────────────────────────────────────────────────

def _bulb_thread_cross_section(bulb_src: trimesh.Trimesh) -> tuple[float, float, float]:
    """The bulb's threaded screw end is the narrow lower-Y end in storage
    (verified by sampling cross sections — the wide bulb-glass lives in
    +Y). Return ``(shaft_x, shaft_z, sample_y)`` measured from a thin slab
    at ``y = y_min + 12 mm`` (well inside the thread region, past any
    rounded tip)."""
    bb = bulb_src.bounds
    sample_y = float(bb[0, 1] + 0.012)
    mask = np.abs(bulb_src.vertices[:, 1] - sample_y) < 0.001
    if int(mask.sum()) < 4:
        sample_y = float(bb[0, 1] + 0.020)
        mask = np.abs(bulb_src.vertices[:, 1] - sample_y) < 0.001
    if int(mask.sum()) < 4:
        # fall back to full bbox; never happens in practice for these meshes.
        return float(bb[1, 0] - bb[0, 0]), float(bb[1, 2] - bb[0, 2]), sample_y
    pts = bulb_src.vertices[mask]
    return float(pts[:, 0].max() - pts[:, 0].min()), float(pts[:, 2].max() - pts[:, 2].min()), sample_y


def main() -> None:
    if not ASSEMBLY_JSON.is_file():
        raise SystemExit(f"missing {ASSEMBLY_JSON} — run step1 first.")
    data = json.loads(ASSEMBLY_JSON.read_text())

    bulb_pairs = data["tasks"]["bulb_screw"]["should_be_assembled"]
    parts_by_idx = {p["idx"]: p["name"] for p in data["parts"]}
    print(f"bulb_screw pairs: {bulb_pairs}")

    role_by_part = {}
    for parent_idx, child_idx in bulb_pairs:
        role_by_part.setdefault(parts_by_idx[parent_idx], "receiver")
        role_by_part.setdefault(parts_by_idx[child_idx], "inserter")

    for part_name, role in sorted(role_by_part.items()):
        _generate_canonical_for_part(part_name, role)

    print("\n=== bulb_screw setup metadata ===")
    parent_idx, child_idx = bulb_pairs[0]
    parent_name = parts_by_idx[parent_idx]
    child_name  = parts_by_idx[child_idx]
    rel_key = f"{parent_idx}->{child_idx}"
    rel_poses_yup = data["assembled_rel_poses_yup"][rel_key]

    chosen = rel_poses_yup[0]
    pos_yup_obj = np.asarray(chosen["pos"], dtype=float)
    rpy_yup     = np.asarray(chosen["rpy_xyz"], dtype=float)

    # Same OBJ-origin → bbox-centroid fix one_leg applies. Without this
    # the inserter's canonical-centroid-anchored URDF lands offset from
    # what the upstream pose intended.
    bulb_src = trimesh.load(
        str(ASSETS_FB / "lamp" / child_name / f"{child_name}.obj"),
        force="mesh", process=False,
    )
    bulb_centroid_in_src = (bulb_src.bounds[0] + bulb_src.bounds[1]) / 2
    R_yup_rel = R.from_euler("xyz", rpy_yup)
    pos_yup_centroid = pos_yup_obj + R_yup_rel.apply(bulb_centroid_in_src)
    # Apply the receiver's canonical rotation (this is the LAMP-specific
    # R_x(-90°), not the one_leg R_x(+90°)).
    pos_canonical_uncentered = R_RECEIVER_STORAGE_TO_CANONICAL @ pos_yup_centroid

    # The receiver mesh gets centered (its bbox centroid moved to origin)
    # by _generate_canonical_for_part, so the bulb's relative pose must
    # also be shifted into that centered receiver frame. one_leg gets away
    # without this correction because square_table_top is symmetric in yup
    # Y; lamp_base is not.
    base_src = trimesh.load(
        str(ASSETS_FB / "lamp" / parent_name / f"{parent_name}.obj"),
        force="mesh", process=False,
    )
    base_centroid_yup = (base_src.bounds[0] + base_src.bounds[1]) / 2
    base_centroid_canonical = R_RECEIVER_STORAGE_TO_CANONICAL @ base_centroid_yup
    pos_canonical = pos_canonical_uncentered - base_centroid_canonical
    print(f"  chosen pose (yup obj-origin): pos={pos_yup_obj.tolist()}, rpy={rpy_yup.tolist()}")
    print(f"  bulb src centroid offset (yup): {bulb_centroid_in_src.tolist()}")
    print(f"  pos_yup centroid              : {pos_yup_centroid.tolist()}")
    print(f"  base centroid (yup→canonical): {np.round(base_centroid_canonical, 5).tolist()}")
    print(f"  pos canonical (uncentered)   : {np.round(pos_canonical_uncentered, 5).tolist()}")
    print(f"  → canonical (z-up, centered) : {np.round(pos_canonical, 5).tolist()}")

    active_hole_xy = (float(pos_canonical[0]), float(pos_canonical[1]))

    shaft_x, shaft_z, sample_y = _bulb_thread_cross_section(bulb_src)
    hole_xy_size = (shaft_x + 2 * HOLE_CLEARANCE_M, shaft_z + 2 * HOLE_CLEARANCE_M)
    print(f"  bulb shaft cross-section at y={sample_y:+.4f}: "
          f"({shaft_x*1000:.1f}, {shaft_z*1000:.1f}) mm")
    print(f"  synthesized hole size (with clearance): "
          f"({hole_xy_size[0]*1000:.1f}, {hole_xy_size[1]*1000:.1f}) mm")
    print(f"  active hole xy center (base canonical): {active_hole_xy}")

    info = {
        "hole_xy_canonical": list(active_hole_xy),
        "hole_xy_size_canonical": list(hole_xy_size),
        "active_corner_pos_yup_obj": chosen["pos"],
        "active_corner_rpy_yup": chosen["rpy_xyz"],
        "bulb_centroid_in_src": bulb_centroid_in_src.tolist(),
        "pos_canonical_centroid": pos_canonical.tolist(),
        "parent_part": parent_name,
        "child_part": child_name,
    }
    SETUP_JSON.parent.mkdir(parents=True, exist_ok=True)
    SETUP_JSON.write_text(json.dumps(info, indent=2))
    print(f"  setup → {SETUP_JSON.relative_to(REPO_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
