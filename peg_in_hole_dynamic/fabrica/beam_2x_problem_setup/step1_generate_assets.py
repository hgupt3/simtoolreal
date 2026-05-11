"""Step 1 of fabrica.beam_2x problem-setup pipeline.

Builds per-problem receptive URDFs that mirror the peg_board_1 approach:
the *active hole* (= the receiver's geometry near the inserter's path) is
represented with the receiver part's existing fine CoACD hulls, while the
rest of the receiver and every non-receiver fixture part is approximated
by axis-aligned `<box>` primitives.

For each ``inserter -> receiver`` pair from
``assembly_order.json["inserts_into"]``:
    * Receiver hulls (already CoACD'd in <receiver>/coacd/) are classified
      by A-frame bbox overlap with the inserter's A-frame bbox + padding:
        - "near" hulls are kept verbatim (referenced as <mesh>) — these are
          the actual hole walls that the inserter contacts during insertion.
        - "far" hulls are dropped from the mesh set; their bulk volume is
          approximated by a 4-box "frame" around the inserter's footprint
          (cropped to the receiver's overall A-frame bbox).
    * Non-receiver fixture parts each get one axis-aligned bbox box.
    * The resulting per-problem URDF is written to
      ``insertion_fixtures/part_<inserter>.urdf``, replacing the old
      "every-part-fully-CoACD'd" URDF that fabrica/problems.py would have
      written previously.

Run:
    .venv/bin/python -m peg_in_hole_dynamic.fabrica.beam_2x_problem_setup.step1_generate_assets
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from xml.dom import minidom

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

REPO_ROOT       = Path(__file__).resolve().parents[3]
ASSEMBLY        = "beam_2x"
ASSEMBLY_DIR    = REPO_ROOT / "assets" / "urdf" / "fabrica" / ASSEMBLY
FIXTURES_DIR    = ASSEMBLY_DIR / "insertion_fixtures"

# Padding around the inserter bbox before testing receiver-hull proximity.
NEAR_PADDING = 0.005   # m


# ─────────────────────────────────────────────────────────────────────────────
# A-frame helpers (same convention as peg_in_hole_dynamic/fabrica/problems.py)
# ─────────────────────────────────────────────────────────────────────────────

def _assembled_pose(transforms: dict, pid: str
                    ) -> Tuple[Tuple[float, float, float],
                               Tuple[float, float, float, float]]:
    """Return ``(xyz, quat_xyzw)`` placing the canonical mesh at its
    assembled-frame pose: translation = original_centroid, rotation =
    inverse(q_a→c)."""
    cx, cy, cz = transforms[pid]["original_centroid"]
    qw, qx, qy, qz = transforms[pid]["assembled_to_canonical_wxyz"]
    rxyzw = R.from_quat([qx, qy, qz, qw]).inv().as_quat()
    return (float(cx), float(cy), float(cz)), (
        float(rxyzw[0]), float(rxyzw[1]), float(rxyzw[2]), float(rxyzw[3]),
    )


def _transform_to_A(mesh: trimesh.Trimesh,
                    pose: Tuple[Tuple[float, float, float],
                                Tuple[float, float, float, float]]
                    ) -> trimesh.Trimesh:
    pos, quat = pose
    Rmat = R.from_quat(quat).as_matrix()
    out = mesh.copy()
    out.vertices = out.vertices @ Rmat.T + np.asarray(pos)
    return out


def _bbox(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    return mesh.bounds[0].copy(), mesh.bounds[1].copy()


def _bbox_overlaps(bb1, bb2, padding: float) -> bool:
    a_min, a_max = bb1
    b_min, b_max = bb2
    return all(
        a_min[k] - padding <= b_max[k] and a_max[k] + padding >= b_min[k]
        for k in range(3)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bulk decomposition (4-box frame around the hole region)
# ─────────────────────────────────────────────────────────────────────────────

def _decompose_around_hole(part_bbox: Tuple[np.ndarray, np.ndarray],
                           hole_bbox: Tuple[np.ndarray, np.ndarray]
                           ) -> List[Tuple[float, float, float, float, float, float]]:
    """Return up to 6 axis-aligned boxes (cx, cy, cz, sx, sy, sz) that tile
    the receiver part's A-frame bbox while skipping the 3D hole region.

    The hole_bbox is the inserter's A-frame bbox (clipped to the part's bbox).
    Outside the hole's Z slab we emit full-XY "below" and "above" boxes; within
    the slab we emit up to 4 XY strips around the hole's XY footprint. This
    keeps the previous axial-insertion behavior (no below/above boxes when the
    inserter's Z spans the full receiver Z) while also covering lateral
    insertions where the inserter touches only a Z slab of the receiver.

    Boxes with non-positive extent are dropped.
    """
    p_min, p_max = part_bbox
    h_min, h_max = hole_bbox
    # Clip the hole bbox to the part bbox so the frame stays inside the part.
    hx_lo = max(h_min[0], p_min[0])
    hx_hi = min(h_max[0], p_max[0])
    hy_lo = max(h_min[1], p_min[1])
    hy_hi = min(h_max[1], p_max[1])
    hz_lo = max(h_min[2], p_min[2])
    hz_hi = min(h_max[2], p_max[2])

    sx_full = p_max[0] - p_min[0]
    sy_full = p_max[1] - p_min[1]
    cx_mid  = (p_min[0] + p_max[0]) / 2
    cy_mid  = (p_min[1] + p_max[1]) / 2

    out: List[Tuple[float, float, float, float, float, float]] = []
    # Below the hole in z: full XY × (p_min[2], hz_lo)
    if hz_lo > p_min[2] + 1e-6:
        sz = hz_lo - p_min[2]
        cz = (p_min[2] + hz_lo) / 2
        out.append((cx_mid, cy_mid, cz, sx_full, sy_full, sz))
    # Above the hole in z: full XY × (hz_hi, p_max[2])
    if p_max[2] > hz_hi + 1e-6:
        sz = p_max[2] - hz_hi
        cz = (hz_hi + p_max[2]) / 2
        out.append((cx_mid, cy_mid, cz, sx_full, sy_full, sz))

    # Within the hole z range, tile XY around the hole footprint.
    if hz_hi > hz_lo + 1e-6:
        cz = (hz_lo + hz_hi) / 2
        sz = hz_hi - hz_lo
        # Top strip: y > hy_hi
        if p_max[1] > hy_hi + 1e-6:
            sy = p_max[1] - hy_hi
            out.append((cx_mid, (hy_hi + p_max[1]) / 2, cz, sx_full, sy, sz))
        # Bottom strip: y < hy_lo
        if hy_lo > p_min[1] + 1e-6:
            sy = hy_lo - p_min[1]
            out.append((cx_mid, (p_min[1] + hy_lo) / 2, cz, sx_full, sy, sz))
        # Left strip: x < hx_lo, y in (hy_lo, hy_hi)
        if hx_lo > p_min[0] + 1e-6:
            sx = hx_lo - p_min[0]
            sy = max(0.0, hy_hi - hy_lo)
            if sy > 1e-6:
                out.append(((p_min[0] + hx_lo) / 2, (hy_lo + hy_hi) / 2, cz, sx, sy, sz))
        # Right strip: x > hx_hi, y in (hy_lo, hy_hi)
        if p_max[0] > hx_hi + 1e-6:
            sx = p_max[0] - hx_hi
            sy = max(0.0, hy_hi - hy_lo)
            if sy > 1e-6:
                out.append(((hx_hi + p_max[0]) / 2, (hy_lo + hy_hi) / 2, cz, sx, sy, sz))
    return out


def _bbox_to_box_primitive(bb: Tuple[np.ndarray, np.ndarray]
                           ) -> Tuple[float, float, float, float, float, float]:
    p_min, p_max = bb
    return (
        float((p_min[0] + p_max[0]) / 2),
        float((p_min[1] + p_max[1]) / 2),
        float((p_min[2] + p_max[2]) / 2),
        float(p_max[0] - p_min[0]),
        float(p_max[1] - p_min[1]),
        float(p_max[2] - p_min[2]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# URDF writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_urdf(out_path: Path, robot_name: str,
                bulk_boxes: List[Tuple[float, float, float, float, float, float]],
                near_hulls: List[Tuple[Path, Tuple[float, float, float],
                                       Tuple[float, float, float, float]]]) -> None:
    """Write a per-problem fabrica.beam_2x receptive URDF.

    `near_hulls` is a list of (mesh_abs_path, xyz, quat_xyzw) — for each
    near-hole receiver hull, we emit a <visual>+<collision> with the
    receiver's A-frame pose baked into the visual <origin> (the mesh file
    referenced is the canonical-frame CoACD hull, unchanged).
    """
    robot = ET.Element("robot", attrib={"name": robot_name})
    link  = ET.SubElement(robot, "link", attrib={"name": "plate"})

    bulk_rgba    = (0.78, 0.55, 0.30, 1.0)
    near_rgba    = (0.30, 0.65, 0.85, 1.0)

    def _add_material(name, rgba):
        m = ET.SubElement(robot, "material", attrib={"name": name})
        ET.SubElement(m, "color", attrib={
            "rgba": f"{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}",
        })

    def _add_origin(parent, xyz, rpy):
        ET.SubElement(parent, "origin", attrib={
            "xyz": f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
            "rpy": f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}",
        })

    _add_material("bulk", bulk_rgba)
    _add_material("near", near_rgba)

    # Bulk boxes (A-frame coords)
    for cx, cy, cz, sx, sy, sz in bulk_boxes:
        for tag in ("visual", "collision"):
            elem = ET.SubElement(link, tag)
            _add_origin(elem, (cx, cy, cz), (0.0, 0.0, 0.0))
            geom = ET.SubElement(elem, "geometry")
            ET.SubElement(geom, "box", attrib={
                "size": f"{sx:.6f} {sy:.6f} {sz:.6f}",
            })
            if tag == "visual":
                ET.SubElement(elem, "material", attrib={"name": "bulk"})

    # Near-hole hulls — mesh stays canonical, visual origin carries the
    # receiver's A-frame pose so the canonical hull lands at the right place.
    for mesh_abs, xyz, quat_xyzw in near_hulls:
        rpy = R.from_quat(quat_xyzw).as_euler("xyz")
        mesh_rel = Path("..") / mesh_abs.relative_to(ASSEMBLY_DIR)
        for tag in ("visual", "collision"):
            elem = ET.SubElement(link, tag)
            _add_origin(elem, xyz, tuple(float(x) for x in rpy))
            geom = ET.SubElement(elem, "geometry")
            ET.SubElement(geom, "mesh", attrib={
                "filename": str(mesh_rel).replace("\\", "/"),
                "scale": "1 1 1",
            })
            if tag == "visual":
                ET.SubElement(elem, "material", attrib={"name": "near"})

    pretty = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  ", encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pretty)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_part_meshes(pid: str) -> Tuple[trimesh.Trimesh, List[Tuple[Path, trimesh.Trimesh]]]:
    """Return (canonical_mesh, [(coacd_hull_abs_path, canonical_hull_mesh), ...])."""
    canonical_obj = ASSEMBLY_DIR / pid / f"{pid}_canonical.obj"
    if not canonical_obj.is_file():
        raise FileNotFoundError(f"Missing canonical mesh for {pid}: {canonical_obj}")
    canonical = trimesh.load_mesh(str(canonical_obj), process=False)

    coacd_dir = ASSEMBLY_DIR / pid / "coacd"
    decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
    hulls = [(p, trimesh.load_mesh(str(p), process=False)) for p in decomp_files]
    return canonical, hulls


def _process_problem(transforms: dict, steps: List[str], inserter_id: str,
                     receiver_id: str) -> None:
    print(f"\n=== fabrica.{ASSEMBLY}.part_{inserter_id} (receiver part_{receiver_id}) ===")
    fixture_pids = steps[: steps.index(inserter_id)]
    print(f"  fixture parts: {fixture_pids}")
    if receiver_id not in fixture_pids:
        print(f"  WARN: receiver {receiver_id} not in fixture {fixture_pids}; skipping")
        return

    # Inserter mesh in A-frame
    inserter_canonical, _ = _load_part_meshes(inserter_id)
    inserter_pose = _assembled_pose(transforms, inserter_id)
    inserter_A = _transform_to_A(inserter_canonical, inserter_pose)
    inserter_bb = _bbox(inserter_A)
    print(f"  inserter A-frame bbox: "
          f"x[{inserter_bb[0][0]:+.3f},{inserter_bb[1][0]:+.3f}] "
          f"y[{inserter_bb[0][1]:+.3f},{inserter_bb[1][1]:+.3f}] "
          f"z[{inserter_bb[0][2]:+.3f},{inserter_bb[1][2]:+.3f}]")

    # Receiver classification
    recv_canonical, recv_hulls = _load_part_meshes(receiver_id)
    recv_pose = _assembled_pose(transforms, receiver_id)
    recv_A = _transform_to_A(recv_canonical, recv_pose)
    recv_part_bb = _bbox(recv_A)

    near_hulls = []   # (abs path to canonical hull, A-frame xyz, A-frame quat)
    n_near = n_far = 0
    for hull_path, hull_canonical in recv_hulls:
        hull_A = _transform_to_A(hull_canonical, recv_pose)
        hull_bb = _bbox(hull_A)
        if _bbox_overlaps(hull_bb, inserter_bb, NEAR_PADDING):
            near_hulls.append((hull_path, recv_pose[0], recv_pose[1]))
            n_near += 1
        else:
            n_far += 1

    print(f"  receiver part_{receiver_id}: {n_near} near hulls (kept), {n_far} far hulls (drop)")

    # Bulk frame around the hole (in receiver's A-frame bbox)
    bulk_boxes = _decompose_around_hole(recv_part_bb, inserter_bb)
    print(f"  bulk frame boxes around hole: {len(bulk_boxes)}")

    # Non-receiver fixture parts → single A-frame bbox each
    other_boxes = []
    for pid in fixture_pids:
        if pid == receiver_id:
            continue
        part_canonical, _ = _load_part_meshes(pid)
        part_A = _transform_to_A(part_canonical, _assembled_pose(transforms, pid))
        other_boxes.append(_bbox_to_box_primitive(_bbox(part_A)))
    print(f"  other-fixture-part boxes: {len(other_boxes)}")

    # Write the URDF (overwrites the old version at the same path)
    bulk_boxes_all = bulk_boxes + other_boxes
    out_urdf = FIXTURES_DIR / f"part_{inserter_id}.urdf"
    robot_name = f"fixture_{ASSEMBLY}_part_{inserter_id}"
    _write_urdf(out_urdf, robot_name, bulk_boxes_all, near_hulls)
    print(f"  → {out_urdf.relative_to(REPO_ROOT)}  "
          f"(bulk {len(bulk_boxes_all)} boxes, near {len(near_hulls)} hulls)")


def main() -> None:
    transforms = json.loads((ASSEMBLY_DIR / "canonical_transforms.json").read_text())
    order      = json.loads((ASSEMBLY_DIR / "assembly_order.json").read_text())
    inserts_into: Dict[str, str] = order["inserts_into"]
    steps: List[str] = order["steps"]

    for inserter_id, receiver_id in inserts_into.items():
        if inserter_id not in steps or receiver_id not in steps:
            continue
        _process_problem(transforms, steps, inserter_id, receiver_id)

    print("\nAll fabrica.beam_2x receptive URDFs regenerated.")


if __name__ == "__main__":
    main()
