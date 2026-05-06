"""Step 2 of peg_board_1 problem-setup pipeline.

Generates the receptive URDF (peg_board_1) and the inserter assets (long pegs)
from the JSON saved by step 1.

Receptive (peg_board_1):
  * Slice the original board mesh into 9 hole columns (one per hole, 5 mm
    XY padding around each hole bbox).
  * Run CoACD on each hole column → multi-hull collision for the hole walls.
  * Tile the rest of the plate with axis-aligned `<box>` primitives — no
    reference to the original board mesh for the bulk. This keeps the URDF
    cheap *and* makes the receptive's XY footprint a clean rectangle.

Inserter (long pegs in the JSON):
  * For each peg used by the JSON, build a canonical mesh (XYZ-centered at
    bbox centroid) and CoACD it.
  * Write a single-link visual URDF (`<peg>.urdf`) and a multi-link CoACD
    URDF (`<peg>_coacd.urdf`).

Run:
    .venv/bin/python -m peg_in_hole_dynamic.fmb.peg_board_problem_setup.step2_generate_assets
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Tuple
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from peg_in_hole_dynamic.fmb.run_coacd import run_coacd_python

REPO_ROOT  = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"
JSON_PATH  = Path(__file__).with_name("peg_board_1_assemblies.json")

BOARD_NAME = "peg_board_1"
BOARD_OBJ  = ASSETS_DIR / "boards" / BOARD_NAME / f"{BOARD_NAME}.obj"

HOLE_PADDING = 0.005   # m — XY pad around each hole's bbox before slicing
PEG_DIR      = ASSETS_DIR / "pegs"
# Hole-patch CoACD: looser threshold + a hard hull cap to avoid pathological
# shapes (hex/square holes) being over-decomposed into hundreds of pieces.
PATCH_COACD_KW = dict(threshold=0.05, max_convex_hull=20, seed=0)
# Peg CoACD: tight threshold preserves the cross-section shape (star/asterisk
# pegs need fine decomposition for proper insertion-clearance contacts).
PEG_COACD_KW   = dict(threshold=0.03, max_convex_hull=-1, seed=0)


def _board_to_A_frame(board: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate the board so its bottom face sits at z=0 and XY is centered
    at origin (A-frame). Mirrors the convention in step1."""
    T = np.array([
        -(board.bounds[0, 0] + board.bounds[1, 0]) / 2,
        -(board.bounds[0, 1] + board.bounds[1, 1]) / 2,
        -board.bounds[0, 2],
    ])
    out = board.copy()
    out.vertices = out.vertices + T
    return out


def _trimesh_to_manifold(m: trimesh.Trimesh):
    """Convert a trimesh.Trimesh into a manifold3d.Manifold."""
    import manifold3d
    mesh = manifold3d.Mesh(
        vert_properties=m.vertices.astype(np.float32),
        tri_verts=m.faces.astype(np.uint32),
    )
    return manifold3d.Manifold(mesh)


def _manifold_to_trimesh(mf) -> trimesh.Trimesh:
    mesh = mf.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)
    if verts.shape[1] > 3:    # property channels beyond xyz; we only added xyz
        verts = verts[:, :3]
    return trimesh.Trimesh(
        vertices=verts,
        faces=np.asarray(mesh.tri_verts, dtype=np.int64),
        process=False,
    )


def _slice_hole_column(board_A: trimesh.Trimesh, cx: float, cy: float,
                       dx: float, dy: float, pad: float) -> trimesh.Trimesh:
    """Crop the A-frame board to the rectangular column around one hole.

    Uses manifold3d's boolean intersection with a vertical box — much more
    robust than half-space slicing, which produced non-manifold output on
    at least one hole because of cap-edge ambiguities at thin features.
    """
    x_lo, x_hi = cx - dx / 2 - pad, cx + dx / 2 + pad
    y_lo, y_hi = cy - dy / 2 - pad, cy + dy / 2 + pad
    z_lo, z_hi = float(board_A.bounds[0, 2]) - 0.001, float(board_A.bounds[1, 2]) + 0.001
    box = trimesh.creation.box(
        extents=(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo),
        transform=trimesh.transformations.translation_matrix([
            (x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2,
        ]),
    )
    a = _trimesh_to_manifold(board_A)
    b = _trimesh_to_manifold(box)
    return _manifold_to_trimesh(a ^ b)   # `^` = intersection in manifold3d


def _decompose_plate_around_one_hole(
    plate_xmin: float, plate_xmax: float,
    plate_ymin: float, plate_ymax: float,
    plate_zmin: float, plate_zmax: float,
    hole_bbox_xy: Tuple[float, float, float, float],
) -> List[Tuple[float, float, float, float, float, float]]:
    """Coarse 4-box decomposition of the plate around ONE active hole.

    Other holes on the same board are intentionally *not* modelled — the bulk
    box ignores them. The peg only inserts into the active hole, so the
    surrounding plate just needs to be a solid frame around that one hole.

    Returns 1-4 axis-aligned boxes (cx, cy, cz, sx, sy, sz):
        +-------------------+
        |        TOP        |
        +----+---------+----+
        | LFT|         |RGT |
        +----+---------+----+
        |       BOTTOM      |
        +-------------------+
    Boxes with zero extent (when the hole touches a plate edge) are dropped.
    """
    cz = (plate_zmin + plate_zmax) / 2
    sz = plate_zmax - plate_zmin
    hx_lo, hx_hi, hy_lo, hy_hi = hole_bbox_xy
    out = []

    # TOP strip: full plate width, y in (hy_hi, plate_ymax)
    if plate_ymax > hy_hi:
        sx = plate_xmax - plate_xmin
        sy = plate_ymax - hy_hi
        out.append(((plate_xmin + plate_xmax) / 2, (hy_hi + plate_ymax) / 2,
                    cz, sx, sy, sz))
    # BOTTOM strip: full plate width, y in (plate_ymin, hy_lo)
    if hy_lo > plate_ymin:
        sx = plate_xmax - plate_xmin
        sy = hy_lo - plate_ymin
        out.append(((plate_xmin + plate_xmax) / 2, (plate_ymin + hy_lo) / 2,
                    cz, sx, sy, sz))
    # LEFT strip: x in (plate_xmin, hx_lo), y in hole y-range
    if hx_lo > plate_xmin:
        sx = hx_lo - plate_xmin
        sy = hy_hi - hy_lo
        out.append(((plate_xmin + hx_lo) / 2, (hy_lo + hy_hi) / 2,
                    cz, sx, sy, sz))
    # RIGHT strip: x in (hx_hi, plate_xmax), y in hole y-range
    if plate_xmax > hx_hi:
        sx = plate_xmax - hx_hi
        sy = hy_hi - hy_lo
        out.append(((hx_hi + plate_xmax) / 2, (hy_lo + hy_hi) / 2,
                    cz, sx, sy, sz))
    return out


def _write_pretty_urdf(root: ET.Element, output_path: Path) -> None:
    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ", encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pretty)


def _write_receptive_urdf(out_urdf: Path, robot_name: str,
                           bulk_boxes: List[Tuple[float, float, float, float, float, float]],
                           hole_hull_files: List[Tuple[str, str]]) -> None:
    """Write a per-problem peg_board_1 receptive URDF.

    All geometry sits under one `<link name="plate">`:
      * one `<visual>` + `<collision>` per bulk box primitive (frame around
        the active hole — other holes ignored, the bulk overlaps where they
        would have been which is fine since this URDF is only used for the
        active-hole problem).
      * one `<visual>` + `<collision>` per active-hole CoACD hull mesh.
    """
    robot = ET.Element("robot", attrib={"name": robot_name})
    link  = ET.SubElement(robot, "link", attrib={"name": "plate"})

    # Distinct material per part_id (= per hole + a single material for bulk)
    palette = [
        (0.85, 0.30, 0.30, 1.0), (0.30, 0.65, 0.85, 1.0), (0.40, 0.75, 0.35, 1.0),
        (0.90, 0.70, 0.20, 1.0), (0.70, 0.40, 0.80, 1.0), (0.30, 0.80, 0.75, 1.0),
        (0.85, 0.55, 0.25, 1.0), (0.55, 0.55, 0.55, 1.0), (0.55, 0.25, 0.65, 1.0),
    ]
    bulk_rgba = (0.78, 0.55, 0.30, 1.0)

    def _add_origin(parent, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
        ET.SubElement(parent, "origin", attrib={
            "xyz": f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
            "rpy": f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}",
        })

    def _add_material(parent, name, rgba):
        m = ET.SubElement(parent, "material", attrib={"name": name})
        ET.SubElement(m, "color", attrib={
            "rgba": f"{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}",
        })

    # Bulk boxes
    _add_material(robot, "bulk", bulk_rgba)
    for cx, cy, cz, sx, sy, sz in bulk_boxes:
        for tag in ("visual", "collision"):
            elem = ET.SubElement(link, tag)
            _add_origin(elem, xyz=(cx, cy, cz))
            geom = ET.SubElement(elem, "geometry")
            ET.SubElement(geom, "box", attrib={
                "size": f"{sx:.6f} {sy:.6f} {sz:.6f}",
            })
            if tag == "visual":
                ET.SubElement(elem, "material", attrib={"name": "bulk"})

    # Hole hull meshes
    for k, (mat_name, mesh_rel) in enumerate(hole_hull_files):
        if mat_name not in {m.attrib["name"] for m in robot.findall("material")}:
            _add_material(robot, mat_name, palette[k % len(palette)])
        for tag in ("visual", "collision"):
            elem = ET.SubElement(link, tag)
            _add_origin(elem, xyz=(0.0, 0.0, 0.0))
            geom = ET.SubElement(elem, "geometry")
            ET.SubElement(geom, "mesh", attrib={"filename": mesh_rel, "scale": "1 1 1"})
            if tag == "visual":
                ET.SubElement(elem, "material", attrib={"name": mat_name})

    _write_pretty_urdf(robot, out_urdf)


def _write_peg_visual_urdf(out_urdf: Path, name: str, mesh_rel: str) -> None:
    """Single-link URDF that references the canonical peg mesh once for both
    visual and collision. Used as the visual-only inserter URDF; for actual
    sim collision use the *_coacd.urdf the generated alongside."""
    robot = ET.Element("robot", attrib={"name": name})
    link  = ET.SubElement(robot, "link", attrib={"name": "base_link"})
    iner  = ET.SubElement(link, "inertial")
    ET.SubElement(iner, "mass",   attrib={"value": "0.05"})
    ET.SubElement(iner, "inertia",
                  attrib={"ixx": "1e-5", "ixy": "0", "ixz": "0",
                          "iyy": "1e-5", "iyz": "0", "izz": "1e-5"})
    for tag in ("visual", "collision"):
        elem = ET.SubElement(link, tag)
        ET.SubElement(elem, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
        geom = ET.SubElement(elem, "geometry")
        ET.SubElement(geom, "mesh", attrib={"filename": mesh_rel, "scale": "1 1 1"})
    _write_pretty_urdf(robot, out_urdf)


def _write_peg_coacd_urdf(out_urdf: Path, name: str, hull_filenames: List[str]) -> None:
    """Multi-link CoACD URDF for an inserter peg (mirrors run_coacd.py output)."""
    robot = ET.Element("robot", attrib={"name": name})
    base  = ET.SubElement(robot, "link", attrib={"name": "base_link"})
    iner  = ET.SubElement(base, "inertial")
    ET.SubElement(iner, "mass",   attrib={"value": "0.05"})
    ET.SubElement(iner, "inertia",
                  attrib={"ixx": "1e-5", "ixy": "0", "ixz": "0",
                          "iyy": "1e-5", "iyz": "0", "izz": "1e-5"})
    palette = [
        (0.9, 0.2, 0.2, 1.0), (0.2, 0.7, 0.2, 1.0), (0.2, 0.3, 0.9, 1.0),
        (0.9, 0.7, 0.1, 1.0), (0.8, 0.3, 0.8, 1.0), (0.1, 0.8, 0.8, 1.0),
        (0.9, 0.5, 0.1, 1.0), (0.5, 0.5, 0.5, 1.0), (0.6, 0.2, 0.6, 1.0),
        (0.3, 0.9, 0.5, 1.0), (0.9, 0.3, 0.5, 1.0), (0.4, 0.6, 0.9, 1.0),
    ]
    for i, fname in enumerate(hull_filenames):
        c = palette[i % len(palette)]
        mat = ET.SubElement(robot, "material", attrib={"name": f"color_{i}"})
        ET.SubElement(mat, "color", attrib={
            "rgba": f"{c[0]} {c[1]} {c[2]} {c[3]}",
        })
        link = ET.SubElement(robot, "link", attrib={"name": f"hull_{i}"})
        iner = ET.SubElement(link, "inertial")
        ET.SubElement(iner, "mass",   attrib={"value": "0.001"})
        ET.SubElement(iner, "inertia",
                      attrib={"ixx": "1e-6", "ixy": "0", "ixz": "0",
                              "iyy": "1e-6", "iyz": "0", "izz": "1e-6"})
        for tag in ("visual", "collision"):
            elem = ET.SubElement(link, tag)
            ET.SubElement(elem, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            geom = ET.SubElement(elem, "geometry")
            ET.SubElement(geom, "mesh", attrib={"filename": fname, "scale": "1 1 1"})
            if tag == "visual":
                ET.SubElement(elem, "material", attrib={"name": f"color_{i}"})
        joint = ET.SubElement(robot, "joint", attrib={
            "name": f"hull_joint_{i}", "type": "fixed",
        })
        ET.SubElement(joint, "parent", attrib={"link": "base_link"})
        ET.SubElement(joint, "child",  attrib={"link": f"hull_{i}"})
        ET.SubElement(joint, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    _write_pretty_urdf(robot, out_urdf)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_board_assets(data: dict) -> Path:
    """Returns the path to the generated receptive URDF (relative to assets/)."""
    print(f"\n=== Receptive: {BOARD_NAME} ===")
    board = trimesh.load_mesh(str(BOARD_OBJ), process=False)
    board_A = _board_to_A_frame(board)
    print(f"  board in A: x={board_A.bounds[:,0].tolist()}, "
          f"y={board_A.bounds[:,1].tolist()}, z={board_A.bounds[:,2].tolist()}")

    out_dir       = ASSETS_DIR / "boards" / BOARD_NAME
    patches_dir   = out_dir / "hole_patches"
    fixtures_dir  = out_dir / "insertion_fixtures"
    if patches_dir.exists():
        shutil.rmtree(patches_dir)
    patches_dir.mkdir(parents=True, exist_ok=True)

    plate_xmin, plate_xmax = float(board_A.bounds[0, 0]), float(board_A.bounds[1, 0])
    plate_ymin, plate_ymax = float(board_A.bounds[0, 1]), float(board_A.bounds[1, 1])
    plate_zmin, plate_zmax = float(board_A.bounds[0, 2]), float(board_A.bounds[1, 2])

    # ── For each hole: slice patch → CoACD → write per-problem receptive URDF ──
    receptive_paths: dict = {}    # hole_id -> rel path from assets/
    for hole_id in sorted(data.keys()):
        info = data[hole_id]
        peg_name = info["peg"]
        cx, cy = info["hole_xy_A"]
        dx, dy = info["hole_bbox"]
        x_lo, x_hi = cx - dx/2 - HOLE_PADDING, cx + dx/2 + HOLE_PADDING
        y_lo, y_hi = cy - dy/2 - HOLE_PADDING, cy + dy/2 + HOLE_PADDING
        hole_bbox_xy = (x_lo, x_hi, y_lo, y_hi)

        # Slice + CoACD this hole's patch
        patch = _slice_hole_column(board_A, cx, cy, dx, dy, HOLE_PADDING)
        patch_path = patches_dir / f"{hole_id}.obj"
        patch.export(str(patch_path))

        coacd_dir = patches_dir / f"{hole_id}_coacd"
        run_coacd_python(patch_path, coacd_dir, **PATCH_COACD_KW)
        hole_meshes_rel = [
            f"../hole_patches/{hole_id}_coacd/{hp.name}"
            for hp in sorted(coacd_dir.glob("decomp_*.obj"))
        ]
        print(f"  {hole_id}: patch v={len(patch.vertices)} f={len(patch.faces)}, "
              f"{len(hole_meshes_rel)} hulls")

        # Coarse 4-box frame around just this hole
        bulk_boxes = _decompose_plate_around_one_hole(
            plate_xmin, plate_xmax, plate_ymin, plate_ymax, plate_zmin, plate_zmax,
            hole_bbox_xy,
        )

        # Per-problem receptive URDF named after the peg (matches the Problem
        # registry's `fmb.peg_board_1.<peg_name>` key).
        out_urdf = fixtures_dir / f"{BOARD_NAME}_{peg_name}.urdf"
        robot_name = f"fixture_{BOARD_NAME}_{peg_name}"
        # Hole hulls all share one material name within this URDF.
        hole_hull_entries = [(hole_id, mesh_rel) for mesh_rel in hole_meshes_rel]
        _write_receptive_urdf(out_urdf, robot_name, bulk_boxes, hole_hull_entries)
        receptive_paths[hole_id] = out_urdf.relative_to(REPO_ROOT / "assets")
        print(f"  → {out_urdf.relative_to(REPO_ROOT)} "
              f"(bulk {len(bulk_boxes)} boxes, hole {len(hole_meshes_rel)} hulls)")

    return receptive_paths


def _proper_perm_to_strict_xyz(extents: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix `P` such that `P @ v_storage` lives in a frame
    with strict extents X > Y > Z. Always a proper rotation (det = +1):
    if the permutation alone gives det = -1, flip the sign of the new Z
    row (the smallest extent — least visible flip).
    """
    axes_desc = np.argsort(-extents)        # e.g. [2, 0, 1] for x>y, [2, 1, 0] for y>x
    P = np.zeros((3, 3))
    for i, a in enumerate(axes_desc):
        P[i, a] = 1.0
    if np.linalg.det(P) < 0:
        P[2, axes_desc[2]] = -1.0           # flip the Z row to make det = +1
    return P


def generate_peg_assets(peg_name: str) -> dict:
    """Build canonical mesh, CoACD, and URDFs for one long peg.

    Canonical pose convention (unified with fabrica beam_2x): the canonical
    mesh is XYZ-centered at the bbox centroid and oriented so that
    extents satisfy **strict X > Y > Z** — long axis along X, longer
    cross-section axis along Y, shorter cross-section axis along Z.

    The rotation that takes a vertex from storage frame to this canonical
    frame is recorded as ``R_storage_to_canonical`` (a 3x3 list) in
    ``canonical_meta.json``. Consumers compose its inverse into the
    assembled-pose quaternion so the world-frame inserted pose is
    independent of any per-peg permutation step2 had to apply.

    Returns the meta dict written to canonical_meta.json.
    """
    print(f"\n=== Inserter: {peg_name} ===")
    peg_dir       = PEG_DIR / peg_name
    obj_path      = peg_dir / f"{peg_name}.obj"
    canonical_obj = peg_dir / f"{peg_name}_canonical.obj"

    raw = trimesh.load_mesh(str(obj_path), process=False)

    # Center at bbox centroid in storage frame.
    centered = raw.copy()
    centered.vertices = centered.vertices - (raw.bounds[0] + raw.bounds[1]) / 2

    # Permute axes so canonical extents satisfy strict X > Y > Z.
    P = _proper_perm_to_strict_xyz(centered.bounds[1] - centered.bounds[0])
    canonical = centered.copy()
    canonical.vertices = (P @ centered.vertices.T).T

    canonical.export(str(canonical_obj))
    e = canonical.bounds[1] - canonical.bounds[0]
    print(f"  canonical extent: ({e[0]*1000:.1f}, {e[1]*1000:.1f}, {e[2]*1000:.1f}) mm  "
          f"(strict X>Y>Z: {e[0] > e[1] > e[2]})")
    print(f"  R_storage_to_canonical = {P.tolist()}")

    # CoACD on the canonical (centered + oriented) mesh
    coacd_dir = peg_dir / "coacd"
    run_coacd_python(canonical_obj, coacd_dir, **PEG_COACD_KW)

    hull_filenames = [hp.name for hp in sorted(coacd_dir.glob("decomp_*.obj"))]

    # Persist the storage→canonical rotation so problems.py can compose its
    # inverse into the assembled-pose quaternion (the saved yaw_deg was
    # authored against the storage-frame mesh).
    meta = {
        "convention": "X > Y > Z strict; long axis on X",
        "R_storage_to_canonical": P.tolist(),
    }
    (peg_dir / "canonical_meta.json").write_text(json.dumps(meta, indent=2))

    # Write URDFs
    visual_urdf = peg_dir / f"{peg_name}.urdf"
    coacd_urdf  = coacd_dir / f"{peg_name}_coacd.urdf"
    _write_peg_visual_urdf(visual_urdf, peg_name, mesh_rel=f"{peg_name}_canonical.obj")
    _write_peg_coacd_urdf(coacd_urdf, f"{peg_name}_coacd", hull_filenames)
    print(f"  visual URDF  → {visual_urdf.relative_to(REPO_ROOT)}")
    print(f"  CoACD  URDF  → {coacd_urdf.relative_to(REPO_ROOT)}  ({len(hull_filenames)} hulls)")
    return meta


def main() -> None:
    if not JSON_PATH.is_file():
        raise SystemExit(f"Missing {JSON_PATH}; run step1 first.")
    data = json.loads(JSON_PATH.read_text())

    # Receptive (board) assets — one URDF per (peg, hole) problem.
    receptive_paths = generate_board_assets(data)

    # Inserter (peg) assets — only for pegs actually used by the JSON
    used_pegs = sorted({info["peg"] for info in data.values()})
    print(f"\nLong pegs used by peg_board_1: {used_pegs}")
    for peg in used_pegs:
        generate_peg_assets(peg)

    print("\nAll assets generated.")
    for hole_id, rel in sorted(receptive_paths.items()):
        peg = data[hole_id]["peg"]
        print(f"  {hole_id} ({peg}): receptive URDF = {rel}")


if __name__ == "__main__":
    main()
