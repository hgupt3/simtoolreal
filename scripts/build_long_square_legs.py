"""Generate longer square_table_leg4 variants.

For each requested body length, emit a sibling part dir under
``assets/urdf/furniture_bench/square_table/`` containing:

  * ``<name>_canonical.obj`` / ``_canonical.stl`` — full watertight leg
    mesh, centered at its bbox centroid (matches the original leg4
    convention so ``pos_canonical_centroid`` is the link-origin pose).
  * ``<name>_canonical.urdf`` — single-link wrapper for the canonical OBJ.
  * ``canonical_meta.json`` — copy of leg4 meta (same R_storage_to_canonical
    and ``role: inserter``).
  * ``sdf_hybrid/<name>_thread.obj`` — thread mesh from leg4, translated
    to the new centered-canonical position.
  * ``sdf_hybrid/<name>_body.obj`` — clean axis-aligned 30 × 30 × L box.
  * ``sdf_hybrid/body_coacd/decomp_0.obj`` — same body box (already convex).
  * ``sdf_hybrid/<name>_sdf_hybrid.urdf`` — bare hybrid URDF (no mass).
  * ``sdf_hybrid/<name>_matchedmass_sdf_hybrid.urdf`` — same hybrid URDF
    with mass + uniform-density inertia computed from the measured mass.

Also emits per-length one_leg setup JSONs into the parent piece dir and
patches problems.py to register the new variants.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import manifold3d
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
PIECE_DIR = REPO / "assets/urdf/furniture_bench/square_table"
SRC_LEG = PIECE_DIR / "square_table_leg4"
CANONICAL_SRC = SRC_LEG / "square_table_leg4_canonical.stl"
SRC_THREAD_OBJ = SRC_LEG / "sdf_hybrid/square_table_leg4_thread.obj"
SRC_META = SRC_LEG / "canonical_meta.json"

# Canonical-frame coords of the source leg (uncentered, before bbox-center
# shift). The boolean union below is also in this uncentered frame; we
# center afterwards.
THREAD_BODY_JUNCTION_X = -0.01875    # m
BODY_HALF_CROSS = 0.015               # m, 30 mm body half-extent

# Mass measurements (kg).
VARIANTS = {
    "square_table_leg4_125mm": {"body_length_m": 0.125, "mass_kg": 0.04092},
    "square_table_leg4_200mm": {"body_length_m": 0.200, "mass_kg": 0.06219},
}


def trimesh_to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    m3d_mesh = manifold3d.Mesh(
        vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32),
    )
    return manifold3d.Manifold(m3d_mesh)


def manifold_to_trimesh(man: manifold3d.Manifold) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    # Make a fully owned copy. `mesh.vert_properties[:, :3]` is a view into
    # the manifold-owned buffer; trimesh.apply_translation later writes
    # through that view and corrupts adjacent properties columns (4th+),
    # so subsequent reads through it report bogus extents. Use .copy() to
    # break the view.
    verts = np.array(mesh.vert_properties[:, :3], dtype=np.float64, copy=True)
    faces = np.array(mesh.tri_verts, dtype=np.int64, copy=True)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.merge_vertices()
    return out


def build_uncentered_canonical(body_length_m: float) -> trimesh.Trimesh:
    """Union(leg4 canonical, clean box body) in the uncentered canonical frame."""
    canonical = trimesh.load(CANONICAL_SRC, force="mesh", process=False)
    canonical.merge_vertices()

    body = trimesh.creation.box(
        extents=[body_length_m, 2 * BODY_HALF_CROSS, 2 * BODY_HALF_CROSS]
    )
    body.apply_translation(
        [THREAD_BODY_JUNCTION_X + body_length_m / 2.0, 0.0, 0.0]
    )

    return manifold_to_trimesh(
        trimesh_to_manifold(canonical) + trimesh_to_manifold(body)
    )


def center_at_bbox(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Translate mesh so its bbox centroid sits at origin; return shift used.

    Implemented via numpy directly because trimesh.apply_translation has a
    bug on manifold3d-derived meshes that corrupts some vertices to bogus
    positions (~1 m offset along x). Building a fresh Trimesh from the
    explicitly-shifted vertex array side-steps it.
    """
    bb = mesh.bounds
    centroid = (bb[0] + bb[1]) / 2.0
    verts = np.array(mesh.vertices, dtype=np.float64, copy=True) - centroid
    out = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    return out, centroid


def write_canonical_urdf(out_path: Path, robot_name: str, obj_name: str) -> None:
    urdf = f"""<?xml version="1.0"?>
<robot name="{robot_name}">
  <link name="{robot_name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{obj_name}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{obj_name}" scale="1 1 1"/></geometry>
    </collision>
    <inertial><density value="800.0"/></inertial>
  </link>
</robot>
"""
    out_path.write_text(urdf)


def _body_coacd_lines(n_hulls: int) -> str:
    out = []
    for i in range(n_hulls):
        out.append(
            f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
            f'<geometry><mesh filename="body_coacd/decomp_{i}.obj" scale="1 1 1"/></geometry>'
            f'<material name="body"/></visual>\n'
            f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>'
            f'<geometry><mesh filename="body_coacd/decomp_{i}.obj" scale="1 1 1"/></geometry>'
            f'</collision>'
        )
    return "\n".join(out)


def write_sdf_hybrid_urdf(
    out_path: Path,
    robot_name: str,
    thread_obj_name: str,
    n_body_hulls: int,
    matched_mass: dict | None = None,
) -> None:
    """Emit sdf_hybrid URDF. If matched_mass is provided, fill in the
    base_link inertial; otherwise use the placeholder mass=0.05 kg.
    """
    if matched_mass is None:
        mass_xml = (
            '      <mass value="0.05"/>\n'
            '      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>'
        )
        mass_comment = ""
    else:
        m = matched_mass["mass_kg"]
        I = matched_mass["inertia"]      # 3x3 in CoG frame
        cog = matched_mass["cog"]         # (x, y, z) in link frame
        mass_xml = (
            f'      <!-- measured: {m*1000:.2f} g -->\n'
            f'      <mass value="{m:.6f}"/>\n'
            f'      <origin xyz="{cog[0]:.6f} {cog[1]:.6f} {cog[2]:.6f}" rpy="0 0 0"/>\n'
            f'      <inertia '
            f'ixx="{I[0,0]:.6e}" ixy="{I[0,1]:.6e}" ixz="{I[0,2]:.6e}" '
            f'iyy="{I[1,1]:.6e}" iyz="{I[1,2]:.6e}" izz="{I[2,2]:.6e}"/>'
        )
        mass_comment = ""

    body_lines = _body_coacd_lines(n_body_hulls)
    urdf = f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="{robot_name}">
  <link name="base_link">
    <inertial>
{mass_xml}
    </inertial>
  </link>
  <material name="thread">
    <color rgba="0.95 0.55 0.10 1.0"/>
  </material>
  <material name="body">
    <color rgba="0.50 0.50 0.50 1.0"/>
  </material>
  <link name="thread_link">
    <visual>
      <origin xyz="0.000000 0.000000 0.000000" rpy="0.000000 0.000000 0.000000"/>
      <geometry>
        <mesh filename="{thread_obj_name}" scale="1 1 1"/>
      </geometry>
      <material name="thread"/>
    </visual>
    <collision>
      <origin xyz="0.000000 0.000000 0.000000" rpy="0.000000 0.000000 0.000000"/>
      <geometry>
        <mesh filename="{thread_obj_name}" scale="1 1 1"/>
      </geometry>
      <sdf resolution="256"/>
    </collision>
  </link>
  <joint name="base_to_thread" type="fixed">
    <parent link="base_link"/>
    <child link="thread_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <link name="body_link">
{body_lines}
  </link>
  <joint name="base_to_body" type="fixed">
    <parent link="base_link"/>
    <child link="body_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""
    out_path.write_text(urdf)


def compute_inertia(mesh: trimesh.Trimesh, mass_kg: float):
    """Uniform-density inertia about CoG, scaled to the measured mass."""
    vol = float(mesh.volume)
    density = mass_kg / vol
    mesh = mesh.copy()
    mesh.density = density
    I = np.asarray(mesh.moment_inertia, dtype=float)  # about CoG, kg·m²
    cog = np.asarray(mesh.center_mass, dtype=float)
    return I, cog, density, vol


def build_variant(name: str, body_length_m: float, mass_kg: float) -> dict:
    out_dir = PIECE_DIR / name
    sdf_dir = out_dir / "sdf_hybrid"
    coacd_body_dir = sdf_dir / "body_coacd"
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)
    coacd_body_dir.mkdir(parents=True, exist_ok=True)

    # --- canonical mesh (centered) ---
    raw = build_uncentered_canonical(body_length_m)
    centered, shift = center_at_bbox(raw)
    centered.export(str(out_dir / f"{name}_canonical.stl"))
    centered.export(str(out_dir / f"{name}_canonical.obj"))

    # --- canonical_meta.json (copy) ---
    shutil.copy2(SRC_META, out_dir / "canonical_meta.json")

    # --- canonical URDF ---
    write_canonical_urdf(
        out_dir / f"{name}_canonical.urdf",
        robot_name=name,
        obj_name=f"{name}_canonical.obj",
    )

    # --- sdf_hybrid thread.obj (translated copy) ---
    thread = trimesh.load(SRC_THREAD_OBJ, force="mesh", process=False)
    # Original thread is in uncentered frame at x ∈ [-43.75, -18.75] mm.
    # Apply the same `-shift` to put it in the new centered frame. Numpy
    # translate to dodge the same trimesh.apply_translation issue.
    thread_verts = np.array(thread.vertices, dtype=np.float64, copy=True) - shift
    thread = trimesh.Trimesh(vertices=thread_verts, faces=thread.faces.copy(), process=False)
    thread_obj_path = sdf_dir / f"{name}_thread.obj"
    thread.export(str(thread_obj_path))

    # --- sdf_hybrid body.obj (clean box in centered frame) ---
    # Body in centered frame: junction at x = -shift[0] - 0.01875.
    body_min_x = -shift[0] + THREAD_BODY_JUNCTION_X
    body_max_x = body_min_x + body_length_m
    body_center_x = (body_min_x + body_max_x) / 2.0
    body = trimesh.creation.box(
        extents=[body_length_m, 2 * BODY_HALF_CROSS, 2 * BODY_HALF_CROSS]
    )
    body.apply_translation([body_center_x, 0.0, 0.0])
    body_obj_path = sdf_dir / f"{name}_body.obj"
    body.export(str(body_obj_path))

    # --- body_coacd/decomp_0.obj (the box is already convex) ---
    body.export(str(coacd_body_dir / "decomp_0.obj"))

    # --- bare sdf_hybrid URDF ---
    write_sdf_hybrid_urdf(
        sdf_dir / f"{name}_sdf_hybrid.urdf",
        robot_name=f"{name}_sdf_hybrid",
        thread_obj_name=f"{name}_thread.obj",
        n_body_hulls=1,
        matched_mass=None,
    )

    # --- matched-mass sdf_hybrid URDF ---
    I, cog, density, vol = compute_inertia(centered, mass_kg)
    write_sdf_hybrid_urdf(
        sdf_dir / f"{name}_matchedmass_sdf_hybrid.urdf",
        robot_name=f"{name}_sdf_hybrid_matchedmass",
        thread_obj_name=f"{name}_thread.obj",
        n_body_hulls=1,
        matched_mass={"mass_kg": mass_kg, "inertia": I, "cog": cog},
    )

    print(
        f"\n{name}: body={body_length_m*1000:.0f} mm, "
        f"mass={mass_kg*1000:.2f} g, volume={vol*1e6:.2f} cm³, "
        f"density={density:.3f} g/cm³"
    )
    print(f"  centered canonical bounds (mm): "
          f"{[round(b*1000, 3) for b in centered.bounds.flatten()]}")
    print(f"  CoG (mm): {[round(c*1000, 4) for c in cog]}")
    print(f"  Inertia diag (kg·m²): "
          f"ixx={I[0,0]:.4e} iyy={I[1,1]:.4e} izz={I[2,2]:.4e}")
    print(f"  shift applied: {[round(s*1000, 3) for s in shift]} mm")
    return {
        "name": name,
        "shift": shift.tolist(),
        "body_length_m": body_length_m,
        "mass_kg": mass_kg,
        "cog": cog.tolist(),
    }


def main() -> None:
    summaries = []
    for name, cfg in VARIANTS.items():
        s = build_variant(name, cfg["body_length_m"], cfg["mass_kg"])
        summaries.append(s)
    print("\n===== summary =====")
    for s in summaries:
        print(f"  {s['name']}: shift={s['shift']}, cog={s['cog']}")


if __name__ == "__main__":
    main()
