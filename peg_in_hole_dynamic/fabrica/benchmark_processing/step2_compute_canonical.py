#!/usr/bin/env python3
"""Compute canonical transforms and generate canonical meshes.

For each part: center at origin, rotate longest bbox extent to X axis.
Outputs canonical_transforms.json, canonical OBJs, and per-part URDFs.

Usage:
    python fabrica/benchmark_processing/step2_compute_canonical.py --assembly beam
    python fabrica/benchmark_processing/step2_compute_canonical.py --assembly beam --part 0
    python fabrica/benchmark_processing/step2_compute_canonical.py --assembly beam --dry-run
    python fabrica/benchmark_processing/step2_compute_canonical.py --assembly beam --viz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_file}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_file}" scale="1 1 1"/></geometry>
{sdf_tag}    </collision>
    <inertial><density value="1250.0"/></inertial>
  </link>
</robot>
"""


def compute_canonical_transform(mesh):
    """Center at origin, rotate longest bbox extent to X, medium to Y, shortest to Z.

    Returns (centroid, canonical_extents, assembled_to_canonical_wxyz).
    """
    centroid = (mesh.bounds[0] + mesh.bounds[1]) / 2  # bbox center, not mass centroid
    bbox = mesh.bounding_box.extents.copy()
    # Tiebreaker: when extents are equal, prefer Y > X > Z as longest axis
    tiebreaker = np.array([0, 1e-10, 2e-10])
    order = np.argsort(-(bbox + tiebreaker))
    canonical_extents = bbox[order]

    if list(order) == [0, 1, 2]:
        mat = np.eye(3)
    else:
        mat = np.zeros((3, 3))
        mat[2, order[2]] = -1.0
        mat[1, order[1]] = 1.0
        mat[0] = np.cross(mat[1], mat[2])

    quat_xyzw = Rotation.from_matrix(mat).as_quat()
    a2c = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    if a2c[0] < 0:
        a2c = -a2c
    return centroid, canonical_extents, a2c


def generate_canonical_mesh(mesh, centroid, a2c_wxyz):
    """Generate canonical mesh: center at origin, rotate by canonical transform."""
    canonical = mesh.copy()
    canonical.vertices -= centroid
    w, x, y, z = a2c_wxyz
    canonical.vertices = Rotation.from_quat([x, y, z, w]).apply(canonical.vertices)
    return canonical


def find_parts(source_dir, part_filter=None):
    if part_filter:
        return [part_filter]
    return sorted([
        d.name for d in source_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.obj").exists()
    ])


def run_viz(assembly, part_filter, port):
    """Viser viewer: two rows of individual parts — assembled frame vs canonical frame."""
    import viser
    from fabrica.viser_utils import COLORS

    source_dir = ASSETS_DIR / assembly
    part_ids = find_parts(source_dir, part_filter)
    if not part_ids:
        print(f"No parts found in {source_dir}")
        return

    # Load and compute
    part_data = {}
    for pid in part_ids:
        mesh = trimesh.load_mesh(str(source_dir / pid / f"{pid}.obj"), process=False)
        centroid, extents, a2c = compute_canonical_transform(mesh)
        canonical = generate_canonical_mesh(mesh, centroid, a2c)
        # Center assembled mesh too
        assembled = mesh.copy()
        assembled.vertices -= centroid
        part_data[pid] = (assembled, canonical, extents, a2c)

    server = viser.ViserServer(host="0.0.0.0", port=port)

    spacing = 0.15
    start_x = -(len(part_ids) - 1) * spacing / 2
    row_assembled_y = 0.15
    row_canonical_y = -0.15

    for i, pid in enumerate(part_ids):
        assembled, canonical, extents, a2c = part_data[pid]
        color = tuple(int(c * 255) for c in COLORS[i % len(COLORS)])
        sx = start_x + i * spacing

        # Top row: assembled frame (centered)
        server.scene.add_mesh_simple(
            f"/assembled/{pid}/mesh",
            vertices=np.array(assembled.vertices, dtype=np.float32),
            faces=np.array(assembled.faces, dtype=np.uint32),
            color=color, position=(sx, row_assembled_y, 0),
        )
        server.scene.add_frame(f"/assembled/{pid}/axes", position=(sx, row_assembled_y, 0),
                               show_axes=True, axes_length=0.04, axes_radius=0.001)
        server.scene.add_label(f"/assembled/{pid}/label", text=f"Part {pid}",
                               position=(sx, row_assembled_y, 0.07))

        # Bottom row: canonical frame
        server.scene.add_mesh_simple(
            f"/canonical/{pid}/mesh",
            vertices=np.array(canonical.vertices, dtype=np.float32),
            faces=np.array(canonical.faces, dtype=np.uint32),
            color=color, position=(sx, row_canonical_y, 0),
        )
        server.scene.add_frame(f"/canonical/{pid}/axes", position=(sx, row_canonical_y, 0),
                               show_axes=True, axes_length=0.04, axes_radius=0.001)
        server.scene.add_label(
            f"/canonical/{pid}/label",
            text=f"Part {pid}\n[{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]",
            position=(sx, row_canonical_y, 0.07),
        )

    # Row labels
    lx = start_x - spacing
    server.scene.add_label("/row_labels/assembled", text="Assembled", position=(lx, row_assembled_y, 0))
    server.scene.add_label("/row_labels/canonical", text="Canonical", position=(lx, row_canonical_y, 0))

    print(f"\nOpen http://localhost:{port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Compute canonical transforms and meshes")
    parser.add_argument("--assembly", type=str, required=True)
    parser.add_argument("--part", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    if args.viz:
        run_viz(args.assembly, args.part, args.port)
        return

    source_dir = ASSETS_DIR / args.assembly
    output_dir = ASSETS_DIR / args.assembly
    part_ids = find_parts(source_dir, args.part)

    if not part_ids:
        print(f"No parts found in {source_dir}")
        return

    print(f"Assembly: {args.assembly}")
    print(f"Parts: {part_ids}\n")

    transforms = {}
    for pid in part_ids:
        obj_path = source_dir / pid / f"{pid}.obj"
        if not obj_path.exists():
            print(f"  WARNING: {obj_path} not found, skipping")
            continue

        mesh = trimesh.load_mesh(str(obj_path), process=False, maintain_order=True)
        centroid, can_ext, a2c = compute_canonical_transform(mesh)

        transforms[pid] = {
            "assembled_to_canonical_wxyz": a2c.tolist(),
            "original_centroid": centroid.tolist(),
            "canonical_extents": can_ext.tolist(),
        }

        print(f"Part {pid}:")
        print(f"  centroid:          [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
        print(f"  canonical_extents: [{can_ext[0]:.6f}, {can_ext[1]:.6f}, {can_ext[2]:.6f}]")
        print(f"  a2c quaternion:    [{a2c[0]:.6f}, {a2c[1]:.6f}, {a2c[2]:.6f}, {a2c[3]:.6f}]\n")

        if not args.dry_run:
            part_dir = output_dir / pid
            part_dir.mkdir(parents=True, exist_ok=True)
            name = f"{args.assembly}_{pid}"
            mesh_file = f"{pid}_canonical.obj"

            generate_canonical_mesh(mesh, centroid, a2c).export(str(part_dir / mesh_file))
            print(f"  Wrote: {part_dir / mesh_file}")

            (part_dir / f"{name}.urdf").write_text(
                URDF_TEMPLATE.format(name=name, mesh_file=mesh_file, sdf_tag=""))
            (part_dir / f"{name}_sdf.urdf").write_text(
                URDF_TEMPLATE.format(name=f"{name}_sdf", mesh_file=mesh_file,
                                     sdf_tag='      <sdf resolution="512"/>\n'))
            print(f"  Wrote: {name}.urdf, {name}_sdf.urdf")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        transforms_path = output_dir / "canonical_transforms.json"
        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)
        print(f"\nWrote: {transforms_path}")
    else:
        print("(dry-run — no files written)")


if __name__ == "__main__":
    main()
