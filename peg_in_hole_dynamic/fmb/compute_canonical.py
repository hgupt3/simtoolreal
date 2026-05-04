#!/usr/bin/env python3
"""Compute canonical transforms and generate canonical meshes for FMB assemblies.

For each part: center at origin, rotate longest bbox extent to X axis.
Outputs canonical_transforms.json, canonical OBJs, and per-part URDFs.

Usage:
    python -m fmb.compute_canonical --assembly fmb_board_1
    python -m fmb.compute_canonical --assembly fmb_board_1 --part board_1_0
    python -m fmb.compute_canonical --assembly fmb_board_1 --dry-run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

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
    centroid = (mesh.bounds[0] + mesh.bounds[1]) / 2
    bbox = mesh.bounding_box.extents.copy()
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


def main():
    parser = argparse.ArgumentParser(description="Compute canonical transforms for FMB assemblies")
    parser.add_argument("--assembly", type=str, required=True)
    parser.add_argument("--part", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
