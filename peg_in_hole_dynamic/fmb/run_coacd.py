#!/usr/bin/env python3
"""Run CoACD convex decomposition on FMB assembly parts.

Usage:
    python -m fmb.run_coacd --assembly fmb_board_1 --part board_1_0
    python -m fmb.run_coacd --assembly fmb_board_1   # all parts
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import trimesh
import tyro

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

COACD_URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="{name}">
{materials}
  <link name="base_link">
    <inertial>
      <density value="1250.0"/>
    </inertial>
  </link>
{hull_links}
</robot>
"""

HULL_COLORS = [
    (0.9, 0.2, 0.2, 1.0), (0.2, 0.7, 0.2, 1.0), (0.2, 0.3, 0.9, 1.0),
    (0.9, 0.7, 0.1, 1.0), (0.8, 0.3, 0.8, 1.0), (0.1, 0.8, 0.8, 1.0),
    (0.9, 0.5, 0.1, 1.0), (0.5, 0.5, 0.5, 1.0), (0.6, 0.2, 0.6, 1.0),
    (0.3, 0.9, 0.5, 1.0), (0.9, 0.3, 0.5, 1.0), (0.4, 0.6, 0.9, 1.0),
]

HULL_LINK_TEMPLATE = """\
  <link name="hull_{i}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="decomp_{i}.obj" scale="1 1 1"/>
      </geometry>
      <material name="color_{i}"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="decomp_{i}.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.001"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
  <joint name="hull_joint_{i}" type="fixed">
    <parent link="base_link"/>
    <child link="hull_{i}"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>"""


def run_coacd_python(
    mesh_path: Path,
    output_dir: Path,
    threshold: float = 0.03,
    max_convex_hull: int = -1,
    seed: int = 0,
) -> List[trimesh.Trimesh]:
    import coacd
    import numpy as np

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_mesh = trimesh.load(mesh_path, force="mesh")
    coacd_mesh = coacd.Mesh(input_mesh.vertices, input_mesh.faces)
    convex_vs_fs_parts = coacd.run_coacd(
        coacd_mesh, threshold=threshold, max_convex_hull=max_convex_hull, seed=seed,
    )

    parts = [trimesh.Trimesh(vs, fs) for vs, fs in convex_vs_fs_parts]

    np.random.seed(0)
    scene = trimesh.Scene()
    for part in parts:
        part.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(part)
    scene.export(str(output_dir / mesh_path.name))

    for i, part in enumerate(parts):
        part.export(str(output_dir / f"decomp_{i}.obj"))

    print(f"  {len(parts)} convex hulls -> {output_dir}")
    return parts


def generate_coacd_urdf(assembly: str, part: str, output_dir: Path) -> Path:
    decomp_files = sorted(output_dir.glob("decomp_*.obj"))
    assert len(decomp_files) > 0, f"No decomp files found in {output_dir}"

    n = len(decomp_files)
    materials = "\n".join(
        f'  <material name="color_{i}">'
        f'<color rgba="{HULL_COLORS[i % len(HULL_COLORS)][0]} '
        f'{HULL_COLORS[i % len(HULL_COLORS)][1]} '
        f'{HULL_COLORS[i % len(HULL_COLORS)][2]} '
        f'{HULL_COLORS[i % len(HULL_COLORS)][3]}"/>'
        f'</material>'
        for i in range(n)
    )
    hull_links = "\n".join(HULL_LINK_TEMPLATE.format(i=i) for i in range(n))

    name = f"{assembly}_{part}_coacd"
    urdf_path = output_dir / f"{name}.urdf"
    urdf_path.write_text(COACD_URDF_TEMPLATE.format(
        name=name, materials=materials, hull_links=hull_links,
    ))
    print(f"  Generated URDF: {urdf_path.name}")
    return urdf_path


@dataclass
class RunCoacdArgs:
    assembly: str = "fmb_board_1"
    """Assembly name."""

    part: str = ""
    """Part ID (e.g., board_1_0). Empty = process all parts."""

    threshold: float = 0.03
    """Concavity threshold (lower = tighter fit)."""

    max_convex_hull: int = -1
    """Maximum number of convex hulls (-1 for unlimited)."""

    seed: int = 0
    """Random seed."""


def main():
    args: RunCoacdArgs = tyro.cli(RunCoacdArgs)
    assembly_dir = ASSETS_DIR / args.assembly

    if args.part:
        part_ids = [args.part]
    else:
        parts_with_canonical = sorted([
            d.name for d in assembly_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}_canonical.obj").exists()
        ])
        if not parts_with_canonical:
            print(f"No canonical meshes found in {assembly_dir}. Run compute_canonical first.")
            return
        part_ids = parts_with_canonical

    print(f"Assembly: {args.assembly}")
    print(f"Parts: {part_ids}\n")

    for pid in part_ids:
        mesh_path = assembly_dir / pid / f"{pid}_canonical.obj"
        output_dir = assembly_dir / pid / "coacd"

        if not mesh_path.exists():
            print(f"  WARNING: {mesh_path} not found, skipping {pid}")
            continue

        print(f"Part {pid}:")
        run_coacd_python(
            mesh_path, output_dir,
            threshold=args.threshold,
            max_convex_hull=args.max_convex_hull,
            seed=args.seed,
        )
        generate_coacd_urdf(args.assembly, pid, output_dir)


if __name__ == "__main__":
    main()
