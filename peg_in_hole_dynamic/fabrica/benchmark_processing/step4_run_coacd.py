#!/usr/bin/env python3
"""Run CoACD convex decomposition on Fabrica assembly parts.

Usage:
    python fabrica/run_coacd.py --assembly beam --part 2
    python fabrica/run_coacd.py --assembly beam --part 6

Visualize existing decomposition:
    python fabrica/run_coacd.py --assembly beam --part 2 --viz
    python fabrica/run_coacd.py --assembly beam --viz --port 8083
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import trimesh
import tyro

from fabrica.viser_utils import ASSETS_DIR


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
    (0.9, 0.2, 0.2, 1.0),
    (0.2, 0.7, 0.2, 1.0),
    (0.2, 0.3, 0.9, 1.0),
    (0.9, 0.7, 0.1, 1.0),
    (0.8, 0.3, 0.8, 1.0),
    (0.1, 0.8, 0.8, 1.0),
    (0.9, 0.5, 0.1, 1.0),
    (0.5, 0.5, 0.5, 1.0),
    (0.6, 0.2, 0.6, 1.0),
    (0.3, 0.9, 0.5, 1.0),
    (0.9, 0.3, 0.5, 1.0),
    (0.4, 0.6, 0.9, 1.0),
    (0.7, 0.9, 0.2, 1.0),
    (0.9, 0.4, 0.7, 1.0),
    (0.2, 0.5, 0.7, 1.0),
    (0.8, 0.6, 0.3, 1.0),
    (0.5, 0.3, 0.9, 1.0),
    (0.3, 0.8, 0.3, 1.0),
    (0.7, 0.2, 0.4, 1.0),
    (0.4, 0.7, 0.6, 1.0),
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


def run_coacd(
    mesh_path: Path,
    output_dir: Path,
    max_convex_hull: int = -1,
    threshold: float = 0.03,
    preprocess_resolution: int = 50,
    preprocess_mode: str = "auto",
    pca: bool = False,
    merge: bool = True,
    resolution: int = 3000,
    mcts_nodes: int = 25,
    mcts_iterations: int = 250,
    mcts_max_depth: int = 4,
    seed: int = 0,
    mode: Literal["subprocess", "python"] = "python",
) -> List[trimesh.Trimesh]:
    """Run COACD on a mesh and return the list of convex hulls."""
    assert mesh_path.exists(), f"Mesh file {mesh_path} does not exist"
    assert mesh_path.suffix == ".obj", f"Mesh file {mesh_path} is not an OBJ file"

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "subprocess":
        from subprocess import run as run_cmd

        output_mesh_path = output_dir / mesh_path.name
        cmd = f"coacd -i {mesh_path} -o {output_mesh_path} -c {max_convex_hull}"
        print(f"Running command: {cmd}")
        run_cmd(cmd, shell=True, check=True)

        assert output_mesh_path.exists(), (
            f"Output mesh file {output_mesh_path} does not exist"
        )
        output_mesh = trimesh.load(output_mesh_path)
        parts = output_mesh.split()
        for i, part in enumerate(parts):
            filename = output_dir / f"decomp_{i}.obj"
            print(f"Saving part {i} to {filename}")
            part.export(filename)
        print(
            f"Decomposition complete. {len(parts)} parts found and saved to {output_dir}."
        )
        return parts
    elif mode == "python":
        import coacd
        import numpy as np

        input_mesh = trimesh.load(mesh_path, force="mesh")
        coacd_mesh = coacd.Mesh(input_mesh.vertices, input_mesh.faces)
        convex_vs_fs_parts = coacd.run_coacd(
            coacd_mesh,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode=preprocess_mode,
            preprocess_resolution=preprocess_resolution,
            pca=pca,
            merge=merge,
            resolution=resolution,
            mcts_nodes=mcts_nodes,
            mcts_iterations=mcts_iterations,
            mcts_max_depth=mcts_max_depth,
            seed=seed,
        )
        parts = []
        for vs, fs in convex_vs_fs_parts:
            parts.append(trimesh.Trimesh(vs, fs))

        np.random.seed(0)
        scene = trimesh.Scene()
        for part in parts:
            part.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(
                np.uint8
            )
            scene.add_geometry(part)
        output_mesh_path = output_dir / mesh_path.name
        print(f"Saving scene to {output_mesh_path}")
        scene.export(output_mesh_path)

        for i, part in enumerate(parts):
            filename = output_dir / f"decomp_{i}.obj"
            print(f"Saving part {i} to {filename}")
            part.export(filename)
        print(
            f"Decomposition complete. {len(parts)} parts found and saved to {output_dir}."
        )
        return parts
    else:
        raise ValueError(f"Invalid mode: {mode}")


def generate_coacd_urdf(assembly: str, part: str, output_dir: Path) -> Path:
    """Generate a multi-body URDF where each hull is both visual and collision.

    Each hull link gets a distinct color so you can see the decomposition
    in IsaacGym's rendered output.
    """
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
    hull_links = "\n".join(
        HULL_LINK_TEMPLATE.format(i=i) for i in range(n)
    )

    name = f"{assembly}_{part}_coacd"
    urdf_content = COACD_URDF_TEMPLATE.format(
        name=name,
        materials=materials,
        hull_links=hull_links,
    )

    urdf_path = output_dir / f"{assembly}_{part}_coacd.urdf"
    urdf_path.write_text(urdf_content)
    print(f"Generated URDF: {urdf_path}")
    return urdf_path


@dataclass
class RunCoacdArgs:
    assembly: str = "beam"
    """Assembly name (e.g., beam)."""

    part: str = "2"
    """Part ID (e.g., 2 or 6)."""

    max_convex_hull: int = -1
    """Maximum number of convex hulls (-1 for unlimited)."""

    threshold: float = 0.03
    """Concavity threshold (lower = tighter fit). Default 0.03 for thin parts."""

    preprocess_resolution: int = 50
    """Voxel resolution for manifold preprocessing."""

    preprocess_mode: str = "auto"
    """Preprocess mode: 'auto', 'on', or 'off'. Use 'off' if mesh is already manifold."""

    pca: bool = False
    """PCA preprocessing — helps with elongated parts."""

    no_merge: bool = False
    """Disable merge postprocessing — preserves more detail around cavities."""

    resolution: int = 3000
    """Sampling resolution for Hausdorff distance calculation."""

    mcts_nodes: int = 25
    """MCTS child nodes per level."""

    mcts_iterations: int = 250
    """MCTS search iterations."""

    mcts_max_depth: int = 4
    """MCTS search tree depth."""

    mode: Literal["subprocess", "python"] = "python"
    """COACD execution mode."""

    assets_dir: Path = ASSETS_DIR
    """Root assets directory."""

    seed: int = 0
    """Random seed for reproducible decomposition."""

    shrink: float = 0.0
    """Post-process: shrink each hull toward its centroid by this fraction (e.g., 0.02 = 2%)."""

    viz: bool = False
    """Launch a viser viewer to visualize existing CoACD decomposition instead of running it."""

    port: int = 8082
    """Port for the viser viewer (only used with --viz)."""


def visualize_coacd(args: RunCoacdArgs):
    """Launch a viser viewer showing CoACD decomposition overlaid on the original mesh."""
    import time

    import numpy as np
    import viser

    from fabrica.viser_utils import load_assembly_parts, SceneManager

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene_mgr = SceneManager()

    # Discover all parts that have a coacd directory.
    parts = load_assembly_parts(args.assets_dir, args.assembly)
    part_ids = [pid for pid, _ in parts]
    if not part_ids:
        print(f"No parts found for assembly '{args.assembly}'")
        return

    # Filter to only parts with existing coacd dirs.
    parts_with_coacd = []
    for pid in part_ids:
        coacd_dir = args.assets_dir / args.assembly / pid / "coacd"
        if coacd_dir.exists() and list(coacd_dir.glob("decomp_*.obj")):
            parts_with_coacd.append(pid)

    if not parts_with_coacd:
        print(f"No CoACD decompositions found for assembly '{args.assembly}'")
        return

    # Default to the specified part if it has coacd, otherwise first available.
    initial_part = args.part if args.part in parts_with_coacd else parts_with_coacd[0]

    # GUI controls.
    part_dropdown = server.gui.add_dropdown(
        "Part",
        options=parts_with_coacd,
        initial_value=initial_part,
    )
    show_original = server.gui.add_checkbox("Show original mesh", initial_value=True)
    show_hulls = server.gui.add_checkbox("Show CoACD hulls", initial_value=True)
    original_opacity = server.gui.add_slider(
        "Original opacity", min=0.05, max=1.0, step=0.05, initial_value=0.3
    )
    stats_text = server.gui.add_markdown("**Stats:** loading...")

    def load_part(part_id: str):
        """Load and display a part's canonical mesh and CoACD hulls."""
        scene_mgr.clear()

        canonical_path = (
            args.assets_dir / args.assembly / part_id / f"{part_id}_canonical.obj"
        )
        coacd_dir = args.assets_dir / args.assembly / part_id / "coacd"
        decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))

        # Load canonical mesh.
        if canonical_path.exists() and show_original.value:
            canon_mesh = trimesh.load(canonical_path, force="mesh")
            vertices = np.array(canon_mesh.vertices, dtype=np.float32)
            faces = np.array(canon_mesh.faces, dtype=np.int32)
            color_rgba = np.array(
                [180, 180, 180, int(original_opacity.value * 255)], dtype=np.uint8
            )
            handle = server.scene.add_mesh_simple(
                f"/canonical/{part_id}",
                vertices=vertices,
                faces=faces,
                color=color_rgba[:3] / 255.0,
                opacity=original_opacity.value,
            )
            scene_mgr.add(handle)

        # Load CoACD hulls.
        total_hull_verts = 0
        total_hull_faces = 0
        if show_hulls.value:
            for i, decomp_file in enumerate(decomp_files):
                hull_mesh = trimesh.load(decomp_file, force="mesh")
                vertices = np.array(hull_mesh.vertices, dtype=np.float32)
                faces = np.array(hull_mesh.faces, dtype=np.int32)
                total_hull_verts += len(vertices)
                total_hull_faces += len(faces)
                color = HULL_COLORS[i % len(HULL_COLORS)]
                handle = server.scene.add_mesh_simple(
                    f"/hulls/{part_id}/decomp_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=color[:3],
                    opacity=0.85,
                )
                scene_mgr.add(handle)

        # Update stats.
        canon_info = ""
        if canonical_path.exists():
            canon_mesh = trimesh.load(canonical_path, force="mesh")
            canon_info = (
                f"  - Original: {len(canon_mesh.vertices)} verts, "
                f"{len(canon_mesh.faces)} faces\n"
            )
        stats_text.content = (
            f"**{args.assembly} / part {part_id}**\n\n"
            f"- Hull count: **{len(decomp_files)}**\n"
            f"{canon_info}"
            f"- Hull totals: {total_hull_verts} verts, {total_hull_faces} faces\n"
        )

    # Initial load.
    load_part(part_dropdown.value)

    # React to GUI changes.
    @part_dropdown.on_update
    def _on_part_change(event: viser.GuiEvent) -> None:
        load_part(part_dropdown.value)

    @show_original.on_update
    def _on_show_original(event: viser.GuiEvent) -> None:
        load_part(part_dropdown.value)

    @show_hulls.on_update
    def _on_show_hulls(event: viser.GuiEvent) -> None:
        load_part(part_dropdown.value)

    @original_opacity.on_update
    def _on_opacity(event: viser.GuiEvent) -> None:
        load_part(part_dropdown.value)

    print(f"Viser CoACD viewer running at http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer.")


def compute_overshoot(mesh_path: Path, output_dir: Path, n_samples: int = 5000) -> float:
    """Compute max Hausdorff overshoot: how far CoACD hulls extend beyond the original mesh.

    Returns max overshoot in meters.
    """
    import numpy as np

    orig = trimesh.load_mesh(str(mesh_path), process=False)
    decomp_files = sorted(output_dir.glob("decomp_*.obj"))
    if not decomp_files:
        return float("inf")

    hull_pts = []
    samples_per = max(1, n_samples // len(decomp_files))
    for f in decomp_files:
        h = trimesh.load_mesh(str(f), process=False)
        pts, _ = trimesh.sample.sample_surface(h, samples_per)
        hull_pts.append(pts)
    hull_pts = np.concatenate(hull_pts)

    _, dists, _ = trimesh.proximity.closest_point(orig, hull_pts)
    return float(dists.max())


def main():
    args: RunCoacdArgs = tyro.cli(RunCoacdArgs)

    if args.viz:
        visualize_coacd(args)
        return

    mesh_path = args.assets_dir / args.assembly / args.part / f"{args.part}_canonical.obj"
    output_dir = args.assets_dir / args.assembly / args.part / "coacd"

    print(f"Input mesh: {mesh_path}")
    print(f"Output dir: {output_dir}")

    run_coacd(
        mesh_path=mesh_path,
        output_dir=output_dir,
        max_convex_hull=args.max_convex_hull,
        threshold=args.threshold,
        preprocess_resolution=args.preprocess_resolution,
        preprocess_mode=args.preprocess_mode,
        pca=args.pca,
        merge=not args.no_merge,
        resolution=args.resolution,
        mcts_nodes=args.mcts_nodes,
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        seed=args.seed,
        mode=args.mode,
    )

    if args.shrink > 0:
        import numpy as np
        decomp_files = sorted(output_dir.glob("decomp_*.obj"))
        for f in decomp_files:
            h = trimesh.load_mesh(str(f), process=False)
            centroid = h.vertices.mean(axis=0)
            h.vertices = centroid + (1.0 - args.shrink) * (h.vertices - centroid)
            h.export(str(f))
        print(f"Shrunk {len(decomp_files)} hulls by {args.shrink * 100:.1f}%")

    generate_coacd_urdf(args.assembly, args.part, output_dir)

    overshoot = compute_overshoot(mesh_path, output_dir)
    print(f"\nMax hull overshoot: {overshoot * 1000:.2f}mm")


if __name__ == "__main__":
    main()
