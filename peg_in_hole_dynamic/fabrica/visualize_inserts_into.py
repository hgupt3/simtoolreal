"""Visualize inserts_into mappings for all assemblies using viser.

Shows each assembly with parts colored by their insertion target.
Arrows drawn from each part's centroid to the centroid of the part it inserts into.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import trimesh
import viser

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

# Distinct colors for target parts
TARGET_COLORS = [
    (0.9, 0.2, 0.2),
    (0.2, 0.7, 0.2),
    (0.2, 0.3, 0.9),
    (0.9, 0.7, 0.1),
    (0.8, 0.3, 0.8),
    (0.1, 0.8, 0.8),
    (0.9, 0.5, 0.1),
    (0.5, 0.5, 0.5),
    (0.6, 0.2, 0.6),
]

BASE_COLOR = (0.3, 0.3, 0.3)  # Gray for the base part


def load_assembly(assembly: str):
    """Load assembly config and meshes."""
    config_path = ASSETS_DIR / assembly / "assembly_order.json"
    config = json.loads(config_path.read_text())
    steps = config["steps"]
    inserts_into = config.get("inserts_into", {})

    meshes = {}
    for pid in steps:
        obj_path = ASSETS_DIR / assembly / pid / f"{pid}.obj"
        if obj_path.exists():
            meshes[pid] = trimesh.load_mesh(str(obj_path), process=False)

    return steps, inserts_into, meshes


def make_arrow_mesh(start, end, shaft_radius=0.001, head_radius=0.003, head_length=0.008):
    """Create a mesh arrow from start to end."""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    direction = direction / length

    shaft_length = max(length - head_length, length * 0.6)

    # Create shaft cylinder
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius,
        height=shaft_length,
        sections=8,
    )

    # Create head cone
    head = trimesh.creation.cone(
        radius=head_radius,
        height=head_length,
        sections=8,
    )

    # Position shaft: centered at midpoint of shaft portion
    shaft.apply_translation([0, 0, shaft_length / 2])
    # Position head: at end of shaft
    head.apply_translation([0, 0, shaft_length + head_length / 2])

    arrow = trimesh.util.concatenate([shaft, head])

    # Rotate to align Z-axis with direction
    z_axis = np.array([0, 0, 1.0])
    if np.allclose(direction, z_axis):
        rotation = np.eye(4)
    elif np.allclose(direction, -z_axis):
        rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        rotation = trimesh.transformations.rotation_matrix(angle, axis)

    arrow.apply_transform(rotation)
    arrow.apply_translation(start)

    return arrow


def main():
    parser = argparse.ArgumentParser(description="Visualize inserts_into mappings")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--assembly", type=str, default=None,
                        help="Show specific assembly (default: all)")
    args = parser.parse_args()

    assemblies = ["beam", "car", "cooling_manifold", "gamepad",
                  "plumbers_block", "stool_circular"]
    if args.assembly:
        assemblies = [args.assembly]

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Viser server at http://localhost:{args.port}")

    # Layout assemblies in a grid
    cols = 3
    spacing = 0.5

    for idx, assembly in enumerate(assemblies):
        config_path = ASSETS_DIR / assembly / "assembly_order.json"
        if not config_path.exists():
            print(f"Skipping {assembly}: no assembly_order.json")
            continue

        steps, inserts_into, meshes = load_assembly(assembly)
        if not meshes:
            print(f"Skipping {assembly}: no meshes")
            continue

        # Grid offset
        row, col = divmod(idx, cols)
        # Center each assembly at its own grid cell
        all_verts = np.concatenate([m.vertices for m in meshes.values()])
        center = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2
        offset = np.array([col * spacing, row * spacing, 0]) - center

        base_pid = steps[0]

        # Assign colors: each unique target gets a color, base is gray
        target_pids = sorted(set(inserts_into.values())) if inserts_into else []
        target_color_map = {pid: TARGET_COLORS[i % len(TARGET_COLORS)]
                           for i, pid in enumerate(target_pids)}

        print(f"\n=== {assembly} ===")
        print(f"  Base: part {base_pid}")
        if inserts_into:
            for pid, target in inserts_into.items():
                print(f"  Part {pid} -> Part {target}")
        else:
            print("  No inserts_into defined")

        # Add assembly label
        label_pos = offset + np.array([0, 0, all_verts.max(axis=0)[2] - center[2] + 0.05])
        server.scene.add_label(
            f"/{assembly}/label",
            text=assembly.replace("_", " ").title(),
            wxyz=np.array([1, 0, 0, 0]),
            position=tuple(label_pos),
        )

        # Add meshes
        for pid, mesh in meshes.items():
            verts = mesh.vertices + offset
            faces = mesh.faces

            if pid == base_pid:
                color = BASE_COLOR
                label = f"Part {pid} (BASE)"
            elif pid in inserts_into:
                target = inserts_into[pid]
                color = target_color_map.get(target, (0.5, 0.5, 0.5))
                label = f"Part {pid} -> {target}"
            else:
                color = (0.5, 0.5, 0.5)
                label = f"Part {pid}"

            r, g, b = color
            colors_array = np.full((len(verts), 3), [int(r*255), int(g*255), int(b*255)],
                                   dtype=np.uint8)

            server.scene.add_mesh_simple(
                f"/{assembly}/parts/{pid}",
                vertices=verts.astype(np.float32),
                faces=faces.astype(np.int32),
                color=(int(r*255), int(g*255), int(b*255)),
                opacity=0.85,
            )

            # Part label at centroid
            centroid = mesh.centroid + offset
            server.scene.add_label(
                f"/{assembly}/parts/{pid}/label",
                text=label,
                wxyz=np.array([1, 0, 0, 0]),
                position=tuple(centroid),
            )

        # Draw arrows from each part to its insertion target
        for pid, target_pid in inserts_into.items():
            if pid not in meshes or target_pid not in meshes:
                continue

            start = meshes[pid].centroid + offset
            end = meshes[target_pid].centroid + offset

            # Shorten arrow: start 30% from source, end 30% from target
            direction = end - start
            arrow_start = start + 0.3 * direction
            arrow_end = end - 0.3 * direction

            arrow_mesh = make_arrow_mesh(arrow_start, arrow_end)
            if arrow_mesh is not None:
                # Arrow color matches the part color
                target_color = target_color_map.get(target_pid, (0.5, 0.5, 0.5))
                r, g, b = target_color
                server.scene.add_mesh_simple(
                    f"/{assembly}/arrows/{pid}_to_{target_pid}",
                    vertices=arrow_mesh.vertices.astype(np.float32),
                    faces=arrow_mesh.faces.astype(np.int32),
                    color=(int(r*255), int(g*255), int(b*255)),
                )

    print("\nVisualization ready. Press Ctrl+C to exit.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
