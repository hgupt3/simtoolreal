#!/usr/bin/env python3
"""Viser viewer to inspect the discrete set of stable resting orientations
computed by compute_stable_rests.py.

Loads `assets/urdf/fabrica/{assembly}/stable_rests/{part_id}.npz` for each
part and renders every rest side by side on a small ground grid, with the
trimesh-reported probability shown in a side panel.

Useful for catching degenerate "balanced on a tiny edge" rests before
they're sampled into scene generation.

Usage:
    python fabrica/scene_generation/visualize_stable_rests.py --assembly beam
    python fabrica/scene_generation/visualize_stable_rests.py --assembly beam --port 8085
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from fabrica.viser_utils import COLORS, SceneManager

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

REST_SPACING = 0.20  # meters between consecutive rests in the grid


def load_assembly_parts(assembly):
    """Return ordered part ids from assembly_order.json."""
    with open(ASSETS_DIR / assembly / "assembly_order.json") as f:
        return json.load(f)["steps"]


def load_part_data(assembly, part_id):
    """Load (mesh, transforms, probabilities) for a part. Returns None if missing."""
    mesh_path = ASSETS_DIR / assembly / part_id / f"{part_id}_canonical.obj"
    rest_path = ASSETS_DIR / assembly / "stable_rests" / f"{part_id}.npz"
    if not mesh_path.exists() or not rest_path.exists():
        return None
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    data = np.load(rest_path)
    return mesh, data["transforms"], data["probabilities"]


def transformed_vertices(mesh, transform):
    """Apply a 4x4 homogeneous transform to mesh vertices."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    homog = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)  # (N, 4)
    out = (transform @ homog.T).T[:, :3]
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Viser viewer for stable resting orientations.",
    )
    parser.add_argument("--assembly", type=str, required=True,
                        help="Assembly name (e.g. 'beam').")
    parser.add_argument("--port", type=int, default=8082,
                        help="Viser port (default 8082).")
    args = parser.parse_args()

    parts = load_assembly_parts(args.assembly)
    print(f"Assembly: {args.assembly}")
    print(f"Parts: {parts}\n")

    # Pre-load every part's data
    part_data = {}
    for pid in parts:
        data = load_part_data(args.assembly, pid)
        if data is None:
            print(f"  WARNING: missing data for part {pid} (run compute_stable_rests.py first)")
            continue
        mesh, transforms, probs = data
        part_data[pid] = (mesh, transforms, probs)
        print(f"  part {pid}: {len(transforms)} rests loaded")

    if not part_data:
        print("\nNo data to visualize. Run compute_stable_rests.py first.")
        return

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene = SceneManager()

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, -1.0, 0.5)
        client.camera.look_at = (0.5, 0.0, 0.0)

    # Static ground reference
    server.scene.add_grid("/ground", width=4.0, height=2.0, cell_size=0.05)

    info_panel = server.gui.add_markdown("")

    def render_part(part_id):
        scene.clear()
        if part_id not in part_data:
            info_panel.content = f"**Part {part_id}**: no data."
            return

        mesh, transforms, probs = part_data[part_id]
        n = len(transforms)

        # Lay out rests in a row along +x
        for i, (T, p) in enumerate(zip(transforms, probs)):
            x_off = i * REST_SPACING
            color = COLORS[i % len(COLORS)]

            # Small "podium" square under each rest for visual reference
            scene.add(server.scene.add_box(
                f"/rests/{i}/podium",
                color=(int(180), int(180), int(180)),
                dimensions=(0.08, 0.08, 0.002),
                position=(x_off, 0.0, -0.001),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            ))

            # Mesh transformed by the stable rest, then translated to its grid slot
            verts = transformed_vertices(mesh, T)
            verts[:, 0] += x_off
            scene.add(server.scene.add_mesh_simple(
                f"/rests/{i}/mesh",
                vertices=verts,
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=color,
            ))

            # Tiny axis frame at each rest origin so we can see "up" easily
            scene.add(server.scene.add_frame(
                f"/rests/{i}/axes",
                position=(x_off, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                show_axes=True,
                axes_length=0.04,
                axes_radius=0.0008,
            ))

        # Side panel: list rest index, probability, and the lift z (centroid height)
        lines = [f"## Part {part_id} — {n} stable rests\n"]
        for i, (T, p) in enumerate(zip(transforms, probs)):
            lines.append(f"- rest {i}: prob={p:.3f}  z_lift={T[2,3]:+.4f} m")
        info_panel.content = "\n".join(lines)
        print(f"\nShowing part {part_id} ({n} rests).")

    # GUI: dropdown to pick part
    dropdown = server.gui.add_dropdown(
        "Part", options=list(part_data.keys()), initial_value=list(part_data.keys())[0],
    )
    dropdown.on_update(lambda _: render_part(dropdown.value))

    render_part(dropdown.value)

    print(f"\nOpen http://localhost:{args.port}")
    print("Use the dropdown to step through parts. Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
