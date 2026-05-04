#!/usr/bin/env python3
"""Visualize FMB board assemblies in a viser web viewer.

Shows each board in assembled, exploded, and per-step cumulative fixture views.

Usage:
    python -m fmb.visualize_assemblies
    python -m fmb.visualize_assemblies --board fmb_board_1
    python -m fmb.visualize_assemblies --port 8082
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

PART_COLORS = [
    (180, 170, 150),  # base: neutral
    (230, 51, 51),
    (51, 179, 51),
    (51, 77, 230),
    (230, 179, 26),
    (204, 77, 204),
    (26, 204, 204),
]


def discover_boards():
    if not ASSETS_DIR.exists():
        return []
    return sorted(
        d.name for d in ASSETS_DIR.iterdir()
        if d.is_dir() and (d / "assembly_order.json").exists()
    )


def load_board(board_name):
    board_dir = ASSETS_DIR / board_name

    with open(board_dir / "assembly_order.json") as f:
        order = json.load(f)

    with open(board_dir / "canonical_transforms.json") as f:
        transforms = json.load(f)

    parts = []
    for pid in order["steps"]:
        obj_path = board_dir / pid / f"{pid}.obj"
        if not obj_path.exists():
            continue
        mesh = trimesh.load(str(obj_path), force="mesh", process=False)
        centroid = np.array(transforms[pid]["original_centroid"])
        a2c_wxyz = np.array(transforms[pid]["assembled_to_canonical_wxyz"])
        parts.append({
            "id": pid,
            "mesh": mesh,
            "centroid": centroid,
            "a2c_wxyz": a2c_wxyz,
        })

    return order, transforms, parts


def add_mesh_at(server, path, mesh, position, color, visible=True):
    verts = (mesh.vertices + position).astype(np.float32)
    return server.scene.add_mesh_simple(
        path,
        vertices=verts,
        faces=mesh.faces.astype(np.int32),
        color=color,
        flat_shading=True,
        visible=visible,
    )


@dataclass
class VizArgs:
    """Visualize FMB board assemblies."""
    board: str = ""
    """Single board name (e.g., fmb_board_1). Empty = all boards."""
    port: int = 8082
    explode_factor: float = 3.0


def main():
    args: VizArgs = tyro.cli(VizArgs)

    boards = [args.board] if args.board else discover_boards()
    if not boards:
        print(f"No boards found in {ASSETS_DIR}")
        return

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True

    board_spacing = 1.0

    for board_idx, board_name in enumerate(boards):
        order, transforms, parts = load_board(board_name)
        if not parts:
            continue

        base_y = board_idx * board_spacing
        steps = order["steps"]
        inserts_into = order.get("inserts_into", {})
        n_steps = len(steps)

        # --- Column 1: Assembled view ---
        col1_x = 0.0
        server.scene.add_label(
            f"/labels/{board_name}/assembled",
            f"{board_name} — assembled",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([col1_x, base_y - 0.05, 0.15]),
        )
        for pidx, part in enumerate(parts):
            color = PART_COLORS[pidx % len(PART_COLORS)]
            offset = np.array([col1_x, base_y, 0.0])
            add_mesh_at(server, f"/assembled/{board_name}/{part['id']}", part["mesh"], offset, color)

        # --- Column 2: Exploded view ---
        col2_x = 0.4
        server.scene.add_label(
            f"/labels/{board_name}/exploded",
            f"{board_name} — exploded",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([col2_x, base_y - 0.05, 0.15]),
        )
        all_verts = np.concatenate([p["mesh"].vertices for p in parts])
        assembly_center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2.0

        for pidx, part in enumerate(parts):
            color = PART_COLORS[pidx % len(PART_COLORS)]
            part_center = part["centroid"]
            direction = part_center - assembly_center
            direction[2] = max(direction[2], 0.005)
            explode_offset = direction * args.explode_factor
            offset = np.array([col2_x, base_y, 0.0]) + explode_offset
            add_mesh_at(server, f"/exploded/{board_name}/{part['id']}", part["mesh"], offset, color)

        # --- Columns 3+: Cumulative fixture steps ---
        # Step 0: just the base
        # Step 1: base + insert_1
        # Step 2: base + insert_1 + insert_2
        # etc.
        col3_x = 0.9
        step_spacing = 0.35

        server.scene.add_label(
            f"/labels/{board_name}/steps",
            f"{board_name} — assembly steps (cumulative fixture)",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([col3_x, base_y - 0.05, 0.15]),
        )

        part_by_id = {p["id"]: (i, p) for i, p in enumerate(parts)}

        for step_idx in range(n_steps):
            sx = col3_x + step_idx * step_spacing
            current_pid = steps[step_idx]

            # Label
            if step_idx == 0:
                step_label = f"Base ({current_pid})"
            else:
                step_label = f"Step {step_idx}: insert {current_pid}"

            server.scene.add_label(
                f"/labels/{board_name}/step_{step_idx}",
                step_label,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.array([sx, base_y - 0.03, 0.10]),
            )

            # Show all parts up to this step as fixture (grey)
            for prev_idx in range(step_idx):
                prev_pid = steps[prev_idx]
                if prev_pid not in part_by_id:
                    continue
                _, prev_part = part_by_id[prev_pid]
                offset = np.array([sx, base_y, 0.0])
                add_mesh_at(
                    server,
                    f"/steps/{board_name}/step_{step_idx}/fixture/{prev_pid}",
                    prev_part["mesh"], offset, (160, 160, 160),
                )

            # Show the current part being inserted (colored, lifted)
            if current_pid in part_by_id:
                pidx, current_part = part_by_id[current_pid]
                color = PART_COLORS[pidx % len(PART_COLORS)]
                if step_idx == 0:
                    # Base sits in place
                    offset = np.array([sx, base_y, 0.0])
                else:
                    # Insertion piece shown lifted above its target position
                    offset = np.array([sx, base_y, 0.06])
                add_mesh_at(
                    server,
                    f"/steps/{board_name}/step_{step_idx}/active/{current_pid}",
                    current_part["mesh"], offset, color,
                )

    print(f"\nFMB assembly viewer running at http://localhost:{args.port}")
    print(f"Boards: {boards}")
    print("Layout: Assembled | Exploded | Assembly steps (cumulative)")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
