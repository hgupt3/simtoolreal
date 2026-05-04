#!/usr/bin/env python3
"""Interactive viser UI to define/edit assembly_order.json.

Define assembly step order, inserts_into mapping, and insertion directions,
then animate the sequential insertions to validate before saving.

Usage:
    python fabrica/benchmark_processing/step1_define_assembly_order.py --assembly beam
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser

from fabrica.viser_utils import COLORS, load_assembly_parts

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

# Scene constants (match eval.py)
TABLE_Z = 0.38
TABLE_SURFACE_Z = TABLE_Z + 0.15
ASSEMBLY_XY = np.array([-0.08, 0.04])

DIRECTION_PRESETS = {
    "top-down [0,0,-1]": [0, 0, -1],
    "bottom-up [0,0,1]": [0, 0, 1],
    "left [1,0,0]": [1, 0, 0],
    "right [-1,0,0]": [-1, 0, 0],
    "front [0,1,0]": [0, 1, 0],
    "back [0,-1,0]": [0, -1, 0],
}


def main():
    parser = argparse.ArgumentParser(description="Define assembly_order.json interactively")
    parser.add_argument("--assembly", required=True)
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    assembly = args.assembly
    parts = load_assembly_parts(ASSETS_DIR, assembly)
    if not parts:
        print(f"No parts found for '{assembly}' in {ASSETS_DIR}")
        return
    part_ids = [pid for pid, _ in parts]
    meshes = {pid: mesh for pid, mesh in parts}
    print(f"Assembly '{assembly}': {len(parts)} parts {part_ids}")

    # Compute table offset to place assembly on table surface
    all_centroids = np.array([meshes[pid].centroid for pid in part_ids])
    overall_centroid_xy = all_centroids[:, :2].mean(axis=0)
    min_z = min(meshes[pid].bounds[0][2] for pid in part_ids)
    table_offset = np.array([
        ASSEMBLY_XY[0] - overall_centroid_xy[0],
        ASSEMBLY_XY[1] - overall_centroid_xy[1],
        TABLE_SURFACE_Z - min_z,
    ])

    # Load existing config if present
    order_path = ASSETS_DIR / assembly / "assembly_order.json"
    if order_path.exists():
        data = json.loads(order_path.read_text())
        steps = data.get("steps", [])
        steps = [s for s in steps if s in part_ids]
        inserts_into = data.get("inserts_into", {})
        insertion_directions = data.get("insertion_directions", {})
    else:
        steps = []
        inserts_into = {}
        insertion_directions = {}

    # Viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    frame_handles = {}
    is_animating = False

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.3, -0.5, 0.9)
        client.camera.look_at = (0.0, 0.0, 0.55)

    # Scene
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_box("/table", color=(180, 130, 70), dimensions=(0.475, 0.4, 0.3),
                         position=(0, 0, TABLE_Z), side="double", opacity=0.7)

    # Robot
    from viser.extras import ViserUrdf
    server.scene.add_frame("/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False)
    robot_urdf = REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"
    if robot_urdf.exists():
        ViserUrdf(server, robot_urdf, root_node_name="/robot")

    # Assembly group frame at table offset
    assembly_frame = server.scene.add_frame("/assembly", wxyz=(1, 0, 0, 0),
                                            position=tuple(table_offset), show_axes=False)

    # Render parts in assembly (on table)
    for i, (pid, mesh) in enumerate(parts):
        f = server.scene.add_frame(f"/assembly/{pid}", wxyz=(1, 0, 0, 0),
                                   position=(0, 0, 0), show_axes=False)
        frame_handles[pid] = f
        server.scene.add_mesh_simple(
            f"/assembly/{pid}/mesh",
            vertices=np.array(mesh.vertices, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=COLORS[i % len(COLORS)],
        )

    # Parts shelf: individual copies laid out in a row behind the table
    shelf_y = -0.2
    shelf_spacing = 0.15
    shelf_start_x = -(len(part_ids) - 1) * shelf_spacing / 2
    for i, (pid, mesh) in enumerate(parts):
        # Center each part at origin of its shelf slot
        centered_verts = np.array(mesh.vertices, dtype=np.float32) - mesh.centroid.astype(np.float32)
        sx = shelf_start_x + i * shelf_spacing
        server.scene.add_mesh_simple(
            f"/shelf/{pid}/mesh",
            vertices=centered_verts,
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=COLORS[i % len(COLORS)],
            position=(sx, shelf_y, TABLE_SURFACE_Z + 0.05),
        )
        server.scene.add_label(
            f"/shelf/{pid}/label",
            text=f"Part {pid}",
            position=(sx, shelf_y, TABLE_SURFACE_Z + 0.12),
        )

    def get_insertion_dir(pid):
        if pid in insertion_directions:
            return np.array(insertion_directions[pid], dtype=float)
        return np.array([0, 0, -1], dtype=float)

    def animate_insertions():
        nonlocal is_animating
        if is_animating or not steps:
            return
        is_animating = True
        approach_dist = 0.15

        # Pull all parts back
        for pid in steps:
            d = get_insertion_dir(pid)
            frame_handles[pid].position = tuple(-d * approach_dist)
        # Hide parts not in steps
        for pid in part_ids:
            if pid not in steps:
                frame_handles[pid].position = (0, 0, -10)  # hide off-screen

        time.sleep(0.5)

        for step_i, pid in enumerate(steps):
            status_text.content = f"**Inserting part {pid}** (step {step_i + 1}/{len(steps)})"
            d = get_insertion_dir(pid)
            start_off = -d * approach_dist

            for fi in range(31):
                t = fi / 30
                a = t * t * (3.0 - 2.0 * t)
                frame_handles[pid].position = tuple(start_off * (1.0 - a))
                time.sleep(1.0 / 30)

            frame_handles[pid].position = (0, 0, 0)
            time.sleep(0.3)

        status_text.content = "**Assembly complete**"
        is_animating = False

    def reset_positions():
        if is_animating:
            return
        for pid in part_ids:
            frame_handles[pid].position = (0, 0, 0)
        status_text.content = ""

    # --- GUI ---
    status_text = server.gui.add_markdown("")

    # Order editing
    with server.gui.add_folder("Assembly Order"):
        order_text = server.gui.add_markdown(f"**{' -> '.join(steps) or '(empty)'}**")
        part_dd = server.gui.add_dropdown("Part", options=part_ids, initial_value=part_ids[0])
        add_btn = server.gui.add_button("Append")
        remove_btn = server.gui.add_button("Remove")
        move_up = server.gui.add_button("Move Earlier")
        move_down = server.gui.add_button("Move Later")

        def refresh():
            order_text.content = f"**{' -> '.join(steps) or '(empty)'}**"

        @add_btn.on_click
        def _(_):
            pid = part_dd.value
            if pid not in steps:
                steps.append(pid)
                refresh()

        @remove_btn.on_click
        def _(_):
            pid = part_dd.value
            if pid in steps:
                steps.remove(pid)
                refresh()

        @move_up.on_click
        def _(_):
            pid = part_dd.value
            if pid in steps:
                idx = steps.index(pid)
                if idx > 0:
                    steps[idx], steps[idx - 1] = steps[idx - 1], steps[idx]
                    refresh()

        @move_down.on_click
        def _(_):
            pid = part_dd.value
            if pid in steps:
                idx = steps.index(pid)
                if idx < len(steps) - 1:
                    steps[idx], steps[idx + 1] = steps[idx + 1], steps[idx]
                    refresh()

    # Inserts-into
    with server.gui.add_folder("Inserts Into"):
        for pid in part_ids:
            opts = ["none"] + [p for p in part_ids if p != pid]
            initial = inserts_into.get(pid, "none")
            if initial not in opts:
                initial = "none"
            dd = server.gui.add_dropdown(f"{pid} ->", options=opts, initial_value=initial)

            def make_cb(p, d):
                @d.on_update
                def _(_):
                    if d.value == "none":
                        inserts_into.pop(p, None)
                    else:
                        inserts_into[p] = d.value
            make_cb(pid, dd)

    # Insertion directions
    with server.gui.add_folder("Insertion Directions"):
        preset_names = list(DIRECTION_PRESETS.keys())
        for pid in part_ids:
            existing = insertion_directions.get(pid)
            initial = "top-down [0,0,-1]"
            if existing is not None:
                for name, vec in DIRECTION_PRESETS.items():
                    if list(existing) == vec:
                        initial = name
                        break
            dd = server.gui.add_dropdown(f"{pid} dir", options=preset_names, initial_value=initial)

            def make_cb(p, d):
                @d.on_update
                def _(_):
                    vec = DIRECTION_PRESETS[d.value]
                    if vec == [0, 0, -1]:
                        insertion_directions.pop(p, None)
                    else:
                        insertion_directions[p] = vec
            make_cb(pid, dd)

    # Animate + Save
    server.gui.add_button("Animate Insertions").on_click(lambda _: animate_insertions())
    server.gui.add_button("Reset View").on_click(lambda _: reset_positions())

    with server.gui.add_folder("Save"):
        save_status = server.gui.add_markdown("")
        save_btn = server.gui.add_button("Save assembly_order.json")

        @save_btn.on_click
        def _(_):
            out = {"steps": list(steps)}
            clean = {k: v for k, v in inserts_into.items() if v != "none"}
            if clean:
                out["inserts_into"] = clean
            if insertion_directions:
                out["insertion_directions"] = insertion_directions
            order_path.parent.mkdir(parents=True, exist_ok=True)
            order_path.write_text(json.dumps(out, indent=2) + "\n")
            save_status.content = f"Saved to `{order_path}`"
            print(f"Saved: {order_path}")

    print(f"\nOpen http://localhost:{args.port}")
    print(f"Order: {' -> '.join(steps) or '(empty)'}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
