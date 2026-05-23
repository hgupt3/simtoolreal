"""Visualize the 7 pose-tracker-eval canonical OBJs in viser.

Loads the canonical meshes from `assets/urdf/pose_tracker_eval/`, places
the kuka_sharpa robot + table in their default poses, and lets the user
pick which object to display on the table via a dropdown.

Object sits with the bottom of its bounding box on the table surface.
Canonical frame is preserved (mesh local frame = canonical), so the
viewer doubles as a sanity check on the orientation each part was
authored in.

Usage:
    python pose_tracker_eval/viewer.py [--port 8083]
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

ASSETS_DIR = Path(__file__).resolve().parent  # this script sits alongside the per-part subdirs
REPO_ROOT = ASSETS_DIR.parent.parent.parent   # …/assets/urdf/pose_tracker_eval -> repo root

OBJECTS = [
    "lpeg",
    "peg",
    "fmb_peg_31",
    "fmb_peg_41",
    "fabrica_beam",
    "fabrica_pillar",
    "beam_3x_part_0",
    "beam_3x_part_2",
    "square_table_leg4",
    "square_table_leg4_125mm",
    "square_table_leg4_200mm",
]

TABLE_URDF = REPO_ROOT / "assets" / "urdf" / "table_narrow.urdf"
ROBOT_URDF = (
    REPO_ROOT
    / "assets"
    / "urdf"
    / "kuka_sharpa_description"
    / "iiwa14_left_sharpa_adjusted_restricted.urdf"
)

TABLE_FRAME_Z = 0.38
TABLE_SURFACE_Z = 0.53
ROBOT_POSITION = (0.0, 0.8, 0.0)
OBJECT_XY = (0.0, 0.0)  # XY center of the table (table frame origin)

HOME_JOINT_POS_IIWA = np.array(
    [
        -1.571,
        1.571 - np.deg2rad(10),
        0.0,
        1.376 + np.deg2rad(10),
        0.0,
        1.485,
        1.308,
    ]
)
HOME_JOINT_POS_SHARPA = np.zeros(22)
HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])


def find_free_port(start: int = 8083, end: int = 8200) -> int:
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free port in [{start}, {end}]")


def load_meshes() -> dict[str, trimesh.Trimesh]:
    meshes: dict[str, trimesh.Trimesh] = {}
    for name in OBJECTS:
        obj_path = ASSETS_DIR / name / f"{name}.obj"
        if not obj_path.exists():
            raise FileNotFoundError(obj_path)
        mesh = trimesh.load(str(obj_path), process=False, force="mesh")
        meshes[name] = mesh
    return meshes


def object_position(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    z_min = float(mesh.bounds[0, 2])
    return (OBJECT_XY[0], OBJECT_XY[1], TABLE_SURFACE_Z - z_min)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="viser port (default: auto-pick free in [8083, 8200])",
    )
    args = parser.parse_args()

    assert TABLE_URDF.exists(), f"TABLE_URDF not found: {TABLE_URDF}"
    assert ROBOT_URDF.exists(), f"ROBOT_URDF not found: {ROBOT_URDF}"

    port = args.port if args.port is not None else find_free_port()
    print(f"[viewer] starting viser on port {port}")
    server = viser.ViserServer(port=port)

    table_frame = server.scene.add_frame(
        "/table",
        show_axes=False,
        position=(0.0, 0.0, TABLE_FRAME_Z),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    ViserUrdf(server, TABLE_URDF, root_node_name="/table")

    server.scene.add_frame(
        "/robot/state",
        show_axes=False,
        position=ROBOT_POSITION,
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    robot_viser = ViserUrdf(server, ROBOT_URDF, root_node_name="/robot/state")
    robot_viser.update_cfg(HOME_JOINT_POS)

    print(f"[viewer] loading {len(OBJECTS)} OBJs from {ASSETS_DIR}")
    meshes = load_meshes()
    # Per-object: a parent frame (carries pose + axes toggle) with the
    # mesh as a child. frame.visible cascades to the mesh, so selection
    # and axes-toggle stay independent.
    frame_handles: dict[str, viser.FrameHandle] = {}
    for name, mesh in meshes.items():
        frame = server.scene.add_frame(
            f"/objects/{name}",
            show_axes=True,
            axes_length=0.08,
            axes_radius=0.003,
            position=object_position(mesh),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        server.scene.add_mesh_trimesh(f"/objects/{name}/mesh", mesh)
        frame.visible = False
        frame_handles[name] = frame
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(
            f"  {name:18s}  verts={len(mesh.vertices):6d}"
            f"  extents=[{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}] m"
        )

    initial = OBJECTS[0]
    frame_handles[initial].visible = True

    dropdown = server.gui.add_dropdown(
        "Object",
        options=OBJECTS,
        initial_value=initial,
    )
    axes_checkbox = server.gui.add_checkbox("Show object axes", initial_value=True)
    table_checkbox = server.gui.add_checkbox("Show table", initial_value=True)

    @dropdown.on_update
    def _(_event) -> None:
        for name in OBJECTS:
            frame_handles[name].visible = name == dropdown.value

    @axes_checkbox.on_update
    def _(_event) -> None:
        for name in OBJECTS:
            frame_handles[name].show_axes = axes_checkbox.value

    @table_checkbox.on_update
    def _(_event) -> None:
        table_frame.visible = table_checkbox.value

    print(f"[viewer] open http://localhost:{port}")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
