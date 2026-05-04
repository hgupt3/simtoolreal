#!/usr/bin/env python3
"""Visualize generated FMB training scenes in a viser web viewer.

Shows the fixture (table + assembled prior parts), the insertion piece at its
start pose, and the goal trajectory waypoints.

Usage:
    python -m fmb.visualize_scenes --assembly fmb_board_1
    python -m fmb.visualize_scenes --assembly fmb_board_1 --part-idx 0 --scene-idx 0
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

TABLE_Z = 0.38
TABLE_TOP_Z = 0.53

FIXTURE_COLOR = (160, 160, 160)
BASE_COLOR = (180, 170, 150)
WAYPOINT_COLOR = (50, 200, 50)
START_COLOR = (230, 51, 51)
GOAL_COLOR = (51, 77, 230)

PART_COLORS = [
    (230, 51, 51),
    (51, 179, 51),
    (51, 77, 230),
    (230, 179, 26),
    (204, 77, 204),
]


def quat_xyzw_to_matrix(q):
    return R.from_quat(q).as_matrix()


def wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]


def quat_inverse_wxyz(q):
    return [q[0], -q[1], -q[2], -q[3]]


def world_assembled_pose(transform_data, table_offset):
    centroid = np.array(transform_data["original_centroid"])
    pos = centroid + np.array(table_offset)
    a2c = transform_data["assembled_to_canonical_wxyz"]
    q_inv = quat_inverse_wxyz(a2c)
    return pos, q_inv


def apply_pose(mesh, pos, quat_xyzw):
    out = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos
    out.apply_transform(T)
    return out


def load_assembly_data(assembly):
    assembly_dir = ASSETS_DIR / assembly
    with open(assembly_dir / "assembly_order.json") as f:
        order = json.load(f)
    with open(assembly_dir / "canonical_transforms.json") as f:
        transforms = json.load(f)
    data = np.load(assembly_dir / "scenes.npz", allow_pickle=True)
    return order, transforms, data


def load_canonical_mesh(assembly, pid):
    path = ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"
    return trimesh.load(str(path), force="mesh", process=False)


def load_coacd_mesh(assembly, pid):
    """Load all CoACD decomposition hulls as a single merged mesh."""
    coacd_dir = ASSETS_DIR / assembly / pid / "coacd"
    decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
    if not decomp_files:
        return load_canonical_mesh(assembly, pid)
    meshes = [trimesh.load(str(f), force="mesh", process=False) for f in decomp_files]
    return trimesh.util.concatenate(meshes)


def load_coacd_hulls(assembly, pid):
    """Load CoACD hulls as separate meshes for per-hull coloring."""
    coacd_dir = ASSETS_DIR / assembly / pid / "coacd"
    decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
    if not decomp_files:
        return [load_canonical_mesh(assembly, pid)]
    return [trimesh.load(str(f), force="mesh", process=False) for f in decomp_files]


HULL_COLORS = [
    (230, 51, 51), (51, 179, 51), (51, 77, 230), (230, 179, 26),
    (204, 77, 204), (26, 204, 204), (230, 128, 26), (128, 128, 128),
    (153, 51, 153), (77, 230, 128), (230, 77, 128), (102, 153, 230),
    (179, 230, 51), (230, 102, 179), (51, 128, 179), (204, 153, 77),
    (128, 77, 230), (77, 204, 77), (179, 51, 102), (102, 179, 153),
]


@dataclass
class VizArgs:
    """Visualize FMB training scenes."""
    assembly: str = "fmb_board_1"
    port: int = 8082
    part_idx: int = 0
    """Initial insertion part index to show (0 = first insertion part)."""
    scene_idx: int = 0
    """Initial scene index."""
    start_idx: int = 0
    """Initial start pose index."""
    collision: bool = False
    """Show CoACD collision meshes instead of visual meshes."""


def main():
    args: VizArgs = tyro.cli(VizArgs)

    order, transforms, data = load_assembly_data(args.assembly)

    insertion_parts = [str(p) for p in data["insertion_parts"].tolist()]
    start_poses = data["start_poses"]       # (P, N, M, 7)
    goals = data["goals"]                   # (P, N, M, T, 7)
    traj_lengths = data["traj_lengths"]     # (P, N, M)
    partial_offsets = data["partial_assembly_offsets"]  # (P, N, 3)

    P, N, M, T, _ = goals.shape
    steps = order["steps"]

    canonical_meshes = {}
    coacd_hulls = {}
    for pid in steps:
        canonical_meshes[pid] = load_canonical_mesh(args.assembly, pid)
        if args.collision:
            coacd_hulls[pid] = load_coacd_hulls(args.assembly, pid)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True

    # GUI controls
    folder = server.gui.add_folder("Scene Controls")
    with folder:
        part_slider = server.gui.add_slider(
            "Part idx", min=0, max=P - 1, step=1, initial_value=min(args.part_idx, P - 1))
        scene_slider = server.gui.add_slider(
            "Scene idx", min=0, max=N - 1, step=1, initial_value=min(args.scene_idx, N - 1))
        start_slider = server.gui.add_slider(
            "Start idx", min=0, max=M - 1, step=1, initial_value=min(args.start_idx, M - 1))
        show_waypoints = server.gui.add_checkbox("Show waypoints", initial_value=True)
        show_goal = server.gui.add_checkbox("Show goal pose", initial_value=True)
        waypoint_slider = server.gui.add_slider(
            "Waypoint idx", min=-1, max=T - 1, step=1, initial_value=-1)
        server.gui.add_markdown("*Waypoint idx: -1 = show at start pose, 0..N = show at waypoint*")
        info_text = server.gui.add_markdown("Loading...")

    scene_handles = []

    def clear_scene():
        for h in scene_handles:
            h.remove()
        scene_handles.clear()

    def render_scene():
        clear_scene()

        p_idx = int(part_slider.value)
        n_idx = int(scene_slider.value)
        m_idx = int(start_slider.value)

        inserting_pid = insertion_parts[p_idx]
        traj_len = int(traj_lengths[p_idx, n_idx, m_idx])
        offset = partial_offsets[p_idx, n_idx]
        table_offset = [offset[0], offset[1], offset[2]]

        # Which parts are in the fixture (all before the inserting part in steps)
        inserting_step_idx = steps.index(inserting_pid)
        fixture_pids = steps[:inserting_step_idx]

        info_text.content = (
            f"**{args.assembly}** | Inserting: **{inserting_pid}** (part {p_idx})\n\n"
            f"Scene {n_idx}, Start {m_idx}, Traj len: {traj_len}\n\n"
            f"Fixture: {fixture_pids}\n\n"
            f"Table offset: [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]"
        )

        # --- Table box (0.475 x 0.4 x 0.3m, centered at TABLE_Z) ---
        table_mesh = trimesh.creation.box(extents=[0.475, 0.4, 0.3])
        table_mesh.vertices += np.array([0, 0, TABLE_Z])
        h = server.scene.add_mesh_simple(
            "/scene/table",
            vertices=table_mesh.vertices.astype(np.float32),
            faces=table_mesh.faces.astype(np.int32),
            color=(209, 143, 89), flat_shading=True)
        scene_handles.append(h)

        # --- Fixture parts (assembled prior parts) ---
        hull_color_idx = 0
        for fix_idx, pid in enumerate(fixture_pids):
            pos, quat_wxyz = world_assembled_pose(transforms[pid], table_offset)
            quat_xyzw = wxyz_to_xyzw(quat_wxyz)

            if args.collision and pid in coacd_hulls:
                for hi, hull in enumerate(coacd_hulls[pid]):
                    world_hull = apply_pose(hull, pos, quat_xyzw)
                    color = HULL_COLORS[hull_color_idx % len(HULL_COLORS)]
                    h = server.scene.add_mesh_simple(
                        f"/scene/fixture/{pid}/hull_{hi}",
                        vertices=world_hull.vertices.astype(np.float32),
                        faces=world_hull.faces.astype(np.int32),
                        color=color, flat_shading=True)
                    scene_handles.append(h)
                    hull_color_idx += 1
            else:
                world_mesh = apply_pose(canonical_meshes[pid], pos, quat_xyzw)
                color = BASE_COLOR if pid == steps[0] else FIXTURE_COLOR
                h = server.scene.add_mesh_simple(
                    f"/scene/fixture/{pid}",
                    vertices=world_mesh.vertices.astype(np.float32),
                    faces=world_mesh.faces.astype(np.int32),
                    color=color, flat_shading=True)
                scene_handles.append(h)

        # --- Insertion piece at start pose or selected waypoint ---
        sp = start_poses[p_idx, n_idx, m_idx]  # (7,) xyz + xyzw quat
        wp_idx = int(waypoint_slider.value)

        if wp_idx < 0 or wp_idx >= traj_len:
            active_pos = sp[:3]
            active_quat_xyzw = sp[3:7]
            active_label = "START"
        else:
            wp = goals[p_idx, n_idx, m_idx, wp_idx]
            active_pos = wp[:3]
            active_quat_xyzw = wp[3:7]
            active_label = f"WP {wp_idx}/{traj_len-1}"

        if args.collision and inserting_pid in coacd_hulls:
            for hi, hull in enumerate(coacd_hulls[inserting_pid]):
                world_hull = apply_pose(hull, active_pos, active_quat_xyzw)
                color = HULL_COLORS[(hull_color_idx + hi) % len(HULL_COLORS)]
                h = server.scene.add_mesh_simple(
                    f"/scene/active/start/hull_{hi}",
                    vertices=world_hull.vertices.astype(np.float32),
                    faces=world_hull.faces.astype(np.int32),
                    color=color, flat_shading=True)
                scene_handles.append(h)
        else:
            active_mesh = apply_pose(canonical_meshes[inserting_pid], active_pos, active_quat_xyzw)
            h = server.scene.add_mesh_simple(
                "/scene/active/start",
                vertices=active_mesh.vertices.astype(np.float32),
                faces=active_mesh.faces.astype(np.int32),
                color=START_COLOR, flat_shading=True)
            scene_handles.append(h)

        h = server.scene.add_label(
            "/scene/labels/active", active_label,
            wxyz=np.array([1., 0., 0., 0.]),
            position=active_pos + np.array([0, 0, 0.05]))
        scene_handles.append(h)

        # --- Goal pose (final assembled position) ---
        if show_goal.value and traj_len > 0:
            gp = goals[p_idx, n_idx, m_idx, traj_len - 1]
            goal_pos = gp[:3]
            goal_quat_xyzw = gp[3:7]

            if args.collision and inserting_pid in coacd_hulls:
                for hi, hull in enumerate(coacd_hulls[inserting_pid]):
                    world_hull = apply_pose(hull, goal_pos, goal_quat_xyzw)
                    h = server.scene.add_mesh_simple(
                        f"/scene/active/goal/hull_{hi}",
                        vertices=world_hull.vertices.astype(np.float32),
                        faces=world_hull.faces.astype(np.int32),
                        color=GOAL_COLOR, flat_shading=True, opacity=0.5)
                    scene_handles.append(h)
            else:
                goal_mesh = apply_pose(canonical_meshes[inserting_pid], goal_pos, goal_quat_xyzw)
                h = server.scene.add_mesh_simple(
                    "/scene/active/goal",
                    vertices=goal_mesh.vertices.astype(np.float32),
                    faces=goal_mesh.faces.astype(np.int32),
                    color=GOAL_COLOR, flat_shading=True, opacity=0.5)
                scene_handles.append(h)

            h = server.scene.add_label(
                "/scene/labels/goal", "GOAL",
                wxyz=np.array([1., 0., 0., 0.]),
                position=goal_pos + np.array([0, 0, 0.05]))
            scene_handles.append(h)

        # --- Trajectory waypoints ---
        if show_waypoints.value and traj_len > 1:
            waypoint_positions = goals[p_idx, n_idx, m_idx, :traj_len, :3]
            for w_idx in range(traj_len):
                wp = waypoint_positions[w_idx]
                t = w_idx / max(traj_len - 1, 1)
                r = int(START_COLOR[0] * (1 - t) + GOAL_COLOR[0] * t)
                g = int(START_COLOR[1] * (1 - t) + GOAL_COLOR[1] * t)
                b = int(START_COLOR[2] * (1 - t) + GOAL_COLOR[2] * t)

                h = server.scene.add_icosphere(
                    f"/scene/waypoints/wp_{w_idx}",
                    radius=0.003,
                    color=(r, g, b),
                    position=wp.astype(np.float32))
                scene_handles.append(h)

    render_scene()

    @part_slider.on_update
    def _on_part(event):
        render_scene()

    @scene_slider.on_update
    def _on_scene(event):
        render_scene()

    @start_slider.on_update
    def _on_start(event):
        render_scene()

    @show_waypoints.on_update
    def _on_wp(event):
        render_scene()

    @show_goal.on_update
    def _on_goal(event):
        render_scene()

    @waypoint_slider.on_update
    def _on_waypoint(event):
        render_scene()

    print(f"\nFMB scene viewer running at http://localhost:{args.port}")
    print(f"Assembly: {args.assembly}")
    print(f"P={P} parts, N={N} scenes, M={M} starts, T={T} max waypoints")
    print("Use sliders to browse scenes. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
