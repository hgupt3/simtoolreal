#!/usr/bin/env python3
"""Viser viewer for single-insertion scenes from ``scenes.npz``.

Loads ``assets/urdf/fabrica/{assembly}/scenes.npz`` (the new single-insertion
schema with axes ``(P, N, M, ...)``) and lets you scroll through every cached
``(insertion_part, scene_idx, start_idx)`` combo.

For the chosen combo, the viewer renders:
  - the welded **partial-assembly fixture** at the scene's
    ``partial_assembly_offset`` (parts strictly before ``insertion_part`` in
    ``assembly_order.steps``);
  - the **inserting part** at the chosen start pose, animated through its
    cached trajectory to the final assembled position via Step Forward / Play.

Use this to hand-validate generated scenes before they're consumed by training.

Usage:
    python -m fabrica.scene_generation.visualize_scenes --assembly beam_2x
    python -m fabrica.scene_generation.visualize_scenes --assembly beam_2x \\
        --scenes-file scenes_smoke.npz --port 8086
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from fabrica.viser_utils import COLORS, SceneManager

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

# Scene constants (must match generate_scenes.py / step3 for visual alignment).
TABLE_Z = 0.38             # world center z of the table box
TABLE_DIM = (0.475, 0.4, 0.3)


def _qxyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _quat_inverse_wxyz(q):
    return [q[0], -q[1], -q[2], -q[3]]


def _slerp(q0, q1, t):
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta = np.arccos(min(dot, 1.0))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def _interp_waypoints(wps, t):
    """Smoothstep-interpolated walk over a list of (pos, wxyz_quat) waypoints."""
    n = len(wps) - 1
    if n <= 0 or t <= 0:
        return wps[0]
    if t >= 1:
        return wps[-1]
    sf = t * n
    si = min(int(sf), n - 1)
    lt = sf - si
    a = lt * lt * (3.0 - 2.0 * lt)
    pos = (1 - a) * wps[si][0] + a * wps[si + 1][0]
    quat = _slerp(wps[si][1], wps[si + 1][1], a)
    return pos, quat


def load_assembly_meta(assembly):
    """Return (assembly_order_steps, inserts_into, canonical_transforms)."""
    with open(ASSETS_DIR / assembly / "assembly_order.json") as f:
        order = json.load(f)
    with open(ASSETS_DIR / assembly / "canonical_transforms.json") as f:
        ct = json.load(f)
    return order["steps"], order.get("inserts_into", {}), ct


def load_canonical_meshes(assembly, part_ids):
    meshes = {}
    for pid in part_ids:
        path = ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"
        meshes[pid] = trimesh.load_mesh(str(path), process=False)
    return meshes


def load_scenes(assembly, scenes_file="scenes.npz"):
    path = ASSETS_DIR / assembly / scenes_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run generate_scenes.py --assembly {assembly} first."
        )
    data = np.load(path, allow_pickle=True)
    return {
        "insertion_parts": [str(p) for p in data["insertion_parts"].tolist()],
        "start_poses": data["start_poses"],                 # (P, N, M, 7) xyzw
        "goals": data["goals"],                              # (P, N, M, T, 7) xyzw
        "traj_lengths": data["traj_lengths"],                # (P, N, M)
        "partial_assembly_offsets": data["partial_assembly_offsets"],  # (P, N, 3)
        "scene_urdf_paths": data["scene_urdf_paths"],        # (P, N) of str
    }


def fixture_world_pose(transforms_pid, table_offset_world):
    """World (pos, wxyz_quat) for a fixture part at the scene's table offset.

    Mirrors ``world_assembled_pose`` in generate_scenes.py — pos = canonical
    centroid + offset, quat = inverse of assembled_to_canonical.
    """
    centroid = np.array(transforms_pid["original_centroid"])
    pos = centroid + np.array(table_offset_world)
    a2c = transforms_pid["assembled_to_canonical_wxyz"]
    quat_wxyz = np.array(_quat_inverse_wxyz(a2c))
    return pos, quat_wxyz


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--assembly", type=str, required=True)
    parser.add_argument("--scenes-file", type=str, default="scenes.npz",
                        help="scenes npz filename inside the assembly dir")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    assembly_order, inserts_into, transforms = load_assembly_meta(args.assembly)
    canonical_meshes = load_canonical_meshes(args.assembly, assembly_order)
    scenes = load_scenes(args.assembly, args.scenes_file)

    insertion_parts = scenes["insertion_parts"]
    start_poses = scenes["start_poses"]                  # (P, N, M, 7)
    goals = scenes["goals"]                              # (P, N, M, T, 7)
    traj_lengths = scenes["traj_lengths"]                # (P, N, M)
    partial_offsets = scenes["partial_assembly_offsets"]  # (P, N, 3)
    P, N, M, _ = start_poses.shape
    print(f"Loaded {args.scenes_file} for {args.assembly}: "
          f"P={P} insertion_parts={insertion_parts}, N={N} scenes, M={M} starts")

    # Pre-build per-(p_idx, n, m) waypoint sequences (start + cached goals).
    # Stored as list of (pos, wxyz_quat) tuples.
    def waypoints_for(p_idx, n, m):
        wps = []
        sp = start_poses[p_idx, n, m]
        wps.append((np.array(sp[:3]), _qxyzw_to_wxyz(sp[3:7])))
        L = int(traj_lengths[p_idx, n, m])
        for k in range(L):
            gp = goals[p_idx, n, m, k]
            wps.append((np.array(gp[:3]), _qxyzw_to_wxyz(gp[3:7])))
        return wps

    # ── Viser setup ──
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    scene_mgr = SceneManager()       # for fixture meshes (rebuilt per scene)
    insert_mgr = SceneManager()      # for inserting part frame (rebuilt per part)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, -1.0, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.5)

    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_frame(
        "/table", position=(0.0, 0.0, TABLE_Z), wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=False,
    )
    server.scene.add_box(
        "/table/wood",
        color=(180, 130, 70),
        dimensions=TABLE_DIM,
        position=(0.0, 0.0, 0.0),
        side="double",
        opacity=0.7,
    )
    server.scene.add_frame(
        "/robot", position=(0.0, 0.8, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=False,
    )
    robot_urdf = (
        REPO_ROOT
        / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    if robot_urdf.exists():
        ViserUrdf(server, robot_urdf, root_node_name="/robot")

    # ── Mutable rendering state ──
    state = dict(
        p_idx=0,                # index into insertion_parts
        n=0,
        m=0,
        animating=False,
        anim_t=0.0,             # current animation parameter [0, 1]
        insert_frame=None,      # viser frame handle for the inserting part
    )

    def _set_insert_pose(pos, wxyz):
        f = state["insert_frame"]
        if f is None:
            return
        f.position = tuple(float(v) for v in pos)
        f.wxyz = tuple(float(v) for v in wxyz)

    def render_scene():
        """Rebuild the fixture meshes + inserting-part frame for the current
        (insertion_part, scene_idx, start_idx) selection."""
        scene_mgr.clear()
        insert_mgr.clear()

        p_idx = state["p_idx"]
        n = state["n"]
        m = state["m"]
        p = insertion_parts[p_idx]
        wps = waypoints_for(p_idx, n, m)

        # Fixture parts = everything strictly before `p` in assembly_order.
        i_in_order = assembly_order.index(p)
        fixture_pids = assembly_order[:i_in_order]
        receiver_pid = inserts_into.get(p, "")
        offset = partial_offsets[p_idx, n]

        # Render each fixture part at its world_assembled_pose for this scene.
        for fpid in fixture_pids:
            mesh = canonical_meshes[fpid]
            pos, q_wxyz = fixture_world_pose(transforms[fpid], offset)
            color = (240, 90, 90) if fpid == receiver_pid else (160, 160, 160)
            f = server.scene.add_frame(
                f"/fixture/{fpid}",
                position=tuple(float(v) for v in pos),
                wxyz=tuple(float(v) for v in q_wxyz),
                show_axes=False,
            )
            scene_mgr.add(f)
            scene_mgr.add(server.scene.add_mesh_simple(
                f"/fixture/{fpid}/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=color,
                opacity=0.95,
            ))

        # Inserting part: per-instance frame + mesh, controlled by animation.
        mesh = canonical_meshes[p]
        f = server.scene.add_frame(
            "/insert",
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            show_axes=True, axes_length=0.05, axes_radius=0.0008,
        )
        insert_mgr.add(f)
        insert_mgr.add(server.scene.add_mesh_simple(
            "/insert/mesh",
            vertices=np.array(mesh.vertices, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=COLORS[p_idx % len(COLORS)],
        ))
        state["insert_frame"] = f

        # Goal ghost — semi-transparent mesh at the user-selected trajectory
        # waypoint (sl_goal). Range 0..L-1 where L = traj length for this
        # (p, n, m). Slider default is the final waypoint.
        L = int(traj_lengths[p_idx, n, m])
        goal_idx = int(min(max(0, sl_goal.value), L - 1))
        state["goal_idx"] = goal_idx
        gp_xyzw = goals[p_idx, n, m, goal_idx]                      # (7,) xyzw
        gpos = gp_xyzw[:3]
        gquat = _qxyzw_to_wxyz(gp_xyzw[3:7])
        gf = server.scene.add_frame(
            "/goal_ghost",
            position=tuple(float(v) for v in gpos),
            wxyz=tuple(float(v) for v in gquat),
            show_axes=False,
        )
        insert_mgr.add(gf)
        insert_mgr.add(server.scene.add_mesh_simple(
            "/goal_ghost/mesh",
            vertices=np.array(mesh.vertices, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=(80, 220, 80),
            opacity=0.35,
        ))

        # Reset animation to the start pose.
        state["anim_t"] = 0.0
        sp_pos, sp_q = wps[0]
        _set_insert_pose(sp_pos, sp_q)
        _update_status()

    def _update_status():
        p_idx = state["p_idx"]
        n = state["n"]
        m = state["m"]
        p = insertion_parts[p_idx]
        i_in_order = assembly_order.index(p)
        fixture_pids = assembly_order[:i_in_order]
        receiver = inserts_into.get(p, "")
        offset = partial_offsets[p_idx, n]
        L = int(traj_lengths[p_idx, n, m])
        status_panel.content = (
            f"## insertion_part `{p}` (idx {p_idx}/{P-1})\n"
            f"### scene `{n}/{N-1}` · start `{m}/{M-1}`\n"
            f"- fixture: `{fixture_pids}` (receiver `{receiver}`)\n"
            f"- partial_assembly_offset: "
            f"`({offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f})`\n"
            f"- trajectory length: `{L}` · ghost waypoint: "
            f"`{state.get('goal_idx', L - 1)}/{L - 1}`\n"
            f"- anim_t: `{state['anim_t']:.2f}`"
        )

    def step_fwd():
        if state["animating"]:
            return
        state["animating"] = True
        wps = waypoints_for(state["p_idx"], state["n"], state["m"])
        nf = max(1, int(dur_slider.value * 30))
        t0 = state["anim_t"]
        t1 = min(1.0, t0 + 1.0 / max(1, len(wps) - 1))  # advance by ~1 waypoint
        for fi in range(nf + 1):
            t = t0 + (t1 - t0) * fi / nf
            pos, q = _interp_waypoints(wps, t)
            _set_insert_pose(pos, q)
            time.sleep(1.0 / 30)
        state["anim_t"] = t1
        _update_status()
        state["animating"] = False

    def step_back():
        if state["animating"]:
            return
        state["animating"] = True
        wps = waypoints_for(state["p_idx"], state["n"], state["m"])
        nf = max(1, int(dur_slider.value * 30))
        t0 = state["anim_t"]
        t1 = max(0.0, t0 - 1.0 / max(1, len(wps) - 1))
        for fi in range(nf + 1):
            t = t0 + (t1 - t0) * fi / nf
            pos, q = _interp_waypoints(wps, t)
            _set_insert_pose(pos, q)
            time.sleep(1.0 / 30)
        state["anim_t"] = t1
        _update_status()
        state["animating"] = False

    def play_all():
        if state["animating"]:
            return
        state["animating"] = True
        wps = waypoints_for(state["p_idx"], state["n"], state["m"])
        total_frames = max(2, int(dur_slider.value * 30))
        t0 = state["anim_t"]
        for fi in range(total_frames + 1):
            t = t0 + (1.0 - t0) * fi / total_frames
            pos, q = _interp_waypoints(wps, t)
            _set_insert_pose(pos, q)
            time.sleep(1.0 / 30)
        state["anim_t"] = 1.0
        _update_status()
        state["animating"] = False

    def reset_anim():
        if state["animating"]:
            return
        wps = waypoints_for(state["p_idx"], state["n"], state["m"])
        sp_pos, sp_q = wps[0]
        _set_insert_pose(sp_pos, sp_q)
        state["anim_t"] = 0.0
        _update_status()

    # ── GUI ──
    dd_part = server.gui.add_dropdown(
        "Insertion part", options=insertion_parts, initial_value=insertion_parts[0],
    )
    sl_scene = server.gui.add_slider("Scene idx", 0, max(0, N - 1), 1, 0)
    sl_start = server.gui.add_slider("Start idx", 0, max(0, M - 1), 1, 0)
    # Goal idx within the trajectory for THIS (part, scene, start): 0 is the
    # first waypoint (lift), -1 (max) is the final assembled pose. The ghost
    # mesh jumps to the selected waypoint so you can inspect the cached
    # trajectory one step at a time. Range is clamped per (p, n, m) at render.
    T_max = int(goals.shape[3])
    sl_goal = server.gui.add_slider(
        "Goal idx (ghost)", 0, max(0, T_max - 1), 1, max(0, T_max - 1),
    )
    dur_slider = server.gui.add_slider(
        "Duration (s)", 0.2, 5.0, 0.1, 1.5,
    )
    status_panel = server.gui.add_markdown("")
    server.gui.add_button("Play (start → goal)").on_click(lambda _: play_all())
    server.gui.add_button("Step Forward").on_click(lambda _: step_fwd())
    server.gui.add_button("Step Back").on_click(lambda _: step_back())
    server.gui.add_button("Reset to start").on_click(lambda _: reset_anim())

    def _on_part(_):
        state["p_idx"] = insertion_parts.index(dd_part.value)
        render_scene()

    def _on_scene(_):
        state["n"] = int(sl_scene.value)
        render_scene()

    def _on_start(_):
        state["m"] = int(sl_start.value)
        render_scene()

    def _on_goal(_):
        render_scene()

    dd_part.on_update(_on_part)
    sl_scene.on_update(_on_scene)
    sl_start.on_update(_on_start)
    sl_goal.on_update(_on_goal)

    render_scene()

    print(f"\nOpen http://localhost:{args.port}")
    print("Pick (insertion part, scene, start) — Play animates the inserting part.")
    print("Ctrl+C to quit.\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
