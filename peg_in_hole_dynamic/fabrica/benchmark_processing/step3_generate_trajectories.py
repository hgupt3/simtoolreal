#!/usr/bin/env python3
"""Generate pick_place.json trajectories for each assembly step.

For each part, generates a crane-style trajectory:
  1. Lift: straight up from start pose to clearance height
  2. Transit: horizontal move + rotation interpolation (slerp)
  3. Approach: translate along insertion direction to goal

Usage:
    python fabrica/benchmark_processing/step3_generate_trajectories.py --assembly beam
    python fabrica/benchmark_processing/step3_generate_trajectories.py --assembly beam --part 2
    python fabrica/benchmark_processing/step3_generate_trajectories.py --viz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

# Scene constants
TABLE_Z = 0.38
TABLE_SURFACE_Z = TABLE_Z + 0.15
CLEARANCE_Z = TABLE_SURFACE_Z + 0.15
START_XY = np.array([0.135, 0.03])
ASSEMBLY_XY = np.array([-0.08, 0.04])
NUM_WAYPOINTS = 12
LIFT_FRAC, TRANSIT_FRAC = 0.25, 0.50


# --- Quaternion helpers (all wxyz) ---

def quat_wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]

def quat_xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]

def quat_inverse_wxyz(q):
    return [q[0], -q[1], -q[2], -q[3]]

def slerp_wxyz(q0, q1, t):
    r0 = Rotation.from_quat(quat_wxyz_to_xyzw(q0))
    r1 = Rotation.from_quat(quat_wxyz_to_xyzw(q1))
    result = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))(t)
    return quat_xyzw_to_wxyz(result.as_quat().tolist())


# --- Config loading ---

def load_assembly_config(assembly):
    with open(ASSETS_DIR / assembly / "canonical_transforms.json") as f:
        transforms = json.load(f)
    with open(ASSETS_DIR / assembly / "assembly_order.json") as f:
        order = json.load(f)
    return transforms, order


# --- Start pose layout ---

def compute_start_poses(canonical_meshes, assembly_order, assembled_y):
    """Bin-pack parts into rows, center each row in X, center all rows in Y around assembled_y."""
    gap_x, gap_y = 0.03, 0.025
    max_row_width = 0.20

    extents = {pid: canonical_meshes[pid].bounding_box.extents for pid in assembly_order}

    # Pack into rows by X width
    rows = []
    row, row_w = [], 0.0
    for pid in assembly_order:
        w = extents[pid][0]
        needed = w + (gap_x if row else 0)
        if row and row_w + needed > max_row_width:
            rows.append(row)
            row, row_w = [pid], w
        else:
            row.append(pid)
            row_w += needed
    if row:
        rows.append(row)

    # Y positions: center all rows around assembled_y
    row_h = [max(extents[pid][1] for pid in r) for r in rows]
    total_y = sum(row_h) + gap_y * (len(rows) - 1)
    cursor_y = assembled_y + total_y / 2.0 - row_h[0] / 2.0
    row_y = []
    for r in range(len(rows)):
        row_y.append(cursor_y)
        if r + 1 < len(rows):
            cursor_y -= row_h[r] / 2.0 + gap_y + row_h[r + 1] / 2.0

    # Place parts
    start_poses = {}
    for r, row_pids in enumerate(rows):
        widths = [extents[pid][0] for pid in row_pids]
        total_w = sum(widths) + gap_x * (len(row_pids) - 1)
        cx = START_XY[0] - total_w / 2.0
        for pid, w in zip(row_pids, widths):
            z = TABLE_SURFACE_Z + extents[pid][2] / 2.0
            start_poses[pid] = np.array([cx + w / 2.0, row_y[r], z])
            cx += w + gap_x
    return start_poses


# --- Trajectory generation ---

def compute_table_offset(assembly, transforms):
    """Compute offset to place assembled parts on table surface.

    Uses actual assembled OBJ bounds to find the true min Z, rather than
    canonical extents (which may not correspond to the Z axis).
    """
    centroids = np.array([transforms[pid]["original_centroid"] for pid in transforms])

    # Find the actual min Z across all parts in assembled frame
    min_z = float("inf")
    for pid in transforms:
        mesh_path = ASSETS_DIR / assembly / pid / f"{pid}.obj"
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        min_z = min(min_z, mesh.bounds[0][2])

    xy_offset = ASSEMBLY_XY - centroids[:, :2].mean(axis=0)
    return np.array([xy_offset[0], xy_offset[1], TABLE_SURFACE_Z - min_z])


def generate_trajectory(start_pos, start_quat, end_pos, end_quat, insertion_dir=None):
    """Generate crane-style trajectory: lift → transit → approach.

    For horizontal insertions (insertion_dir nearly perpendicular to Z),
    adds a descent phase so transit stays at clearance height:
      lift → transit (at clearance) → descent → horizontal approach
    """
    if insertion_dir is None:
        insertion_dir = np.array([0.0, 0.0, -1.0])
    else:
        insertion_dir = np.array(insertion_dir, dtype=float)
        insertion_dir /= np.linalg.norm(insertion_dir)

    horizontal_insertion = abs(insertion_dir[2]) <= 0.01

    n_lift = max(1, int(NUM_WAYPOINTS * LIFT_FRAC))
    n_transit = max(1, int(NUM_WAYPOINTS * TRANSIT_FRAC))
    n_approach = NUM_WAYPOINTS - n_lift - n_transit

    # Approach start: back along insertion dir from end
    if not horizontal_insertion:
        approach_dist = abs((CLEARANCE_Z - end_pos[2]) / insertion_dir[2])
    else:
        approach_dist = CLEARANCE_Z - end_pos[2]
    approach_start = end_pos - insertion_dir * approach_dist

    waypoints = []
    lift_target = np.array([start_pos[0], start_pos[1], CLEARANCE_Z])

    # Lift
    for i in range(1, n_lift + 1):
        t = i / n_lift
        waypoints.append((start_pos + t * (lift_target - start_pos), start_quat))

    if horizontal_insertion:
        # Transit stays at clearance height
        transit_target = np.array([approach_start[0], approach_start[1], CLEARANCE_Z])
        for i in range(1, n_transit + 1):
            t = i / n_transit
            pos = lift_target + t * (transit_target - lift_target)
            quat = slerp_wxyz(start_quat, end_quat, t)
            waypoints.append((pos, quat))

        # Descent from clearance to approach start Z
        n_descent = max(1, n_approach // 2)
        n_horiz = n_approach - n_descent
        for i in range(1, n_descent + 1):
            t = i / n_descent
            pos = transit_target + t * (approach_start - transit_target)
            waypoints.append((pos, end_quat))

        # Horizontal approach along insertion dir
        for i in range(1, n_horiz + 1):
            t = i / n_horiz
            waypoints.append((approach_start + t * (end_pos - approach_start), end_quat))
    else:
        # Transit (may descend for angled/vertical insertions)
        for i in range(1, n_transit + 1):
            t = i / n_transit
            pos = lift_target + t * (approach_start - lift_target)
            quat = slerp_wxyz(start_quat, end_quat, t)
            waypoints.append((pos, quat))

        # Approach
        for i in range(1, n_approach + 1):
            t = i / n_approach
            waypoints.append((approach_start + t * (end_pos - approach_start), end_quat))

    # Format as pick_place.json
    start_pose = list(start_pos) + quat_wxyz_to_xyzw(start_quat)
    goals = [
        [round(v, 6) for v in list(pos) + quat_wxyz_to_xyzw(quat)]
        for pos, quat in waypoints
    ]
    return {"start_pose": [round(v, 6) for v in start_pose], "goals": goals}


# --- Viz ---

def run_viz(port):
    """Interactive trajectory playback viewer."""
    import viser
    from viser.extras import ViserUrdf
    from fabrica.viser_utils import COLORS, SceneManager

    ALL_ASSEMBLIES = ["beam", "car", "cooling_manifold", "duct",
                      "gamepad", "plumbers_block", "stool_circular"]

    # Load all assemblies that have trajectories
    all_data = {}
    for name in ALL_ASSEMBLIES:
        order_path = ASSETS_DIR / name / "assembly_order.json"
        if not order_path.exists():
            continue
        steps = json.loads(order_path.read_text()).get("steps", [])
        parts = []
        for pid in steps:
            traj_path = ASSETS_DIR / name / "trajectories" / pid / "pick_place.json"
            mesh_path = ASSETS_DIR / name / pid / f"{pid}_canonical.obj"
            if traj_path.exists() and mesh_path.exists():
                mesh = trimesh.load_mesh(str(mesh_path), process=False)
                traj = json.loads(traj_path.read_text())
                parts.append((pid, mesh, traj))
        if parts:
            all_data[name] = parts
            print(f"  {name}: {len(parts)} parts")

    if not all_data:
        print("No assemblies with trajectories found")
        return

    server = viser.ViserServer(host="0.0.0.0", port=port)
    scene = SceneManager()

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.0, -1.0, 1.0)
        client.camera.look_at = (0.0, 0.0, 0.5)

    # Static scene
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    server.scene.add_frame("/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False)
    server.scene.add_box("/table/wood", color=(180, 130, 70), dimensions=(0.475, 0.4, 0.3),
                         position=(0, 0, 0), side="double", opacity=0.9)
    robot_urdf = REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"
    server.scene.add_frame("/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False)
    if robot_urdf.exists():
        ViserUrdf(server, robot_urdf, root_node_name="/robot")

    # Animation state
    frames = {}
    waypoints = {}
    order = []
    step = [0]
    animating = [False]

    def _qxyzw_to_wxyz(q):
        return np.array([q[3], q[0], q[1], q[2]])

    def show_assembly(name):
        scene.clear()
        frames.clear()
        waypoints.clear()
        order.clear()
        step[0] = 0

        parts = all_data[name]
        for i, (pid, mesh, traj) in enumerate(parts):
            order.append(pid)
            offset = -np.array(mesh.centroid, dtype=np.float32)
            f = server.scene.add_frame(f"/assembly/{pid}", wxyz=(1, 0, 0, 0), position=(0, 0, 0),
                                       show_axes=True, axes_length=0.1, axes_radius=0.001)
            scene.add(f)
            frames[pid] = f
            mf = server.scene.add_frame(f"/assembly/{pid}/mf", position=tuple(offset),
                                        wxyz=(1, 0, 0, 0), show_axes=False)
            scene.add(mf)
            scene.add(server.scene.add_mesh_simple(
                f"/assembly/{pid}/mf/mesh",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=COLORS[i % len(COLORS)],
            ))
            # Build waypoint list: start_pose + goals
            wps = []
            sp = traj["start_pose"]
            wps.append((np.array(sp[:3]), _qxyzw_to_wxyz(sp[3:7])))
            for gp in traj["goals"]:
                wps.append((np.array(gp[:3]), _qxyzw_to_wxyz(gp[3:7])))
            waypoints[pid] = wps

        _set_poses(0, 0.0)
        _update_status()

    def _slerp(q0, q1, t):
        dot = np.dot(q0, q1)
        if dot < 0:
            q1, dot = -q1, -dot
        if dot > 0.9995:
            r = q0 + t * (q1 - q0)
            return r / np.linalg.norm(r)
        theta = np.arccos(min(dot, 1.0))
        return (np.sin((1-t)*theta) * q0 + np.sin(t*theta) * q1) / np.sin(theta)

    def _interp(wps, t):
        n = len(wps) - 1
        if n <= 0 or t <= 0: return wps[0]
        if t >= 1: return wps[-1]
        sf = t * n
        si = min(int(sf), n - 1)
        lt = sf - si
        a = lt * lt * (3.0 - 2.0 * lt)
        return ((1-a)*wps[si][0] + a*wps[si+1][0], _slerp(wps[si][1], wps[si+1][1], a))

    def _set_poses(s, t):
        for i, pid in enumerate(order):
            wps = waypoints[pid]
            if i < s: pos, q = wps[-1]
            elif i == s and t > 0: pos, q = _interp(wps, t)
            else: pos, q = wps[0]
            frames[pid].position = tuple(pos)
            frames[pid].wxyz = tuple(q)

    def _update_status():
        n = len(order)
        if step[0] >= n:
            status.content = f"Step {n}/{n} - Complete"
        else:
            status.content = f"Step {step[0]}/{n} - next: {order[step[0]]}"

    def step_fwd():
        if animating[0] or step[0] >= len(order): return
        animating[0] = True
        dur = dur_slider.value
        nf = max(1, int(dur * 30))
        for fi in range(nf + 1):
            _set_poses(step[0], fi / nf)
            time.sleep(1.0 / 30)
        step[0] += 1
        _set_poses(step[0], 0.0)
        _update_status()
        animating[0] = False

    def step_back():
        if animating[0] or step[0] <= 0: return
        animating[0] = True
        step[0] -= 1
        nf = max(1, int(dur_slider.value * 30))
        for fi in range(nf + 1):
            _set_poses(step[0], 1.0 - fi / nf)
            time.sleep(1.0 / 30)
        _set_poses(step[0], 0.0)
        _update_status()
        animating[0] = False

    def play_all():
        if animating[0]: return
        while step[0] < len(order):
            step_fwd()
            if step[0] < len(order): time.sleep(0.3)

    def reset():
        if animating[0]: return
        step[0] = 0
        _set_poses(0, 0.0)
        _update_status()

    # GUI
    names = list(all_data.keys())
    dd = server.gui.add_dropdown("Assembly", options=names, initial_value=names[0])
    dur_slider = server.gui.add_slider("Duration (s)", min=0.2, max=3.0, step=0.1, initial_value=1.0)
    status = server.gui.add_markdown("")
    server.gui.add_button("Play All").on_click(lambda _: play_all())
    server.gui.add_button("Step Forward").on_click(lambda _: step_fwd())
    server.gui.add_button("Step Back").on_click(lambda _: step_back())
    server.gui.add_button("Reset").on_click(lambda _: reset())
    dd.on_update(lambda _: show_assembly(dd.value))

    show_assembly(names[0])
    print(f"\nOpen http://localhost:{port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate pick_place trajectories")
    parser.add_argument("--assembly", type=str)
    parser.add_argument("--part", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    if args.viz:
        run_viz(args.port)
        return

    if not args.assembly:
        parser.error("--assembly is required when not using --viz")

    transforms, order_config = load_assembly_config(args.assembly)
    assembly_order = order_config["steps"]
    insertion_directions = order_config.get("insertion_directions", {})

    offset = compute_table_offset(args.assembly, transforms)
    print(f"Assembly: {args.assembly}")
    print(f"Table offset: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]\n")

    # Load canonical meshes
    canonical_meshes = {}
    for pid in assembly_order:
        canonical_meshes[pid] = trimesh.load_mesh(
            str(ASSETS_DIR / args.assembly / pid / f"{pid}_canonical.obj"), process=False)

    start_poses = compute_start_poses(canonical_meshes, assembly_order, ASSEMBLY_XY[1])
    parts = [args.part] if args.part else assembly_order

    for pid in parts:
        name = f"{args.assembly}_{pid}"
        traj_dir = ASSETS_DIR / args.assembly / "trajectories" / pid
        traj_file = traj_dir / "pick_place.json"

        if traj_file.exists() and not args.force:
            print(f"  Skipping {name} (exists, use --force)")
            continue

        # End pose
        centroid = np.array(transforms[pid]["original_centroid"])
        end_pos = centroid + offset
        a2c = transforms[pid]["assembled_to_canonical_wxyz"]
        end_quat = quat_inverse_wxyz(a2c)

        # Start pose (identity rotation in canonical frame)
        start_pos = start_poses[pid]
        start_quat = [1.0, 0.0, 0.0, 0.0]

        traj = generate_trajectory(start_pos, start_quat, end_pos, end_quat,
                                   insertion_dir=insertion_directions.get(pid))

        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_file.write_text(json.dumps(traj, indent=4) + "\n")
        print(f"  {name}: {len(traj['goals'])} waypoints")
        print(f"    start: pos=[{start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.4f}]")
        print(f"    end:   pos=[{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
