#!/usr/bin/env python3
"""Precompute table URDFs with completed parts baked in for each assembly step.

For each assembly step and each collision method (vhacd, sdf, coacd), generates
a scene URDF with all previously-completed parts as fixed collision geometry.

With --viz, skips generation and launches a viser viewer to validate the scene
URDFs and trajectories interactively (dropdown to select part/step, shows table,
placed parts, active part at start/goal poses, trajectory waypoints).

Usage:
    python fabrica/benchmark_processing/step5_generate_table_urdfs.py
    python fabrica/benchmark_processing/step5_generate_table_urdfs.py --assembly car
    python fabrica/benchmark_processing/step5_generate_table_urdfs.py --viz
    python fabrica/benchmark_processing/step5_generate_table_urdfs.py --viz --port 8083
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"
TABLE_Z = 0.38


def _load_assembly_config(assembly: str) -> dict:
    path = ASSETS_DIR / assembly / "assembly_order.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_assembly_order(assembly: str) -> List[str]:
    config = _load_assembly_config(assembly)
    return config.get("steps", [])


def _load_trajectory(assembly: str, pid: str):
    path = ASSETS_DIR / assembly / "trajectories" / pid / "pick_place.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _quat_xyzw_to_rpy(q) -> Tuple[float, float, float]:
    from scipy.spatial.transform import Rotation
    r = Rotation.from_quat([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
    roll, pitch, yaw = r.as_euler('xyz')
    return roll, pitch, yaw


def _quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _rpy_to_quat_wxyz(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


TABLE_LINK = """\
  <link name="box">
    <visual>
      <material name="wood"><color rgba="0.82 0.56 0.35 1.0"/></material>
      <origin xyz="0 0 0"/>
      <geometry><box size="0.475 0.4 0.3"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry><box size="0.475 0.4 0.3"/></geometry>
    </collision>
    <inertial>
      <mass value="500"/>
      <friction value="1.0"/>
      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>
    </inertial>
  </link>"""


def _part_goal_offset(assembly: str, pid: str):
    """Return (rx, ry, rz, roll, pitch, yaw) for a completed part relative to table."""
    traj = _load_trajectory(assembly, pid)
    if traj is None:
        return None
    goal_pose = traj["goals"][-1]
    rx, ry, rz = goal_pose[0], goal_pose[1], goal_pose[2] - TABLE_Z
    rpy = _quat_xyzw_to_rpy(goal_pose[3:7])
    return rx, ry, rz, rpy[0], rpy[1], rpy[2]


def _generate_vhacd_table(assembly: str, active_pid: str, completed_pids: List[str],
                          env_dir: Path):
    """VHACD table: single link per completed part with raw mesh (IsaacGym applies VHACD)."""
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="table_{assembly}_{active_pid}_scene">',
        TABLE_LINK,
    ]

    for pid in completed_pids:
        offset = _part_goal_offset(assembly, pid)
        if offset is None:
            continue
        rx, ry, rz, roll, pitch, yaw = offset
        mesh_rel = f"../../{pid}/{pid}_canonical.obj"

        lines.extend([
            f'  <link name="part_{pid}">',
            '    <visual>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            f'      <material name="placed_{pid}"><color rgba="0.6 0.6 0.6 1.0"/></material>',
            '    </visual>',
            '    <collision>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            '    </collision>',
            '  </link>',
            f'  <joint name="part_{pid}_joint" type="fixed">',
            '    <parent link="box"/>',
            f'    <child link="part_{pid}"/>',
            f'    <origin xyz="{rx} {ry} {rz}" rpy="{roll} {pitch} {yaw}"/>',
            '  </joint>',
        ])

    lines.append('</robot>')
    out_path = env_dir / "scene.urdf"
    out_path.write_text('\n'.join(lines))
    return out_path


def _generate_sdf_table(assembly: str, active_pid: str, completed_pids: List[str],
                        env_dir: Path, sdf_resolution: int = 512):
    """SDF table: single link per completed part with raw mesh + SDF tag."""
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="table_{assembly}_{active_pid}_scene_sdf">',
        TABLE_LINK,
    ]

    for pid in completed_pids:
        offset = _part_goal_offset(assembly, pid)
        if offset is None:
            continue
        rx, ry, rz, roll, pitch, yaw = offset
        mesh_rel = f"../../{pid}/{pid}_canonical.obj"

        lines.extend([
            f'  <link name="part_{pid}">',
            '    <visual>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            f'      <material name="placed_{pid}"><color rgba="0.6 0.6 0.6 1.0"/></material>',
            '    </visual>',
            '    <collision>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
            f'      <sdf resolution="{sdf_resolution}"/>',
            '    </collision>',
            '  </link>',
            f'  <joint name="part_{pid}_joint" type="fixed">',
            '    <parent link="box"/>',
            f'    <child link="part_{pid}"/>',
            f'    <origin xyz="{rx} {ry} {rz}" rpy="{roll} {pitch} {yaw}"/>',
            '  </joint>',
        ])

    lines.append('</robot>')
    out_path = env_dir / "scene_sdf.urdf"
    out_path.write_text('\n'.join(lines))
    return out_path


def _generate_coacd_table(assembly: str, active_pid: str, completed_pids: List[str],
                          env_dir: Path, coacd_target_pids=None):
    """CoACD table: uses CoACD decomp for the insertion target part(s), raw mesh for others.

    Args:
        coacd_target_pids: The part(s) the active part inserts into. Can be a
            single string or a list of strings. Only these parts get full CoACD
            decomposition. Other completed parts use a single raw mesh
            (IsaacGym auto-computes convex hull). If None, all parts use CoACD
            (backward compatible).
    """
    # Normalize to a set for fast lookup
    if coacd_target_pids is None:
        target_set = None  # means "all"
    elif isinstance(coacd_target_pids, str):
        target_set = {coacd_target_pids}
    else:
        target_set = set(coacd_target_pids)

    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="table_{assembly}_{active_pid}_scene_coacd">',
        TABLE_LINK,
    ]

    for pid in completed_pids:
        offset = _part_goal_offset(assembly, pid)
        if offset is None:
            continue
        rx, ry, rz, roll, pitch, yaw = offset

        use_coacd = (target_set is None) or (pid in target_set)

        if use_coacd:
            # Full CoACD decomposition: multiple sub-links per part
            coacd_dir = ASSETS_DIR / assembly / pid / "coacd"
            decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
            if not decomp_files:
                print(f"  Warning: no CoACD decomps for {assembly}/{pid}, skipping")
                continue

            for i, decomp_file in enumerate(decomp_files):
                link_name = f"part_{pid}_hull_{i}"
                mesh_rel = f"../../{pid}/coacd/{decomp_file.name}"

                lines.extend([
                    f'  <link name="{link_name}">',
                    '    <visual>',
                    '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                    f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
                    f'      <material name="placed_{pid}_{i}"><color rgba="0.6 0.6 0.6 1.0"/></material>',
                    '    </visual>',
                    '    <collision>',
                    '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                    f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
                    '    </collision>',
                    '  </link>',
                    f'  <joint name="{link_name}_joint" type="fixed">',
                    '    <parent link="box"/>',
                    f'    <child link="{link_name}"/>',
                    f'    <origin xyz="{rx} {ry} {rz}" rpy="{roll} {pitch} {yaw}"/>',
                    '  </joint>',
                ])
        else:
            # Raw mesh: single link (IsaacGym auto-computes convex hull)
            mesh_rel = f"../../{pid}/{pid}_canonical.obj"
            lines.extend([
                f'  <link name="part_{pid}">',
                '    <visual>',
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
                f'      <material name="placed_{pid}"><color rgba="0.6 0.6 0.6 1.0"/></material>',
                '    </visual>',
                '    <collision>',
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>',
                '    </collision>',
                '  </link>',
                f'  <joint name="part_{pid}_joint" type="fixed">',
                '    <parent link="box"/>',
                f'    <child link="part_{pid}"/>',
                f'    <origin xyz="{rx} {ry} {rz}" rpy="{roll} {pitch} {yaw}"/>',
                '  </joint>',
            ])

    lines.append('</robot>')
    out_path = env_dir / "scene_coacd.urdf"
    out_path.write_text('\n'.join(lines))
    return out_path


def _generate_empty_table_coacd():
    """Generate empty_table_coacd.urdf (just the box, no parts)."""
    lines = [
        '<?xml version="1.0"?>',
        '<robot name="empty_table_coacd">',
        TABLE_LINK,
        '</robot>',
    ]
    out_path = ASSETS_DIR / "empty_table_coacd.urdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines))
    print(f"  Generated: {out_path}")
    return out_path


def generate_all_tables(assembly: str):
    """Generate table URDFs for every step in the assembly order."""
    config = _load_assembly_config(assembly)
    order = config.get("steps", [])
    inserts_into = config.get("inserts_into", {})

    if not order:
        print(f"No assembly order found for {assembly}")
        return

    # Filter to parts with trajectories
    available = []
    for pid in order:
        traj_path = ASSETS_DIR / assembly / "trajectories" / pid / "pick_place.json"
        if traj_path.exists():
            available.append(pid)

    if not available:
        print(f"No trajectories found for {assembly}")
        return

    print(f"Assembly: {assembly}")
    print(f"  Available parts (in order): {available}")
    if inserts_into:
        print(f"  Adjacency map: {inserts_into}")

    # Generate empty table coacd variant
    _generate_empty_table_coacd()

    for step_idx, active_pid in enumerate(available):
        completed_pids = available[:step_idx]
        env_dir = ASSETS_DIR / assembly / "environments" / active_pid
        env_dir.mkdir(parents=True, exist_ok=True)

        coacd_target = inserts_into.get(active_pid, None)
        if isinstance(coacd_target, list):
            target_label = ", ".join(coacd_target)
        else:
            target_label = coacd_target or "all"
        print(f"\n  Step {step_idx}: Part {active_pid} (completed: {completed_pids}, "
              f"CoACD target: {target_label})")

        vhacd_path = _generate_vhacd_table(assembly, active_pid, completed_pids, env_dir)
        print(f"    VHACD: {vhacd_path.name}")

        sdf_path = _generate_sdf_table(assembly, active_pid, completed_pids, env_dir)
        print(f"    SDF:   {sdf_path.name}")

        coacd_path = _generate_coacd_table(assembly, active_pid, completed_pids, env_dir,
                                           coacd_target_pids=coacd_target)
        print(f"    CoACD: {coacd_path.name}")

    print(f"\nDone. Generated URDFs for {len(available)} steps.")


# ---------------------------------------------------------------------------
# Visualization mode (--viz)
# ---------------------------------------------------------------------------

def _parse_scene_urdf(urdf_path):
    """Parse a scene URDF and return list of (mesh_path, xyz, rpy, link_name)."""
    if not urdf_path.exists():
        return []

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent

    joint_origins = {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        joint_origins[child] = (xyz, rpy)

    parts = []
    for link in root.findall("link"):
        name = link.get("name")
        if name == "box":
            continue

        visual = link.find("visual")
        if visual is None:
            continue
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            continue

        mesh_rel = mesh_elem.get("filename")
        mesh_path = (urdf_dir / mesh_rel).resolve()

        xyz, rpy = joint_origins.get(name, ([0, 0, 0], [0, 0, 0]))
        parts.append((mesh_path, xyz, rpy, name))

    return parts


def run_viz(assembly: str, port: int):
    """Launch a viser viewer to validate scene URDFs and trajectories."""
    import time
    import trimesh
    import viser

    order = _load_assembly_order(assembly)

    available = []
    for pid in order:
        traj = _load_trajectory(assembly, pid)
        if traj is not None:
            available.append(pid)

    if not available:
        print(f"No parts with trajectories found for assembly '{assembly}'")
        return

    print(f"Assembly: {assembly}")
    print(f"Available parts: {available}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    scene_handles = []

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = (0.3, -0.5, 0.9)
        client.camera.look_at = (0.0, 0.0, 0.55)

    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    def clear_scene():
        for h in scene_handles:
            h.remove()
        scene_handles.clear()

    URDF_NAMES = {
        "coacd": "scene_coacd.urdf",
        "sdf": "scene_sdf.urdf",
        "vhacd": "scene.urdf",
    }

    def show_part(pid, method="coacd"):
        clear_scene()
        step_idx = available.index(pid)
        traj = _load_trajectory(assembly, pid)

        # Table box
        h = server.scene.add_box(
            "/validate/table",
            color=(180, 130, 70),
            dimensions=(0.475, 0.4, 0.3),
            position=(0, 0, TABLE_Z),
            side="double",
            opacity=0.7,
        )
        scene_handles.append(h)

        # Load and render scene URDF (completed parts)
        env_dir = ASSETS_DIR / assembly / "environments" / pid
        scene_urdf = env_dir / URDF_NAMES[method]
        scene_parts = _parse_scene_urdf(scene_urdf)

        import random
        rng = random.Random(42)
        for mesh_path, xyz, rpy, link_name in scene_parts:
            if not mesh_path.exists():
                print(f"  WARNING: Missing mesh {mesh_path}")
                continue
            mesh = trimesh.load_mesh(str(mesh_path), process=False)

            world_pos = (xyz[0], xyz[1], xyz[2] + TABLE_Z)
            world_quat = _rpy_to_quat_wxyz(rpy[0], rpy[1], rpy[2])

            color = (rng.randint(80, 230), rng.randint(80, 230), rng.randint(80, 230))
            h = server.scene.add_mesh_simple(
                f"/validate/scene/{link_name}",
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=color,
                opacity=0.6,
                position=world_pos,
                wxyz=tuple(world_quat),
            )
            scene_handles.append(h)

        # Load canonical mesh for active part
        canonical_path = ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"
        if not canonical_path.exists():
            print(f"  WARNING: No canonical mesh for part {pid}")
            return
        canonical_mesh = trimesh.load_mesh(str(canonical_path), process=False)
        verts = np.array(canonical_mesh.vertices, dtype=np.float32)
        faces = np.array(canonical_mesh.faces, dtype=np.uint32)

        # Start pose
        sp = traj["start_pose"]
        start_pos = tuple(sp[:3])
        start_quat = tuple(_quat_xyzw_to_wxyz(sp[3:7]))
        h = server.scene.add_mesh_simple(
            "/validate/active/start",
            vertices=verts, faces=faces,
            color=(50, 200, 50),
            opacity=0.5,
            position=start_pos,
            wxyz=start_quat,
        )
        scene_handles.append(h)
        h = server.scene.add_label(
            "/validate/active/start_label",
            text="START",
            position=(sp[0], sp[1], sp[2] + 0.02),
        )
        scene_handles.append(h)

        # Goal pose
        gp = traj["goals"][-1]
        goal_pos = tuple(gp[:3])
        goal_quat = tuple(_quat_xyzw_to_wxyz(gp[3:7]))
        h = server.scene.add_mesh_simple(
            "/validate/active/goal",
            vertices=verts, faces=faces,
            color=(200, 50, 50),
            opacity=0.5,
            position=goal_pos,
            wxyz=goal_quat,
        )
        scene_handles.append(h)
        h = server.scene.add_label(
            "/validate/active/goal_label",
            text="GOAL",
            position=(gp[0], gp[1], gp[2] + 0.02),
        )
        scene_handles.append(h)

        # Trajectory waypoints as numbered spheres
        all_waypoints = [traj["start_pose"]] + traj["goals"]
        for wi, wp in enumerate(all_waypoints):
            color = (
                (50, 200, 50) if wi == 0
                else (200, 50, 50) if wi == len(all_waypoints) - 1
                else (50, 100, 200)
            )
            h = server.scene.add_icosphere(
                f"/validate/waypoints/wp_{wi}",
                radius=0.003,
                color=color,
                position=(wp[0], wp[1], wp[2]),
            )
            scene_handles.append(h)
            h = server.scene.add_label(
                f"/validate/waypoints/wp_{wi}_label",
                text=str(wi),
                position=(wp[0], wp[1], wp[2] + 0.01),
            )
            scene_handles.append(h)

        # Connect waypoints with lines
        points = np.array([[wp[0], wp[1], wp[2]] for wp in all_waypoints], dtype=np.float32)
        if len(points) > 1:
            line_points = np.stack([points[:-1], points[1:]], axis=1)
            h = server.scene.add_line_segments(
                "/validate/trajectory_line",
                points=line_points,
                colors=(100, 100, 100),
                line_width=2.0,
            )
            scene_handles.append(h)

        # Info panel
        completed = available[:step_idx]
        info = (
            f"**Part {pid}** (step {step_idx}/{len(available)})\n\n"
            f"Completed parts in scene: {completed if completed else 'none'}\n\n"
            f"Start: [{sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f}]\n\n"
            f"Goal:  [{gp[0]:.4f}, {gp[1]:.4f}, {gp[2]:.4f}]\n\n"
            f"Goal quat (xyzw): [{gp[3]:.4f}, {gp[4]:.4f}, {gp[5]:.4f}, {gp[6]:.4f}]\n\n"
            f"Waypoints: {len(traj['goals'])}\n\n"
            f"Scene URDF: {scene_urdf.name} ({method}, {len(scene_parts)} collision links)"
        )
        info_text.content = info

    # GUI controls
    part_options = [f"Part {pid} (step {i})" for i, pid in enumerate(available)]
    dropdown = server.gui.add_dropdown(
        "Part", options=part_options, initial_value=part_options[0]
    )
    method_dd = server.gui.add_dropdown(
        "Collision", options=["coacd", "sdf", "vhacd"], initial_value="coacd"
    )
    info_text = server.gui.add_markdown("Loading...")

    def refresh(_=None):
        idx = part_options.index(dropdown.value)
        pid = available[idx]
        show_part(pid, method=method_dd.value)

    dropdown.on_update(refresh)
    method_dd.on_update(refresh)
    refresh()

    print(f"\nOpen http://localhost:{port}")
    print("Select parts from dropdown to validate scene + trajectory.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute table URDFs for assembly evaluation")
    parser.add_argument("--assembly", default="beam", help="Assembly name (default: beam)")
    parser.add_argument("--viz", action="store_true",
                        help="Skip generation and launch viser viewer to validate scenes")
    parser.add_argument("--port", type=int, default=8082,
                        help="Viser server port for --viz mode (default: 8082)")
    args = parser.parse_args()

    if args.viz:
        run_viz(args.assembly, args.port)
    else:
        generate_all_tables(args.assembly)
