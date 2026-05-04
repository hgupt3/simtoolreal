#!/usr/bin/env python3
"""Generate single-insertion training scenes for FMB board assemblies.

Sampling procedure per (part, scene, start):
  1. Sample insertion piece start pose: XY uniform in ±START_HALF_WIDTHS,
     random stable-rest orientation + yaw, on the table surface.
  2. Sample board position: anywhere on the table (clipped to not fall off).
  3. Check piece doesn't collide with the board at start.
  4. Generate crane trajectory (lift → transit → descend to goal).
  5. If any check fails, resample board position (outer loop) and/or piece
     position (inner loop).

Usage:
    python -m fmb.generate_scenes --assembly fmb_board_1 \
        --num-scenes-per-part 100 --num-starts-per-scene 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from fabrica.scene_generation.trajectory_generation import generate_variable_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

# --- Scene constants (matching SimToolReal) ---
TABLE_Z = 0.38
TABLE_TOP_Z = 0.53
TABLE_HALF_X = 0.2375
TABLE_HALF_Y = 0.2
EDGE_INSET = 0.02

# --- Sampling ranges ---
START_HALF_WIDTHS = (0.10, 0.10)
PROXIMITY_MARGIN = 0.01
TARGET_SPACING = 0.05
LIFT_INITIAL = TABLE_TOP_Z + 0.20
LIFT_INCREMENT = 0.05
MAX_LIFT_ITERATIONS = 5
MAX_PLACEMENT_RETRIES = 500
MAX_SCENE_RETRIES = 50


# --- Helpers ---

def _wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]

def _matrix_to_wxyz(R_mat):
    xyzw = R.from_matrix(R_mat).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]

def _wxyz_to_matrix(q):
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def _wxyz_to_rpy(q):
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz").tolist()

def quat_inverse_wxyz(q):
    return [q[0], -q[1], -q[2], -q[3]]

def make_transform(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = _wxyz_to_matrix(quat_wxyz)
    T[:3, 3] = pos
    return T

def transform_mesh(mesh, pos, quat_wxyz):
    out = mesh.copy()
    out.apply_transform(make_transform(pos, quat_wxyz))
    return out

def compose_link_pose(stable_rest_T, yaw_rad, x, y):
    R_rest = stable_rest_T[:3, :3]
    t_rest = stable_rest_T[:3, 3]
    R_yaw = R.from_euler("z", yaw_rad).as_matrix()
    R_link = R_yaw @ R_rest
    p_link = R_yaw @ t_rest + np.array([x, y, TABLE_TOP_Z])
    return p_link, _matrix_to_wxyz(R_link)

def world_assembled_pose(transform_data, table_offset):
    centroid = np.array(transform_data["original_centroid"])
    pos = centroid + np.array(table_offset)
    a2c = transform_data["assembled_to_canonical_wxyz"]
    return pos, quat_inverse_wxyz(a2c)


# --- Collision helpers ---

def make_collision_manager(meshes_dict):
    cm = trimesh.collision.CollisionManager()
    for name, m in meshes_dict.items():
        cm.add_object(name, m)
    return cm


def check_piece_clear(piece_mesh_world, placed_cm):
    piece_cm = trimesh.collision.CollisionManager()
    piece_cm.add_object("piece", piece_mesh_world)
    dist = piece_cm.min_distance_other(placed_cm)
    return dist > PROXIMITY_MARGIN


def lift_transit_clear(canonical_mesh, placed_cm, waypoints, lift_transit_end):
    piece_cm = trimesh.collision.CollisionManager()
    piece_cm.add_object("piece", canonical_mesh)
    for i in range(lift_transit_end):
        pos = waypoints[i, :3]
        qx, qy, qz, qw = waypoints[i, 3:7]
        piece_cm.set_transform("piece", make_transform(pos, [qw, qx, qy, qz]))
        if piece_cm.min_distance_other(placed_cm) < PROXIMITY_MARGIN:
            return False
    return True


# --- Fixture URDF generation (all parts CoACD) ---

def _fixture_urdf_header(assembly, p, n):
    return (
        '<?xml version="1.0"?>\n'
        f'<robot name="table_{assembly}_{p}_scene_{n:04d}">\n'
        '  <link name="box">\n'
        '    <visual>\n'
        '      <material name="wood"><color rgba="0.82 0.56 0.35 1.0"/></material>\n'
        '      <origin xyz="0 0 0"/>\n'
        '      <geometry><box size="0.475 0.4 0.3"/></geometry>\n'
        '    </visual>\n'
        '    <collision>\n'
        '      <origin xyz="0 0 0"/>\n'
        '      <geometry><box size="0.475 0.4 0.3"/></geometry>\n'
        '    </collision>\n'
        '    <inertial>\n'
        '      <mass value="500"/>\n'
        '      <friction value="1.0"/>\n'
        '      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>\n'
        '    </inertial>\n'
        '  </link>\n'
    )

def _urdf_fixed_link(link_name, mesh_rel, xyz, rpy, mat_idx, sdf_tag=""):
    x, y, z = xyz
    rr, rp, ry = rpy
    return (
        f'  <link name="{link_name}">\n'
        f'    <visual>\n'
        f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>\n'
        f'      <material name="placed_{mat_idx}"><color rgba="0.6 0.6 0.6 1.0"/></material>\n'
        f'    </visual>\n'
        f'    <collision>\n'
        f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>\n'
        f'{sdf_tag}'
        f'    </collision>\n'
        f'  </link>\n'
        f'  <joint name="{link_name}_joint" type="fixed">\n'
        f'    <parent link="box"/>\n'
        f'    <child link="{link_name}"/>\n'
        f'    <origin xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="{rr:.6f} {rp:.6f} {ry:.6f}"/>\n'
        f'  </joint>\n'
    )

def write_fixture_urdf(assembly, p, n, table_offset_world, fixture_pids, transforms):
    dst_dir = ASSETS_DIR / assembly / "insertion_scenes" / p
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"scene_{n:04d}.urdf"

    lines = [_fixture_urdf_header(assembly, p, n)]
    mat_idx = 0

    for pid in fixture_pids:
        world_pos, world_quat = world_assembled_pose(transforms[pid], table_offset_world)
        origin_xyz = (world_pos[0], world_pos[1], world_pos[2] - TABLE_Z)
        origin_rpy = _wxyz_to_rpy(world_quat)

        coacd_dir = ASSETS_DIR / assembly / pid / "coacd"
        decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
        if decomp_files:
            for i, decomp in enumerate(decomp_files):
                mesh_rel = f"../../{pid}/coacd/{decomp.name}"
                lines.append(_urdf_fixed_link(
                    f"part_{pid}_hull_{i}", mesh_rel, origin_xyz, origin_rpy, mat_idx))
                mat_idx += 1
        else:
            mesh_rel = f"../../{pid}/{pid}_canonical.obj"
            lines.append(_urdf_fixed_link(
                f"part_{pid}_canonical", mesh_rel, origin_xyz, origin_rpy, mat_idx))
            mat_idx += 1

    lines.append("</robot>\n")
    dst.write_text("".join(lines))

    # Also write SDF variant: single canonical mesh per part with SDF tag.
    dst_sdf = dst_dir / f"scene_{n:04d}_sdf.urdf"
    sdf_lines = [_fixture_urdf_header(assembly, p, n)]
    sdf_mat_idx = 0
    sdf_tag = '      <sdf resolution="512"/>\n'
    for pid in fixture_pids:
        world_pos, world_quat = world_assembled_pose(transforms[pid], table_offset_world)
        origin_xyz = (world_pos[0], world_pos[1], world_pos[2] - TABLE_Z)
        origin_rpy = _wxyz_to_rpy(world_quat)
        mesh_rel = f"../../{pid}/{pid}_canonical.obj"
        sdf_lines.append(_urdf_fixed_link(
            f"part_{pid}_sdf", mesh_rel, origin_xyz, origin_rpy, sdf_mat_idx, sdf_tag=sdf_tag))
        sdf_mat_idx += 1
    sdf_lines.append("</robot>\n")
    dst_sdf.write_text("".join(sdf_lines))

    return str(dst.relative_to(REPO_ROOT / "assets"))


# --- Main generation ---

def load_assembly_config(assembly):
    with open(ASSETS_DIR / assembly / "canonical_transforms.json") as f:
        transforms = json.load(f)
    with open(ASSETS_DIR / assembly / "assembly_order.json") as f:
        order = json.load(f)
    return transforms, order


def compute_fixture_bounds(transforms, fixture_pids, canonical_meshes):
    if not fixture_pids:
        return TABLE_TOP_Z, 0.0, 0.0
    min_z = float("inf")
    x_bounds, y_bounds = [], []
    for pid in fixture_pids:
        pos, quat = world_assembled_pose(transforms[pid], (0, 0, 0))
        wm = transform_mesh(canonical_meshes[pid], pos, quat)
        lo, hi = wm.bounds
        min_z = min(min_z, lo[2])
        x_bounds.extend([lo[0], hi[0]])
        y_bounds.extend([lo[1], hi[1]])
    z_offset = TABLE_TOP_Z - min_z
    half_x = max(abs(min(x_bounds)), abs(max(x_bounds)))
    half_y = max(abs(min(y_bounds)), abs(max(y_bounds)))
    return z_offset, half_x, half_y


def generate_scenes(assembly, N, M, seed):
    rng = np.random.default_rng(seed)
    transforms, order = load_assembly_config(assembly)
    steps = order["steps"]
    inserts_into = order.get("inserts_into", {})
    insertion_parts = steps[1:]
    P = len(insertion_parts)

    canonical_meshes = {}
    for pid in steps:
        canonical_meshes[pid] = trimesh.load_mesh(
            str(ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"), process=False)

    stable_rests = {}
    for pid in steps:
        rp = ASSETS_DIR / assembly / "stable_rests" / f"{pid}.npz"
        if rp.exists():
            stable_rests[pid] = np.load(rp)["transforms"]

    # Precompute per-part fixture metadata.
    per_part_meta = {}
    for p in insertion_parts:
        fixture_pids = steps[:steps.index(p)]
        z_offset, half_x, half_y = compute_fixture_bounds(
            transforms, fixture_pids, canonical_meshes)
        max_off_x = max(0.001, (TABLE_HALF_X - EDGE_INSET) - half_x)
        max_off_y = max(0.001, (TABLE_HALF_Y - EDGE_INSET) - half_y)
        per_part_meta[p] = dict(
            fixture_pids=fixture_pids, z_offset=z_offset,
            max_off_x=max_off_x, max_off_y=max_off_y,
            half_x=half_x, half_y=half_y)

    print(f"Assembly: {assembly}")
    print(f"Insertion parts: {insertion_parts}")
    for p in insertion_parts:
        m = per_part_meta[p]
        print(f"  {p}: fixture={m['fixture_pids']} "
              f"board_offset_range=±({m['max_off_x']:.3f}, {m['max_off_y']:.3f})")
    print(f"Sampling: {N} scenes × {M} starts per part (seed={seed})\n", flush=True)

    start_poses = np.zeros((P, N, M, 7), dtype=np.float32)
    per_pnm_waypoints = [[[None]*M for _ in range(N)] for _ in range(P)]
    traj_lengths = np.zeros((P, N, M), dtype=np.int32)
    partial_offsets = np.zeros((P, N, 3), dtype=np.float32)
    scene_urdf_paths = np.empty((P, N), dtype=object)

    for p_idx, p in enumerate(insertion_parts):
        meta = per_part_meta[p]
        fixture_pids = meta["fixture_pids"]
        n_rests = len(stable_rests.get(p, []))

        for n in range(N):
            # Outer retry loop: resample board position if piece placement fails.
            scene_ok = False
            for scene_retry in range(MAX_SCENE_RETRIES):
                offx = float(rng.uniform(-meta["max_off_x"], meta["max_off_x"]))
                offy = float(rng.uniform(-meta["max_off_y"], meta["max_off_y"]))
                table_offset = (offx, offy, meta["z_offset"])

                # Build AABB collision set for the fixture at this offset.
                placed_meshes = {}
                for pid in fixture_pids:
                    pos, quat = world_assembled_pose(transforms[pid], table_offset)
                    wm = transform_mesh(canonical_meshes[pid], pos, quat)
                    lo, hi = wm.bounds
                    aabb = trimesh.creation.box(extents=(hi - lo))
                    aabb.apply_translation(0.5 * (lo + hi))
                    placed_meshes[f"fixture_{pid}"] = aabb

                has_placed = bool(placed_meshes)
                placed_cm = make_collision_manager(placed_meshes) if has_placed else None
                goal_pose = world_assembled_pose(transforms[p], table_offset)
                insertion_dir = None

                # Try placing all M starts for this board position.
                starts_found = 0
                temp_starts = np.zeros((M, 7), dtype=np.float32)
                temp_waypoints = [None] * M
                temp_traj_lens = np.zeros(M, dtype=np.int32)

                for m in range(M):
                    placed = False
                    for attempt in range(MAX_PLACEMENT_RETRIES):
                        stable_idx = int(rng.integers(n_rests)) if n_rests > 0 else 0
                        stable_T = stable_rests[p][stable_idx] if n_rests > 0 else np.eye(4)
                        yaw = float(rng.uniform(-np.pi, np.pi))
                        x = float(rng.uniform(-START_HALF_WIDTHS[0], START_HALF_WIDTHS[0]))
                        y = float(rng.uniform(-START_HALF_WIDTHS[1], START_HALF_WIDTHS[1]))

                        start_pos, start_quat = compose_link_pose(stable_T, yaw, x, y)

                        # Check bbox within table.
                        piece_world = transform_mesh(canonical_meshes[p], start_pos, start_quat)
                        bb_min, bb_max = piece_world.bounds
                        if (bb_min[0] < -(TABLE_HALF_X - EDGE_INSET) or
                            bb_max[0] > (TABLE_HALF_X - EDGE_INSET) or
                            bb_min[1] < -(TABLE_HALF_Y - EDGE_INSET) or
                            bb_max[1] > (TABLE_HALF_Y - EDGE_INSET)):
                            continue

                        # Check piece doesn't collide with fixture.
                        if has_placed and not check_piece_clear(piece_world, placed_cm):
                            continue

                        # Generate trajectory and check lift/transit is clear.
                        end_pos, end_quat = goal_pose
                        chosen_wps = None
                        for lift_iter in range(MAX_LIFT_ITERATIONS):
                            cz = LIFT_INITIAL + lift_iter * LIFT_INCREMENT
                            wps, lt_end = generate_variable_trajectory(
                                start_pos, start_quat, end_pos, end_quat,
                                clearance_z=cz, insertion_dir=insertion_dir,
                                target_spacing=TARGET_SPACING)
                            if not has_placed or lift_transit_clear(
                                    canonical_meshes[p], placed_cm, wps, lt_end):
                                chosen_wps = wps
                                break

                        if chosen_wps is None:
                            continue

                        temp_starts[m, 0:3] = start_pos
                        temp_starts[m, 3:7] = _wxyz_to_xyzw(start_quat)
                        temp_waypoints[m] = chosen_wps
                        temp_traj_lens[m] = len(chosen_wps)
                        starts_found += 1
                        placed = True
                        break

                    if not placed:
                        break  # This board position can't accommodate all M starts.

                if starts_found == M:
                    # Success — commit this scene.
                    urdf_rel = write_fixture_urdf(
                        assembly, p, n, table_offset, fixture_pids, transforms)
                    scene_urdf_paths[p_idx, n] = urdf_rel
                    partial_offsets[p_idx, n] = np.array(table_offset, dtype=np.float32)
                    start_poses[p_idx, n] = temp_starts
                    for m in range(M):
                        per_pnm_waypoints[p_idx][n][m] = temp_waypoints[m]
                    traj_lengths[p_idx, n] = temp_traj_lens
                    scene_ok = True
                    break

            if not scene_ok:
                raise RuntimeError(
                    f"[part={p}, scene={n}] failed after {MAX_SCENE_RETRIES} board resamples.")

            if (n + 1) % 10 == 0 or n == N - 1:
                print(f"  [{p}] scene {n+1}/{N}: offset=({offx:+.3f},{offy:+.3f}) "
                      f"traj_len=[{int(traj_lengths[p_idx,n].min())}, "
                      f"{int(traj_lengths[p_idx,n].max())}] "
                      f"(board retries: {scene_retry})", flush=True)

    max_traj_len = int(traj_lengths.max())
    goals = np.zeros((P, N, M, max_traj_len, 7), dtype=np.float32)
    for pi in range(P):
        for n in range(N):
            for m in range(M):
                traj = per_pnm_waypoints[pi][n][m]
                goals[pi, n, m, :len(traj), :] = traj

    return dict(
        insertion_parts=np.array(insertion_parts, dtype=object),
        start_poses=start_poses, goals=goals, traj_lengths=traj_lengths,
        partial_assembly_offsets=partial_offsets, scene_urdf_paths=scene_urdf_paths)


def main():
    ap = argparse.ArgumentParser(description="Generate FMB training scenes")
    ap.add_argument("--assembly", required=True)
    ap.add_argument("--num-scenes-per-part", type=int, default=100)
    ap.add_argument("--num-starts-per-scene", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-name", default="scenes.npz")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = ASSETS_DIR / args.assembly / args.output_name
    if out_path.exists() and not args.force:
        print(f"{out_path} exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    result = generate_scenes(
        args.assembly, args.num_scenes_per_part,
        args.num_starts_per_scene, args.seed)

    np.savez(out_path, **result)

    print(f"\nSaved {out_path.relative_to(REPO_ROOT)}")
    print(f"  insertion_parts : {result['insertion_parts'].tolist()}")
    print(f"  start_poses     : {result['start_poses'].shape}")
    print(f"  goals           : {result['goals'].shape}")
    print(f"  traj_lengths    : min={result['traj_lengths'].min()} max={result['traj_lengths'].max()}")


if __name__ == "__main__":
    main()
