#!/usr/bin/env python3
"""Generate single-insertion training scenes for a fabrica assembly.

For each insertion part ``p`` in ``assembly_order.steps[1:]`` (the base, index 0,
is never trained — the partial assembly is treated as a static fixture), this
script emits:

1. ``N`` partial-assembly positions per insertion part (``table_offset_xy``
   uniform in ``±PARTIAL_HALF_WIDTHS``, clipped to keep the fixture inside the
   table).
2. One static-fixture URDF per ``(p, scene)`` under
   ``assets/urdf/fabrica/<assembly>/insertion_scenes/{p}/scene_{n:04d}.urdf``.
   The URDF welds the table base with the full chain of parts before ``p`` in
   ``assembly_order``. The **receiver** part (per ``inserts_into[p]``) is
   represented by COACD hulls for precise contact; the **rest** of the chain
   uses the single canonical mesh per part (no V-HACD).
3. ``M`` cached start poses per ``(p, scene)`` — inserting-part XY uniform in
   ``±START_HALF_WIDTHS`` around the table origin, with stable-rest orientation
   and yaw. Each is rejection-filtered for bounds + 2 cm fixture proximity +
   clear lift/transit trajectory.
4. ``scenes.npz`` with the full ``(P, N, M, ...)`` tensor layout described in
   ``assets/urdf/fabrica/<assembly>/`` (below).

Output: ``assets/urdf/fabrica/<assembly>/scenes.npz`` with keys
  - ``insertion_parts``          : object (P,)             — part ids, e.g. ["2","0","3","1"]
  - ``start_poses``              : float32 (P, N, M, 7)     — xyz + xyzw quat
  - ``goals``                    : float32 (P, N, M, T, 7)  — zero-padded past traj_lengths
  - ``traj_lengths``             : int32   (P, N, M)
  - ``partial_assembly_offsets`` : float32 (P, N, 3)        — world XYZ of partial-assembly origin
  - ``scene_urdf_paths``         : object  (P, N)           — str, relative to assets/

Usage:
    python -m fabrica.scene_generation.generate_scenes \\
        --assembly beam_2x --num-scenes-per-part 100 --num-starts-per-scene 100

    python -m fabrica.scene_generation.generate_scenes \\
        --assembly beam_2x --num-scenes-per-part 20 --num-starts-per-scene 20 \\
        --seed 42 --output-name scenes_val.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from fabrica.benchmark_processing.step3_generate_trajectories import (
    load_assembly_config,
    quat_inverse_wxyz,
)
from fabrica.scene_generation.trajectory_generation import generate_variable_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

# --- Scene constants (world frame) ---

TABLE_Z = 0.38                          # world Z of table base-link origin
TABLE_TOP_Z = 0.53                      # world Z of the table top surface
TABLE_HALF_X = 0.2375                   # half of the 0.475 m table extent in x
TABLE_HALF_Y = 0.2                      # half of the 0.4 m table extent in y
EDGE_INSET = 0.05                       # safety margin from table edges (m)

# --- Peg-in-hole-style sampling ranges ---
# Both start and partial-assembly offset are centered at the table origin (0,0).
START_HALF_WIDTHS = (0.10, 0.10)        # inserting-part XY jitter
PARTIAL_HALF_WIDTHS = (0.15, 0.11)      # partial-assembly offset range
PROXIMITY_MARGIN = 0.02                 # min clearance: inserting part vs. fixture (m)

# --- Trajectory generation ---
TARGET_SPACING = 0.025                  # ~2.5 cm between consecutive waypoints
LIFT_INITIAL = TABLE_TOP_Z + 0.20       # initial lift altitude (m)
LIFT_INCREMENT = 0.05                   # raise lift altitude by this much per retry
MAX_LIFT_ITERATIONS = 5
MAX_PLACEMENT_RETRIES = 200


# ----------------------------------------------------------------------------
# Quaternion / pose helpers
# ----------------------------------------------------------------------------

def _wxyz_to_xyzw(q_wxyz):
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


def _matrix_to_wxyz(R_mat):
    rot = R.from_matrix(R_mat)
    xyzw = rot.as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def _wxyz_to_matrix(q_wxyz):
    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot.as_matrix()


def _wxyz_to_rpy(q_wxyz):
    rot = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot.as_euler("xyz").tolist()


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
    """Compose URDF link pose for an active part on the table.

    The stable rest 4x4 places the canonical mesh on z=0 in a stable
    orientation. Then yaw rotates about world z, then (x, y) translates in
    the table plane and TABLE_TOP_Z lifts the bottom onto the surface.
    """
    R_rest = stable_rest_T[:3, :3]
    t_rest = stable_rest_T[:3, 3]
    R_yaw = R.from_euler("z", yaw_rad).as_matrix()
    R_link = R_yaw @ R_rest
    p_link = R_yaw @ t_rest + np.array([x, y, TABLE_TOP_Z])
    return p_link, _matrix_to_wxyz(R_link)


def world_assembled_pose(transforms_pid, table_offset):
    """World (pos, quat_wxyz) for a part in its final fixture pose."""
    centroid = np.array(transforms_pid["original_centroid"])
    pos = centroid + np.array(table_offset)
    a2c = transforms_pid["assembled_to_canonical_wxyz"]
    quat_wxyz = quat_inverse_wxyz(a2c)
    return pos, quat_wxyz


# ----------------------------------------------------------------------------
# Collision managers
# ----------------------------------------------------------------------------

def make_placed_cm(placed_meshes_world):
    cm = trimesh.collision.CollisionManager()
    for name, m in placed_meshes_world.items():
        cm.add_object(name, m)
    return cm


def make_active_cm(canonical_mesh):
    cm = trimesh.collision.CollisionManager()
    cm.add_object("active", canonical_mesh)
    return cm


def update_active(active_cm, pos, quat_wxyz):
    active_cm.set_transform("active", make_transform(pos, quat_wxyz))


def active_to_placed_distance(active_cm, placed_cm):
    return float(active_cm.min_distance_other(placed_cm))


def lift_transit_clear(active_cm, placed_cm, waypoints, lift_transit_end):
    """True iff every lift+transit waypoint is >= PROXIMITY_MARGIN from placed."""
    for i in range(lift_transit_end):
        pos = waypoints[i, :3]
        qx, qy, qz, qw = waypoints[i, 3:7]
        update_active(active_cm, pos, [qw, qx, qy, qz])
        if active_to_placed_distance(active_cm, placed_cm) < PROXIMITY_MARGIN:
            return False
    return True


# ----------------------------------------------------------------------------
# Partial-assembly helpers
# ----------------------------------------------------------------------------

def compute_partial_assembly_metadata(assembly, transforms, fixture_pids, canonical_meshes):
    """Compute (z_offset, half_extent_xy) for the partial-assembly fixture.

    z_offset: value for ``table_offset_world[2]`` so the assembly's lowest
        fixture-part Z lands exactly on TABLE_TOP_Z.
    half_extent_xy: max absolute |x|,|y| of the fixture bbox at zero offset,
        used to clip the partial-assembly XY sampling so the fixture never
        overflows the table interior.
    """
    if not fixture_pids:
        # Inserting directly against the base — no fixture parts. Should not
        # happen in the current beam assembly (the base "6" is always in the
        # chain for parts 2/0/3/1), but handle it gracefully.
        return TABLE_TOP_Z, 0.0, 0.0

    min_z = float("inf")
    x_bounds = []
    y_bounds = []
    for pid in fixture_pids:
        # Place the fixture part at canonical assembled pose with zero offset.
        pos, quat = world_assembled_pose(transforms[pid], (0.0, 0.0, 0.0))
        world_mesh = transform_mesh(canonical_meshes[pid], pos, quat)
        lo, hi = world_mesh.bounds
        min_z = min(min_z, lo[2])
        x_bounds.extend([lo[0], hi[0]])
        y_bounds.extend([lo[1], hi[1]])

    z_offset = TABLE_TOP_Z - min_z
    half_x = max(abs(min(x_bounds)), abs(max(x_bounds)))
    half_y = max(abs(min(y_bounds)), abs(max(y_bounds)))
    return z_offset, half_x, half_y


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


def _urdf_fixed_link_block(link_name, mesh_rel, origin_xyz, origin_rpy, material_idx):
    """URDF fragment: one <link> + one fixed <joint> parenting 'box'."""
    x, y, z = origin_xyz
    rr, rp, ry = origin_rpy
    return (
        f'  <link name="{link_name}">\n'
        f'    <visual>\n'
        f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>\n'
        f'      <material name="placed_{material_idx}"><color rgba="0.6 0.6 0.6 1.0"/></material>\n'
        f'    </visual>\n'
        f'    <collision>\n'
        f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="{mesh_rel}" scale="1 1 1"/></geometry>\n'
        f'    </collision>\n'
        f'  </link>\n'
        f'  <joint name="{link_name}_joint" type="fixed">\n'
        f'    <parent link="box"/>\n'
        f'    <child link="{link_name}"/>\n'
        f'    <origin xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="{rr:.6f} {rp:.6f} {ry:.6f}"/>\n'
        f'  </joint>\n'
    )


def write_fixture_urdf(
    assembly,
    p,
    n,
    table_offset_world,
    fixture_pids,
    receiver_pid,
    transforms,
):
    """Bake the per-scene static-fixture URDF.

    The receiver (``inserts_into[p]``) is represented by COACD decomp hulls
    for precise insertion contact. The rest of the chain uses the single
    canonical mesh per part (no V-HACD).

    Returns the URDF path relative to ``assets/``.
    """
    dst_dir = ASSETS_DIR / assembly / "insertion_scenes" / p
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"scene_{n:04d}.urdf"

    lines = [_fixture_urdf_header(assembly, p, n)]
    material_idx = 0

    for pid in fixture_pids:
        world_pos, world_quat_wxyz = world_assembled_pose(transforms[pid], table_offset_world)
        # URDF origin = world_pos - (0, 0, TABLE_Z), since the box link sits
        # at world Z = TABLE_Z when SimToolReal places the table.
        origin_xyz = (world_pos[0], world_pos[1], world_pos[2] - TABLE_Z)
        origin_rpy = _wxyz_to_rpy(world_quat_wxyz)

        if pid == receiver_pid:
            # Receiver: one link per COACD decomp hull.
            coacd_dir = ASSETS_DIR / assembly / pid / "coacd"
            decomp_files = sorted(coacd_dir.glob("decomp_*.obj"))
            if not decomp_files:
                raise FileNotFoundError(
                    f"No COACD decomp files in {coacd_dir}. Run run_coacd.py first."
                )
            for i, decomp in enumerate(decomp_files):
                mesh_rel = f"../../{pid}/coacd/{decomp.name}"
                link = f"part_{pid}_hull_{i}"
                lines.append(
                    _urdf_fixed_link_block(link, mesh_rel, origin_xyz, origin_rpy, material_idx)
                )
                material_idx += 1
        else:
            # Rest of fixture: one link per part, single canonical mesh.
            mesh_rel = f"../../{pid}/{pid}_canonical.obj"
            link = f"part_{pid}_canonical"
            lines.append(
                _urdf_fixed_link_block(link, mesh_rel, origin_xyz, origin_rpy, material_idx)
            )
            material_idx += 1

    lines.append("</robot>\n")
    dst.write_text("".join(lines))
    return str(dst.relative_to(REPO_ROOT / "assets"))


# ----------------------------------------------------------------------------
# Per-part start-pose sampling
# ----------------------------------------------------------------------------

def sample_part_placement(
    rng,
    pid,
    canonical_mesh,
    stable_rests_for_pid,
    placed_meshes_world,
    goal_pose,
    insertion_dir,
):
    """Sample one (start_pose, trajectory) for the inserting part.

    Rejects on:
      1. rotated bbox outside ±(TABLE_HALF - EDGE_INSET).
      2. start pose < PROXIMITY_MARGIN from the partial-assembly fixture.
      3. no lift/transit trajectory clears the fixture (iterates lift height).

    Returns (start_pos, start_quat_wxyz, waypoints, lift_transit_end) on success
    or None if all MAX_PLACEMENT_RETRIES attempts failed.
    """
    n_rests = len(stable_rests_for_pid)
    end_pos, end_quat_wxyz = goal_pose

    has_placed = bool(placed_meshes_world)
    placed_cm = make_placed_cm(placed_meshes_world) if has_placed else None
    active_cm = make_active_cm(canonical_mesh)

    n_bbox_fail = 0
    n_prox_fail = 0
    n_traj_fail = 0

    for attempt in range(MAX_PLACEMENT_RETRIES):
        stable_idx = int(rng.integers(n_rests))
        stable_T = stable_rests_for_pid[stable_idx]
        yaw = float(rng.uniform(-np.pi, np.pi))
        x = float(rng.uniform(-START_HALF_WIDTHS[0], START_HALF_WIDTHS[0]))
        y = float(rng.uniform(-START_HALF_WIDTHS[1], START_HALF_WIDTHS[1]))

        start_pos, start_quat_wxyz = compose_link_pose(stable_T, yaw, x, y)

        # (1) Rotated bbox inside the table (with edge inset).
        active_mesh_world = transform_mesh(canonical_mesh, start_pos, start_quat_wxyz)
        bb_min, bb_max = active_mesh_world.bounds
        if (
            bb_min[0] < -(TABLE_HALF_X - EDGE_INSET)
            or bb_max[0] > (TABLE_HALF_X - EDGE_INSET)
            or bb_min[1] < -(TABLE_HALF_Y - EDGE_INSET)
            or bb_max[1] > (TABLE_HALF_Y - EDGE_INSET)
        ):
            n_bbox_fail += 1
            continue

        # (2) Start pose clear of the fixture (mirror peg-in-hole PROXIMITY_MARGIN).
        if has_placed:
            update_active(active_cm, start_pos, start_quat_wxyz)
            if active_to_placed_distance(active_cm, placed_cm) < PROXIMITY_MARGIN:
                n_prox_fail += 1
                continue

        # (3) Find a lift altitude that keeps the whole trajectory clear.
        chosen_waypoints = None
        chosen_end = None
        for lift_iter in range(MAX_LIFT_ITERATIONS):
            clearance_z = LIFT_INITIAL + lift_iter * LIFT_INCREMENT
            waypoints, lift_transit_end = generate_variable_trajectory(
                start_pos,
                start_quat_wxyz,
                end_pos,
                end_quat_wxyz,
                clearance_z=clearance_z,
                insertion_dir=insertion_dir,
                target_spacing=TARGET_SPACING,
            )
            if not has_placed or lift_transit_clear(
                active_cm, placed_cm, waypoints, lift_transit_end
            ):
                chosen_waypoints = waypoints
                chosen_end = lift_transit_end
                break

        if chosen_waypoints is None:
            n_traj_fail += 1
            continue

        return start_pos, start_quat_wxyz, chosen_waypoints, chosen_end

    print(
        f"  [debug] part {pid} after {MAX_PLACEMENT_RETRIES} attempts: "
        f"bbox_fail={n_bbox_fail}, prox_fail={n_prox_fail}, traj_fail={n_traj_fail}",
        flush=True,
    )
    return None


# ----------------------------------------------------------------------------
# Scene generation
# ----------------------------------------------------------------------------

def generate_single_insertion_scenes(assembly, n_scenes_per_part, m_starts_per_scene, seed):
    rng = np.random.default_rng(seed)

    transforms, order_config = load_assembly_config(assembly)
    assembly_order = order_config["steps"]
    inserts_into = order_config.get("inserts_into", {})
    insertion_directions = order_config.get("insertion_directions", {})

    insertion_parts = assembly_order[1:]  # drop the base
    P = len(insertion_parts)
    N = n_scenes_per_part
    M = m_starts_per_scene

    # Load canonical meshes (same ones Isaac Gym loads).
    canonical_meshes = {}
    for pid in assembly_order:
        mesh_path = ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"
        canonical_meshes[pid] = trimesh.load_mesh(str(mesh_path), process=False)

    # Load stable rests.
    stable_rests = {}
    for pid in assembly_order:
        rests_path = ASSETS_DIR / assembly / "stable_rests" / f"{pid}.npz"
        if not rests_path.exists():
            raise FileNotFoundError(
                f"Missing stable rests for part {pid}: {rests_path}\n"
                f"Run compute_stable_rests.py --assembly {assembly} first."
            )
        stable_rests[pid] = np.load(rests_path)["transforms"]

    # Precompute per-part fixture metadata: Z offset + XY half-extent.
    per_part_meta = {}
    for p in insertion_parts:
        fixture_pids = assembly_order[: assembly_order.index(p)]
        z_offset, half_x, half_y = compute_partial_assembly_metadata(
            assembly, transforms, fixture_pids, canonical_meshes
        )
        per_part_meta[p] = dict(
            fixture_pids=fixture_pids,
            receiver_pid=inserts_into.get(p),
            z_offset=z_offset,
            half_x=half_x,
            half_y=half_y,
        )

    print(f"Assembly: {assembly}")
    print(f"Insertion parts: {insertion_parts}")
    for p in insertion_parts:
        m = per_part_meta[p]
        print(
            f"  part {p!r}: fixture={m['fixture_pids']} receiver={m['receiver_pid']!r} "
            f"fixture_half_xy=({m['half_x']:.3f}, {m['half_y']:.3f}) z_offset={m['z_offset']:.4f}"
        )
    print(f"Sampling: {N} scenes × {M} starts per part (seed={seed})\n", flush=True)

    # Output buffers.
    start_poses = np.zeros((P, N, M, 7), dtype=np.float32)           # xyzw
    per_pnm_waypoints = [[[None] * M for _ in range(N)] for _ in range(P)]
    traj_lengths = np.zeros((P, N, M), dtype=np.int32)
    partial_assembly_offsets = np.zeros((P, N, 3), dtype=np.float32)
    scene_urdf_paths = np.empty((P, N), dtype=object)

    for p_idx, p in enumerate(insertion_parts):
        meta = per_part_meta[p]
        fixture_pids = meta["fixture_pids"]
        receiver_pid = meta["receiver_pid"]

        # Clip PARTIAL_HALF_WIDTHS by the fixture's XY extent so the fixture
        # bbox never leaves the table interior.
        xy_clip_x = max(0.0, (TABLE_HALF_X - EDGE_INSET) - meta["half_x"])
        xy_clip_y = max(0.0, (TABLE_HALF_Y - EDGE_INSET) - meta["half_y"])
        part_half_x = min(PARTIAL_HALF_WIDTHS[0], xy_clip_x)
        part_half_y = min(PARTIAL_HALF_WIDTHS[1], xy_clip_y)
        if part_half_x <= 0 or part_half_y <= 0:
            raise RuntimeError(
                f"Part {p}: fixture is wider than the table interior "
                f"(half_xy={meta['half_x']:.3f}, {meta['half_y']:.3f}; "
                f"inset interior={TABLE_HALF_X-EDGE_INSET:.3f}, {TABLE_HALF_Y-EDGE_INSET:.3f})"
            )

        for n in range(N):
            offx = float(rng.uniform(-part_half_x, part_half_x))
            offy = float(rng.uniform(-part_half_y, part_half_y))
            table_offset_world = (offx, offy, meta["z_offset"])

            # Build the fixture collision set at this scene's offset.
            # Each fixture part is approximated by its world-frame AABB
            # primitive (8 verts) instead of the canonical mesh (~12k verts) —
            # ~150x faster trimesh distance checks. Mirrors peg-in-hole's
            # `hole_block_mesh()` (peg_in_hole/scene_generation/generate_scenes.py:201-211).
            placed_meshes = {}
            for pid in fixture_pids:
                pos, quat = world_assembled_pose(transforms[pid], table_offset_world)
                world_mesh = transform_mesh(canonical_meshes[pid], pos, quat)
                lo, hi = world_mesh.bounds
                aabb = trimesh.creation.box(extents=(hi - lo))
                aabb.apply_translation(0.5 * (lo + hi))
                placed_meshes[f"fixture_{pid}"] = aabb

            # Bake the fixture URDF for this (p, n).
            urdf_rel = write_fixture_urdf(
                assembly, p, n, table_offset_world, fixture_pids, receiver_pid, transforms
            )
            scene_urdf_paths[p_idx, n] = urdf_rel
            partial_assembly_offsets[p_idx, n] = np.array(table_offset_world, dtype=np.float32)

            # Inserting-part goal pose in world coords.
            goal_pose = world_assembled_pose(transforms[p], table_offset_world)

            for m in range(M):
                result = sample_part_placement(
                    rng,
                    p,
                    canonical_meshes[p],
                    stable_rests[p],
                    placed_meshes,
                    goal_pose,
                    insertion_directions.get(p),
                )
                if result is None:
                    raise RuntimeError(
                        f"[part={p}, scene={n}, start={m}] failed to place after "
                        f"{MAX_PLACEMENT_RETRIES} attempts."
                    )
                start_pos, start_quat_wxyz, waypoints, _ = result
                start_poses[p_idx, n, m, 0:3] = start_pos
                start_poses[p_idx, n, m, 3:7] = _wxyz_to_xyzw(start_quat_wxyz)
                per_pnm_waypoints[p_idx][n][m] = waypoints
                traj_lengths[p_idx, n, m] = len(waypoints)

            if (n + 1) % 10 == 0 or n == N - 1:
                print(
                    f"  [{p}] scene {n+1}/{N}: offset=({offx:+.3f},{offy:+.3f}) "
                    f"traj_len_range=[{int(traj_lengths[p_idx, n].min())}, "
                    f"{int(traj_lengths[p_idx, n].max())}]",
                    flush=True,
                )

    # Pad goals to uniform max length.
    max_traj_len = int(traj_lengths.max())
    goals = np.zeros((P, N, M, max_traj_len, 7), dtype=np.float32)
    for p_idx in range(P):
        for n in range(N):
            for m in range(M):
                traj = per_pnm_waypoints[p_idx][n][m]
                goals[p_idx, n, m, : len(traj), :] = traj

    return dict(
        insertion_parts=np.array(insertion_parts, dtype=object),
        start_poses=start_poses,
        goals=goals,
        traj_lengths=traj_lengths,
        partial_assembly_offsets=partial_assembly_offsets,
        scene_urdf_paths=scene_urdf_paths,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--assembly", required=True, help="e.g., beam_2x")
    ap.add_argument("--num-scenes-per-part", type=int, default=100)
    ap.add_argument("--num-starts-per-scene", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output-name",
        default="scenes.npz",
        help="output filename within the assembly dir (default: scenes.npz)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    out_path = ASSETS_DIR / args.assembly / args.output_name
    if out_path.exists() and not args.force:
        print(f"{out_path} exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    result = generate_single_insertion_scenes(
        args.assembly,
        args.num_scenes_per_part,
        args.num_starts_per_scene,
        args.seed,
    )

    np.savez(
        out_path,
        insertion_parts=result["insertion_parts"],
        start_poses=result["start_poses"],
        goals=result["goals"],
        traj_lengths=result["traj_lengths"],
        partial_assembly_offsets=result["partial_assembly_offsets"],
        scene_urdf_paths=result["scene_urdf_paths"],
    )

    print(f"\nSaved {out_path.relative_to(REPO_ROOT)}")
    print(f"  insertion_parts          : {result['insertion_parts'].tolist()}")
    print(f"  start_poses              : {result['start_poses'].shape} {result['start_poses'].dtype}")
    print(f"  goals                    : {result['goals'].shape} {result['goals'].dtype}")
    print(f"  traj_lengths             : {result['traj_lengths'].shape} {result['traj_lengths'].dtype}")
    print(f"  partial_assembly_offsets : {result['partial_assembly_offsets'].shape}")
    print(f"  scene_urdf_paths         : {result['scene_urdf_paths'].shape}")
    print(
        f"  traj_length min/max: {int(result['traj_lengths'].min())} / "
        f"{int(result['traj_lengths'].max())}"
    )


if __name__ == "__main__":
    main()
