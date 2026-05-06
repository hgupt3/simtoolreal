#!/usr/bin/env python3
"""Validate an insertion Problem's physics in Isaac Gym.

Given a `--problem` name (e.g. `fabrica.beam_2x.part_2`,
`fmb.peg_board_1.peg_46`, `peg.tol0p5mm`), load the receptive + inserter
URDFs from `PROBLEM_REGISTRY`, teleport the inserter from its pre-insert
pose to the final insert pose along a linear interpolation, then let it
settle. Report how well the inserter tracks the commanded path (a clean
hole admits the peg without large pose error) and optionally write an MP4
of the run.

Usage:
    python -m peg_in_hole_dynamic.validate_insertions \
        --problem fabrica.beam_2x.part_2

    python -m peg_in_hole_dynamic.validate_insertions \
        --problem fmb.peg_board_1.peg_46 --no-video

Outputs (under ``debug_output/validate_insertions/<problem>/<timestamp>/``):
    * ``<problem>.mp4``        if recording video
    * ``results.json``         pose-tracking + settle stats

The receptive is loaded at world ``(0, 0, TABLE_TOP_Z + hole_z_offset)``,
matching what the env does at episode reset (with random XY zeroed for a
deterministic check).
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import List, Tuple

# isaacgym must be imported before torch
from isaacgym import gymapi, gymtorch

import imageio
import numpy as np
import torch

from peg_in_hole_dynamic import PROBLEM_REGISTRY
from dextoolbench.objects import NAME_TO_OBJECT

REPO_ROOT  = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"

# Match peg_in_hole_dynamic_env / SimToolReal table conventions.
TABLE_RESET_Z    = 0.38
TABLE_HALF_HEIGHT = 0.15
TABLE_TOP_Z      = TABLE_RESET_Z + TABLE_HALF_HEIGHT       # 0.53 m

OBJECT_BASE_SIZE = 0.04
KEYPOINT_SCALE   = 1.5
KEYPOINT_DIRS    = np.array([
    [+1, +1, +1], [+1, +1, -1], [-1, -1, +1], [-1, -1, -1],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),         2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),         2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _slerp(q0, q1, t):
    q0 = np.array(q0, np.float64)
    q1 = np.array(q1, np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    r = s0 * q0 + s1 * q1
    return r / np.linalg.norm(r)


def _interp_two_poses(start, end, n_steps):
    """Linearly interpolate position and slerp orientation between two
    7-vector poses (xyz + xyzw quat). Returns n_steps poses, end-inclusive."""
    poses = []
    for s in range(n_steps):
        t = s / max(n_steps - 1, 1)
        pos = start[:3] + t * (end[:3] - start[:3])
        quat = _slerp(start[3:7], end[3:7], t)
        poses.append(np.concatenate([pos, quat]))
    return poses


def _keypoints_max_dist(pose_a, pose_b, kp_offsets):
    def _kps(p):
        Rm = _quat_xyzw_to_matrix(p[3:7])
        return p[:3][None, :] + (Rm @ kp_offsets.T).T
    return float(np.linalg.norm(_kps(pose_a) - _kps(pose_b), axis=1).max())


def _compute_kp_offsets(insertion_object_name: str) -> np.ndarray:
    """Half-extents-scaled corner offsets, matching the env's keypoint convention.
    Falls back to a generic 4cm cube if the object's scale isn't queryable."""
    obj = NAME_TO_OBJECT.get(insertion_object_name)
    if obj is None:
        half = np.array([OBJECT_BASE_SIZE / 2] * 3)
    else:
        half = np.array([obj.scale[i] * OBJECT_BASE_SIZE / 2 for i in range(3)])
    return KEYPOINT_DIRS * half[None, :] * KEYPOINT_SCALE


# ─────────────────────────────────────────────────────────────────────────────
# Pose plumbing
# ─────────────────────────────────────────────────────────────────────────────

def _problem_world_poses(problem) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (receptive_pos, receptive_quat, inserter_pre_pose7, inserter_final_pose7)
    in world frame with the receptive at origin XY on the table."""
    recv_pos = np.array([0.0, 0.0, TABLE_TOP_Z + problem.hole_z_offset])
    recv_quat = np.array([0.0, 0.0, 0.0, 1.0])

    insert_pos_rel = np.array(problem.insert_pose_rel_receptive[:3])
    insert_quat    = np.array(problem.insert_pose_rel_receptive[3:7])
    insertion_dir  = np.array(problem.insertion_direction, dtype=np.float64)
    insertion_dir /= max(np.linalg.norm(insertion_dir), 1e-9)
    pre_offset     = float(problem.pre_insert_offset)

    final_world_pos = recv_pos + insert_pos_rel
    pre_world_pos   = final_world_pos - insertion_dir * pre_offset

    pre_pose7   = np.concatenate([pre_world_pos,   insert_quat])
    final_pose7 = np.concatenate([final_world_pos, insert_quat])
    return recv_pos, recv_quat, pre_pose7, final_pose7


def _camera_for_pose(world_pos):
    cam_pos    = gymapi.Vec3(0.35, -0.45, 0.95)
    cam_target = gymapi.Vec3(float(world_pos[0]), float(world_pos[1]), float(world_pos[2]) + 0.05)
    return cam_pos, cam_target


# ─────────────────────────────────────────────────────────────────────────────
# Main validation routine
# ─────────────────────────────────────────────────────────────────────────────

def validate(problem_name: str, *, steps_pre_to_final: int = 200,
             settle_frames: int = 180, record_video: bool = True,
             output_dir: Path = None) -> dict:
    if problem_name not in PROBLEM_REGISTRY:
        raise KeyError(
            f"Unknown problem {problem_name!r}; "
            f"known: {sorted(PROBLEM_REGISTRY)[:10]}..."
        )
    problem = PROBLEM_REGISTRY[problem_name]
    obj = NAME_TO_OBJECT.get(problem.insertion_object_name)
    if obj is None:
        raise KeyError(
            f"insertion_object_name {problem.insertion_object_name!r} "
            "not in NAME_TO_OBJECT — register it before validating."
        )

    receptive_abs = ASSETS_DIR / problem.receptive_urdf
    inserter_abs  = Path(obj.urdf_path)
    if not receptive_abs.is_file():
        raise FileNotFoundError(receptive_abs)
    if not inserter_abs.is_file():
        raise FileNotFoundError(inserter_abs)

    print(f"Problem:        {problem_name}")
    print(f"  receptive:    {problem.receptive_urdf}")
    print(f"  inserter:     {inserter_abs.relative_to(ASSETS_DIR)}")
    print(f"  insert pose:  {tuple(round(v, 4) for v in problem.insert_pose_rel_receptive)}")
    print(f"  pre_insert_offset = {problem.pre_insert_offset}")
    print(f"  insertion_direction = {tuple(problem.insertion_direction)}")
    print(f"  hole_z_offset = {problem.hole_z_offset}")

    recv_pos, recv_quat, pre_pose7, final_pose7 = _problem_world_poses(problem)
    print(f"  pre-insert world pose:  ({pre_pose7[0]:+.4f}, {pre_pose7[1]:+.4f}, {pre_pose7[2]:+.4f})")
    print(f"  final-insert world pose: ({final_pose7[0]:+.4f}, {final_pose7[1]:+.4f}, {final_pose7[2]:+.4f})")

    kp_offsets = _compute_kp_offsets(problem.insertion_object_name)

    # ── Sim setup ──
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 192
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.005
    sim_params.physx.bounce_threshold_velocity = 0.02
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.max_depenetration_velocity = 5.0
    sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024
    sim_params.physx.default_buffer_size_multiplier = 25.0
    sim_params.physx.num_threads = 0
    sim_params.physx.use_gpu = True

    gym = gymapi.acquire_gym()
    graphics_device = 0 if record_video else -1
    sim = gym.create_sim(0, graphics_device, gymapi.SIM_PHYSX, sim_params)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Receptive (fixed base, multi-link/box+mesh URDFs)
    recv_options = gymapi.AssetOptions()
    recv_options.fix_base_link = True
    recv_options.collapse_fixed_joints = True
    recv_asset = gym.load_asset(
        sim, str(receptive_abs.parent), receptive_abs.name, recv_options,
    )

    # Inserter (dynamic, CoACD-driven URDF)
    ins_options = gymapi.AssetOptions()
    ins_options.collapse_fixed_joints = True
    ins_options.replace_cylinder_with_capsule = True
    ins_asset = gym.load_asset(
        sim, str(inserter_abs.parent), inserter_abs.name, ins_options,
    )
    print(f"  receptive: {gym.get_asset_rigid_body_count(recv_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(recv_asset)} shapes | "
          f"inserter: {gym.get_asset_rigid_body_count(ins_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(ins_asset)} shapes")

    spacing = 0.5
    env = gym.create_env(
        sim, gymapi.Vec3(-spacing, -spacing, 0),
        gymapi.Vec3(spacing, spacing, spacing), 1,
    )

    recv_pose = gymapi.Transform()
    recv_pose.p = gymapi.Vec3(*recv_pos)
    recv_pose.r = gymapi.Quat(*recv_quat)
    gym.create_actor(env, recv_asset, recv_pose, "receptive", 0, 0)

    ins_pose = gymapi.Transform()
    ins_pose.p = gymapi.Vec3(*pre_pose7[:3])
    ins_pose.r = gymapi.Quat(*pre_pose7[3:7])
    gym.create_actor(env, ins_asset, ins_pose, "inserter", 0, 0, 0)

    cam_handle = None
    if record_video:
        cam_props = gymapi.CameraProperties()
        cam_props.width = 1280
        cam_props.height = 960
        cam_props.use_collision_geometry = True
        cam_handle = gym.create_camera_sensor(env, cam_props)
        cp, ct = _camera_for_pose(final_pose7[:3])
        gym.set_camera_location(cam_handle, env, cp, ct)

    gym.prepare_sim(sim)
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    INSERTER_IDX = 1

    # Pre → final teleport trajectory
    traj = _interp_two_poses(pre_pose7, final_pose7, steps_pre_to_final)
    total_steps = len(traj) + settle_frames

    deltas, kp_dists, frames = [], [], []
    max_delta = 0.0
    max_kp = 0.0

    for step in range(total_steps):
        if step < len(traj):
            desired = traj[step]
            root_states[INSERTER_IDX, 0:7] = torch.tensor(
                desired, dtype=torch.float32, device="cuda",
            )
            root_states[INSERTER_IDX, 7:13] = 0.0
            actor_indices = torch.tensor([INSERTER_IDX], dtype=torch.int32, device="cuda")
            gym.set_actor_root_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(root_states),
                gymtorch.unwrap_tensor(actor_indices), 1,
            )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if record_video:
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
        gym.refresh_actor_root_state_tensor(sim)

        if step < len(traj):
            actual = root_states[INSERTER_IDX, 0:7].cpu().numpy()
            d = float(np.linalg.norm(traj[step][:3] - actual[:3]))
            kp = _keypoints_max_dist(actual, traj[step], kp_offsets)
            deltas.append(d)
            kp_dists.append(kp)
            max_delta = max(max_delta, d)
            max_kp = max(max_kp, kp)
            if step % 50 == 0:
                print(f"  step {step:4d}/{total_steps}: z={actual[2]:.4f}  "
                      f"Δpos={d*1000:.2f}mm  kp={kp*1000:.2f}mm")

        if record_video and step % 5 == 0 and cam_handle is not None:
            img = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
            if img.size > 0:
                img = img.reshape(960, 1280, 4)
                frames.append(img[:, :, :3].copy())

    # Settle check
    gym.refresh_actor_root_state_tensor(sim)
    final_actual = root_states[INSERTER_IDX, 0:7].cpu().numpy()
    settle_kp = _keypoints_max_dist(final_actual, final_pose7, kp_offsets)

    gym.destroy_sim(sim)

    result = {
        "problem":           problem_name,
        "max_pos_delta_mm":  max_delta * 1000,
        "max_kp_dist_mm":    max_kp * 1000,
        "settle_kp_mm":      settle_kp * 1000,
        "collision":         max_kp > 0.001,           # > 1mm = the inserter
                                                       # was pushed off-track
                                                       # by collision
    }

    # Persist artifacts
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if record_video and frames:
            status = "COLLISION" if result["collision"] else "OK"
            video_name = f"{problem_name.replace('.', '_')}_kp{result['max_kp_dist_mm']:.1f}mm_{status}.mp4"
            video_path = output_dir / video_name
            imageio.mimsave(str(video_path), frames, fps=12)
            print(f"  video: {video_path}")
        json_path = output_dir / "results.json"
        json_path.write_text(json.dumps(result, indent=2))
        print(f"  results: {json_path}")

    print()
    status = "COLLISION" if result["collision"] else "OK"
    print(f"  {status}  Δpos_max={result['max_pos_delta_mm']:.2f}mm  "
          f"kp_max={result['max_kp_dist_mm']:.2f}mm  "
          f"settle_kp={result['settle_kp_mm']:.2f}mm")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True,
                        help="Name in PROBLEM_REGISTRY, e.g. fabrica.beam_2x.part_2")
    parser.add_argument("--steps-pre-to-final", type=int, default=200,
                        help="Teleport steps from pre-insert to final pose.")
    parser.add_argument("--settle-frames", type=int, default=180,
                        help="Free-sim frames after reaching final pose.")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override default debug_output/validate_insertions/<problem>/<ts>")
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parent
            / "debug_output" / "validate_insertions"
            / args.problem.replace(".", "_") / timestamp
        )

    validate(
        args.problem,
        steps_pre_to_final=args.steps_pre_to_final,
        settle_frames=args.settle_frames,
        record_video=not args.no_video,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
