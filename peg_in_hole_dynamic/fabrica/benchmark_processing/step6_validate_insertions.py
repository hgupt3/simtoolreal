#!/usr/bin/env python3
"""Insertion debug using training-style scene URDFs.

Usage:
    python fabrica/benchmark_processing/step6_validate_insertions.py --assembly beam --part 2 --method coacd
    python fabrica/benchmark_processing/step6_validate_insertions.py --assembly beam --part 0 --method coacd
    python fabrica/benchmark_processing/step6_validate_insertions.py --assembly beam --part 6 --method sdf
"""

import argparse
import datetime
import json
from pathlib import Path

from isaacgym import gymapi, gymtorch

import imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TABLE_Z = 0.38

KEYPOINT_SCALE = 1.5
KEYPOINT_DIRS = np.array([
    [1, 1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
], dtype=np.float64)


def load_canonical_extents(assembly: str, part: str) -> np.ndarray:
    """Load per-part canonical extents from canonical_transforms.json."""
    path = REPO_ROOT / f"assets/urdf/fabrica/{assembly}/canonical_transforms.json"
    if not path.exists():
        raise FileNotFoundError(f"No canonical_transforms.json for {assembly}: {path}")
    with open(path) as f:
        data = json.load(f)
    if part not in data:
        raise KeyError(f"Part {part} not in canonical_transforms.json for {assembly}")
    return np.array(data[part]["canonical_extents"])


def compute_keypoint_offsets(extents: np.ndarray) -> np.ndarray:
    """Compute keypoint offsets from part extents."""
    return KEYPOINT_DIRS * extents[None, :] * KEYPOINT_SCALE / 2


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def compute_keypoints(pose_xyzw, keypoint_offsets):
    pos = pose_xyzw[:3]
    R = quat_xyzw_to_matrix(pose_xyzw[3:7])
    return pos[None, :] + (R @ keypoint_offsets.T).T


def keypoints_max_dist(pose_actual, pose_desired, keypoint_offsets):
    kp_actual = compute_keypoints(pose_actual, keypoint_offsets)
    kp_desired = compute_keypoints(pose_desired, keypoint_offsets)
    dists = np.linalg.norm(kp_actual - kp_desired, axis=1)
    return dists.max()


def slerp(q0, q1, t):
    q0 = np.array(q0, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def interpolate_waypoints(waypoints, steps_per):
    poses = []
    for i in range(len(waypoints) - 1):
        wp0 = np.array(waypoints[i])
        wp1 = np.array(waypoints[i + 1])
        for s in range(steps_per):
            t = s / steps_per
            pos = wp0[:3] + t * (wp1[:3] - wp0[:3])
            quat = slerp(wp0[3:7], wp1[3:7], t)
            poses.append(np.concatenate([pos, quat]))
    poses.append(np.array(waypoints[-1]))
    return poses


def load_trajectory(assembly, part, start_waypoint, end_waypoint):
    traj_path = REPO_ROOT / f"assets/urdf/fabrica/{assembly}/trajectories/{part}/pick_place.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    with open(traj_path) as f:
        traj = json.load(f)

    goals = traj["goals"]
    end = end_waypoint + 1 if end_waypoint is not None else len(goals)
    waypoints = goals[start_waypoint:end]

    if len(waypoints) < 2:
        raise ValueError(
            f"Need at least 2 waypoints, got {len(waypoints)} "
            f"(goals has {len(goals)} entries, slice [{start_waypoint}:{end}])"
        )

    print(f"Loaded {len(waypoints)} waypoints from {traj_path} "
          f"(goals[{start_waypoint}:{end}] of {len(goals)})")
    return waypoints


def compute_camera_for_trajectory(waypoints):
    positions = np.array([wp[:3] for wp in waypoints])
    center = positions.mean(axis=0)
    span = positions.max(axis=0) - positions.min(axis=0)
    max_span = max(span.max(), 0.15)

    cam_pos = gymapi.Vec3(
        center[0] + max_span * 0.6,
        center[1] - max_span * 1.2,
        center[2] + max_span * 0.3,
    )
    cam_target = gymapi.Vec3(center[0], center[1], center[2])
    return cam_pos, cam_target


def main():
    parser = argparse.ArgumentParser(
        description="Debug insertion using training-style scene URDFs")
    parser.add_argument("--assembly", type=str, default="beam")
    parser.add_argument("--part", type=str, default="2")
    parser.add_argument("--method", type=str, default="coacd",
                        choices=["vhacd", "coacd", "sdf"])
    parser.add_argument("--start-waypoint", type=int, default=0)
    parser.add_argument("--end-waypoint", type=int, default=None)
    parser.add_argument("--steps-per-waypoint", type=int, default=50)
    parser.add_argument("--settle-frames", type=int, default=120)
    parser.add_argument("--output-dir", type=str, default="fabrica/debug_output/insertion")
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()

    assembly = args.assembly
    part = args.part

    # Load per-part extents for keypoint computation
    extents = load_canonical_extents(assembly, part)
    keypoint_offsets = compute_keypoint_offsets(extents)
    print(f"Part {part} canonical extents: {extents} (keypoint scale: {KEYPOINT_SCALE})")

    timestamp = args.timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) / assembly / timestamp / f"{assembly}_{part}" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    waypoints = load_trajectory(assembly, part, args.start_waypoint, args.end_waypoint)

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 16
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.005
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.00625
    sim_params.physx.max_depenetration_velocity = 5.0
    sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024
    sim_params.physx.num_threads = 0
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "Failed to create sim"

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    scene_root = f"assets/urdf/fabrica/{assembly}/environments/{part}"
    scene_options = gymapi.AssetOptions()
    scene_options.fix_base_link = True
    scene_options.collapse_fixed_joints = True

    if args.method == "vhacd":
        scene_file = "scene.urdf"
        scene_options.vhacd_enabled = True
    elif args.method == "coacd":
        scene_file = "scene_coacd.urdf"
    elif args.method == "sdf":
        scene_file = "scene_sdf.urdf"
        scene_options.thickness = 0.0

    scene_asset = gym.load_asset(sim, scene_root, scene_file, scene_options)
    print(f"Scene ({args.method}): {gym.get_asset_rigid_body_count(scene_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(scene_asset)} shapes")

    part_options = gymapi.AssetOptions()
    part_options.collapse_fixed_joints = True
    part_options.replace_cylinder_with_capsule = True

    if args.method == "vhacd":
        part_root = f"assets/urdf/fabrica/{assembly}/{part}"
        part_file = f"{assembly}_{part}.urdf"
        part_options.vhacd_enabled = True
    elif args.method == "coacd":
        part_root = f"assets/urdf/fabrica/{assembly}/{part}/coacd"
        part_file = f"{assembly}_{part}_coacd.urdf"
    elif args.method == "sdf":
        part_root = f"assets/urdf/fabrica/{assembly}/{part}"
        part_file = f"{assembly}_{part}_sdf.urdf"
        part_options.thickness = 0.0

    part_asset = gym.load_asset(sim, part_root, part_file, part_options)
    print(f"Part {part} ({args.method}): {gym.get_asset_rigid_body_count(part_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(part_asset)} shapes")

    env_spacing = 0.5
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    scene_pose = gymapi.Transform()
    scene_pose.p = gymapi.Vec3(0.0, 0.0, TABLE_Z)
    scene_pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, scene_asset, scene_pose, "scene", 0, 0)

    wp0 = waypoints[0]
    part_pose = gymapi.Transform()
    part_pose.p = gymapi.Vec3(wp0[0], wp0[1], wp0[2])
    part_pose.r = gymapi.Quat(wp0[3], wp0[4], wp0[5], wp0[6])
    part_handle = gym.create_actor(env, part_asset, part_pose, f"part{part}", 0, 0, 0)

    if args.method == "sdf":
        shape_props = gym.get_actor_rigid_shape_properties(env, part_handle)
        for sp in shape_props:
            sp.thickness = 0.0
        gym.set_actor_rigid_shape_properties(env, part_handle, shape_props)

    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 960
    cam_props.use_collision_geometry = True
    cam_handle = gym.create_camera_sensor(env, cam_props)
    cam_pos, cam_target = compute_camera_for_trajectory(waypoints)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

    gym.prepare_sim(sim)

    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    PART_IDX = 1

    traj_poses = interpolate_waypoints(waypoints, args.steps_per_waypoint)
    total_teleport_steps = len(traj_poses)
    total_steps = total_teleport_steps + args.settle_frames
    print(f"Trajectory: {total_teleport_steps} teleport steps + {args.settle_frames} settle = {total_steps} total")

    frames_dir = output_dir / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    deltas = []
    kp_dists = []
    settle_pose_start = None
    settle_drift_pos = []
    settle_drift_kp = []
    final_desired_pose = np.array(traj_poses[-1])

    print(f"Running insertion test ({assembly} part {part}, {args.method}, training scene)...")
    for step in range(total_steps):
        if step < total_teleport_steps:
            desired = traj_poses[step]
            root_states[PART_IDX, 0:7] = torch.tensor(desired, dtype=torch.float32, device="cuda")
            root_states[PART_IDX, 7:13] = 0.0

            actor_indices = torch.tensor([PART_IDX], dtype=torch.int32, device="cuda")
            gym.set_actor_root_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(root_states),
                gymtorch.unwrap_tensor(actor_indices), 1
            )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        gym.refresh_actor_root_state_tensor(sim)
        actual = root_states[PART_IDX, 0:7].cpu().numpy()

        if step < total_teleport_steps:
            desired_np = traj_poses[step]
            delta = np.linalg.norm(desired_np[:3] - actual[:3])
            kp_dist = keypoints_max_dist(actual, desired_np, keypoint_offsets)
            deltas.append(delta)
            kp_dists.append(kp_dist)
            if step % 25 == 0:
                print(f"  Step {step:4d}/{total_steps}: desired_z={desired_np[2]:.4f} "
                      f"actual_z={actual[2]:.4f} pos_delta={delta*1000:.3f}mm "
                      f"kp_max={kp_dist*1000:.3f}mm")
        else:
            settle_step = step - total_teleport_steps
            if settle_pose_start is None:
                settle_pose_start = actual.copy()
            drift_from_start = np.linalg.norm(actual[:3] - settle_pose_start[:3])
            drift_from_desired = np.linalg.norm(actual[:3] - final_desired_pose[:3])
            kp_drift = keypoints_max_dist(actual, final_desired_pose, keypoint_offsets)
            settle_drift_pos.append(drift_from_start)
            settle_drift_kp.append(kp_drift)
            if settle_step % 25 == 0:
                print(f"  Settle {settle_step:4d}/{args.settle_frames}: "
                      f"pos=({actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.4f}) "
                      f"drift_from_start={drift_from_start*1000:.3f}mm "
                      f"pos_err_vs_desired={drift_from_desired*1000:.3f}mm "
                      f"kp_err_vs_desired={kp_drift*1000:.3f}mm")

        if step % 5 == 0:
            color_image = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
            if color_image.size > 0:
                color_image = color_image.reshape(cam_props.height, cam_props.width, 4)
                frames.append(color_image[:, :, :3].copy())
                frame_path = frames_dir / f"frame_{step:04d}.png"
                imageio.imwrite(str(frame_path), color_image[:, :, :3])

    if deltas:
        deltas = np.array(deltas)
        kp_dists = np.array(kp_dists)
        print(f"\n--- Collision Delta Summary ({assembly} part {part}, {args.method}, training scene) ---")
        print(f"  Position error:")
        print(f"    Max:  {deltas.max()*1000:.3f} mm")
        print(f"    Mean: {deltas.mean()*1000:.3f} mm")
        print(f"    Final: {deltas[-1]*1000:.3f} mm")
        print(f"  Keypoint max dist (training metric):")
        print(f"    Max:  {kp_dists.max()*1000:.3f} mm")
        print(f"    Mean: {kp_dists.mean()*1000:.3f} mm")
        print(f"    Final: {kp_dists[-1]*1000:.3f} mm")
        print(f"  Steps with pos_delta > 1mm: {(deltas > 0.001).sum()}/{len(deltas)}")
        print(f"  Steps with kp_max > 1mm:    {(kp_dists > 0.001).sum()}/{len(kp_dists)}")
        print(f"  Steps with kp_max > 4mm:    {(kp_dists > 0.004).sum()}/{len(kp_dists)}")
        if deltas.max() < 0.001:
            print("  Result: No collisions detected")
        else:
            print(f"  Result: COLLISIONS DETECTED (max pos delta {deltas.max()*1000:.1f}mm, max kp {kp_dists.max()*1000:.1f}mm)")

    if settle_drift_pos:
        sd_pos = np.array(settle_drift_pos)
        sd_kp = np.array(settle_drift_kp)
        print(f"\n--- Settle Stability Summary ({assembly} part {part}, {args.method}, {args.settle_frames} free-physics frames) ---")
        print(f"  Drift from start of settle (no teleport applied):")
        print(f"    Max:  {sd_pos.max()*1000:.3f} mm")
        print(f"    Final: {sd_pos[-1]*1000:.3f} mm")
        print(f"  Pose error vs trajectory final desired (after settle):")
        print(f"    Final pos: {np.linalg.norm(root_states[PART_IDX, 0:3].cpu().numpy() - final_desired_pose[:3])*1000:.3f} mm")
        print(f"    Final kp_max: {sd_kp[-1]*1000:.3f} mm")
        if sd_pos.max() < 0.001:
            print("  Result: STABLE — part holds position with no teleport")
        elif sd_pos.max() < 0.005:
            print(f"  Result: minor drift {sd_pos.max()*1000:.1f}mm (likely settling into contact)")
        else:
            print(f"  Result: UNSTABLE — part drifts {sd_pos.max()*1000:.1f}mm without teleport")

    gym.refresh_actor_root_state_tensor(sim)
    scene_final = root_states[0, 0:7].cpu().numpy()
    part_final = root_states[PART_IDX, 0:7].cpu().numpy()
    print(f"\nFinal scene pose: pos={scene_final[:3]}, quat={scene_final[3:7]}")
    print(f"Final part {part} pose: pos={part_final[:3]}, quat={part_final[3:7]}")

    contact_tensor = gym.acquire_net_contact_force_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    contacts = gymtorch.wrap_tensor(contact_tensor)
    print(f"Final contact force on scene: {contacts[0].cpu().numpy()}")
    print(f"Final contact force on part {part}: {contacts[PART_IDX].cpu().numpy()}")

    if frames:
        final_pos_mm = deltas[-1] * 1000 if len(deltas) > 0 else 0.0
        final_kp_mm = kp_dists[-1] * 1000 if len(kp_dists) > 0 else 0.0
        video_path = output_dir / f"{assembly}_{part}_{args.method}_pos{final_pos_mm:.1f}mm_kp{final_kp_mm:.1f}mm.mp4"
        imageio.mimsave(str(video_path), frames, fps=12)
        print(f"\nSaved {len(frames)} frames to {frames_dir}")
        print(f"Video: {video_path}")
    else:
        print("WARNING: No frames captured. Check graphics_device_id >= 0.")

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
