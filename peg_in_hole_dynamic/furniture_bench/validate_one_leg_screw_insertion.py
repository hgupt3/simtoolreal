#!/usr/bin/env python3
"""Validate the FurnitureBench one-leg task along a screw trajectory.

This is a render/debug validator, not a training env. It loads the current
``furniture_bench.one_leg_sdf_hybrid`` problem, teleports the leg along a
threaded path from a screw-preinsert pose to the assembled pose, lets it
settle, and writes an MP4 plus a small JSON report.

The physics asset keeps the SDF-hybrid collisions. The camera intentionally
renders visual geometry instead of collision geometry:

  * receptive visual: only the local hole-detail patch
  * inserter visual: full canonical leg mesh

Usage:
    python -m peg_in_hole_dynamic.furniture_bench.validate_one_leg_screw_insertion
"""

from __future__ import annotations

import argparse
import datetime
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

# isaacgym must be imported before torch
from isaacgym import gymapi, gymtorch

import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dextoolbench.objects import NAME_TO_OBJECT
from peg_in_hole_dynamic import PROBLEM_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_FB = ASSETS_DIR / "urdf" / "furniture_bench" / "square_table"
PROBLEM_NAME = "furniture_bench.one_leg_sdf_hybrid"

TABLE_RESET_Z = 0.38
TABLE_HALF_HEIGHT = 0.15
TABLE_TOP_Z = TABLE_RESET_Z + TABLE_HALF_HEIGHT

DEFAULT_THREAD_PITCH_MM = 9.37368684342171


def _xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_angle_error_deg(q_actual: np.ndarray, q_desired: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(q_actual * q_desired, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _finite_difference_jerk(pos: np.ndarray, dt: float) -> np.ndarray:
    if len(pos) < 4:
        return np.zeros((0,), dtype=np.float64)
    jerk = np.diff(pos, n=3, axis=0) / (dt ** 3)
    return np.linalg.norm(jerk, axis=1)


def _angular_velocity(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    if len(quat_xyzw) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    rots = R.from_quat(quat_xyzw)
    rel = rots[:-1].inv() * rots[1:]
    return rel.as_rotvec() / dt


def _jerk_summary(values: np.ndarray, prefix: str) -> dict:
    if len(values) == 0:
        return {
            f"{prefix}_rms": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_rms": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _motion_metrics(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    dt: float,
    commanded_steps: int,
) -> dict:
    actual_cmd = actual[:commanded_steps]
    desired_cmd = desired[:commanded_steps]
    pos_err_mm = np.linalg.norm(actual_cmd[:, :3] - desired_cmd[:, :3], axis=1) * 1000.0
    rot_err_deg = _quat_angle_error_deg(actual_cmd[:, 3:7], desired_cmd[:, 3:7])

    actual_pos_jerk = _finite_difference_jerk(actual_cmd[:, :3], dt)
    error_pos_jerk = _finite_difference_jerk(actual_cmd[:, :3] - desired_cmd[:, :3], dt)

    actual_ang_vel = _angular_velocity(actual_cmd[:, 3:7], dt)
    desired_ang_vel = _angular_velocity(desired_cmd[:, 3:7], dt)
    actual_ang_jerk = _finite_difference_jerk(actual_ang_vel, dt)
    error_ang_jerk = _finite_difference_jerk(actual_ang_vel - desired_ang_vel, dt)

    out = {
        "pos_error_rms_mm": float(np.sqrt(np.mean(pos_err_mm ** 2))),
        "pos_error_p95_mm": float(np.percentile(pos_err_mm, 95)),
        "pos_error_max_mm": float(np.max(pos_err_mm)),
        "rot_error_rms_deg": float(np.sqrt(np.mean(rot_err_deg ** 2))),
        "rot_error_p95_deg": float(np.percentile(rot_err_deg, 95)),
        "rot_error_max_deg": float(np.max(rot_err_deg)),
    }
    out.update(_jerk_summary(actual_pos_jerk, "actual_pos_jerk_m_s3"))
    out.update(_jerk_summary(error_pos_jerk, "tracking_error_pos_jerk_m_s3"))
    out.update(_jerk_summary(actual_ang_jerk, "actual_ang_jerk_rad_s3"))
    out.update(_jerk_summary(error_ang_jerk, "tracking_error_ang_jerk_rad_s3"))
    return out


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    out = s0 * q0 + s1 * q1
    return out / np.linalg.norm(out)


def _problem_world_pose(problem) -> Tuple[np.ndarray, R]:
    recv_pos = np.array([0.0, 0.0, TABLE_TOP_Z + problem.hole_z_offset], dtype=np.float64)
    final_pos = recv_pos + np.asarray(problem.insert_pose_rel_receptive[:3], dtype=np.float64)
    final_rot = R.from_quat(problem.insert_pose_rel_receptive[3:7])
    return final_pos, final_rot


def _screw_trajectory(
    problem,
    *,
    pitch_mm: float,
    pre_offset_mm: float,
    steps: int,
    yaw_sign: float,
) -> np.ndarray:
    final_pos, final_rot = _problem_world_pose(problem)
    insertion_dir = np.asarray(problem.insertion_direction, dtype=np.float64)
    insertion_dir /= max(np.linalg.norm(insertion_dir), 1e-9)

    poses = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        backoff_m = (1.0 - t) * pre_offset_mm / 1000.0
        pos = final_pos - insertion_dir * backoff_m
        yaw_deg = yaw_sign * 360.0 * (backoff_m * 1000.0) / pitch_mm
        rot = R.from_euler("z", yaw_deg, degrees=True) * final_rot
        poses.append(np.concatenate([pos, rot.as_quat()]))
    return np.asarray(poses, dtype=np.float64)


def _straight_trajectory(problem, *, pre_offset_mm: float, steps: int) -> np.ndarray:
    final_pos, final_rot = _problem_world_pose(problem)
    insertion_dir = np.asarray(problem.insertion_direction, dtype=np.float64)
    insertion_dir /= max(np.linalg.norm(insertion_dir), 1e-9)

    poses = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        backoff_m = (1.0 - t) * pre_offset_mm / 1000.0
        pos = final_pos - insertion_dir * backoff_m
        poses.append(np.concatenate([pos, final_rot.as_quat()]))
    return np.asarray(poses, dtype=np.float64)


def _interp_to_pose(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
    out = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        pos = start[:3] + t * (end[:3] - start[:3])
        quat = _slerp(start[3:7], end[3:7], t)
        out.append(np.concatenate([pos, quat]))
    return np.asarray(out, dtype=np.float64)


def _absolutize_mesh_filenames(root: ET.Element, urdf_dir: Path) -> None:
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = (urdf_dir / path).resolve()
        mesh.set("filename", path.as_posix())


def _write_temp_receptive_urdf(src_urdf: Path, out_urdf: Path) -> None:
    tree = ET.parse(str(src_urdf))
    root = tree.getroot()
    _absolutize_mesh_filenames(root, src_urdf.parent)

    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            filename = mesh.get("filename", "") if mesh is not None else ""
            if not filename.endswith("one_leg_hole_detail.obj"):
                link.remove(visual)

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_urdf), encoding="utf-8", xml_declaration=True)


def _write_temp_inserter_urdf(src_urdf: Path, out_urdf: Path) -> None:
    tree = ET.parse(str(src_urdf))
    root = tree.getroot()
    _absolutize_mesh_filenames(root, src_urdf.parent)

    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)

    base = root.find("link[@name='base_link']")
    if base is None:
        raise ValueError(f"{src_urdf} has no base_link")

    visual = ET.SubElement(base, "visual")
    geom = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geom, "mesh")
    mesh.set(
        "filename",
        (ASSETS_FB / "square_table_leg4" / "square_table_leg4_canonical.obj").as_posix(),
    )
    mat = ET.SubElement(visual, "material", {"name": "leg_visual"})
    ET.SubElement(mat, "color", {"rgba": "0.85 0.70 0.50 1.0"})

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_urdf), encoding="utf-8", xml_declaration=True)


def _camera_for_pose(world_pos: np.ndarray) -> Tuple[gymapi.Vec3, gymapi.Vec3]:
    return (
        gymapi.Vec3(0.10, -0.18, 0.68),
        gymapi.Vec3(float(world_pos[0]), float(world_pos[1]), float(world_pos[2] - 0.012)),
    )


def validate(
    *,
    mode: str,
    pitch_mm: float,
    pre_offset_mm: float,
    steps: int,
    settle_frames: int,
    yaw_sign: float,
    output_dir: Path,
) -> dict:
    problem = PROBLEM_REGISTRY[PROBLEM_NAME]
    obj = NAME_TO_OBJECT[problem.insertion_object_name]

    source_receptive = ASSETS_DIR / problem.receptive_urdf
    source_inserter = Path(obj.urdf_path)
    render_dir = output_dir / "render_urdfs"
    receptive_urdf = render_dir / "one_leg_receptive_visual_hole_only.urdf"
    inserter_urdf = render_dir / "one_leg_inserter_visual_full_leg.urdf"
    _write_temp_receptive_urdf(source_receptive, receptive_urdf)
    _write_temp_inserter_urdf(source_inserter, inserter_urdf)

    if mode == "screw":
        trajectory = _screw_trajectory(
            problem,
            pitch_mm=pitch_mm,
            pre_offset_mm=pre_offset_mm,
            steps=steps,
            yaw_sign=yaw_sign,
        )
        total_yaw_deg = yaw_sign * 360.0 * pre_offset_mm / pitch_mm
    elif mode == "straight":
        trajectory = _straight_trajectory(
            problem,
            pre_offset_mm=pre_offset_mm,
            steps=steps,
        )
        total_yaw_deg = 0.0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    final_pose = trajectory[-1]

    print(f"Problem: {PROBLEM_NAME}")
    print(f"  physics source receptive: {source_receptive.relative_to(ASSETS_DIR)}")
    print(f"  physics source inserter:  {source_inserter.relative_to(ASSETS_DIR)}")
    print(f"  render/physics URDFs:     {render_dir}")
    print(f"  mode: {mode}")
    print(f"  pitch: {pitch_mm:.4f} mm/turn")
    print(f"  pre-offset: {pre_offset_mm:.3f} mm")
    print(f"  total yaw: {total_yaw_deg:.3f} deg")
    print("  camera: visual geometry, not collision geometry")

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
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    table_options = gymapi.AssetOptions()
    table_options.fix_base_link = True
    table_asset = gym.create_box(sim, 0.475, 0.4, 0.3, table_options)

    recv_options = gymapi.AssetOptions()
    recv_options.fix_base_link = True
    recv_options.collapse_fixed_joints = True
    recv_asset = gym.load_asset(
        sim, str(receptive_urdf.parent), receptive_urdf.name, recv_options,
    )

    ins_options = gymapi.AssetOptions()
    ins_options.collapse_fixed_joints = True
    ins_options.replace_cylinder_with_capsule = True
    ins_asset = gym.load_asset(
        sim, str(inserter_urdf.parent), inserter_urdf.name, ins_options,
    )
    print(f"  receptive: {gym.get_asset_rigid_body_count(recv_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(recv_asset)} shapes")
    print(f"  inserter:  {gym.get_asset_rigid_body_count(ins_asset)} bodies, "
          f"{gym.get_asset_rigid_shape_count(ins_asset)} shapes")

    env = gym.create_env(
        sim, gymapi.Vec3(-0.5, -0.5, 0.0),
        gymapi.Vec3(0.5, 0.5, 0.5),
        1,
    )

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, TABLE_RESET_Z)
    table_pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

    recv_pose = gymapi.Transform()
    recv_pose.p = gymapi.Vec3(0.0, 0.0, TABLE_TOP_Z + problem.hole_z_offset)
    recv_pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, recv_asset, recv_pose, "receptive", 0, 0)

    ins_pose = gymapi.Transform()
    ins_pose.p = gymapi.Vec3(*trajectory[0, :3])
    ins_pose.r = gymapi.Quat(*trajectory[0, 3:7])
    ins_actor = gym.create_actor(env, ins_asset, ins_pose, "inserter", 0, 0, 0)

    for i in range(gym.get_asset_rigid_body_count(ins_asset)):
        gym.set_rigid_body_color(
            env, ins_actor, i, gymapi.MESH_VISUAL, gymapi.Vec3(0.86, 0.70, 0.48),
        )

    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 960
    cam_props.use_collision_geometry = False
    cam_handle = gym.create_camera_sensor(env, cam_props)
    cam_pos, cam_target = _camera_for_pose(final_pose[:3])
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

    gym.prepare_sim(sim)
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor)
    device = root_states.device
    inserter_idx = gym.get_actor_index(env, ins_actor, gymapi.DOMAIN_SIM)

    frames = []
    actual_log = []
    desired_log = []
    max_pos_delta = 0.0
    max_kp_delta = 0.0
    total_steps = len(trajectory) + settle_frames

    for step in range(total_steps):
        if step < len(trajectory):
            desired = trajectory[step]
            root_states[inserter_idx, 0:7] = torch.as_tensor(
                desired, dtype=torch.float32, device=device,
            )
            root_states[inserter_idx, 7:13] = 0.0
            actor_indices = torch.tensor([inserter_idx], dtype=torch.int32, device=device)
            gym.set_actor_root_state_tensor_indexed(
                sim,
                gymtorch.unwrap_tensor(root_states),
                gymtorch.unwrap_tensor(actor_indices),
                1,
            )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.refresh_actor_root_state_tensor(sim)

        actual_full = root_states[inserter_idx, 0:7].cpu().numpy().copy()
        desired_full = trajectory[step] if step < len(trajectory) else final_pose
        actual_log.append(actual_full)
        desired_log.append(desired_full.copy())

        if step < len(trajectory):
            actual = actual_full
            desired = trajectory[step]
            pos_delta = float(np.linalg.norm(actual[:3] - desired[:3]))
            rot_delta = np.linalg.norm(
                _xyzw_to_matrix(actual[3:7]) - _xyzw_to_matrix(desired[3:7]),
                ord="fro",
            )
            max_pos_delta = max(max_pos_delta, pos_delta)
            max_kp_delta = max(max_kp_delta, rot_delta)
            if step % 50 == 0:
                if mode == "screw":
                    yaw_backoff = (
                        yaw_sign * 360.0
                        * (1.0 - step / max(len(trajectory) - 1, 1))
                        * pre_offset_mm / pitch_mm
                    )
                else:
                    yaw_backoff = 0.0
                print(
                    f"  step {step:4d}/{total_steps}: z={actual[2]:.5f} "
                    f"yaw_backoff={yaw_backoff:.1f}deg "
                    f"pos_delta={pos_delta * 1000:.3f}mm"
                )

        if step % 3 == 0:
            img = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
            if img.size > 0:
                frames.append(img.reshape(cam_props.height, cam_props.width, 4)[:, :, :3].copy())

    final_actual = root_states[inserter_idx, 0:7].cpu().numpy()
    final_pos_delta = float(np.linalg.norm(final_actual[:3] - final_pose[:3]))

    gym.destroy_sim(sim)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / (
        f"{PROBLEM_NAME.replace('.', '_')}_{mode}_pitch{pitch_mm:.2f}mm.mp4"
    )
    imageio.mimsave(str(video_path), frames, fps=20)

    result = {
        "problem": PROBLEM_NAME,
        "mode": mode,
        "pitch_mm": pitch_mm,
        "pre_offset_mm": pre_offset_mm,
        "yaw_sign": yaw_sign,
        "total_yaw_deg": total_yaw_deg,
        "steps": int(steps),
        "settle_frames": int(settle_frames),
        "max_pos_delta_mm": max_pos_delta * 1000.0,
        "max_rot_matrix_fro_delta": max_kp_delta,
        "final_pos_delta_mm": final_pos_delta * 1000.0,
        "video": str(video_path),
        "rendered_collision_geometry": False,
    }
    actual_arr = np.asarray(actual_log, dtype=np.float64)
    desired_arr = np.asarray(desired_log, dtype=np.float64)
    result.update(_motion_metrics(
        actual_arr,
        desired_arr,
        dt=float(sim_params.dt),
        commanded_steps=len(trajectory),
    ))
    (output_dir / "results.json").write_text(json.dumps(result, indent=2))
    (output_dir / "trajectory.json").write_text(json.dumps(trajectory.tolist()))
    (output_dir / "pose_log.json").write_text(json.dumps({
        "actual": actual_arr.tolist(),
        "desired": desired_arr.tolist(),
        "dt": float(sim_params.dt),
        "commanded_steps": len(trajectory),
    }))

    print(f"  video: {video_path}")
    print(f"  results: {output_dir / 'results.json'}")
    print(
        "  jerk rms: "
        f"actual_pos={result['actual_pos_jerk_m_s3_rms']:.3f} m/s^3, "
        f"tracking_error_pos={result['tracking_error_pos_jerk_m_s3_rms']:.3f} m/s^3, "
        f"actual_ang={result['actual_ang_jerk_rad_s3_rms']:.3f} rad/s^3"
    )
    print(f"  final_pos_delta={final_pos_delta * 1000.0:.3f}mm")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("screw", "straight"), default="screw",
        help="screw follows z+yaw; straight descends at fixed final yaw.",
    )
    parser.add_argument("--pitch-mm", type=float, default=DEFAULT_THREAD_PITCH_MM)
    parser.add_argument(
        "--turns", type=float, default=None,
        help="Override --pitch-mm by specifying total turns over --pre-offset-mm.",
    )
    parser.add_argument("--pre-offset-mm", type=float, default=25.0)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--settle-frames", type=int, default=90)
    parser.add_argument("--yaw-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir else
        REPO_ROOT / "peg_in_hole_dynamic" / "debug_output"
        / "validate_one_leg_screw_insertion" / timestamp
    )

    pitch_mm = args.pitch_mm
    if args.turns is not None:
        if args.turns <= 0.0:
            raise SystemExit("--turns must be positive")
        pitch_mm = args.pre_offset_mm / args.turns

    validate(
        mode=args.mode,
        pitch_mm=pitch_mm,
        pre_offset_mm=args.pre_offset_mm,
        steps=args.steps,
        settle_frames=args.settle_frames,
        yaw_sign=args.yaw_sign,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
