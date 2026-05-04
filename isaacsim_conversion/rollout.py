"""
Full policy rollout in Isaac Sim.
Phase 9 of the plan: obs → policy → action → physics loop.

Usage:
    source .venv_isaacsim/bin/activate
    PYTHONPATH=. python isaacsim_conversion/rollout.py \
        --assembly beam --part_id 2 --collision_method coacd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deployment.rl_player import RlPlayer
from isaacsim_conversion.isaacsim_env import _log
from isaacgymenvs.utils.observation_action_utils_sharpa import (
    JOINT_NAMES_ISAACGYM,
    N_OBS,
    _compute_keypoint_positions,
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)


def launch_app():
    """Create AppLauncher FIRST, before any isaaclab/omni imports."""
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_source",
        choices=["fabrica", "dextoolbench"],
        default="fabrica",
        help="Which asset/trajectory layout to use.",
    )
    parser.add_argument("--assembly", default="beam")
    parser.add_argument("--part_id", default="2")
    parser.add_argument("--collision_method", default="coacd")
    parser.add_argument("--object_category", default="hammer")
    parser.add_argument("--object_name", default="claw_hammer")
    parser.add_argument("--task_name", default="swing_down")
    parser.add_argument("--max_steps", type=int, default=6000, help="Max sim steps (100s at 60Hz)")
    parser.add_argument("--video_dir", default="rollout_videos", help="Directory to save video")
    parser.add_argument("--video_fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--checkpoint", default="pretrained_policy/model.pth", help="Policy checkpoint path")
    parser.add_argument("--config", default="pretrained_policy/config.yaml", help="Policy config path")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    return app_launcher.app, args


# Must happen at module level, before any other imports
_app, _args = launch_app()


def main():
    args = _args
    app = _app
    repo_root = Path(__file__).parent.parent
    device = "cuda"
    import isaaclab.sim as sim_utils

    # --- Locate assets ---
    robot_urdf = str(repo_root / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf")

    # Object/table/task selection
    from dextoolbench.objects import NAME_TO_OBJECT

    if args.task_source == "fabrica":
        table_urdf = str(
            repo_root
            / f"assets/urdf/fabrica/{args.assembly}/environments/{args.part_id}/scene_{args.collision_method}.urdf"
        )
        traj_path = repo_root / f"assets/urdf/fabrica/{args.assembly}/trajectories/{args.part_id}/pick_place.json"
        object_name = f"{args.assembly}_{args.part_id}_{args.collision_method}"
        import peg_in_hole_dynamic.fabrica.objects  # noqa: registers fabrica parts
    else:
        table_urdf = str(
            repo_root
            / f"assets/urdf/dextoolbench/environments/{args.object_category}/{args.object_name}/{args.task_name}.urdf"
        )
        traj_path = repo_root / f"dextoolbench/trajectories/{args.object_category}/{args.object_name}/{args.task_name}.json"
        object_name = args.object_name

    obj_info = NAME_TO_OBJECT.get(object_name)
    assert obj_info is not None, f"Object '{object_name}' not found in NAME_TO_OBJECT"
    object_urdf = obj_info.urdf_path

    print(f"Robot URDF:  {robot_urdf}")
    print(f"Table URDF:  {table_urdf}")
    print(f"Object URDF: {object_urdf}")
    print(f"Trajectory:  {traj_path}")

    # --- Load trajectory ---
    with open(traj_path) as f:
        traj = json.load(f)
    goals = traj["goals"]  # list of [x,y,z,qx,qy,qz,qw]
    start_pose = traj["start_pose"]  # [x,y,z,qx,qy,qz,qw]
    print(f"Trajectory: {len(goals)} goals, start_pose={start_pose}")

    # --- Create Isaac Sim environment ---
    from isaacsim_conversion.isaacsim_env import IsaacSimEnv
    env = IsaacSimEnv(
        robot_urdf=robot_urdf,
        table_urdf=table_urdf,
        object_urdf=object_urdf,
        headless=True,
        app=app,
    )

    start_pos = np.array(start_pose[:3], dtype=np.float32)
    start_quat_xyzw = np.array(start_pose[3:7], dtype=np.float32)

    # --- Load policy ---
    print("\nLoading policy...")
    checkpoint_path = str(repo_root / args.checkpoint) if not args.checkpoint.startswith("/") else args.checkpoint
    config_path = str(repo_root / args.config) if not args.config.startswith("/") else args.config
    _log(f"Loading policy: checkpoint={checkpoint_path}, config={config_path}")
    player = RlPlayer(
        num_observations=140,
        num_actions=29,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        num_envs=1,
    )

    # --- Load URDF for FK ---
    urdf = create_urdf_object("iiwa14_left_sharpa_adjusted_restricted")

    # --- Config from training ---
    obs_list = [
        "joint_pos", "joint_vel", "prev_action_targets", "palm_pos",
        "palm_rot", "object_rot", "fingertip_pos_rel_palm",
        "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
    ]
    import yaml
    with open(config_path) as f:
        policy_cfg = yaml.safe_load(f)
    if args.task_source == "fabrica":
        # Fabrica transfer uses fixedSize from the training config.
        fixed_size = policy_cfg.get("task", {}).get("env", {}).get("fixedSize", [0.141, 0.03025, 0.0271])
        object_scales = np.array([fixed_size], dtype=np.float32)
        _log(f"Object scales (fixedSize): {object_scales[0]}")
    else:
        # DexToolBench deployment/eval uses the object grasp bounding box scale from NAME_TO_OBJECT.
        object_scales = np.array([obj_info.scale], dtype=np.float32)
        _log(f"Object scales (NAME_TO_OBJECT.scale): {object_scales[0]}")
    hand_moving_average = 0.1
    arm_moving_average = 0.1
    dof_speed_scale = 1.5
    dt = 1 / 60
    success_steps = 10
    keypoint_tolerance = 0.01 * 1.5  # targetSuccessTolerance * keypointScale

    # --- Set up video recording ---
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"rollout_{args.assembly}_{args.part_id}.mp4"
    frames = []

    # Record trajectory data for debugging/visualization
    trajectory_log = []

    # Set up camera for video recording (requires --enable_cameras flag)
    has_camera = False
    if getattr(args, 'enable_cameras', False):
        try:
            from isaaclab.sensors import Camera, CameraCfg
            camera_cfg = CameraCfg(
                prim_path="/World/RecordCamera",
                update_period=0,
                height=480,
                width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    # Match Isaac Gym: cam_pos=(0, -1, 1.03), cam_target=(0, 0, 0.53)
                    # opengl convention: -Z=forward, +Y=up
                    pos=(0.0, -1.0, 1.03),
                    rot=(0.8507, 0.5257, 0.0, 0.0),  # wxyz
                    convention="opengl",
                ),
            )
            camera = Camera(cfg=camera_cfg)
            # Camera needs sim reset to initialize internal buffers (_timestamp etc.)
            env.sim.reset()
            has_camera = True
            _log(f"Camera created for video recording to {video_dir}")
        except Exception as e:
            _log(f"Camera setup failed: {e}")
    else:
        _log("Video recording disabled (pass --enable_cameras to enable)")

    # Place object at start pose (must be after camera reset)
    env.set_object_pose(start_pos, start_quat_xyzw)

    # --- Set robot to default pose (match Isaac Gym) ---
    env.reset_robot_to_default_pose(render=False)
    q_init, _ = env.get_robot_state()
    _log(f"Robot default pose set. Arm joints: {q_init[:7]}")
    prev_targets = q_init[None].copy()  # (1, 29)

    current_goal_idx = 0
    goal_pose = np.array(goals[current_goal_idx], dtype=np.float32)[None]  # (1, 7)
    near_goal_steps = 0

    player.player.init_rnn()  # fresh LSTM state

    _log(f"\n=== Starting rollout: {args.max_steps} steps, {len(goals)} goals ===\n")

    # --- Main loop ---
    for step_i in range(args.max_steps):
        # 1. Extract state
        q, qd = env.get_robot_state()
        object_pose = env.get_object_pose_xyzw()[None]  # (1, 7)

        # 2. Compute observation
        obs = compute_observation(
            q=q[None], qd=qd[None], prev_action_targets=prev_targets,
            object_pose=object_pose, goal_object_pose=goal_pose,
            object_scales=object_scales, urdf=urdf, obs_list=obs_list,
        )

        # 3. Policy inference
        obs_tensor = torch.from_numpy(obs).float().to(device)
        action = player.get_normalized_action(obs_tensor, deterministic_actions=True)

        # 4. Compute joint targets
        targets = compute_joint_pos_targets(
            actions=action.cpu().numpy(),
            prev_targets=prev_targets,
            hand_moving_average=hand_moving_average,
            arm_moving_average=arm_moving_average,
            hand_dof_speed_scale=dof_speed_scale,
            dt=dt,
        )
        prev_targets = targets

        # 5. Apply and step
        env.set_joint_position_targets(targets[0])
        env.step(render=True)  # Always render for camera

        # 6. Capture frame (every 2nd step = 30fps video from 60Hz sim)
        if has_camera and step_i % 2 == 0:
            camera.update(dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frames.append(rgb[0].cpu().numpy()[:, :, :3])

        # 7. Goal switching
        object_kps = _compute_keypoint_positions(object_pose, object_scales)
        goal_kps = _compute_keypoint_positions(goal_pose, object_scales)
        keypoints_max_dist = np.max(np.linalg.norm(
            object_kps[0] - goal_kps[0], axis=-1
        ))

        if keypoints_max_dist < keypoint_tolerance:
            near_goal_steps += 1
        else:
            near_goal_steps = 0  # forceConsecutiveNearGoalSteps=True

        if near_goal_steps >= success_steps:
            _log(f"[step {step_i}] Goal {current_goal_idx} REACHED! (dist={keypoints_max_dist:.4f})")
            current_goal_idx += 1
            near_goal_steps = 0
            if current_goal_idx >= len(goals):
                _log(f"\n=== ALL {len(goals)} GOALS REACHED at step {step_i}! ===")
                break
            goal_pose = np.array(goals[current_goal_idx], dtype=np.float32)[None]
            _log(f"  -> Advancing to goal {current_goal_idx}/{len(goals)}")

        # Record trajectory data
        trajectory_log.append({
            "step": step_i,
            "q": q.copy(),
            "object_pose": object_pose[0].copy(),
            "kp_dist": keypoints_max_dist,
            "goal_idx": current_goal_idx,
        })

        # Periodic logging
        if step_i % 60 == 0:
            obj_z = object_pose[0, 2]
            _log(
                f"[step {step_i:5d}] goal={current_goal_idx}/{len(goals)}, "
                f"kp_dist={keypoints_max_dist:.4f}, obj_z={obj_z:.3f}, "
                f"near_goal={near_goal_steps}/{success_steps}"
            )

    _log(f"\nRollout complete. Final goal: {current_goal_idx}/{len(goals)}")

    # Save trajectory data
    if trajectory_log:
        traj_file = video_dir / "trajectory.npz"
        np.savez(
            str(traj_file),
            steps=np.array([t["step"] for t in trajectory_log]),
            q=np.array([t["q"] for t in trajectory_log]),
            object_poses=np.array([t["object_pose"] for t in trajectory_log]),
            kp_dists=np.array([t["kp_dist"] for t in trajectory_log]),
            goal_idxs=np.array([t["goal_idx"] for t in trajectory_log]),
        )
        _log(f"Trajectory saved: {traj_file} ({len(trajectory_log)} steps)")

    # Save video
    if frames:
        import imageio
        _log(f"Saving {len(frames)} frames to {video_path}")
        imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
        _log(f"Video saved: {video_path}")
    else:
        _log("WARNING: No frames captured for video")

    env.close()


if __name__ == "__main__":
    main()
