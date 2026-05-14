#!/usr/bin/env python
"""Spawn Isaac Lab env at fixed-init and dump one preprocessed student depth.

Mirrors the "Init pose=fixed" path the GUI eval uses, so the output is the
same depth view the policy receives at training/eval time. Pair with
``deployment/capture_real_first_depth.py`` to compare real vs sim.

Example:
    .venv_isaacsim/bin/python -u peg_in_hole_dynamic/capture_sim_depth.py \\
        --out-dir /tmp/depth_compare
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Real-robot home pose used by deployment/home_robot.py. Robot is driven here
# at the start of every real-world session, so the sim depth must be captured
# with the robot at the same joint state for the comparison to be fair.
# Order = canonical (concat[iiwa_joint_1..7, joint_0.0..21.0]).
HOME_JOINT_POS_IIWA = np.array([
    -1.571,
    1.571 - np.deg2rad(10),
    0.0,
    1.376 + np.deg2rad(10),
    0.0,
    1.485,
    1.308,
], dtype=np.float64)
HOME_JOINT_POS_SHARPA = np.zeros(22, dtype=np.float64)
HOME_JOINT_POS_CANON = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])

# NOTE: peg z=0.541 (resting on the real-world table top the user measured)
# rather than the training fixed_start_pose z=0.63 (hovering ~0.10 m above).
# This is for a snap-image comparison test only.
_FIXED_INIT_OVERRIDES = {
    "env.reset.fixed_start_pose": [-0.10, 0.0, 0.541, 1.0, 0.0, 0.0, 0.0],
    "env.peg_in_hole.hole_x_range": [0.10, 0.10],
    "env.peg_in_hole.hole_y_range": [0.0, 0.0],
    "env.reset.reset_position_noise_x": 0.0,
    "env.reset.reset_position_noise_y": 0.0,
    "env.reset.reset_position_noise_z": 0.0,
    "env.reset.reset_dof_pos_random_interval_arm": 0.0,
    "env.reset.reset_dof_pos_random_interval_fingers": 0.0,
    "env.reset.reset_dof_vel_random_interval": 0.0,
    "env.reset.table_reset_z_range": 0.0,
}


def _save_frame(out_dir: Path, name: str, frame: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    png_path = out_dir / f"{name}.png"
    np.savez_compressed(npz_path, image=frame)
    from PIL import Image

    img = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(png_path)
    return png_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg.tol0p5mm")
    p.add_argument("--goal-mode", default="finalGoalOnly")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--out-dir", type=str, default="/tmp/depth_compare")
    p.add_argument("--name", type=str, default="sim_depth")
    p.add_argument("--no-fixed-init", action="store_true",
                   help="Skip the fixed-init overrides; use task yaml defaults.")
    return p.parse_args()


def _reseed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import isaacsimenvs  # noqa: F401  (registers gym tasks)
    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _instantiate_env,
        _load_env_cfg,
    )

    cfg = _load_env_cfg(args.task)
    extra = {} if args.no_fixed_init else dict(_FIXED_INIT_OVERRIDES)
    _apply_env_overrides(
        cfg,
        problem=args.problem,
        goal_mode=args.goal_mode,
        random_goal_fraction=0.0,
        insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005,
        num_envs=int(args.num_envs),
        sim_device=args.sim_device,
        sdf=False,
        keep_dr=False,
        extra_overrides=extra,
    )
    # Force all DR knobs off so the sim depth we save is the deterministic
    # canonical view (no cam_pose_rand, no depth_aug, no obs/action delay).
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False

    print(f"=> spawning env (problem={args.problem}, goal_mode={args.goal_mode}, "
          f"fixed_init={not args.no_fixed_init}, seed={args.seed})", flush=True)

    _reseed(args.seed)
    env = _instantiate_env(args.task, cfg)

    # Prime via two resets so the camera renders the post-reset scene.
    _reseed(args.seed)
    env.reset()
    _reseed(args.seed)
    env.reset()

    # Override the sim robot pose to the lab's HOME_JOINT_POS so the depth we
    # save matches the pose the real robot is driven to before the policy
    # starts. env.robot expects joint pos in LAB order; HOME_JOINT_POS_CANON
    # is in canonical order (concat[iiwa1..7, joint_0.0..21.0]), so reorder
    # via env._perm_canon_to_lab before writing.
    import torch

    home_canon = torch.tensor(
        HOME_JOINT_POS_CANON, device=env.device, dtype=torch.float32
    )
    home_lab = home_canon[env._perm_canon_to_lab]
    n_env = int(env.num_envs)
    pos_lab = home_lab.unsqueeze(0).expand(n_env, -1).contiguous()
    vel_lab = torch.zeros_like(pos_lab)
    env_ids = torch.arange(n_env, device=env.device, dtype=torch.long)
    env.robot.write_joint_state_to_sim(pos_lab, vel_lab, env_ids=env_ids)
    # Seed prev/cur targets so proprio's prev_action_targets reflects home.
    if hasattr(env, "_prev_targets"):
        env._prev_targets[env_ids] = pos_lab
    if hasattr(env, "_cur_targets"):
        env._cur_targets[env_ids] = pos_lab

    # One zero-action step so the camera renders the home-pose scene.
    zero_action = torch.zeros(n_env, int(env.action_space.shape[-1]),
                              device=env.device)
    env.step(zero_action)

    obs = env.get_student_obs()
    image = obs["image"]  # (N, C, H, W) float32, already preprocessed
    print(f"=> obs image shape={tuple(image.shape)} dtype={image.dtype} "
          f"min={float(image.min()):.3f} mean={float(image.mean()):.3f} "
          f"max={float(image.max()):.3f}", flush=True)

    frame = image[0, 0].detach().float().cpu().numpy()
    out_dir = Path(args.out_dir).resolve()
    png = _save_frame(out_dir, args.name, frame)
    print(f"=> saved {png}  shape={frame.shape}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
