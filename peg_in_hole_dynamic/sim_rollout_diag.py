#!/usr/bin/env python
"""Run the depth student in sim and log the same diagnostics the deployment
node prints: per-step mu range, target-q deviation, prev-q deviation.

Reproduces the deployment safety filter check ``max|target - q| > 0.5 rad``
so we can answer "is this expected in sim too?" empirically.

Usage:
    .venv_isaacsim/bin/python -u peg_in_hole_dynamic/sim_rollout_diag.py \\
        --checkpoint hardware_rollouts/2026-05-13_camera_noise_checkpoints/no_delays_no_camnoise/model.pth \\
        --num-steps 120
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


N_ARM = 7
N_HAND = 22
N_ACT = 29

IIWA_NAMES = [f"iiwa_joint_{i + 1}" for i in range(N_ARM)]
SHARPA_NAMES = [f"joint_{i}.0" for i in range(N_HAND)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg.tol0p5mm")
    p.add_argument("--goal-mode", default="finalGoalOnly")
    p.add_argument("--teacher-checkpoint", default=str(
        REPO_ROOT / "train_dir" / "isaacsimenvs" / "play2win_peg_insertion"
        / "lpeg_tol0p5mm_finetune_rgf0_2026-05-11_16-50-01"
        / "0_lpeg_tol0p5mm_finetune_rgf0_2026-05-11_16-50-01" / "best"
        / "model.pth"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-fixed-init", action="store_true",
                   help="Default uses fixed init to match deployment HOME pose.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    largs, _ = lp.parse_known_args([])
    largs.headless = True
    largs.enable_cameras = True
    AppLauncher(largs)

    import math
    import torch
    import gymnasium as gym
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info
    from isaacsimenvs.dagger.teacher import Teacher
    from isaacsimenvs.dagger import networks as _net  # noqa: F401

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides, _configure_agent, _instantiate_env, _load_env_cfg,
    )
    from peg_in_hole_dynamic.eval_student_isaacsim import _build_student

    cfg = _load_env_cfg(args.task)
    _apply_env_overrides(
        cfg, problem=args.problem, goal_mode=args.goal_mode,
        random_goal_fraction=0.0, insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005, num_envs=int(args.num_envs),
        sim_device=args.device, sdf=False, keep_dr=False,
        extra_overrides=({} if args.no_fixed_init else {
            "env.reset.fixed_start_pose": [-0.10, 0.0, 0.63, 1.0, 0.0, 0.0, 0.0],
            "env.peg_in_hole.hole_x_range": [0.10, 0.10],
            "env.peg_in_hole.hole_y_range": [0.0, 0.0],
            "env.reset.reset_position_noise_x": 0.0,
            "env.reset.reset_position_noise_y": 0.0,
            "env.reset.reset_position_noise_z": 0.0,
            "env.reset.reset_dof_pos_random_interval_arm": 0.0,
            "env.reset.reset_dof_pos_random_interval_fingers": 0.0,
            "env.reset.reset_dof_vel_random_interval": 0.0,
            "env.reset.table_reset_z_range": 0.0,
        }),
    )
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); np.random.seed(args.seed)
    env = _instantiate_env(args.task, cfg)
    agent_cfg = _configure_agent(
        args.task, "rl_games_sapg_cfg_entry_point",
        rl_device=args.device, num_envs=int(args.num_envs),
        deterministic=True, games=1, extra_overrides={},
    )
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=args.device,
                                   clip_obs=clip_obs, clip_actions=clip_actions)

    env_info_teacher = teacher_env_info(wrapped)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    student_agent_cfg = load_cfg_from_registry(args.task, "rl_games_dagger_sapg_cfg_entry_point")
    dagger = student_agent_cfg["params"]["config"].get("dagger", {})
    teacher = Teacher(
        task_id=str(dagger.get("teacher_task_id", "Isaacsimenvs-PegInHole-Direct-v0")),
        agent_key=str(dagger.get("teacher_agent_key", "rl_games_sapg_cfg_entry_point")),
        checkpoint_path=args.teacher_checkpoint, num_envs=int(args.num_envs),
        rl_device=args.device, env_info=env_info_teacher,
    )

    net_params = student_agent_cfg["params"]["network"]
    obs0 = env.get_student_obs()
    proprio_dim = int(obs0["proprio"].shape[-1])
    image_channels = int(net_params.get("image_channels", 1))
    image_hw = tuple(net_params["image_hw"])
    has_block_id = bool(net_params.get("has_block_id", True))
    num_blocks = int(student_agent_cfg["params"]["config"].get("expl_coef_block_size", 1))
    if proprio_dim != int(net_params.get("proprio_dim", -1)):
        net_params = dict(net_params); net_params["proprio_dim"] = proprio_dim
    student = _build_student(
        net_params=net_params, action_dim=int(env.action_space.shape[-1]),
        image_channels=image_channels, image_hw=image_hw, proprio_dim=proprio_dim,
        has_block_id=has_block_id, num_blocks=max(num_blocks, 1),
        student_checkpoint=args.checkpoint, device=args.device,
    )
    student.eval()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); np.random.seed(args.seed)
    wrapped.reset()

    rnn_state = None
    print(f"# sim rollout diagnostic: {args.checkpoint}")
    print(f"# num_envs={args.num_envs}  num_steps={args.num_steps}  "
          f"fixed_init={not args.no_fixed_init}")
    print(f"# columns: step | max|mu| @joint | max|target-q| @joint | "
          f"max|prev-q| @joint | mu top-5 saturated joints")

    # Track per-step the env's policy-side metrics. We need access to:
    #   - the action the policy emits (mu)
    #   - the post-EMA target ( env._cur_targets in lab order -> canon order )
    #   - the proprio q (canon order)
    for step in range(int(args.num_steps)):
        out = env.get_student_obs()
        image = out["image"]; proprio = out["proprio"]
        if has_block_id:
            block_id = torch.zeros(image.shape[0], 1, device=image.device)
            flat = torch.cat([image.flatten(1), proprio, block_id], dim=-1)
        else:
            flat = torch.cat([image.flatten(1), proprio], dim=-1)
        with torch.no_grad():
            mu, _ls, _v, rnn_state = student({"obs": flat, "rnn_states": rnn_state, "seq_length": 1})
        mu = torch.clamp(mu, -1.0, 1.0)
        wrapped.step(mu)

        # Pull policy-side values. env._cur_targets is in LAB order; permute
        # to canon to compare with q.
        perm_lab_to_canon = env._perm_lab_to_canon
        cur_canon = env._cur_targets[:, perm_lab_to_canon][0].detach().cpu().numpy()
        prev_canon = env._prev_targets[:, perm_lab_to_canon][0].detach().cpu().numpy()
        q_canon = env.robot.data.joint_pos[:, perm_lab_to_canon][0].detach().cpu().numpy()
        mu_np = mu[0].detach().cpu().numpy()

        dev_tgt = cur_canon - q_canon
        dev_prev = prev_canon - q_canon
        sat_idx = list(np.argsort(-np.abs(mu_np))[:5])

        if step < 30 or step % 5 == 0 or step == args.num_steps - 1:
            sat_str = ",".join(
                f"{i}={'IIWA' if i < N_ARM else 'HAND'}:{mu_np[i]:+.2f}" for i in sat_idx
            )
            i_mu = int(np.argmax(np.abs(mu_np)))
            i_t = int(np.argmax(np.abs(dev_tgt)))
            i_p = int(np.argmax(np.abs(dev_prev)))
            print(
                f"step={step:3d} | "
                f"max|mu|={float(np.abs(mu_np).max()):+.3f}@{i_mu} | "
                f"max|tgt-q|={float(np.abs(dev_tgt).max()):+.3f}@{i_t} | "
                f"max|prev-q|={float(np.abs(dev_prev).max()):+.3f}@{i_p} | "
                f"sat: {sat_str}"
            )

    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
