"""Smoke test: load the teacher via the same path as DAggerA2CAgent, then
roll the depth-student env using ONLY teacher actions. Print per-step stats
on the teacher's μ output to confirm whether the teacher produces NaN.

The intent is to isolate whether the NaN we observed in the BC training
(`dagger/teacher_action_l2 = NaN` for 47k+ minibatches, `dagger/L_D = NaN`,
student weights stuck at init via mixed-precision GradScaler) comes from:
  (a) the Teacher class loading path (this smoke),
  (b) the env's teacher_obs feeding NaN to the teacher,
  (c) some downstream interaction in calc_gradients / experience buffer.

Run with `.venv_isaacsim/bin/python -u isaacsimenvs/dagger/smoke_teacher_only.py
--teacher-run-dir <path>`. Prints to stdout — no wandb.
"""
from __future__ import annotations

import argparse
import math
import os
import sys


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    parser.add_argument(
        "--teacher-task-id",
        default="Isaacsimenvs-PegInHole-Direct-v0",
        help="Same default as DAggerA2CAgent: load teacher cfg from the state-MLP task.",
    )
    parser.add_argument("--teacher-run-dir", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--problem", default="Lpeg.tol0p5mm")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--override-block-id", type=float, default=None,
        help="Replace `player.intr_reward_coef_embd` with all-`X` after teacher init. "
             "Use 0.0 for least-exploratory block, 50.0 for most.",
    )
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args_cli).app

    import gymnasium as gym
    import torch
    from pathlib import Path

    import isaacsimenvs  # noqa: F401 — gym.register side effects
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info
    from isaacsimenvs.dagger.teacher import Teacher, latest_checkpoint

    @hydra_task_config_with_yaml(args_cli.task, "rl_games_dagger_sapg_cfg_entry_point")
    def run(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.scene.replicate_physics = True
        env_cfg.peg_in_hole.problem = args_cli.problem
        env_cfg.sim.device = args_cli.sim_device

        env = gym.make(args_cli.task, cfg=env_cfg)
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
        wrapped = register_rlgames_env(env, rl_device=args_cli.rl_device,
                                       clip_obs=clip_obs, clip_actions=clip_actions)

        info = teacher_env_info(wrapped)
        print(f"[smoke] teacher env_info from central wrapper: "
              f"obs={info['observation_space']} action={info['action_space']}", flush=True)

        ckpt = (Path(args_cli.teacher_checkpoint) if args_cli.teacher_checkpoint
                else latest_checkpoint(args_cli.teacher_run_dir))
        teacher = Teacher(
            task_id=args_cli.teacher_task_id,
            agent_key="rl_games_sapg_cfg_entry_point",
            checkpoint_path=ckpt,
            num_envs=args_cli.num_envs,
            rl_device=args_cli.rl_device,
            env_info=info,
        )
        print(f"[smoke] teacher loaded from {ckpt}", flush=True)
        if args_cli.override_block_id is not None:
            teacher.pin_block_id(args_cli.override_block_id)
            print(f"[smoke] OVERRIDE intr_reward_coef_embd to constant {args_cli.override_block_id}",
                  flush=True)
        embd = teacher.intr_reward_coef_embd
        print(f"[smoke] teacher.intr_reward_coef_embd: shape={None if embd is None else embd.shape} "
              f"vals={None if embd is None else embd.flatten()[:8].tolist()} "
              f"min={None if embd is None else float(embd.min())} "
              f"max={None if embd is None else float(embd.max())} "
              f"has_nan={None if embd is None else bool(embd.isnan().any())}", flush=True)

        obs_dict, _ = env.reset()
        print(f"[smoke] reset obs keys: {list(obs_dict.keys())}", flush=True)
        for key, val in obs_dict.items():
            t = val
            print(f"  obs['{key}']: shape={tuple(t.shape)} "
                  f"min={float(t.min()):.4f} max={float(t.max()):.4f} "
                  f"mean={float(t.mean()):.4f} has_nan={bool(t.isnan().any())} "
                  f"has_inf={bool(t.isinf().any())}", flush=True)

        # Roll using teacher actions and track functional success.
        unwrapped = env.unwrapped
        total_successes = torch.zeros(args_cli.num_envs, dtype=torch.long, device=args_cli.rl_device)
        total_resets = 0
        for step in range(args_cli.steps):
            teacher_obs = obs_dict["teacher_obs"]

            with torch.no_grad():
                action = teacher.get_action(teacher_obs)

            obs_dict, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            cur_succ = unwrapped._successes.long() if hasattr(unwrapped, "_successes") else None
            if cur_succ is not None:
                total_successes = torch.maximum(total_successes, cur_succ)

            if done.any():
                done_idx = done.nonzero(as_tuple=False).flatten()
                teacher.reset_idx(done_idx)
                total_resets += int(done_idx.numel())

            if (step + 1) % max(1, args_cli.steps // 10) == 0 or step == args_cli.steps - 1:
                if cur_succ is not None:
                    print(f"[step {step+1:4d}/{args_cli.steps}] "
                          f"action |a|>1={float((action.abs() > 1).float().mean()):.3f} "
                          f"has_nan={bool(action.isnan().any())} "
                          f"peak_succ_per_env={total_successes.tolist()} "
                          f"resets={total_resets}",
                          flush=True)
                else:
                    print(f"[step {step+1:4d}/{args_cli.steps}] action has_nan={bool(action.isnan().any())}",
                          flush=True)

        peak = int(total_successes.max()) if hasattr(unwrapped, "_successes") else -1
        succ_envs = int((total_successes > 0).sum())
        print(f"\n[smoke] DONE — peak_successes_any_env={peak} "
              f"envs_with_any_success={succ_envs}/{args_cli.num_envs} "
              f"total_resets={total_resets}",
              flush=True)

    run()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
