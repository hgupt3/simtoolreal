"""Fixed-init teacher eval. Rolls the SAPG teacher on the depth-student env
and reports per-episode success rate. Optionally records env 0's student depth
input as an mp4 for visual inspection.

Two modes (both via this one script):
  - Aggregate-success mode (default): many envs, headless, prints overall
    success rate after each env has completed `--num-episodes` episodes.
  - Depth-video mode: `--num-envs 1 --save-depth-video <path>.mp4` writes one
    episode of the exact (70x70) depth tensor the student CNN reads, for
    visual inspection of the obs the student is supervised on.

The success metric is `env._termination_reasons["max_successes"]` (set in
peg_in_hole_env.py at the end of `_get_dones`). An episode counts as a success
iff that flag fires on the step the episode ends, i.e. the env hit
`_successes >= env_max_goals` (not a fall, not a hand_far timeout).

Run with `.venv_isaacsim/bin/python -u isaacsimenvs/dagger/eval_teacher_fixed_init.py
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
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Min completed episodes per env before we stop.")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Hard cap on total step count (safety).")
    parser.add_argument(
        "--override-block-id", type=float, default=None,
        help="Pin `intr_reward_coef_embd` to a constant. 50.0 matches the dagger trainer default.",
    )
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--save-depth-video", default=None,
                        help="If set, write env 0's student depth (70x70) for one episode "
                             "as an mp4 to this path. Implies --num-envs 1.")
    parser.add_argument("--depth-video-upscale", type=int, default=6)
    parser.add_argument("--depth-video-fps", type=int, default=30)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if args_cli.save_depth_video is not None and args_cli.num_envs != 1:
        print(f"[eval_fixed_init] --save-depth-video set, forcing --num-envs 1 "
              f"(was {args_cli.num_envs}).", flush=True)
        args_cli.num_envs = 1

    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args_cli).app

    import gymnasium as gym
    import numpy as np
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
        print(f"[eval_fixed_init] teacher env_info: obs={info['observation_space']} "
              f"action={info['action_space']}", flush=True)

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
        print(f"[eval_fixed_init] teacher loaded from {ckpt}", flush=True)
        if args_cli.override_block_id is not None:
            teacher.pin_block_id(args_cli.override_block_id)
            print(f"[eval_fixed_init] pinned block_id to {args_cli.override_block_id}",
                  flush=True)

        unwrapped = env.unwrapped
        device = args_cli.rl_device
        num_envs = args_cli.num_envs
        max_ep_len = int(unwrapped.max_episode_length)

        episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        success_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        first_success_steps = torch.full(
            (num_envs,), fill_value=-1, dtype=torch.long, device=device,
        )

        save_video = args_cli.save_depth_video is not None
        depth_frames: list[np.ndarray] = []
        image_h = int(env_cfg.student_obs.image_input_height)
        image_w = int(env_cfg.student_obs.image_input_width)
        image_numel = image_h * image_w  # depth = 1 channel

        obs_dict, _ = env.reset()
        print(f"[eval_fixed_init] reset obs keys: {list(obs_dict.keys())} "
              f"max_episode_length={max_ep_len}", flush=True)

        step = 0
        stop_after_video = False
        while step < args_cli.max_steps:
            teacher_obs = obs_dict["teacher_obs"]
            with torch.no_grad():
                action = teacher.get_action(teacher_obs)

            if save_video:
                # Slice env 0's depth out of the obs the teacher just consumed.
                # obs_dict["policy"] is [image_flat, proprio]; image comes first.
                # Avoids re-running the camera (a fresh get_student_obs() call
                # forces another sim.render() + camera readback).
                img_flat = obs_dict["policy"][0, :image_numel]
                frame = img_flat.reshape(image_h, image_w).clamp(0.0, 1.0).detach().cpu().numpy()
                frame_u8 = (frame * 255.0).astype(np.uint8)
                k = max(1, int(args_cli.depth_video_upscale))
                if k > 1:
                    frame_u8 = np.kron(frame_u8, np.ones((k, k), dtype=np.uint8))
                depth_frames.append(frame_u8)

            obs_dict, _, terminated, truncated, _ = env.step(action)
            step += 1
            ep_steps += 1

            term_reasons = getattr(unwrapped, "_termination_reasons", None)
            if term_reasons is not None:
                max_succ = term_reasons["max_successes"].to(device)
            else:
                max_succ = torch.zeros(num_envs, dtype=torch.bool, device=device)

            # Track first-success step per current episode.
            first_now = max_succ & (first_success_steps < 0)
            if first_now.any():
                first_success_steps[first_now] = ep_steps[first_now]

            done = (terminated | truncated).to(device)
            if done.any():
                done_idx = done.nonzero(as_tuple=False).flatten()
                # Was this episode a success?
                ep_was_success = max_succ[done_idx]
                episode_count[done_idx] += 1
                success_count[done_idx] += ep_was_success.long()

                ep_steps[done_idx] = 0
                first_success_steps[done_idx] = -1
                teacher.reset_idx(done_idx)

                if save_video:
                    # In video mode we want exactly one episode of env 0.
                    stop_after_video = bool(done[0].item())

            if step % 100 == 0 or step == 1:
                ep_min = int(episode_count.min())
                ep_mean = float(episode_count.float().mean())
                succ_so_far = (
                    float(success_count.sum()) / max(1, int(episode_count.sum()))
                    if episode_count.sum() > 0
                    else 0.0
                )
                print(
                    f"[step {step:5d}] episodes/env min={ep_min} mean={ep_mean:.2f} "
                    f"success_rate_so_far={succ_so_far:.3f} "
                    f"action |a|max={float(action.abs().max()):.3f} "
                    f"has_nan={bool(action.isnan().any())}",
                    flush=True,
                )

            if stop_after_video:
                print(f"[eval_fixed_init] env 0 episode ended at step {step}; "
                      f"stopping video capture.", flush=True)
                break

            if not save_video and int(episode_count.min()) >= args_cli.num_episodes:
                print(f"[eval_fixed_init] all {num_envs} envs reached "
                      f"{args_cli.num_episodes} episodes at step {step}.", flush=True)
                break

        total_eps = int(episode_count.sum())
        total_succ = int(success_count.sum())
        succ_rate = total_succ / max(1, total_eps)
        print(f"\n[eval_fixed_init] === SUMMARY ===", flush=True)
        print(f"  num_envs              = {num_envs}", flush=True)
        print(f"  total_episodes        = {total_eps}", flush=True)
        print(f"  total_successes       = {total_succ}", flush=True)
        print(f"  success_rate          = {succ_rate:.4f}", flush=True)
        print(f"  episodes_per_env: min={int(episode_count.min())} "
              f"max={int(episode_count.max())} "
              f"mean={float(episode_count.float().mean()):.2f}", flush=True)
        print(f"  successes_per_env: min={int(success_count.min())} "
              f"max={int(success_count.max())} "
              f"mean={float(success_count.float().mean()):.2f}", flush=True)

        if save_video and depth_frames:
            try:
                import imageio.v2 as imageio
            except ImportError:
                import imageio  # type: ignore[no-redef]
            out_path = Path(args_cli.save_depth_video).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(out_path), depth_frames, fps=int(args_cli.depth_video_fps))
            first_png = out_path.with_suffix(".first.png")
            imageio.imwrite(str(first_png), depth_frames[0])
            print(f"\n[eval_fixed_init] wrote depth video ({len(depth_frames)} frames) "
                  f"to {out_path}", flush=True)
            print(f"[eval_fixed_init] first-frame PNG at {first_png}", flush=True)

    run()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
