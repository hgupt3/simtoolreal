#!/usr/bin/env python3
"""Offline 5-way eval of one student checkpoint across the {delays, cam-pose-
rand, depth-aug} toggle grid.

One env instance (num_envs=N), one student checkpoint. For each of 5
settings we mutate `env.cfg.{student_obs,domain_randomization}.*` flags,
re-seed torch (same seed for every setting), reset, and roll out one
episode per env. Same seed before reset → identical robot/object/hole/goal
init states across settings, so episodes are paired comparisons.

Camera pose differs across settings *by design* when `use_camera_pose_rand`
is True — that's part of what the toggle does. All other reset randomness
(joint noise, object pose, etc.) is identical across settings.

Usage:
    .venv_isaacsim/bin/python -u peg_in_hole_dynamic/offline_eval_5way.py \\
        --teacher-checkpoint <path> --student-checkpoint <path> \\
        --num-envs 10 --num-episodes 1 --seed 42 [--headless]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Order matches the eval_student_isaacsim.py STUDENT_POLICY_ORDER.
SETTINGS = [
    ("no_delays_no_camnoise",    {"delays": False, "camera_pose_rand": False, "depth_aug": False}),
    ("camrand_off_depthaug_off", {"delays": True,  "camera_pose_rand": False, "depth_aug": False}),
    ("camrand_on_depthaug_off",  {"delays": True,  "camera_pose_rand": True,  "depth_aug": False}),
    ("camrand_off_depthaug_on",  {"delays": True,  "camera_pose_rand": False, "depth_aug": True}),
    ("camrand_on_depthaug_on",   {"delays": True,  "camera_pose_rand": True,  "depth_aug": True}),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--agent", default="rl_games_dagger_sapg_cfg_entry_point")
    p.add_argument("--problem", default="Lpeg.tol0p5mm")
    p.add_argument("--goal-mode", default="preInsertAndFinal")
    p.add_argument("--teacher-checkpoint", required=True,
                   help="Teacher .pth (used only to set up the SAPG block-id channel).")
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=10,
                   help="One reset per env per setting = num_envs episodes per setting.")
    p.add_argument("--num-episodes", type=int, default=1,
                   help="Rollouts per env. Total episodes/setting = num_envs * num_episodes.")
    p.add_argument("--max-steps-per-episode", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--delay-max", type=int, default=3,
        help="Max frames for obs_delay / action_delay / camera_delay queues "
             "(allocated at env init; must be set up-front). Training used 3. "
             "Bump up for stress-test sanity checks (e.g. --delay-max 60).",
    )
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--output-json", default=None,
                   help="If set, write a structured summary here.")
    return p.parse_args()


def _reseed(seed: int) -> None:
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _apply_toggles(env, *, delays: bool, camera_pose_rand: bool, depth_aug: bool) -> None:
    """Mutate the env cfg flags. All three are read at obs/reset time so
    they take effect on the next env.reset() / env.step()."""
    so = env.cfg.student_obs
    dr = env.cfg.domain_randomization
    so.use_camera_delay = bool(delays)
    so.use_camera_pose_rand = bool(camera_pose_rand)
    so.use_depth_aug = bool(depth_aug)
    dr.use_obs_delay = bool(delays)
    dr.use_action_delay = bool(delays)


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    # --- AppLauncher ---
    from isaaclab.app import AppLauncher
    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = bool(args.headless)
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import torch
    import gymnasium as gym
    import isaacsimenvs  # noqa: F401  triggers gym.register
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info
    from isaacsimenvs.dagger.teacher import Teacher
    from isaacsimenvs.dagger import networks as _net  # noqa: F401  registers depth_cnn_lstm

    # Reuse helpers verbatim
    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
    )
    from peg_in_hole_dynamic.eval_student_isaacsim import _build_student

    # --- env cfg (max delays so the queues exist; the toggle gates whether they're used) ---
    cfg = _load_env_cfg(args.task)
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
        extra_overrides={},
    )
    # Size the delay queues at env init. Each queue is a fixed-length FIFO,
    # so the max has to be set up-front; flipping `use_*_delay` True later
    # samples a delay uniformly in [0, max] per env per step.
    dmax = max(1, int(args.delay_max))
    cfg.student_obs.camera_delay_max = dmax
    cfg.domain_randomization.obs_delay_max = dmax
    cfg.domain_randomization.action_delay_max = dmax
    # Stash defaults for the camera-pose / depth-aug numerical knobs (medium preset).
    cfg.student_obs.use_camera_delay = True
    cfg.student_obs.use_camera_pose_rand = True
    cfg.student_obs.use_depth_aug = True

    env = _instantiate_env(args.task, cfg)
    agent_cfg = _configure_agent(
        args.task, "rl_games_sapg_cfg_entry_point",
        rl_device=args.rl_device, num_envs=int(args.num_envs),
        deterministic=True, games=1, extra_overrides={},
    )
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=args.rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

    # --- teacher (so we can size the SAPG block-id; we don't actually act with it here) ---
    env_info_teacher = teacher_env_info(wrapped)
    student_agent_cfg = gym.spec(args.task).kwargs.get("rl_games_dagger_sapg_cfg_entry_point", None)
    # Fallback: load directly via isaaclab loader.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    student_agent_cfg = load_cfg_from_registry(args.task, "rl_games_dagger_sapg_cfg_entry_point")
    dagger_block = student_agent_cfg["params"]["config"].get("dagger", {})
    teacher = Teacher(
        task_id=str(dagger_block.get("teacher_task_id", "Isaacsimenvs-PegInHole-Direct-v0")),
        agent_key=str(dagger_block.get("teacher_agent_key", "rl_games_sapg_cfg_entry_point")),
        checkpoint_path=args.teacher_checkpoint,
        num_envs=int(args.num_envs),
        rl_device=args.rl_device,
        env_info=env_info_teacher,
    )

    # --- student ---
    net_params = student_agent_cfg["params"]["network"]
    student_obs0 = env.get_student_obs()
    actual_proprio_dim = int(student_obs0["proprio"].shape[-1])
    image_channels = int(net_params.get("image_channels", 1))
    image_hw = tuple(net_params["image_hw"])
    has_block_id = bool(net_params.get("has_block_id", True))
    num_blocks = int(student_agent_cfg["params"]["config"].get("expl_coef_block_size", 1))
    if actual_proprio_dim != int(net_params.get("proprio_dim", -1)):
        net_params = dict(net_params)
        net_params["proprio_dim"] = actual_proprio_dim
    student = _build_student(
        net_params=net_params,
        action_dim=int(env.action_space.shape[-1]),
        image_channels=image_channels,
        image_hw=image_hw,
        proprio_dim=int(net_params["proprio_dim"]),
        has_block_id=has_block_id,
        num_blocks=max(num_blocks, 1),
        student_checkpoint=args.student_checkpoint,
        device=args.rl_device,
    )
    student.eval()

    # --- rollout helpers ---
    def _student_action(rnn_state) -> tuple[torch.Tensor, tuple | None]:
        out = env.get_student_obs()
        image = out["image"]
        proprio = out["proprio"]
        if has_block_id:
            block_id = torch.zeros(image.shape[0], 1, device=image.device)
            flat = torch.cat([image.flatten(1), proprio, block_id], dim=-1)
        else:
            flat = torch.cat([image.flatten(1), proprio], dim=-1)
        with torch.no_grad():
            mu, _ls, _v, rnn_state = student({"obs": flat, "rnn_states": rnn_state, "seq_length": 1})
        return mu, rnn_state

    def _run_one_setting(name: str, toggles: dict) -> dict:
        """Reset all envs (paired init via seed re-seed), roll out, return per-env max succ."""
        _apply_toggles(env, delays=toggles["delays"],
                       camera_pose_rand=toggles["camera_pose_rand"],
                       depth_aug=toggles["depth_aug"])
        _reseed(args.seed)
        student.reset_default_state() if hasattr(student, "reset_default_state") else None
        teacher.reset()
        wrapped.reset()  # warm-up reset; the toggle change applies on this reset

        # Final reset for actual rollout, with the same seed
        _reseed(args.seed)
        wrapped.reset()
        rnn_state = None
        # Each env runs up to max_steps_per_episode. env terminates per-env via
        # env._successes / dones; we just collect the per-env final success count.
        max_succ = torch.zeros(int(args.num_envs), device=args.rl_device)
        for step in range(int(args.max_steps_per_episode)):
            act, rnn_state = _student_action(rnn_state)
            wrapped.step(act)
            max_succ = torch.maximum(max_succ, env._successes.float())
        max_goals = float(env.env_max_goals[0].item())
        succ_rate = (max_succ / max(max_goals, 1.0)).clamp(0.0, 1.0)
        return {
            "name": name,
            "toggles": toggles,
            "max_succ_per_env": max_succ.cpu().tolist(),
            "succ_rate_per_env": succ_rate.cpu().tolist(),
            "mean_succ": float(succ_rate.mean().item()),
            "max_goals": max_goals,
        }

    print(f"=> teacher loaded from '{args.teacher_checkpoint}'", flush=True)
    print(f"=> student loaded from '{args.student_checkpoint}'", flush=True)
    print(f"=> num_envs={args.num_envs}  max_steps={args.max_steps_per_episode}  seed={args.seed}")
    print(f"=> testing the student across {len(SETTINGS)} env settings", flush=True)
    results = []
    for name, toggles in SETTINGS:
        t0 = time.time()
        try:
            res = _run_one_setting(name, toggles)
        except Exception as exc:
            print(f"[setting] {name}: FAILED -- {exc}", flush=True)
            traceback.print_exc()
            res = {"name": name, "toggles": toggles, "error": str(exc), "mean_succ": float("nan")}
        dt = time.time() - t0
        results.append(res)
        print(f"[setting] {name:32s} mean_succ={res.get('mean_succ', float('nan')):.4f}  "
              f"({dt:.1f}s)", flush=True)

    # Summary table
    print("\n" + "=" * 88)
    print(f"OFFLINE EVAL SUMMARY  (student={Path(args.student_checkpoint).parent.name},  "
          f"N={args.num_envs}, seed={args.seed})")
    print("=" * 88)
    print(f"{'setting':32s} {'delays':>7} {'campose':>8} {'depth_aug':>10} {'mean_succ':>10}")
    print("-" * 88)
    for r in results:
        t = r["toggles"]
        print(f"{r['name']:32s} {str(t['delays']):>7} {str(t['camera_pose_rand']):>8} "
              f"{str(t['depth_aug']):>10} {r.get('mean_succ', float('nan')):>10.4f}")
    print("=" * 88)

    if args.output_json:
        with open(args.output_json, "w") as fp:
            json.dump({
                "student_checkpoint": str(args.student_checkpoint),
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "num_envs": int(args.num_envs),
                "seed": int(args.seed),
                "max_steps_per_episode": int(args.max_steps_per_episode),
                "results": results,
            }, fp, indent=2)
        print(f"=> wrote {args.output_json}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
