#!/usr/bin/env python3
"""Offline eval of one student checkpoint under table pose / scale randomization.

Knobs tested (sequentially, sharing one env instance):
  - reset-time table xy noise   (env.reset.table_reset_xy_range_m)
  - reset-time table yaw noise  (env.reset.table_reset_yaw_range_deg)

Table SCALE variants are USD-level (baked at scene init), so this script bakes
them once up-front via CLI args; flip scaling on/off by re-running the script.

Same seed reseeded before every setting => paired init across settings (robot,
object, hole, goal). With scale ON each env_i still gets its same scaled USD
across all settings within the run -- paired comparison holds.

Usage:
    .venv_isaacsim/bin/python -u peg_in_hole_dynamic/offline_eval_table_dr.py \\
        --teacher-checkpoint <path> --student-checkpoint <path> \\
        --num-envs 10 --seed 42 \\
        --table-scale-x 1.0,1.0 --table-scale-y 1.0,1.0 --table-scale-n 1
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


# Reset-time settings only. Scale is baked at init from CLI args.
SETTINGS = [
    ("no_table_dr",       {"xy_m": 0.00, "yaw_deg": 0.0}),
    ("table_xy_3cm",      {"xy_m": 0.03, "yaw_deg": 0.0}),
    ("table_yaw_5deg",    {"xy_m": 0.00, "yaw_deg": 5.0}),
    ("table_xy_3cm_yaw5", {"xy_m": 0.03, "yaw_deg": 5.0}),
]


def _parse_pair(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg.tol0p5mm")
    p.add_argument("--goal-mode", default="finalGoalOnly")
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--max-steps-per-episode", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay-max", type=int, default=3)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--table-scale-x", type=_parse_pair, default=(1.0, 1.0),
                   help="Scale range for table X dim (e.g. 0.85,1.15).")
    p.add_argument("--table-scale-y", type=_parse_pair, default=(1.0, 1.0),
                   help="Scale range for table Y dim.")
    p.add_argument("--table-scale-n", type=int, default=1,
                   help="Number of scaled-table USD variants to bake (1 = none).")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def _reseed(seed: int) -> None:
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _apply_table_toggles(env, *, xy_m: float, yaw_deg: float) -> None:
    env.cfg.reset.table_reset_xy_range_m = (float(xy_m), float(xy_m))
    env.cfg.reset.table_reset_yaw_range_deg = float(yaw_deg)


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher
    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = bool(args.headless)
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import torch
    import gymnasium as gym
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info
    from isaacsimenvs.dagger.teacher import Teacher
    from isaacsimenvs.dagger import networks as _net  # noqa: F401

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
    )
    from peg_in_hole_dynamic.eval_student_isaacsim import _build_student

    # --- env cfg ---
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
    dmax = max(1, int(args.delay_max))
    cfg.student_obs.camera_delay_max = dmax
    cfg.domain_randomization.obs_delay_max = dmax
    cfg.domain_randomization.action_delay_max = dmax
    # Match the "no_delays_no_camnoise" training distribution as the baseline.
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False

    # Bake table scale variants up-front (USD-time; CANNOT be toggled at runtime).
    cfg.assets.table_scale_range_x = tuple(args.table_scale_x)
    cfg.assets.table_scale_range_y = tuple(args.table_scale_y)
    cfg.assets.table_scale_num_variants = int(args.table_scale_n)

    # Start with the reset-time knobs at zero -- mutated per setting below.
    cfg.reset.table_reset_xy_range_m = (0.0, 0.0)
    cfg.reset.table_reset_yaw_range_deg = 0.0

    print(f"=> table_scale_x={args.table_scale_x}  table_scale_y={args.table_scale_y}  "
          f"n_variants={args.table_scale_n}", flush=True)

    env = _instantiate_env(args.task, cfg)
    agent_cfg = _configure_agent(
        args.task, "rl_games_sapg_cfg_entry_point",
        rl_device=args.rl_device, num_envs=int(args.num_envs),
        deterministic=True, games=1, extra_overrides={},
    )
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=args.rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

    env_info_teacher = teacher_env_info(wrapped)
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
        _apply_table_toggles(env, xy_m=toggles["xy_m"], yaw_deg=toggles["yaw_deg"])
        _reseed(args.seed)
        if hasattr(student, "reset_default_state"):
            student.reset_default_state()
        teacher.reset()
        wrapped.reset()  # warm-up reset; new table-dr setting takes effect

        _reseed(args.seed)
        wrapped.reset()
        rnn_state = None
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
    print(f"=> running {len(SETTINGS)} table-DR settings (reset-time only)", flush=True)
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
        print(f"[setting] {name:24s} xy={toggles['xy_m']:.2f}m  yaw={toggles['yaw_deg']:.1f}deg  "
              f"mean_succ={res.get('mean_succ', float('nan')):.4f}  ({dt:.1f}s)", flush=True)

    scale_label = (f"scale_x={args.table_scale_x}  scale_y={args.table_scale_y}  "
                   f"n_variants={args.table_scale_n}")
    print("\n" + "=" * 92)
    print(f"TABLE-DR OFFLINE EVAL  (student={Path(args.student_checkpoint).parent.name},  "
          f"N={args.num_envs}, seed={args.seed})")
    print(scale_label)
    print("=" * 92)
    print(f"{'setting':24s} {'xy_m':>8} {'yaw_deg':>10} {'mean_succ':>10}")
    print("-" * 92)
    for r in results:
        t = r["toggles"]
        print(f"{r['name']:24s} {t['xy_m']:>8.2f} {t['yaw_deg']:>10.1f} "
              f"{r.get('mean_succ', float('nan')):>10.4f}")
    print("=" * 92)

    if args.output_json:
        with open(args.output_json, "w") as fp:
            json.dump({
                "student_checkpoint": str(args.student_checkpoint),
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "num_envs": int(args.num_envs),
                "seed": int(args.seed),
                "max_steps_per_episode": int(args.max_steps_per_episode),
                "table_scale_x": list(args.table_scale_x),
                "table_scale_y": list(args.table_scale_y),
                "table_scale_num_variants": int(args.table_scale_n),
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
