#!/usr/bin/env python3
"""Teacher-only robustness eval: state-obs SAPG policy under the same
delays / cam-rand / depth-aug / table xy / table yaw / table scale grid as
offline_eval_robustness.py.

Cam-rand and depth-aug are no-ops for the teacher (state obs has no camera
channel), but kept in the matrix so rows align with the student eval.
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


SETTINGS = [
    ("baseline",        {"delays": False, "cam": False, "depth": False, "xy_m": 0.00, "yaw_deg": 0.0}),
    ("delays",          {"delays": True,  "cam": False, "depth": False, "xy_m": 0.00, "yaw_deg": 0.0}),
    ("cam_pose_rand",   {"delays": False, "cam": True,  "depth": False, "xy_m": 0.00, "yaw_deg": 0.0}),
    ("depth_aug",       {"delays": False, "cam": False, "depth": True,  "xy_m": 0.00, "yaw_deg": 0.0}),
    ("table_xy_3cm",    {"delays": False, "cam": False, "depth": False, "xy_m": 0.03, "yaw_deg": 0.0}),
    ("table_yaw_5deg",  {"delays": False, "cam": False, "depth": False, "xy_m": 0.00, "yaw_deg": 5.0}),
    ("table_xy_yaw",    {"delays": False, "cam": False, "depth": False, "xy_m": 0.03, "yaw_deg": 5.0}),
    ("all_realistic",   {"delays": True,  "cam": True,  "depth": True,  "xy_m": 0.03, "yaw_deg": 5.0}),
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
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--max-steps-per-episode", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay-max", type=int, default=3)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--table-scale-x", type=_parse_pair, default=(1.0, 1.0))
    p.add_argument("--table-scale-y", type=_parse_pair, default=(1.0, 1.0))
    p.add_argument("--table-scale-n", type=int, default=1)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _reseed(seed: int) -> None:
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _apply_setting(env, t: dict) -> None:
    so = env.cfg.student_obs
    dr = env.cfg.domain_randomization
    so.use_camera_delay = bool(t["delays"])
    dr.use_obs_delay = bool(t["delays"])
    dr.use_action_delay = bool(t["delays"])
    so.use_camera_pose_rand = bool(t["cam"])
    so.use_depth_aug = bool(t["depth"])
    env.cfg.reset.table_reset_xy_range_m = (float(t["xy_m"]), float(t["xy_m"]))
    env.cfg.reset.table_reset_yaw_range_deg = float(t["yaw_deg"])


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
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info
    from isaacsimenvs.dagger.teacher import Teacher

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
    )
    from peg_in_hole_dynamic.eval_student_isaacsim import _extract_teacher_obs

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
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.assets.table_scale_range_x = tuple(args.table_scale_x)
    cfg.assets.table_scale_range_y = tuple(args.table_scale_y)
    cfg.assets.table_scale_num_variants = int(args.table_scale_n)
    cfg.reset.table_reset_xy_range_m = (0.0, 0.0)
    cfg.reset.table_reset_yaw_range_deg = 0.0

    print(f"=> TEACHER eval  scale_x={args.table_scale_x}  scale_y={args.table_scale_y}  "
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

    def _run_one_setting(name: str, toggles: dict) -> dict:
        _apply_setting(env, toggles)
        _reseed(args.seed)
        teacher.reset()
        wrapped.reset()
        _reseed(args.seed)
        obs = wrapped.reset()
        max_succ = torch.zeros(int(args.num_envs), device=args.rl_device)
        for _step in range(int(args.max_steps_per_episode)):
            tobs = _extract_teacher_obs(obs)
            act = teacher.get_action(tobs)
            step_out = wrapped.step(act)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
            max_succ = torch.maximum(max_succ, env._successes.float())
        max_goals = float(env.env_max_goals[0].item())
        succ_rate = (max_succ / max(max_goals, 1.0)).clamp(0.0, 1.0)
        return {
            "name": name,
            "toggles": toggles,
            "mean_succ": float(succ_rate.mean().item()),
            "succ_rate_per_env": succ_rate.cpu().tolist(),
            "max_succ_per_env": max_succ.cpu().tolist(),
            "max_goals": max_goals,
        }

    print(f"=> teacher loaded from '{args.teacher_checkpoint}'", flush=True)
    print(f"=> {len(SETTINGS)} settings, num_envs={args.num_envs}, seed={args.seed}", flush=True)

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
        print(f"[TEACHER] {name:18s} mean_succ={res.get('mean_succ', float('nan')):.4f}  "
              f"({dt:.1f}s)", flush=True)

    with open(args.output_json, "w") as fp:
        json.dump({
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
