"""Smoke test: does `student_obs.camera_update_period_s=0.03333` actually
latch the depth tensor on alternate 60Hz policy steps (= 30Hz render)?

Spawns a minimal Isaac Sim env (n=4) with the 30Hz override, steps it N
times with zero actions, captures the raw raycaster depth tensor at every
step, and prints a per-step diff so we can see whether the sensor is
genuinely updating every other step.

Compare against the 60Hz default by passing --period 0.0.

Usage:
    .venv_isaacsim/bin/python peg_in_hole_dynamic/smoke_test_camera_period.py \\
        --period 0.03333 --steps 12
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg_matchedmass.tol0p5mm")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument(
        "--period",
        type=float,
        default=0.03333,
        help="camera_update_period_s. 0.0 = every step (60Hz); 0.03333 = every "
        "other step (30Hz at our 60Hz policy cadence).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-device", default="cuda:0")
    args = p.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    AppLauncher(launcher_args)

    import torch
    import isaacsimenvs  # noqa: F401

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _instantiate_env,
        _load_env_cfg,
    )

    cfg = _load_env_cfg(args.task)
    _apply_env_overrides(
        cfg,
        problem=args.problem,
        goal_mode="finalGoalOnly",
        random_goal_fraction=0.0,
        insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005,
        num_envs=int(args.num_envs),
        sim_device=args.sim_device,
        sdf=False,
        keep_dr=False,
        extra_overrides={},
    )

    # Match the production sub's camera settings.
    cfg.student_obs.camera_backend = "raycaster"
    cfg.student_obs.image_width = 70
    cfg.student_obs.image_height = 70
    cfg.student_obs.image_input_width = 70
    cfg.student_obs.image_input_height = 70
    cfg.student_obs.crop_enabled = False
    cfg.student_obs.horizontal_aperture = 14.524
    # PinholeCameraPatternCfg (raycaster) uses DIMENSIONLESS offsets:
    # offset = (c - n/2) / f_px. So h_off = (-9.98 - 35) / 115.67 = -0.389
    # and v_off = (47.13 - 35) / 115.67 = +0.105.
    cfg.student_obs.horizontal_aperture_offset = -0.389
    cfg.student_obs.vertical_aperture_offset = 0.105
    cfg.student_obs.camera_update_period_s = float(args.period)
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False

    print(
        f"=> smoke: num_envs={args.num_envs} steps={args.steps} "
        f"camera_update_period_s={args.period}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = _instantiate_env(args.task, cfg)
    env.reset()
    env.reset()  # second reset primes the sensor

    n_env = int(env.num_envs)
    action_dim = int(env.action_space.shape[-1])
    # Small random actions so the scene actually moves between steps -- with
    # zero actions a static robot + peg-on-table looks identical step-to-step
    # and we couldn't tell SAME-because-latched from SAME-because-no-motion.
    torch.manual_seed(args.seed)
    rand_actions = 0.1 * torch.randn(args.steps, n_env, action_dim, device=env.device)

    prev_depth = None
    print(
        "\n  step | depth_changed | max_abs_diff_to_prev | mean_depth",
        flush=True,
    )
    print("  -----|---------------|----------------------|-----------", flush=True)

    for step in range(args.steps):
        # Pre-step diagnostics: where does the sensor stand?
        cam = env.student_camera
        pre_ts = float(cam._timestamp[0].item())
        pre_last = float(cam._timestamp_last_update[0].item())
        pre_outdated = int(cam._is_outdated.sum().item())
        env.step(rand_actions[step])
        # Go through the production path: get_student_obs() calls
        # read_student_camera_image() which now calls camera.update(dt,
        # force_recompute=False) -- honors cfg.update_period.
        env.get_student_obs()
        post_ts = float(cam._timestamp[0].item())
        post_last = float(cam._timestamp_last_update[0].item())
        post_outdated = int(cam._is_outdated.sum().item())
        print(
            f"   [diag step {step}] pre: ts={pre_ts:.4f} last={pre_last:.4f} "
            f"outd={pre_outdated} | post: ts={post_ts:.4f} last={post_last:.4f} "
            f"outd={post_outdated} period={cam.cfg.update_period}",
            flush=True,
        )
        raw = env.student_camera.data.output.get("distance_to_image_plane")
        if raw is None:
            print("  raw depth output unavailable", flush=True)
            break
        if raw.dim() == 4 and raw.shape[-1] == 1:
            raw = raw.squeeze(-1)
        # Use env 0 for the diff signal.
        cur = raw[0].detach().float().cpu().clone()
        finite = torch.isfinite(cur)
        mean_depth = float(cur[finite].mean()) if finite.any() else float("nan")
        if prev_depth is None:
            print(
                f"  {step:4d} | (first sample) |                      | "
                f"{mean_depth:.4f}",
                flush=True,
            )
        else:
            diff = (cur - prev_depth).abs()
            both_finite = torch.isfinite(cur) & torch.isfinite(prev_depth)
            max_diff = float(diff[both_finite].max()) if both_finite.any() else 0.0
            # "Changed" = any pixel differs by more than 1e-6.
            changed = max_diff > 1e-6
            mark = "CHANGED " if changed else "SAME    "
            print(
                f"  {step:4d} | {mark}      | {max_diff:.6f}            | "
                f"{mean_depth:.4f}",
                flush=True,
            )
        prev_depth = cur

    print(
        f"\n  Expected with period={args.period}:\n"
        f"    0.0     : depth CHANGED on every step (60Hz)\n"
        f"    0.03333 : depth changes on every OTHER step (30Hz, alternating SAME/CHANGED)\n",
        flush=True,
    )

    try:
        env.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
