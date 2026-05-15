"""Quick throughput sweep: env.step() FPS at various num_envs x camera period.

Spawns the env with the production student-sub raycaster config, warms up
for 20 steps (skip Isaac Sim JIT compiles), then times the next 50 steps.
Reports FPS = num_envs * steps / wall_seconds.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg_matchedmass.tol0p5mm")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--period", type=float, default=0.0,
                   help="camera_update_period_s. 0.0=60Hz, 0.03333=30Hz.")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", default="/tmp/smoke_throughput.csv")
    args = p.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    la, _ = lp.parse_known_args([])
    la.headless = True
    la.enable_cameras = True
    AppLauncher(la)

    import torch
    import isaacsimenvs  # noqa: F401
    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides, _instantiate_env, _load_env_cfg,
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
        sim_device="cuda:0",
        sdf=False,
        keep_dr=False,
        extra_overrides={},
    )

    # Production student-sub raycaster config.
    cfg.student_obs.camera_backend = "raycaster"
    cfg.student_obs.image_width = 70
    cfg.student_obs.image_height = 70
    cfg.student_obs.image_input_width = 70
    cfg.student_obs.image_input_height = 70
    cfg.student_obs.crop_enabled = False
    cfg.student_obs.horizontal_aperture = 14.524
    cfg.student_obs.horizontal_aperture_offset = -0.389
    cfg.student_obs.vertical_aperture_offset = 0.105
    cfg.student_obs.camera_update_period_s = float(args.period)
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False

    print(
        f"=> throughput: num_envs={args.num_envs} period={args.period} "
        f"warmup={args.warmup} measure={args.measure}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    env = _instantiate_env(args.task, cfg)
    env.reset()
    env.reset()

    n_env = int(env.num_envs)
    action_dim = int(env.action_space.shape[-1])
    actions = 0.05 * torch.randn(args.warmup + args.measure, n_env, action_dim,
                                  device=env.device)

    # Warmup
    print("=> warming up...", flush=True)
    for step in range(args.warmup):
        env.step(actions[step])
        env.get_student_obs()

    # Measure
    print("=> measuring...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(args.measure):
        env.step(actions[args.warmup + step])
        env.get_student_obs()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    samples = n_env * args.measure
    fps_total = samples / elapsed
    step_ms = (elapsed / args.measure) * 1000.0

    print(
        f"=> RESULT num_envs={n_env} period={args.period}  "
        f"steps={args.measure}  elapsed={elapsed:.2f}s  "
        f"fps_total={fps_total:.0f}  "
        f"step_ms={step_ms:.2f}",
        flush=True,
    )

    # Append to CSV.
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(args.out_csv).exists()
    with open(args.out_csv, "a") as f:
        if write_header:
            f.write("num_envs,period,fps_total,step_ms,elapsed_s\n")
        f.write(f"{n_env},{args.period},{fps_total:.0f},{step_ms:.2f},{elapsed:.2f}\n")

    try:
        env.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
