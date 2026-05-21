"""Boot Isaac Sim, reset N envs, step physics until objects settle, dump poses.

For one fig 4 task: spawn N parallel envs with the env's own randomization,
let the object fall + settle under gravity (zero policy actions), then read
out the final object pose and goal pose. Save to JSON.

Usage:
    .venv_isaacsim/bin/python plot_figures/fig4/dump_settled_poses.py \\
        --task Isaacsimenvs-PegInHole-Direct-v0 \\
        --problem Lpeg.tol0p5mm \\
        --output plot_figures/fig4/inputs/settled_poses_peg_in_hole.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaacsimenvs-PegInHole-Direct-v0")
    parser.add_argument("--problem", default="Lpeg.tol0p5mm")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = gym.spec(args.task)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})

    cfg.scene.num_envs = int(args.num_envs)
    cfg.peg_in_hole.problem = args.problem
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"
    # Disable DR noise but keep init randomization (that's the point of the dump).
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.object_state_xyz_noise_std = 0.0
    cfg.domain_randomization.object_state_rotation_noise_degrees = 0.0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0

    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    n = args.num_envs
    device = getattr(base, "device", "cuda:0")
    action_dim = cfg.action_space     # int — matches smoke_test_pih_direct_roi_video.py
    print(f"[dump] num_envs={n} action_dim={action_dim} device={device}", flush=True)

    obs, info = env.reset(seed=args.seed)
    print(f"[dump] reset complete; settling for {args.settle_steps} steps", flush=True)
    zero_action = torch.zeros((n, action_dim), device=device, dtype=torch.float32)
    for i in range(args.settle_steps):
        env.step(zero_action)
        if i % 20 == 0:
            print(f"[dump] step {i}/{args.settle_steps}", flush=True)

    object_pose = base.object.data.root_pose_w.detach().cpu().numpy()
    goal_pose = base.goal_viz.data.root_pose_w.detach().cpu().numpy()
    env_origins = base.scene.env_origins.detach().cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "task": args.task,
        "problem": args.problem,
        "num_envs": int(n),
        "object_poses_w": object_pose.tolist(),
        "goal_poses_w": goal_pose.tolist(),
        "env_origins": env_origins.tolist(),
    }, indent=2))
    print(f"wrote {args.output} ({n} poses)")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
