"""Smoke test for PegInHole depth-student observations.

Runs a tiny headless camera env and verifies:
  - normal DirectRLEnv observations still expose policy/critic tensors
  - env.unwrapped.get_student_obs() exposes image/proprio tensors
  - optional student camera and bundled-observation delay queues are created

Example:
    .venv_isaacsim/bin/python isaacsimenvs/tests/test_peg_in_hole_depth_student_obs.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=90)
    parser.add_argument("--problem", type=str, default="peg.tol0p5mm")
    parser.add_argument("--crop_top_left", type=int, nargs=2, default=(90, 0), metavar=("X", "Y"))
    parser.add_argument("--crop_bottom_right", type=int, nargs=2, default=(160, 70), metavar=("X", "Y"))
    parser.add_argument("--camera_delay_max", type=int, default=3)
    parser.add_argument("--student_obs_delay_max", type=int, default=2)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import torch
    import yaml

    import isaacsimenvs  # noqa: F401  (registers gym envs)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task = "Isaacsimenvs-PegInHoleDepthStudent-Direct-v0"
    spec = gym.spec(task)
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})

    cfg.scene.num_envs = args.num_envs
    cfg.peg_in_hole.problem = args.problem
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"

    cfg.student_obs.image_width = args.image_width
    cfg.student_obs.image_height = args.image_height
    crop_x0, crop_y0 = args.crop_top_left
    crop_x1, crop_y1 = args.crop_bottom_right
    cfg.student_obs.image_input_width = crop_x1 - crop_x0
    cfg.student_obs.image_input_height = crop_y1 - crop_y0
    cfg.student_obs.crop_enabled = True
    cfg.student_obs.crop_top_left = (crop_x0, crop_y0)
    cfg.student_obs.crop_bottom_right = (crop_x1, crop_y1)
    cfg.student_obs.use_camera_delay = args.camera_delay_max > 0
    cfg.student_obs.camera_delay_max = args.camera_delay_max
    cfg.student_obs.use_student_obs_delay = args.student_obs_delay_max > 0
    cfg.student_obs.student_obs_delay_max = args.student_obs_delay_max

    env = gym.make(task, cfg=cfg)
    inner = env.unwrapped

    obs, _ = env.reset()
    assert "policy" in obs and "critic" in obs, obs.keys()

    actions = torch.zeros(
        (inner.num_envs, cfg.action_space),
        device=inner.device,
        dtype=torch.float32,
    )
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        if (
            torch.isnan(obs["policy"]).any()
            or torch.isnan(obs["critic"]).any()
            or torch.isnan(reward).any()
        ):
            raise RuntimeError("NaN detected in teacher obs or reward")

    student_obs = inner.get_student_obs()
    expected_image_shape = (
        args.num_envs,
        1,
        crop_y1 - crop_y0,
        crop_x1 - crop_x0,
    )
    assert student_obs["image"].shape == expected_image_shape, student_obs["image"].shape
    assert student_obs["proprio"].shape == (args.num_envs, 87), student_obs["proprio"].shape
    assert not torch.isnan(student_obs["image"]).any()
    assert not torch.isnan(student_obs["proprio"]).any()

    if args.camera_delay_max > 0:
        assert hasattr(inner, "_student_camera_queue")
    if args.student_obs_delay_max > 0:
        assert hasattr(inner, "_student_obs_queue")

    image = student_obs["image"]
    print(
        "[smoke] student image "
        f"shape={tuple(image.shape)} min={float(image.min()):.4f} "
        f"max={float(image.max()):.4f} nonzero={(image > 0).float().mean().item():.4f}"
    )
    print(f"[smoke] student proprio shape={tuple(student_obs['proprio'].shape)}")
    print("PegInHole depth student obs smoke test OK")

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
