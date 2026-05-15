"""Smoke-test the student-camera ROI pipeline two ways at two resolutions:

  1) post-crop @ 1x:   capture 160x90,  crop [90,0]..[160,70]   -> 70x70
  2) direct-ROI @ 1x:  capture 70x70 with adjusted apertures    -> 70x70
  3) post-crop @ 3x:   capture 480x270, crop [270,0]..[480,210] -> 210x210
  4) direct-ROI @ 3x:  capture 210x210 with adjusted apertures  -> 210x210

The scene is locked (fixed_start_pose, hole_x/y range collapsed, reset
noises 0, DR off, seed pinned) so cases that are supposed to be identical
can be diffed numerically. Each invocation captures one config (pass --tag)
and saves a normalized PNG plus a raw .npy of env 0's student depth.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "videos" / "smoke_test_camera_roi"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True,
                        help="Used to name the output .png/.npy files.")
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--horizontal-aperture", type=float, default=33.19737869997174)
    parser.add_argument("--horizontal-aperture-offset", type=float, default=0.0035)
    parser.add_argument("--vertical-aperture-offset", type=float, default=0.4418)
    parser.add_argument("--crop-enabled", action="store_true")
    parser.add_argument("--no-crop", action="store_true",
                        help="Force crop_enabled=False (overrides --crop-enabled).")
    parser.add_argument("--crop-top-left", type=int, nargs=2, default=[0, 0],
                        metavar=("X", "Y"))
    parser.add_argument("--crop-bottom-right", type=int, nargs=2, default=[0, 0],
                        metavar=("X", "Y"))
    parser.add_argument("--input-width", type=int, required=True,
                        help="Final policy-facing image width (after crop).")
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=2,
                        help=">=2 because some Isaac plumbing assumes batch>=2.")
    parser.add_argument("--num-warmup-steps", type=int, default=4,
                        help="Number of env.step(zero) calls before capture.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    from PIL import Image

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task = "Isaacsimenvs-PegInHoleDepthStudent-Direct-v0"
    spec = gym.spec(task)
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})

    # --- Lock the scene so the same physical render comes out across configs ---
    cfg.scene.num_envs = int(args.num_envs)
    cfg.peg_in_hole.problem = "Lpeg.tol0p5mm"
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"
    cfg.peg_in_hole.hole_x_range = [0.10, 0.10]
    cfg.peg_in_hole.hole_y_range = [0.0, 0.0]
    cfg.reset.fixed_start_pose = [-0.10, 0.0, 0.63, 1.0, 0.0, 0.0, 0.0]
    cfg.reset.reset_position_noise_x = 0.0
    cfg.reset.reset_position_noise_y = 0.0
    cfg.reset.reset_position_noise_z = 0.0
    cfg.reset.reset_dof_pos_random_interval_arm = 0.0
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
    cfg.reset.reset_dof_vel_random_interval = 0.0
    cfg.reset.table_reset_z_range = 0.0
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False

    # --- Camera config ---
    cfg.student_obs.image_width = int(args.image_width)
    cfg.student_obs.image_height = int(args.image_height)
    cfg.student_obs.horizontal_aperture = float(args.horizontal_aperture)
    cfg.student_obs.horizontal_aperture_offset = float(args.horizontal_aperture_offset)
    cfg.student_obs.vertical_aperture_offset = float(args.vertical_aperture_offset)
    crop_enabled = bool(args.crop_enabled) and not bool(args.no_crop)
    cfg.student_obs.crop_enabled = crop_enabled
    cfg.student_obs.crop_top_left = tuple(args.crop_top_left)
    cfg.student_obs.crop_bottom_right = tuple(args.crop_bottom_right)
    cfg.student_obs.image_input_width = int(args.input_width)
    cfg.student_obs.image_input_height = int(args.input_height)

    print(
        f"[smoke {args.tag}] cam={args.image_width}x{args.image_height} "
        f"horiz_aperture={args.horizontal_aperture:.4f}mm "
        f"haperture_off={args.horizontal_aperture_offset:.4f}mm "
        f"vaperture_off={args.vertical_aperture_offset:.4f}mm "
        f"crop={crop_enabled} top_left={tuple(args.crop_top_left)} "
        f"bottom_right={tuple(args.crop_bottom_right)} "
        f"input={args.input_width}x{args.input_height}",
        flush=True,
    )

    env = gym.make(task, cfg=cfg)
    inner = env.unwrapped
    actions = torch.zeros(
        (inner.num_envs, cfg.action_space), device=inner.device, dtype=torch.float32
    )

    env.reset()
    for _ in range(int(args.num_warmup_steps)):
        env.step(actions)

    student_obs = inner.get_student_obs()
    # (B, 1, H, W) → take env 0, channel 0
    student_depth = student_obs["image"][0, 0].detach().float().cpu().numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.output_dir / f"{args.tag}.npy"
    png_path = args.output_dir / f"{args.tag}.png"
    np.save(npy_path, student_depth)
    norm = np.clip(student_depth, 0.0, 1.0)
    Image.fromarray((norm * 255.0).astype(np.uint8)).save(png_path)
    print(
        f"[smoke {args.tag}] shape={student_depth.shape} "
        f"min={student_depth.min():.4f} max={student_depth.max():.4f} "
        f"mean={student_depth.mean():.4f}",
        flush=True,
    )
    print(f"[smoke {args.tag}] wrote {npy_path}", flush=True)
    print(f"[smoke {args.tag}] wrote {png_path}", flush=True)

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
