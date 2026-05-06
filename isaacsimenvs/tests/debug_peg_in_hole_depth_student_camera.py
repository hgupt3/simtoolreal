"""Debug PegInHole depth-student camera framing.

This script is intentionally artifact-oriented, not a strict unit test. It
creates a tiny depth-student PegInHole env, captures the student depth images,
saves PNG grids, and prints per-env depth statistics for visual inspection.

Fast fixed-problem check:
    .venv_isaacsim/bin/python isaacsimenvs/tests/debug_peg_in_hole_depth_student_camera.py

Different dynamic problem:
    .venv_isaacsim/bin/python isaacsimenvs/tests/debug_peg_in_hole_depth_student_camera.py \\
      --problem furniture_bench.one_leg_sdf_hybrid_sparse --num_envs 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "videos"
    / "test_videos"
    / "peg_in_hole_depth_student"
)


def _depth_tensor_to_nhw(depth):
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    elif depth.ndim != 3:
        raise RuntimeError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")
    return depth.detach().float().cpu().numpy()


def _make_grid(images, *, pad: int = 2):
    import numpy as np

    images = np.clip(images, 0.0, 1.0)
    n, h, w = images.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.full(
        (rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad),
        255,
        dtype=np.uint8,
    )
    images_u8 = (images * 255.0).round().astype(np.uint8)
    for i in range(n):
        row = i // cols
        col = i % cols
        y0 = row * (h + pad)
        x0 = col * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = images_u8[i]
    return grid


def _save_depth_grid(path: Path, images) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_make_grid(images)).save(path)


def _print_depth_stats(prefix: str, raw_depth_m, student_depth, near: float, far: float) -> None:
    import numpy as np

    for env_id in range(raw_depth_m.shape[0]):
        raw = raw_depth_m[env_id]
        student = student_depth[env_id]
        finite = np.isfinite(raw)
        if finite.any():
            raw_valid = raw[finite]
            raw_min = float(raw_valid.min())
            raw_max = float(raw_valid.max())
            raw_mean = float(raw_valid.mean())
            in_window = float(((raw >= near) & (raw <= far) & finite).mean())
        else:
            raw_min = raw_max = raw_mean = float("nan")
            in_window = 0.0
        nonzero = float((student > 0.0).mean())
        sat_low = float((student <= 1e-6).mean())
        sat_high = float((student >= 1.0 - 1e-6).mean())
        print(
            f"{prefix} env={env_id:02d} "
            f"raw_m[min={raw_min:.3f}, mean={raw_mean:.3f}, max={raw_max:.3f}, "
            f"in_window={in_window:.3f}] "
            f"student[nonzero={nonzero:.3f}, sat_low={sat_low:.3f}, "
            f"sat_high={sat_high:.3f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resets", type=int, default=1)
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=90)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--problem", type=str, default="peg.tol0p5mm")
    parser.add_argument(
        "--goal_mode",
        choices=("preInsertAndFinal", "finalGoalOnly"),
        default="preInsertAndFinal",
    )
    parser.add_argument("--depth_min_m", type=float, default=None)
    parser.add_argument("--depth_max_m", type=float, default=None)
    parser.add_argument(
        "--crop_top_left",
        type=int,
        nargs=2,
        default=(90, 0),
        metavar=("X", "Y"),
    )
    parser.add_argument(
        "--crop_bottom_right",
        type=int,
        nargs=2,
        default=(160, 70),
        metavar=("X", "Y"),
    )
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--camera_delay_max", type=int, default=0)
    parser.add_argument("--student_obs_delay_max", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import numpy as np
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
    cfg.peg_in_hole.goal_mode = args.goal_mode

    cfg.student_obs.image_modality = "depth"
    cfg.student_obs.image_width = args.image_width
    cfg.student_obs.image_height = args.image_height
    cfg.student_obs.crop_enabled = not args.no_crop
    cfg.student_obs.crop_top_left = tuple(args.crop_top_left)
    cfg.student_obs.crop_bottom_right = tuple(args.crop_bottom_right)
    if cfg.student_obs.crop_enabled:
        x0, y0 = cfg.student_obs.crop_top_left
        x1, y1 = cfg.student_obs.crop_bottom_right
        cfg.student_obs.image_input_width = x1 - x0
        cfg.student_obs.image_input_height = y1 - y0
    else:
        cfg.student_obs.image_input_width = args.image_width
        cfg.student_obs.image_input_height = args.image_height
    cfg.student_obs.use_camera_delay = args.camera_delay_max > 0
    cfg.student_obs.camera_delay_max = args.camera_delay_max
    cfg.student_obs.use_student_obs_delay = args.student_obs_delay_max > 0
    cfg.student_obs.student_obs_delay_max = args.student_obs_delay_max
    if args.depth_min_m is not None:
        cfg.student_obs.depth_min_m = args.depth_min_m
    if args.depth_max_m is not None:
        cfg.student_obs.depth_max_m = args.depth_max_m

    env = gym.make(task, cfg=cfg)
    inner = env.unwrapped
    actions = torch.zeros(
        (inner.num_envs, cfg.action_space),
        device=inner.device,
        dtype=torch.float32,
    )

    out_dir = Path(args.output_dir)
    print(f"[debug] writing depth grids to {out_dir}")
    print(
        "[debug] camera cfg "
        f"problem={cfg.peg_in_hole.problem} "
        f"goal_mode={cfg.peg_in_hole.goal_mode} "
        f"pos={cfg.student_obs.camera_pos} quat_wxyz={cfg.student_obs.camera_quat_wxyz} "
        f"window=[{cfg.student_obs.depth_min_m}, {cfg.student_obs.depth_max_m}] "
        f"crop_enabled={cfg.student_obs.crop_enabled} "
        f"crop_top_left={cfg.student_obs.crop_top_left} "
        f"crop_bottom_right={cfg.student_obs.crop_bottom_right} "
        f"student_input={cfg.student_obs.image_input_width}x{cfg.student_obs.image_input_height}"
    )

    for reset_idx in range(args.resets):
        env.reset()
        for _ in range(args.steps):
            env.step(actions)

        student_obs = inner.get_student_obs()
        student_depth = student_obs["image"][:, 0].detach().float().cpu().numpy()

        raw_depth = inner.student_camera.data.output["distance_to_image_plane"]
        raw_depth_m = _depth_tensor_to_nhw(raw_depth)
        near = float(cfg.student_obs.depth_min_m)
        far = float(cfg.student_obs.depth_max_m)
        raw_window = np.nan_to_num((raw_depth_m - near) / (far - near), nan=0.0)
        raw_window = np.clip(raw_window, 0.0, 1.0)

        student_path = out_dir / f"reset_{reset_idx:02d}_student_depth.png"
        raw_path = out_dir / f"reset_{reset_idx:02d}_raw_depth_window.png"
        _save_depth_grid(student_path, student_depth)
        _save_depth_grid(raw_path, raw_window)
        print(f"[debug] wrote {student_path}")
        print(f"[debug] wrote {raw_path}")
        _print_depth_stats(
            f"[debug reset={reset_idx:02d}]",
            raw_depth_m,
            student_depth,
            near,
            far,
        )

    camera_data = inner.student_camera.data
    if hasattr(camera_data, "pos_w"):
        print(f"[debug] camera pos_w[0]={camera_data.pos_w[0].detach().cpu().tolist()}")
    if hasattr(camera_data, "quat_w_ros"):
        print(
            f"[debug] camera quat_w_ros[0]="
            f"{camera_data.quat_w_ros[0].detach().cpu().tolist()}"
        )

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
