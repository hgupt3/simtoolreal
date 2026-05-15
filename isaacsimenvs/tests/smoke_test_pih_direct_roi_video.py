"""Record a depth video of the real PegInHoleDepthStudent env using the
direct-ROI raycaster pipeline: `camera_backend='raycaster'` rendering at
exactly the policy's input resolution (default 210×210), no 480×270→crop.

This is the configuration we benchmarked at 3.07× the throughput and 14 GiB
less GPU memory than tiled@480×270→crop. The video is just to eyeball that
the resulting depth image is sensible while the robot is actually moving.

Usage:
    .venv_isaacsim/bin/python isaacsimenvs/tests/smoke_test_pih_direct_roi_video.py \\
        --num-steps 150 --action-scale 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "videos" / "smoke_test_pih_direct_roi_video"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    parser.add_argument("--problem", default="Lpeg.tol0p5mm")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=("raycaster", "tiled"),
        default="raycaster",
        help="raycaster: render direct-ROI at (W_roi x H_roi). "
        "tiled: render full-frame and crop (the existing deployed pipeline).",
    )
    # Capture-frame dimensions (used to compute direct-ROI intrinsics from
    # the cropped pipeline's coordinates).
    parser.add_argument("--full-image-width", type=int, default=480)
    parser.add_argument("--full-image-height", type=int, default=270)
    # ROI within the full frame (matches the existing hi-res sub crop).
    parser.add_argument("--crop-x0", type=int, default=270)
    parser.add_argument("--crop-y0", type=int, default=0)
    parser.add_argument("--crop-x1", type=int, default=480)
    parser.add_argument("--crop-y1", type=int, default=210)
    # Source intrinsics from the env cfg defaults; can be overridden.
    parser.add_argument("--focal-length", type=float, default=24.0)
    parser.add_argument("--full-horizontal-aperture", type=float, default=33.19737869997174)
    # IMPORTANT: the deployed rasterizer (TiledCamera/Replicator) SILENTLY
    # DROPS aperture offsets — see sensors.py:110 in IsaacSim. So even
    # though the env cfg sets h=0.0035, v=0.4418 (ZED principal-point cal),
    # the policy actually saw a centered principal point during training.
    # To match what the trained policy sees, default the "full-frame"
    # offsets here to 0.0 and recompute the direct-ROI offsets from a
    # centered (cx_full=W/2, cy_full=H/2) principal point.
    parser.add_argument("--full-horizontal-aperture-offset", type=float, default=0.0)
    parser.add_argument("--full-vertical-aperture-offset", type=float, default=0.0)
    parser.add_argument("--num-warmup-steps", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--action-scale", type=float, default=0.4)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    spec = gym.spec(args.task)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})

    cfg.scene.num_envs = int(args.num_envs)
    cfg.peg_in_hole.problem = args.problem
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"

    # Lock the scene so two runs with different backends produce bit-identical
    # robot/hole/peg states given identical RNG-seeded actions.
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

    # --- Compute direct-ROI intrinsics from the cropped-pipeline equivalent ---
    # The cropped pipeline captures (W_full × H_full), then takes a
    # (W_roi × H_roi) crop at [crop_x0:crop_x1, crop_y0:crop_y1]. The
    # equivalent direct-render at (W_roi × H_roi) keeps the same focal
    # length and per-pixel pitch by shrinking the aperture proportionally,
    # then shifts the principal point via aperture offsets so it lands at
    # the same world direction.
    #
    # CRITICAL note on units (this trips up direct comparison against the
    # rasterizer's StudentObs config):
    #   Tiled / USD Camera offsets are in SCENE UNITS at the sensor plane
    #   (cm @ Isaac Lab scale). PinholeCameraPatternCfg's offsets are
    #   DIMENSIONLESS — the formula in ray_caster_camera._compute_intrinsic_
    #   matrices is `c_x = horizontal_aperture_offset * f_x + W/2`, so
    #   offset = (c_x_px - W/2) / f_x_px.
    W_full = int(args.full_image_width)
    H_full = int(args.full_image_height)
    W_roi = int(args.crop_x1) - int(args.crop_x0)
    H_roi = int(args.crop_y1) - int(args.crop_y0)
    f_cm = float(args.focal_length)
    H_ap_full = float(args.full_horizontal_aperture)
    V_ap_full = H_ap_full * (H_full / W_full)  # square pixels
    # Original full-frame intrinsics in pixels (rasterizer convention)
    fx_full_px = W_full * f_cm / H_ap_full
    fy_full_px = H_full * f_cm / V_ap_full
    # The full-frame offsets in scene units (cm) come from the cropped
    # pipeline's StudentObsCfg — they apply to the rasterizer. Convert to
    # the px shift via the sensor pitch.
    pitch_full_h = H_ap_full / W_full  # cm/px
    pitch_full_v = V_ap_full / H_full
    cx_full = W_full / 2.0 + float(args.full_horizontal_aperture_offset) / pitch_full_h
    cy_full = H_full / 2.0 + float(args.full_vertical_aperture_offset) / pitch_full_v
    # Principal point inside the crop (= inside the new direct-ROI image)
    cx_roi = cx_full - int(args.crop_x0)
    cy_roi = cy_full - int(args.crop_y0)
    # New aperture preserves per-pixel pitch: H_ap_roi / W_roi == H_ap_full / W_full
    H_ap_roi = H_ap_full * (W_roi / W_full)
    V_ap_roi = H_ap_roi * (H_roi / W_roi)
    # Raycaster offsets are dimensionless: (c - n/2) / f_n
    fx_roi_px = W_roi * f_cm / H_ap_roi
    fy_roi_px = H_roi * f_cm / V_ap_roi
    h_off_roi = (cx_roi - W_roi / 2.0) / fx_roi_px
    v_off_roi = (cy_roi - H_roi / 2.0) / fy_roi_px

    print(
        f"[direct-roi-vid] derived intrinsics for direct-ROI render:\n"
        f"  full frame  : {W_full}x{H_full}, H_ap={H_ap_full:.4f} cm, "
        f"fx_px={fx_full_px:.2f} px\n"
        f"  full pp (px): ({cx_full:.2f}, {cy_full:.2f})  "
        f"(from rasterizer offsets h={args.full_horizontal_aperture_offset:.4f} cm, "
        f"v={args.full_vertical_aperture_offset:.4f} cm)\n"
        f"  ROI crop    : x=[{args.crop_x0},{args.crop_x1}], y=[{args.crop_y0},{args.crop_y1}] "
        f"-> {W_roi}x{H_roi}\n"
        f"  ROI pp (px) : ({cx_roi:.2f}, {cy_roi:.2f})  "
        f"(ROI center=({W_roi/2:.1f}, {H_roi/2:.1f}))\n"
        f"  ROI H_ap    : {H_ap_roi:.4f} cm, fx_px={fx_roi_px:.2f} px "
        f"(same per-px pitch as full frame: {pitch_full_h*1e4:.3f} um/px)\n"
        f"  ROI offsets : h_off={h_off_roi:.6f}, v_off={v_off_roi:.6f} (dimensionless)",
        flush=True,
    )

    if str(args.backend) == "raycaster":
        cfg.student_obs.camera_backend = "raycaster"
        cfg.student_obs.image_width = W_roi
        cfg.student_obs.image_height = H_roi
        cfg.student_obs.crop_enabled = False
        cfg.student_obs.image_input_width = W_roi
        cfg.student_obs.image_input_height = H_roi
        cfg.student_obs.focal_length = float(args.focal_length)
        cfg.student_obs.horizontal_aperture = float(H_ap_roi)
        cfg.student_obs.horizontal_aperture_offset = float(h_off_roi)
        cfg.student_obs.vertical_aperture_offset = float(v_off_roi)
    else:
        # Tiled / deployed pipeline: full-frame capture + crop. Aperture
        # offsets are silently dropped by Replicator, so we leave them at
        # whatever the yaml has (effective principal point = image center).
        cfg.student_obs.camera_backend = "tiled"
        cfg.student_obs.image_width = W_full
        cfg.student_obs.image_height = H_full
        cfg.student_obs.crop_enabled = True
        cfg.student_obs.crop_top_left = (int(args.crop_x0), int(args.crop_y0))
        cfg.student_obs.crop_bottom_right = (int(args.crop_x1), int(args.crop_y1))
        cfg.student_obs.image_input_width = W_roi
        cfg.student_obs.image_input_height = H_roi
        cfg.student_obs.focal_length = float(args.focal_length)
        cfg.student_obs.horizontal_aperture = float(H_ap_full)
        cfg.student_obs.horizontal_aperture_offset = float(args.full_horizontal_aperture_offset)
        cfg.student_obs.vertical_aperture_offset = float(args.full_vertical_aperture_offset)

    print(
        f"[direct-roi-vid] backend={args.backend} cam={W_roi}x{H_roi} "
        f"n_envs={args.num_envs} warmup={args.num_warmup_steps} steps={args.num_steps} "
        f"action_scale={args.action_scale}",
        flush=True,
    )

    env = gym.make(args.task, cfg=cfg)
    inner = env.unwrapped
    device = inner.device
    action_dim = cfg.action_space
    env.reset()

    zero_act = torch.zeros((inner.num_envs, action_dim), device=device, dtype=torch.float32)
    for _ in range(int(args.num_warmup_steps)):
        env.step(zero_act)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_frames: list[np.ndarray] = []
    raw_frames: list[np.ndarray] = []
    raw_values: list[np.ndarray] = []

    rng = np.random.default_rng(int(args.seed))

    def _gray_to_rgb(g: np.ndarray) -> np.ndarray:
        return np.repeat(g[..., None], 3, axis=-1)

    for step_i in range(int(args.num_steps)):
        a_np = rng.uniform(-1.0, 1.0, size=(inner.num_envs, action_dim)).astype(np.float32)
        a = torch.from_numpy(a_np).to(device) * float(args.action_scale)
        env.step(a)

        obs = inner.get_student_obs()
        image = obs["image"][0, 0].detach().float().cpu().numpy()
        norm_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        norm_frames.append(_gray_to_rgb(norm_u8))

        raw = inner.student_camera.data.output["distance_to_image_plane"][0].detach().float().cpu().numpy()
        if raw.ndim == 3 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        raw_values.append(raw.copy())

        if step_i % 15 == 0:
            finite = np.isfinite(raw)
            stats = (
                f"min={float(np.nanmin(raw)):.3f} "
                f"max={float(np.nanmax(raw)):.3f} "
                f"mean={float(np.nanmean(raw)):.3f}"
                if finite.any()
                else "all-non-finite"
            )
            print(
                f"[direct-roi-vid] step {step_i:03d}  norm[min={image.min():.3f} "
                f"max={image.max():.3f}]  raw {stats}",
                flush=True,
            )

    raw_stack = np.stack(raw_values, axis=0)
    finite_mask = np.isfinite(raw_stack)
    if finite_mask.any():
        lo = float(raw_stack[finite_mask].min())
        hi = float(raw_stack[finite_mask].max())
    else:
        lo, hi = 0.0, 1.0
    for raw in raw_values:
        d = np.where(np.isfinite(raw), raw, hi)
        norm = np.clip((d - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        raw_frames.append(_gray_to_rgb((norm * 255.0).astype(np.uint8)))

    tag = "raycaster_direct" if args.backend == "raycaster" else "tiled_crop"
    norm_mp4 = out_dir / f"{tag}_{W_roi}x{H_roi}_norm.mp4"
    raw_mp4 = out_dir / f"{tag}_{W_roi}x{H_roi}_raw.mp4"
    raw_npy = out_dir / f"{tag}_{W_roi}x{H_roi}_raw.npy"
    np.save(str(raw_npy), np.stack(raw_values, axis=0))
    imageio.mimwrite(str(norm_mp4), norm_frames, fps=int(args.video_fps))
    imageio.mimwrite(str(raw_mp4), raw_frames, fps=int(args.video_fps))

    print(
        f"[direct-roi-vid] wrote {norm_mp4} "
        f"(policy-input window-normalized depth, {len(norm_frames)} frames)",
        flush=True,
    )
    print(
        f"[direct-roi-vid] wrote {raw_mp4} "
        f"(raw distance_to_image_plane autoscaled to [{lo:.3f}, {hi:.3f}] m, "
        f"{len(raw_frames)} frames)",
        flush=True,
    )

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
