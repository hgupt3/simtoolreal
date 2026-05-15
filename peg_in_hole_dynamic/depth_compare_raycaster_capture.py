"""Capture sim depth at the real-robot HOME pose with the raycaster
backend + the real ZED HD1080 intrinsics (fx=fy=115.67, cx=80.02, cy=47.13
at the 160x90 retrieve resolution). The rasterizer silently drops aperture
offsets, so the existing capture_sim_depth.py's `--match-real-cy` variant
couldn't actually shift cy — the raycaster honors them, so this gets a
real principal-point match.

Saves alongside the existing depth_compare/ outputs for direct comparison
with real_depth_raw.npz.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg.tol0p5mm")
    p.add_argument("--goal-mode", default="finalGoalOnly")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--match-real-cy",
        action="store_true",
        help="Set vertical_aperture_offset dimensionless so cy = 47.13 px "
        "at the 160x90 retrieve resolution (real ZED HD1080 cal).",
    )
    p.add_argument(
        "--direct-width",
        type=int,
        default=None,
        help="If set together with --direct-height, render a direct ROI of "
        "this exact size (no crop). Skips the 160x90->70x70 crop pipeline. "
        "Use 70 to match the policy's input window, or 210 for 3x hi-res.",
    )
    p.add_argument(
        "--direct-height",
        type=int,
        default=None,
        help="See --direct-width.",
    )
    p.add_argument(
        "--out-name",
        default="sim_depth_homepose_raycaster_matched",
        help="Base name written to /depth_compare/ (no extension).",
    )
    p.add_argument(
        "--out-dir",
        default="/share/portal/kk837/depthbasedRL/peg_in_hole_dynamic/depth_compare",
    )
    args = p.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    la, _ = lp.parse_known_args([])
    la.headless = True
    la.enable_cameras = True
    app = AppLauncher(la).app

    import torch
    import isaacsimenvs  # noqa: F401

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _instantiate_env,
        _load_env_cfg,
    )
    from peg_in_hole_dynamic.capture_sim_depth import (
        _FIXED_INIT_OVERRIDES,
        HOME_JOINT_POS_CANON,
        _reseed,
    )

    cfg = _load_env_cfg(args.task)
    _apply_env_overrides(
        cfg,
        problem=args.problem,
        goal_mode=args.goal_mode,
        random_goal_fraction=0.0,
        insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005,
        num_envs=1,
        sim_device="cuda:0",
        sdf=False,
        keep_dr=False,
        extra_overrides=dict(_FIXED_INIT_OVERRIDES),
    )

    # All DR / pose-rand off — deterministic canonical view.
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False

    # --- Force the raycaster backend ---
    cfg.student_obs.camera_backend = "raycaster"

    # --- Real ZED HD1080 intrinsics at 160x90 (lab serial 15107) ---
    #   fx = fy = 115.67 px
    #   cx = 80.02 px (image center 80 -> sub-pixel offset, ignore)
    #   cy = 47.13 px (image center 45 -> 2.13 px shift)
    if args.direct_width is not None and args.direct_height is not None:
        # Direct-ROI render: bypass the full 160x90 capture + crop. The crop
        # the deployed pipeline applies is [crop_x0:crop_x1, crop_y0:crop_y1]
        # = [90:160, 0:70]. We shrink the aperture so the per-pixel pitch
        # matches the full-frame pitch, then shift the principal point via
        # the (dimensionless) aperture offsets so it lands at the real ZED's
        # cy=47.13 inside the new direct-ROI frame.
        FULL_W, FULL_H = 160, 90
        CROP_X0, CROP_X1 = 90, 160      # deployed crop window
        CROP_Y0, CROP_Y1 = 0, 70
        ZED_CX, ZED_CY = 80.02, 47.13   # principal point in 160x90 frame

        Wd, Hd = int(args.direct_width), int(args.direct_height)
        f_cm = float(cfg.student_obs.focal_length)
        H_ap_full = float(cfg.student_obs.horizontal_aperture)

        # Map the deployed crop to a target ROI at the chosen direct size.
        scale = Wd / (CROP_X1 - CROP_X0)
        assert abs(scale - Hd / (CROP_Y1 - CROP_Y0)) < 1e-6, (
            "--direct-width/height must match the crop window aspect (70x70 "
            "for baseline, 210x210 for 3x hi-res)."
        )
        # New aperture: same per-pixel pitch as the full frame.
        H_ap_d = H_ap_full * Wd / (FULL_W * scale)
        # Principal point in the new direct-ROI frame.
        cx_d = (ZED_CX - CROP_X0) * scale
        cy_d = (ZED_CY - CROP_Y0) * scale
        fx_d_px = Wd * f_cm / H_ap_d
        fy_d_px = Hd * f_cm / (H_ap_d * (Hd / Wd))   # square pixels
        h_off_d = (cx_d - Wd / 2.0) / fx_d_px
        v_off_d = (cy_d - Hd / 2.0) / fy_d_px

        cfg.student_obs.image_width = Wd
        cfg.student_obs.image_height = Hd
        cfg.student_obs.image_input_width = Wd
        cfg.student_obs.image_input_height = Hd
        cfg.student_obs.crop_enabled = False
        cfg.student_obs.horizontal_aperture = H_ap_d
        cfg.student_obs.horizontal_aperture_offset = float(h_off_d)
        cfg.student_obs.vertical_aperture_offset = float(v_off_d)
        print(
            f"=> direct-ROI: {Wd}x{Hd}, H_ap={H_ap_d:.3f} cm, "
            f"fx_px={fx_d_px:.2f}, principal_point=({cx_d:.2f}, {cy_d:.2f}) "
            f"(ZED cy=47.13 baked in), h_off={h_off_d:.6f}, v_off={v_off_d:.6f}",
            flush=True,
        )
    elif args.match_real_cy:
        # PinholeCameraPatternCfg expects DIMENSIONLESS offsets:
        #   offset = (c - n/2) / f_px
        # so v_off = (47.13 - 45) / 115.67 = 0.01841.
        # The (sub-pixel) h_off = (80.02 - 80) / 115.67 ~= 0.000173, dropped.
        cfg.student_obs.vertical_aperture_offset = (47.13 - 45.0) / 115.67
        cfg.student_obs.horizontal_aperture_offset = 0.0
        print(
            f"=> raycaster v_offset={cfg.student_obs.vertical_aperture_offset:.6f} "
            f"(dimensionless, places cy at row 47.13)",
            flush=True,
        )
    else:
        # Centered principal point (matches what the tiled rasterizer
        # effectively produces, since it drops offsets).
        cfg.student_obs.horizontal_aperture_offset = 0.0
        cfg.student_obs.vertical_aperture_offset = 0.0
        print("=> raycaster offsets forced to 0 (geometric-center cy=45)", flush=True)

    print(
        f"=> camera_pos={tuple(cfg.student_obs.camera_pos)}\n"
        f"=> camera_quat_wxyz={tuple(cfg.student_obs.camera_quat_wxyz)}\n"
        f"=> image={cfg.student_obs.image_width}x{cfg.student_obs.image_height}, "
        f"focal_length={cfg.student_obs.focal_length}, "
        f"H_ap={cfg.student_obs.horizontal_aperture}",
        flush=True,
    )

    _reseed(args.seed)
    env = _instantiate_env(args.task, cfg)
    _reseed(args.seed)
    env.reset()
    _reseed(args.seed)
    env.reset()

    # Write the deployment HOME joint pose into the sim robot.
    home_canon = torch.tensor(HOME_JOINT_POS_CANON, device=env.device, dtype=torch.float32)
    home_lab = home_canon[env._perm_canon_to_lab]
    n_env = int(env.num_envs)
    pos_lab = home_lab.unsqueeze(0).expand(n_env, -1).contiguous()
    vel_lab = torch.zeros_like(pos_lab)
    env_ids = torch.arange(n_env, device=env.device, dtype=torch.long)
    env.robot.write_joint_state_to_sim(pos_lab, vel_lab, env_ids=env_ids)
    if hasattr(env, "_prev_targets"):
        env._prev_targets[env_ids] = pos_lab
    if hasattr(env, "_cur_targets"):
        env._cur_targets[env_ids] = pos_lab
    zero_action = torch.zeros(n_env, int(env.action_space.shape[-1]), device=env.device)
    env.step(zero_action)

    obs = env.get_student_obs()
    image = obs["image"][0, 0].detach().float().cpu().numpy()
    raw = env.student_camera.data.output["distance_to_image_plane"][0].detach().float().cpu().numpy()
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{args.out_name}.npz", image=image)
    np.savez_compressed(out_dir / f"{args.out_name}_raw.npz", depth_m=raw)

    # PNGs
    from PIL import Image
    img_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img_u8).save(out_dir / f"{args.out_name}.png")
    finite = np.isfinite(raw) & (raw < 50)
    if finite.any():
        lo, hi = float(raw[finite].min()), float(raw[finite].max())
        d = np.where(finite, raw, hi)
        raw_u8 = (np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(raw_u8).save(out_dir / f"{args.out_name}_raw.png")
        print(
            f"=> raw range [{lo:.3f}, {hi:.3f}] m, mean={float(raw[finite].mean()):.3f} m",
            flush=True,
        )

    print(f"=> wrote {out_dir}/{args.out_name}.npz + .png", flush=True)
    print(f"=> wrote {out_dir}/{args.out_name}_raw.npz + .png", flush=True)

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
