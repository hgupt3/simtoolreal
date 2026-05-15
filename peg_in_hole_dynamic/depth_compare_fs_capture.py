"""Capture sim depth at the real-robot HOME pose with the
camera_backend='foundation_stereo' path: render a stereo TiledCamera pair
at the team's deployment resolution (384x224), run Fast-FoundationStereo
to recover disparity, convert to metric depth, downsample to the policy's
160x90 retrieve resolution, then dump both the policy-input crop (70x70
in [0,1]) and the raw 160x90 depth (in meters) under depth_compare/.

Mirrors capture_sim_depth.py / depth_compare_raycaster_capture.py so the
output drops alongside the tiled / raycaster variants for direct
comparison with the real ZED capture.
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
        "--model-dir",
        default="third_party/Fast-FoundationStereo/weights/23-36-37",
    )
    p.add_argument(
        "--engine-dir",
        default="third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4",
        help="If both feature_runner.engine and post_runner.engine exist here, "
        "TRT inference is used; otherwise the PyTorch .pth path is taken.",
    )
    p.add_argument("--valid-iters", type=int, default=4)
    p.add_argument("--stereo-width", type=int, default=384)
    p.add_argument("--stereo-height", type=int, default=224)
    p.add_argument(
        "--out-name",
        default="sim_depth_homepose_fs_23-36-37_iters4_384x224",
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

    # --- Force the Fast-FoundationStereo backend ---
    cfg.student_obs.camera_backend = "foundation_stereo"
    # Tier 1 render-quality upgrade so FS sees clean RGB (matches the team's
    # ~32-128 SPP Omniverse path-traced training distribution closer than
    # the default 1-SPP RTX RealTime path used by other backends).
    cfg.sim.render.rendering_mode = "quality"
    cfg.sim.render.antialiasing_mode = "DLAA"
    cfg.sim.render.enable_dl_denoiser = True
    cfg.sim.render.samples_per_pixel = 16
    cfg.sim.render.enable_direct_lighting = True
    cfg.sim.render.enable_reflections = True
    cfg.sim.render.enable_global_illumination = True
    cfg.sim.render.enable_shadows = True
    cfg.sim.render.enable_ambient_occlusion = True
    cfg.student_obs.fs_model_dir = str(args.model_dir)
    cfg.student_obs.fs_engine_dir = str(args.engine_dir) if args.engine_dir else ""
    cfg.student_obs.fs_valid_iters = int(args.valid_iters)
    cfg.student_obs.fs_stereo_width = int(args.stereo_width)
    cfg.student_obs.fs_stereo_height = int(args.stereo_height)

    # Keep the policy retrieve resolution at the standard 160x90 baseline so
    # the cropped 70x70 image is directly comparable to the other variants.
    cfg.student_obs.fs_downsample_to_policy_res = True

    print(
        f"=> FS sim capture\n"
        f"   model_dir   : {cfg.student_obs.fs_model_dir}\n"
        f"   engine_dir  : {cfg.student_obs.fs_engine_dir or '(none -> PyTorch fallback)'}\n"
        f"   valid_iters : {cfg.student_obs.fs_valid_iters}\n"
        f"   stereo      : {cfg.student_obs.fs_stereo_width}x{cfg.student_obs.fs_stereo_height}\n"
        f"   policy res  : {cfg.student_obs.image_width}x{cfg.student_obs.image_height}",
        flush=True,
    )

    _reseed(args.seed)
    env = _instantiate_env(args.task, cfg)
    _reseed(args.seed)
    env.reset()
    _reseed(args.seed)
    env.reset()

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
    # The downsampled (160x90) raw depth in meters is what fed into the
    # window-normalize pipeline. _last_student_image_noisy is the cropped
    # 70x70 policy view in [0, 1]; the 160x90 raw m view we need to grab
    # by hand via a second pass.
    cfg.student_obs.fs_downsample_to_policy_res = True
    # Use the FS module directly to grab the 160x90 raw depth in meters
    # (the policy depth modality has been preprocessed into [0,1], so we
    # need the FS output before _preprocess_student_depth).
    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import _run_foundation_stereo
    raw_m = _run_foundation_stereo(env, cfg.student_obs)
    raw = raw_m[0, 0].detach().float().cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{args.out_name}.npz", image=image)
    np.savez_compressed(out_dir / f"{args.out_name}_raw.npz", depth_m=raw)

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
            f"=> raw 160x90 depth: range [{lo:.3f}, {hi:.3f}] m, "
            f"mean={float(raw[finite].mean()):.3f} m, "
            f"finite_frac={finite.mean():.3f}",
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
