"""Dump the inputs that Fast-FoundationStereo sees + the depth it produces.

Goal: sanity-check that the stereo TiledCamera pair is rendering the right
scene at the right baseline / cy, so we can rule out "garbage in, garbage
out" as the cause of FS's poor depth output.

Outputs (under depth_compare/fs_inputs/):
  left_rgb.png            384x224 RGB from /World/envs/env_0/StudentCameraLeft
  right_rgb.png           384x224 RGB from /World/envs/env_0/StudentCameraRight
  stereo_pair.png         side-by-side left | right for blink comparison
  fs_disparity_raw.png    raw FS disparity (224 rows x 384 cols, normalized
                          autoscale to its actual finite range)
  fs_depth_raw.png        FS depth in meters at the stereo resolution
                          (224 rows x 384 cols, autoscaled)
  fs_depth_policy.png     FS depth downsampled to the policy's 160x90
                          retrieve (same as what the policy ultimately sees)
  fs_inputs.npz           numpy arrays for all of the above
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
    p.add_argument("--model-dir",
                   default="third_party/Fast-FoundationStereo/weights/23-36-37")
    p.add_argument("--engine-dir",
                   default="third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4")
    p.add_argument("--valid-iters", type=int, default=4)
    p.add_argument("--stereo-width", type=int, default=384)
    p.add_argument("--stereo-height", type=int, default=224)
    p.add_argument("--out-dir", default="/share/portal/kk837/depthbasedRL/peg_in_hole_dynamic/depth_compare/fs_inputs")
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
    from PIL import Image

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides, _instantiate_env, _load_env_cfg,
    )
    from peg_in_hole_dynamic.capture_sim_depth import (
        _FIXED_INIT_OVERRIDES, HOME_JOINT_POS_CANON, _reseed,
    )

    cfg = _load_env_cfg(args.task)
    _apply_env_overrides(
        cfg,
        problem=args.problem, goal_mode=args.goal_mode,
        random_goal_fraction=0.0, insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005, num_envs=1, sim_device="cuda:0",
        sdf=False, keep_dr=False,
        extra_overrides=dict(_FIXED_INIT_OVERRIDES),
    )
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False

    cfg.student_obs.camera_backend = "foundation_stereo"
    # Upgrade sim render quality so FS gets in-distribution RGB. See
    # offline_eval_one_backend.py for the rationale. Tier 1 = quality preset
    # + DLAA + DL denoiser + 16 SPP. No path-tracing.
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
    cfg.student_obs.fs_engine_dir = str(args.engine_dir)
    cfg.student_obs.fs_valid_iters = int(args.valid_iters)
    cfg.student_obs.fs_stereo_width = int(args.stereo_width)
    cfg.student_obs.fs_stereo_height = int(args.stereo_height)
    cfg.student_obs.fs_downsample_to_policy_res = True

    print(
        f"=> FS inputs capture\n"
        f"   model_dir   : {cfg.student_obs.fs_model_dir}\n"
        f"   engine_dir  : {cfg.student_obs.fs_engine_dir}\n"
        f"   iters       : {cfg.student_obs.fs_valid_iters}\n"
        f"   stereo      : {cfg.student_obs.fs_stereo_width}x{cfg.student_obs.fs_stereo_height}\n"
        f"   v_off       : {cfg.student_obs.vertical_aperture_offset} cm (default)",
        flush=True,
    )

    _reseed(args.seed)
    env = _instantiate_env(args.task, cfg)
    _reseed(args.seed); env.reset()
    _reseed(args.seed); env.reset()

    home_canon = torch.tensor(HOME_JOINT_POS_CANON, device=env.device, dtype=torch.float32)
    home_lab = home_canon[env._perm_canon_to_lab]
    n_env = int(env.num_envs)
    pos_lab = home_lab.unsqueeze(0).expand(n_env, -1).contiguous()
    vel_lab = torch.zeros_like(pos_lab)
    env_ids = torch.arange(n_env, device=env.device, dtype=torch.long)
    env.robot.write_joint_state_to_sim(pos_lab, vel_lab, env_ids=env_ids)
    if hasattr(env, "_prev_targets"): env._prev_targets[env_ids] = pos_lab
    if hasattr(env, "_cur_targets"): env._cur_targets[env_ids] = pos_lab
    zero_action = torch.zeros(n_env, int(env.action_space.shape[-1]), device=env.device)
    env.step(zero_action)

    # Trigger the FS pipeline so env.student_camera_{left,right} and the FS
    # module have current outputs to read from.
    obs = env.get_student_obs()
    print(f"=> policy image  : {tuple(obs['image'].shape)} (cropped 70x70)", flush=True)

    # Pull left + right RGB at the stereo resolution.
    left_rgba  = env.student_camera_left.data.output["rgb"][0].detach().cpu().numpy()
    right_rgba = env.student_camera_right.data.output["rgb"][0].detach().cpu().numpy()
    left_rgb  = left_rgba[..., :3].astype(np.uint8)
    right_rgb = right_rgba[..., :3].astype(np.uint8)
    print(f"=> left  RGB    : shape={left_rgb.shape}  min={left_rgb.min()} max={left_rgb.max()}")
    print(f"=> right RGB    : shape={right_rgb.shape}  min={right_rgb.min()} max={right_rgb.max()}")

    # Re-run FS to grab the disparity directly (the env path immediately
    # converts to depth + downsamples).
    from isaacsimenvs.perception.fast_foundation_stereo import FastFoundationStereoModule
    fs = env._fs_module
    left_t  = env.student_camera_left.data.output["rgb"][..., :3].permute(0,3,1,2).float().contiguous()
    right_t = env.student_camera_right.data.output["rgb"][..., :3].permute(0,3,1,2).float().contiguous()

    fx_px = float(cfg.student_obs.fs_stereo_width) * float(cfg.student_obs.focal_length) \
            / float(cfg.student_obs.horizontal_aperture)
    print(f"=> fx_px @ stereo: {fx_px:.2f} px")
    print(f"=> baseline      : {cfg.student_obs.fs_stereo_baseline_m} m")

    depth_stereo = fs(left_t, right_t,
                      fx_px=fx_px,
                      baseline_m=float(cfg.student_obs.fs_stereo_baseline_m))
    depth_stereo_np = depth_stereo[0, 0].detach().cpu().numpy()
    # disparity = fx * baseline / depth
    disparity_stereo_np = fx_px * float(cfg.student_obs.fs_stereo_baseline_m) / np.clip(depth_stereo_np, 1e-3, None)
    finite_d = np.isfinite(depth_stereo_np) & (depth_stereo_np < 50)
    print(f"=> FS depth (stereo res, finite_frac={finite_d.mean():.3f}):")
    if finite_d.any():
        d = depth_stereo_np[finite_d]
        print(f"     range [{d.min():.3f}, {d.max():.3f}] m  mean={d.mean():.3f} m")
        print(f"     p25={np.percentile(d, 25):.3f}  p50={np.percentile(d, 50):.3f}  p75={np.percentile(d, 75):.3f}")
    print(f"=> FS disparity  : range [{disparity_stereo_np.min():.2f}, {disparity_stereo_np.max():.2f}] px")

    # Downsample to policy resolution for comparability with the other captures.
    import torch.nn.functional as F
    depth_policy = F.interpolate(
        depth_stereo, size=(int(cfg.student_obs.image_height), int(cfg.student_obs.image_width)),
        mode="bilinear", antialias=True,
    )
    depth_policy_np = depth_policy[0, 0].detach().cpu().numpy()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(left_rgb).save(out_dir / "left_rgb.png")
    Image.fromarray(right_rgb).save(out_dir / "right_rgb.png")
    Image.fromarray(np.concatenate([left_rgb, right_rgb], axis=1)).save(out_dir / "stereo_pair.png")

    # Autoscale disparity / depth pngs.
    def _autoscale_u8(arr, fill=None):
        m = np.isfinite(arr) & (arr < 50) if fill is None else np.ones_like(arr, dtype=bool)
        if not m.any():
            return np.zeros(arr.shape, dtype=np.uint8)
        lo, hi = float(arr[m].min()), float(arr[m].max())
        a = np.where(m, arr, hi)
        return (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)

    Image.fromarray(_autoscale_u8(disparity_stereo_np)).save(out_dir / "fs_disparity_raw.png")
    Image.fromarray(_autoscale_u8(depth_stereo_np)).save(out_dir / "fs_depth_raw.png")
    Image.fromarray(_autoscale_u8(depth_policy_np)).save(out_dir / "fs_depth_policy.png")

    np.savez_compressed(
        out_dir / "fs_inputs.npz",
        left_rgb=left_rgb, right_rgb=right_rgb,
        fs_disparity_stereo=disparity_stereo_np.astype(np.float32),
        fs_depth_stereo=depth_stereo_np.astype(np.float32),
        fs_depth_policy=depth_policy_np.astype(np.float32),
        fx_px=np.float32(fx_px),
        baseline_m=np.float32(cfg.student_obs.fs_stereo_baseline_m),
    )
    print(f"=> wrote {out_dir}/", flush=True)
    for f in sorted(out_dir.iterdir()):
        print(f"     {f.name}", flush=True)

    env.close()
    del app
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
