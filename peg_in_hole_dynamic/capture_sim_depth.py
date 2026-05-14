#!/usr/bin/env python
"""Spawn Isaac Lab env at fixed-init and dump one preprocessed student depth.

Mirrors the "Init pose=fixed" path the GUI eval uses, so the output is the
same depth view the policy receives at training/eval time. Pair with
``deployment/capture_real_first_depth.py`` to compare real vs sim.

Example:
    .venv_isaacsim/bin/python -u peg_in_hole_dynamic/capture_sim_depth.py \\
        --out-dir /tmp/depth_compare
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


# Real-robot home pose used by deployment/home_robot.py. Robot is driven here
# at the start of every real-world session, so the sim depth must be captured
# with the robot at the same joint state for the comparison to be fair.
# Order = canonical (concat[iiwa_joint_1..7, joint_0.0..21.0]).
HOME_JOINT_POS_IIWA = np.array([
    -1.571,
    1.571 - np.deg2rad(10),
    0.0,
    1.376 + np.deg2rad(10),
    0.0,
    1.485,
    1.308,
], dtype=np.float64)
HOME_JOINT_POS_SHARPA = np.zeros(22, dtype=np.float64)
HOME_JOINT_POS_CANON = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])

# NOTE: peg z=0.541 (resting on the real-world table top the user measured)
# rather than the training fixed_start_pose z=0.63 (hovering ~0.10 m above).
# This is for a snap-image comparison test only.
_FIXED_INIT_OVERRIDES = {
    "env.reset.fixed_start_pose": [-0.10, 0.0, 0.541, 1.0, 0.0, 0.0, 0.0],
    "env.peg_in_hole.hole_x_range": [0.10, 0.10],
    "env.peg_in_hole.hole_y_range": [0.0, 0.0],
    "env.reset.reset_position_noise_x": 0.0,
    "env.reset.reset_position_noise_y": 0.0,
    "env.reset.reset_position_noise_z": 0.0,
    "env.reset.reset_dof_pos_random_interval_arm": 0.0,
    "env.reset.reset_dof_pos_random_interval_fingers": 0.0,
    "env.reset.reset_dof_vel_random_interval": 0.0,
    "env.reset.table_reset_z_range": 0.0,
}


def _save_frame(out_dir: Path, name: str, frame: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    png_path = out_dir / f"{name}.png"
    np.savez_compressed(npz_path, image=frame)
    from PIL import Image

    img = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(png_path)
    return png_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg.tol0p5mm")
    p.add_argument("--goal-mode", default="finalGoalOnly")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--out-dir", type=str, default="/tmp/depth_compare")
    p.add_argument("--name", type=str, default="sim_depth")
    p.add_argument("--no-fixed-init", action="store_true",
                   help="Skip the fixed-init overrides; use task yaml defaults.")
    p.add_argument("--match-real-cy", action="store_true",
                   help="Render at 160x94 so the principal point lands at "
                        "y=47, matching the real ZED's cy=47.13 (HD1080 "
                        "calibration). VISUALISATION ONLY -- changes V-FOV "
                        "from 42.6 deg to 44.2 deg, so the student trained "
                        "on 160x90 will see OOD obs if you reuse this cfg "
                        "for inference.")
    return p.parse_args()


def _reseed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    app = AppLauncher(launcher_args).app

    import isaacsimenvs  # noqa: F401  (registers gym tasks)
    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _instantiate_env,
        _load_env_cfg,
    )

    cfg = _load_env_cfg(args.task)
    extra = {} if args.no_fixed_init else dict(_FIXED_INIT_OVERRIDES)
    _apply_env_overrides(
        cfg,
        problem=args.problem,
        goal_mode=args.goal_mode,
        random_goal_fraction=0.0,
        insertion_success_tolerance=0.010,
        retract_success_tolerance=0.005,
        num_envs=int(args.num_envs),
        sim_device=args.sim_device,
        sdf=False,
        keep_dr=False,
        extra_overrides=extra,
    )
    # Force all DR knobs off so the sim depth we save is the deterministic
    # canonical view (no cam_pose_rand, no depth_aug, no obs/action delay).
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False

    # Match the real ZED's principal point.  HD1080 calibration on serial
    # 15107 gave cx=80.02, cy=47.13 after scaling to 160x90.  Sim's
    # PinholeCameraCfg always renders with optical axis at image center, so
    # to put the principal point at y=47 we render a taller image (160x94 =
    # cy 47 by symmetry) and keep the crop window at y=[0, 70).  Result: the
    # 70-tall policy crop has the optical axis at row 47, matching real.
    # WARNING: this only matches FOV for visualisation; do not use this 94
    # render path for policy inference (student was trained on 160x90).
    if args.match_real_cy:
        # Render at 160x94 so the optical axis lands at y=47 (= 94/2).
        # Crop window stays y=[0,70) -> 70 input rows, optical axis row 47.
        # image_input_height must remain 70 to match the crop output.
        cfg.student_obs.image_height = 94
        print("=> NOTE: image_height bumped 90 -> 94 so the y=[0,70) crop "
              "puts the optical axis at row 47 (matches real ZED cy=47.13).",
              flush=True)

    print(f"=> spawning env (problem={args.problem}, goal_mode={args.goal_mode}, "
          f"fixed_init={not args.no_fixed_init}, seed={args.seed})", flush=True)

    _reseed(args.seed)
    env = _instantiate_env(args.task, cfg)

    # Prime via two resets so the camera renders the post-reset scene.
    _reseed(args.seed)
    env.reset()
    _reseed(args.seed)
    env.reset()

    # Override the sim robot pose to the lab's HOME_JOINT_POS so the depth we
    # save matches the pose the real robot is driven to before the policy
    # starts. env.robot expects joint pos in LAB order; HOME_JOINT_POS_CANON
    # is in canonical order (concat[iiwa1..7, joint_0.0..21.0]), so reorder
    # via env._perm_canon_to_lab before writing.
    import torch

    home_canon = torch.tensor(
        HOME_JOINT_POS_CANON, device=env.device, dtype=torch.float32
    )
    home_lab = home_canon[env._perm_canon_to_lab]
    n_env = int(env.num_envs)
    pos_lab = home_lab.unsqueeze(0).expand(n_env, -1).contiguous()
    vel_lab = torch.zeros_like(pos_lab)
    env_ids = torch.arange(n_env, device=env.device, dtype=torch.long)
    env.robot.write_joint_state_to_sim(pos_lab, vel_lab, env_ids=env_ids)
    # Seed prev/cur targets so proprio's prev_action_targets reflects home.
    if hasattr(env, "_prev_targets"):
        env._prev_targets[env_ids] = pos_lab
    if hasattr(env, "_cur_targets"):
        env._cur_targets[env_ids] = pos_lab

    # One zero-action step so the camera renders the home-pose scene.
    zero_action = torch.zeros(n_env, int(env.action_space.shape[-1]),
                              device=env.device)
    env.step(zero_action)

    obs = env.get_student_obs()
    image = obs["image"]  # (N, C, H, W) float32, already preprocessed
    print(f"=> obs image shape={tuple(image.shape)} dtype={image.dtype} "
          f"min={float(image.min()):.3f} mean={float(image.mean()):.3f} "
          f"max={float(image.max()):.3f}", flush=True)

    out_dir = Path(args.out_dir).resolve()

    # 1) Policy-preprocessed 70x70 depth (the same thing the student sees).
    frame = image[0, 0].detach().float().cpu().numpy()
    png = _save_frame(out_dir, args.name, frame)
    print(f"=> saved {png}  shape={frame.shape}", flush=True)

    # 2) Raw uncropped 160x90 depth (in meters) straight from the sim sensor,
    #    before _apply_depth_noise / _crop_student_image / _preprocess_student_depth.
    #    Mirrors what capture_real_first_depth.py saves as <name>_raw.{npz,png}.
    try:
        cam_out = env.student_camera.data.output
        raw_depth_t = cam_out.get("distance_to_image_plane")
        if raw_depth_t is None:
            print("=> raw depth output unavailable (skipping raw dump).", flush=True)
        else:
            # Shapes: (N, H, W) or (N, H, W, 1).
            if raw_depth_t.dim() == 4 and raw_depth_t.shape[-1] == 1:
                raw_depth_t = raw_depth_t.squeeze(-1)
            raw_depth = raw_depth_t[0].detach().float().cpu().numpy()
            raw_name = f"{args.name}_raw"
            raw_npz = out_dir / f"{raw_name}.npz"
            raw_png = out_dir / f"{raw_name}.png"
            np.savez_compressed(raw_npz, depth_m=raw_depth)
            # Use a wide visualization window so both near and far structure is visible.
            from PIL import Image

            vis_near, vis_far = 0.30, 1.50
            valid = np.isfinite(raw_depth)
            safe = np.where(valid, raw_depth, vis_far)
            norm = (safe - vis_near) / max(vis_far - vis_near, 1e-6)
            img = (np.clip(norm, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            Image.fromarray(img).save(raw_png)
            print(
                f"=> saved {raw_png}  shape={raw_depth.shape} "
                f"valid_m=[{float(raw_depth[valid].min()) if valid.any() else float('nan'):.3f}, "
                f"{float(raw_depth[valid].max()) if valid.any() else float('nan'):.3f}] "
                f"finite_frac={float(valid.mean()):.3f}",
                flush=True,
            )
    except Exception as exc:
        print(f"=> raw depth dump failed: {exc!r}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
