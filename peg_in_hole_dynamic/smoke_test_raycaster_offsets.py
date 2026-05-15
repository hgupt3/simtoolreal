"""Smoke test: verify raycaster's aperture_offset is dimensionless.

Spawns a tiny env (n=4) with the same direct-70x70 raycaster config the
student subs use, calls the actual `read_student_camera_image()` pipeline
(matches training), and dumps both the raw depth (m) and policy view to
disk. Run twice -- once with the CM values from the production sub, once
with the DIMENSIONLESS values implied by the offset formula -- and compare.

Usage:
    .venv_isaacsim/bin/python peg_in_hole_dynamic/smoke_test_raycaster_offsets.py \\
        --h-off -0.933 --v-off 0.2517 --tag cm_units
    .venv_isaacsim/bin/python peg_in_hole_dynamic/smoke_test_raycaster_offsets.py \\
        --h-off -0.389 --v-off 0.105 --tag dimensionless
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg_matchedmass.tol0p5mm")
    p.add_argument("--h-off", type=float, required=True,
                   help="horizontal_aperture_offset value to test")
    p.add_argument("--v-off", type=float, required=True,
                   help="vertical_aperture_offset value to test")
    p.add_argument("--tag", required=True,
                   help="Output filename tag (e.g. 'cm_units' or 'dimensionless').")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir",
                   default="/tmp/smoke_raycaster_offsets")
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

    # Reproduce the production student sub raycaster config.
    cfg.student_obs.camera_backend = "raycaster"
    cfg.student_obs.image_width = 70
    cfg.student_obs.image_height = 70
    cfg.student_obs.image_input_width = 70
    cfg.student_obs.image_input_height = 70
    cfg.student_obs.crop_enabled = False
    cfg.student_obs.horizontal_aperture = 14.524
    cfg.student_obs.horizontal_aperture_offset = float(args.h_off)
    cfg.student_obs.vertical_aperture_offset = float(args.v_off)
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False

    print(
        f"=> tag={args.tag} h_off={args.h_off} v_off={args.v_off} "
        f"H_ap={cfg.student_obs.horizontal_aperture}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = _instantiate_env(args.task, cfg)
    env.reset()
    env.reset()

    n_env = int(env.num_envs)
    zero_action = torch.zeros(n_env, int(env.action_space.shape[-1]), device=env.device)
    env.step(zero_action)

    # This is the production path (forces sensor update via force_recompute=True).
    obs = env.get_student_obs()
    image = obs["image"][0, 0].detach().float().cpu().numpy()    # policy view, [0,1]

    # Also grab the raw raycaster depth in meters.
    raw_t = env.student_camera.data.output.get("distance_to_image_plane")
    if raw_t.dim() == 4 and raw_t.shape[-1] == 1:
        raw_t = raw_t.squeeze(-1)
    raw = raw_t[0].detach().float().cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    # Policy view ([0,1]) -> grayscale png.
    img_u8 = (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(img_u8).save(out_dir / f"{args.tag}_policy.png")

    # Raw depth (m) -> wide-range visualization 0.3-1.5m.
    finite = np.isfinite(raw)
    vis_near, vis_far = 0.30, 1.50
    safe = np.where(finite, raw, vis_far)
    norm = np.clip((safe - vis_near) / max(vis_far - vis_near, 1e-9), 0, 1)
    Image.fromarray((norm * 255).round().astype(np.uint8)).save(
        out_dir / f"{args.tag}_raw.png"
    )

    if finite.any():
        print(
            f"=> tag={args.tag} raw depth (m): "
            f"min={float(raw[finite].min()):.3f} "
            f"max={float(raw[finite].max()):.3f} "
            f"mean={float(raw[finite].mean()):.3f} "
            f"finite_frac={float(finite.mean()):.3f}",
            flush=True,
        )
    else:
        print(f"=> tag={args.tag} raw depth: ALL NaN/inf (no rays hit)", flush=True)

    print(
        f"=> tag={args.tag} policy view: "
        f"min={float(image.min()):.3f} max={float(image.max()):.3f} "
        f"mean={float(image.mean()):.3f}",
        flush=True,
    )
    print(f"=> saved {out_dir}/{args.tag}_policy.png and {args.tag}_raw.png",
          flush=True)

    try:
        env.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
