"""Render init-state distribution overlays IN IsaacSim for fig 4 panel (a).

For each task:
    1. Boot Isaac Sim with the env (single env to keep memory light).
    2. Add a TiledCamera positioned top-down over the env.
    3. Loop N=10 times:
        - Reset the env (random object + goal pose per env config).
        - Step physics for SETTLE_STEPS to let the object settle.
        - Capture the camera RGB.
    4. PIL-composite the 10 captures with alpha blending → one PNG.

Outputs:
    plot_figures/fig4/inputs/init_dist_<task>_sim.png

Usage:
    .venv_isaacsim/bin/python plot_figures/fig4/render_in_sim.py \\
        --task Isaacsimenvs-PegInHole-Direct-v0 \\
        --problem Lpeg.tol0p5mm \\
        --slug peg_in_hole
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaacsimenvs-PegInHole-Direct-v0")
    parser.add_argument("--problem", default="Lpeg.tol0p5mm")
    parser.add_argument("--slug", required=True, help="filename slug, e.g. peg_in_hole")
    parser.add_argument("--num-resets", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=150)
    parser.add_argument("--cam-height", type=float, default=1.0,
                        help="camera height above env origin (m)")
    parser.add_argument("--cam-offset-y", type=float, default=-0.35,
                        help="camera y-offset for slight iso angle (m, negative pulls camera toward -y)")
    parser.add_argument("--cam-xmag", type=float, default=0.40,
                        help="orthographic half-width (m); fov-equivalent")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
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
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = gym.spec(args.task)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})

    cfg.scene.num_envs = 1
    cfg.peg_in_hole.problem = args.problem
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.object_state_xyz_noise_std = 0.0
    cfg.domain_randomization.object_state_rotation_noise_degrees = 0.0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0

    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    device = base.device
    action_dim = cfg.action_space
    zero_action = torch.zeros((1, action_dim), device=device, dtype=torch.float32)

    print(f"[render-in-sim] env constructed; adding top-down camera", flush=True)

    # Add a near-top-down camera in env_0's region with a slight iso tilt
    # (better aesthetics than pure straight-down). Camera sits above the
    # fixture, shifted along -y, tilted to look at the table top.
    env_origin = base.scene.env_origins[0].detach().cpu().numpy()
    # Camera offset relative to env_0 (parent prim is env_0).
    cam_pos_local = (0.0, args.cam_offset_y, args.cam_height)
    # Compute look-at orientation: camera at cam_pos_local looking toward
    # (0, 0, 0.53) (table top center). For a slight iso, we want pitch ~70°
    # down from horizontal, yaw 0. Convert to quaternion (wxyz).
    import math
    # vector from cam_pos_local to target (table top)
    target = np.array([0.0, 0.0, 0.53])
    eye = np.array(cam_pos_local)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    # camera looks along -Z in its local frame; we need rotation that sends
    # local -Z to world fwd. Use lookat helper or compute manually:
    z_world = -fwd                # camera's +Z (opengl convention: -Z is forward)
    up_guess = np.array([0.0, 1.0, 0.0])
    x_world = np.cross(up_guess, z_world); x_world /= np.linalg.norm(x_world)
    y_world = np.cross(z_world, x_world)
    R = np.stack([x_world, y_world, z_world], axis=1)   # 3x3
    # rotation matrix → quaternion (wxyz)
    tr = R.trace()
    qw = math.sqrt(max(0.0, 1 + tr)) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw) if qw > 1e-6 else 0.0
    qy = (R[0, 2] - R[2, 0]) / (4 * qw) if qw > 1e-6 else 0.0
    qz = (R[1, 0] - R[0, 1]) / (4 * qw) if qw > 1e-6 else 0.0
    cam_rot_wxyz = (float(qw), float(qx), float(qy), float(qz))

    cam_cfg = CameraCfg(
        prim_path="/World/envs/env_0/TopDownCam",
        update_period=0.0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=cam_pos_local,
            rot=cam_rot_wxyz,
            convention="opengl",
        ),
    )
    cam = Camera(cam_cfg)

    # Run sim steps to initialize camera buffers.
    base.sim.reset()
    for _ in range(2):
        base.sim.step()
    cam.update(dt=0.0)

    OUT = Path(__file__).resolve().parent / "inputs"
    OUT.mkdir(parents=True, exist_ok=True)

    captures = []
    for i in range(args.num_resets):
        env.reset(seed=args.seed + i)
        for _ in range(args.settle_steps):
            env.step(zero_action)
        # Trigger camera render
        cam.update(dt=base.step_dt)
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()  # (H, W, 3 or 4)
        captures.append(rgb.astype(np.uint8))
        print(f"[render-in-sim] capture {i + 1}/{args.num_resets}: shape={rgb.shape}", flush=True)

    # Composite: keep capture[0] as the clean background (robot, table,
    # fixture all stay full-intensity), and OVERLAY pixels from each
    # subsequent capture only where they DIFFER from the background
    # (i.e., where the object/goal moved). Use MIN inside the diff mask so
    # all positions of the dark object accumulate.
    bg = captures[0][..., :3].astype(np.float32)
    composite = bg.copy()
    for c in captures[1:]:
        c_f = c[..., :3].astype(np.float32)
        diff = np.linalg.norm(c_f - bg, axis=-1)            # (H, W)
        mask = (diff > 15)[..., None]                       # (H, W, 1)
        composite = np.where(mask, np.minimum(composite, c_f), composite)
    composite = composite.clip(0, 255).astype(np.uint8)
    out_path = OUT / f"init_dist_{args.slug}_sim.png"
    Image.fromarray(composite).save(out_path)
    print(f"wrote {out_path}", flush=True)

    env.close()
    app.close()


if __name__ == "__main__":
    main()
