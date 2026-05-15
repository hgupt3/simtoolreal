"""Scope-A smoke test: render the same minimal static scene with
a TiledCamera (RTX rasterizer) and a RayCasterCamera (CUDA mesh raycast)
at the same pose and same intrinsics, and compare the depth outputs.

The scene is intentionally trivial — a ground plane and a static cube —
so we can verify two things:
  1) Both sensors see the same geometry (depth values match within the
     anti-aliasing tolerance of the rasterizer).
  2) The RayCaster path honors `horizontal_aperture_offset` /
     `vertical_aperture_offset` while the TiledCamera path silently
     ignores them (Omniverse limitation surfaced via the warning at
     isaaclab/sim/spawners/sensors/sensors.py:110).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "videos" / "smoke_test_raycaster_vs_tiled"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--focal-length", type=float, default=24.0)
    parser.add_argument("--horizontal-aperture", type=float, default=33.19737869997174)
    parser.add_argument("--horizontal-aperture-offset", type=float, default=0.0035)
    parser.add_argument("--vertical-aperture-offset", type=float, default=0.4418)
    # Simple top-down view: camera at (0, 0, 1.5) looking straight down at
    # the origin. Both cameras' rays land squarely on the ground+cube.
    parser.add_argument("--cam-pos", type=float, nargs=3, default=[0.0, 0.0, 1.5])
    parser.add_argument("--cam-look-at", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--cube-pos", type=float, nargs=3, default=[0.15, 0.10, 0.05])
    parser.add_argument("--cube-size", type=float, default=0.20)
    parser.add_argument("--depth-min-m", type=float, default=0.0)
    parser.add_argument("--depth-max-m", type=float, default=3.0)
    parser.add_argument("--num-warmup-steps", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Appended to tiled/raycaster output filenames (e.g. '_offset0').")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args).app

    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
    from isaaclab.sensors.ray_caster import (
        MultiMeshRayCasterCamera,
        MultiMeshRayCasterCameraCfg,
        patterns,
    )
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.terrains.utils import create_prim_from_mesh

    # --- Simulation context (blank stage) ---
    sim_utils.create_new_stage()
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim_utils.get_current_stage()

    # --- Ground plane large enough that the angled camera rays still land
    # on it (the existing camera pose points ~horizontally toward +Y, so
    # a small 2x2 plane is mostly missed). ---
    plane_mesh = make_plane(size=(20.0, 20.0), height=0.0, center_zero=True)
    create_prim_from_mesh("/World/ground", plane_mesh)

    # --- Static cube as "peg": place near where the camera actually looks
    # (forward axis ~ (0.29, 0.96, -0.03) from cam pos (-0.5, -0.64, 1.02)),
    # so projection from cam to z=0 lands near (9, 31). Drop the cube at a
    # sane workspace point and put it on the ground. ---
    s = float(args.cube_size)
    cube_mesh = trimesh.creation.box(extents=(s, s, s))
    cube_mesh.apply_translation(args.cube_pos)
    create_prim_from_mesh("/World/peg", cube_mesh)

    # Let USD finish processing the new prims before sensor init touches them.
    sim_utils.update_stage()

    # --- Camera prim parents ---
    # TiledCamera spawns its own UsdGeomCamera at this path — do NOT pre-create.
    # RayCasterCamera attaches to an existing prim; pre-create as Xform. We
    # bake the camera pose later via set_world_poses() after sim.play().
    sim_utils.create_prim("/World/RayCasterCamera", "Xform")

    # --- TiledCamera ---
    tiled_cfg = TiledCameraCfg(
        prim_path="/World/TiledCamera",
        update_period=0,
        update_latest_camera_pose=True,
        height=int(args.height),
        width=int(args.width),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args.focal_length),
            horizontal_aperture=float(args.horizontal_aperture),
            horizontal_aperture_offset=float(args.horizontal_aperture_offset),
            vertical_aperture_offset=float(args.vertical_aperture_offset),
            clipping_range=(0.1, 5.0),
        ),
        # Pose is set later via set_world_poses_from_view; use identity here.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )
    tiled = TiledCamera(cfg=tiled_cfg)

    # --- MultiMeshRayCasterCamera (same pose + intrinsics, two static targets) ---
    ray_pattern = patterns.PinholeCameraPatternCfg(
        focal_length=float(args.focal_length),
        horizontal_aperture=float(args.horizontal_aperture),
        horizontal_aperture_offset=float(args.horizontal_aperture_offset),
        vertical_aperture_offset=float(args.vertical_aperture_offset),
        width=int(args.width),
        height=int(args.height),
    )
    multi_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/RayCasterCamera",
        update_period=0,
        # Pose is baked into the parent Xform above; use identity offset.
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
        # Pass plain strings (internally converted with track_mesh_transforms=False
        # per the docstring) — matches the working pattern in
        # .venv_isaacsim/.../test/sensors/test_multi_mesh_ray_caster_camera.py.
        mesh_prim_paths=["/World/ground", "/World/peg"],
        pattern_cfg=ray_pattern,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",  # clip non-hit to max_distance instead of NaN
        max_distance=10.0,
    )
    ray_cam = MultiMeshRayCasterCamera(cfg=multi_cfg)

    # --- Bring up sim + sensors ---
    sim.reset()
    sim.play()

    # Bake the camera pose AFTER sim.play() — set_world_poses is the path
    # the working multi-mesh raycaster test uses (vs the OffsetCfg path,
    # which doesn't always propagate in headless mode).
    # Warp kernels require CUDA tensors; place them on the sensor's device.
    # Use set_world_poses_from_view — both cameras accept it, both compute
    # the orientation from eyes→target with the stage's up axis. In Isaac
    # Lab 5.1+ with Fabric, the pose write only reaches the rasterizer when
    # `_sync_usd_on_fabric_write=True` (same flag the env's
    # `scene_utils._apply_camera_pose_rand_at_reset` sets at runtime).
    dev = tiled.device
    eyes = torch.tensor([args.cam_pos], dtype=torch.float32, device=dev)
    targets = torch.tensor([args.cam_look_at], dtype=torch.float32, device=dev)
    for cam in (tiled, ray_cam):
        view = getattr(cam, "_view", None)
        if view is not None and hasattr(view, "_sync_usd_on_fabric_write"):
            view._sync_usd_on_fabric_write = True
    tiled.set_world_poses_from_view(eyes=eyes, targets=targets)
    ray_cam.set_world_poses_from_view(eyes=eyes, targets=targets)

    # Force a `.data` access to trigger `_update_buffers_impl`, which is what
    # actually populates `_ray_starts_w` / `_ray_directions_w` (the bare
    # buffers stay at their zero-init state until then).
    _ = ray_cam.data.output["distance_to_image_plane"]
    _ = tiled.data.output["distance_to_image_plane"]

    try:
        ray_starts = ray_cam._ray_starts_w[0, 0].cpu().tolist()
        ray_dirs = ray_cam._ray_directions_w[0, 0].cpu().tolist()
        mid_idx = (int(args.height) // 2) * int(args.width) + int(args.width) // 2
        mid_dir = ray_cam._ray_directions_w[0, mid_idx].cpu().tolist()
        print(
            f"[smoke-cmp] raycaster ray[0]: start={ray_starts} dir={ray_dirs}  "
            f"mid_dir={mid_dir}",
            flush=True,
        )
    except (AttributeError, IndexError) as exc:
        print(f"[smoke-cmp] ray buffer access failed: {exc}", flush=True)
    print(
        f"[smoke-cmp] raycaster quat_w_ros={ray_cam.data.quat_w_ros[0].cpu().tolist()} "
        f"tiled quat_w_ros={tiled.data.quat_w_ros[0].cpu().tolist()}",
        flush=True,
    )
    try:
        n_meshes = (
            int(ray_cam._mesh_ids_wp[0].shape[0]) if hasattr(ray_cam, "_mesh_ids_wp") else -1
        )
        print(
            f"[smoke-cmp] raycaster meshes registered per env: {n_meshes}",
            flush=True,
        )
    except Exception as exc:
        print(f"[smoke-cmp] mesh count read failed: {exc}", flush=True)

    for _ in range(int(args.num_warmup_steps)):
        sim.step()
        tiled.update(sim_cfg.dt)
        ray_cam.update(sim_cfg.dt)

    print(
        f"[smoke-cmp] tiled pos_w={tiled.data.pos_w[0].cpu().tolist()} "
        f"raycaster pos_w={ray_cam.data.pos_w[0].cpu().tolist()}",
        flush=True,
    )

    # --- Read depth from both ---
    tiled_depth = tiled.data.output["distance_to_image_plane"][0].detach().float().cpu().numpy()
    # TiledCamera depth shape: (H, W, 1)
    if tiled_depth.ndim == 3 and tiled_depth.shape[-1] == 1:
        tiled_depth = tiled_depth[..., 0]
    ray_depth = ray_cam.data.output["distance_to_image_plane"][0].detach().float().cpu().numpy()
    if ray_depth.ndim == 3 and ray_depth.shape[-1] == 1:
        ray_depth = ray_depth[..., 0]

    # --- Save raw + normalized PNGs (use the same depth window for fair visual compare) ---
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(tag: str, depth_m: np.ndarray) -> None:
        np.save(out_dir / f"{tag}.npy", depth_m)
        finite = np.isfinite(depth_m)
        d = depth_m.copy()
        d[~finite] = args.depth_max_m  # render nan/inf as far
        norm = np.clip(
            (d - args.depth_min_m) / max(args.depth_max_m - args.depth_min_m, 1e-6),
            0.0,
            1.0,
        )
        Image.fromarray((norm * 255.0).astype(np.uint8)).save(out_dir / f"{tag}.png")
        print(
            f"[smoke-cmp] {tag}: shape={depth_m.shape} "
            f"finite={float(finite.mean()):.3f} "
            f"min={float(np.nanmin(depth_m)):.4f}m max={float(np.nanmax(depth_m)):.4f}m "
            f"mean={float(np.nanmean(depth_m)):.4f}m",
            flush=True,
        )

    _save(f"tiled{args.output_suffix}", tiled_depth)
    _save(f"raycaster{args.output_suffix}", ray_depth)

    # --- Numerical diff (only where both are finite) ---
    both_finite = np.isfinite(tiled_depth) & np.isfinite(ray_depth)
    if both_finite.any():
        diff = (tiled_depth - ray_depth)[both_finite]
        print(
            f"[smoke-cmp] diff (tiled - raycaster): "
            f"|mean|={float(np.abs(diff).mean()):.4f}m "
            f"|max|={float(np.abs(diff).max()):.4f}m "
            f"rmse={float(np.sqrt((diff ** 2).mean())):.4f}m "
            f"npix={int(both_finite.sum())}",
            flush=True,
        )
        # Save a diff PNG centered at 0
        diff_full = np.where(both_finite, tiled_depth - ray_depth, 0.0)
        max_abs = max(float(np.abs(diff_full).max()), 1e-6)
        diff_norm = np.clip(diff_full / max_abs * 0.5 + 0.5, 0.0, 1.0)
        Image.fromarray((diff_norm * 255.0).astype(np.uint8)).save(out_dir / "diff_tiled_minus_raycaster.png")
        print(
            f"[smoke-cmp] diff PNG: signed (gray=0, white=+{max_abs:.4f}m, black=-{max_abs:.4f}m)",
            flush=True,
        )
    else:
        print("[smoke-cmp] no pixels finite in both — comparison skipped.", flush=True)

    sim._timeline.stop()
    sim.clear_all_callbacks()
    sim.clear_instance()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
