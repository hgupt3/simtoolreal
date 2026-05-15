"""Dynamic scene smoke test: a rigid cube falls under PhysX gravity onto a
ground plane, observed simultaneously by a TiledCamera (rasterizer) and a
MultiMeshRayCasterCamera (CUDA raycaster). Both cameras share the same
pose and intrinsics, the cube is spawned as a `MeshCuboidCfg` so its
geometry is triangulated (the raycaster requires UsdGeom.Mesh prims), and
the raycaster's cube target is configured with `track_mesh_transforms=True`
so its world pose is refreshed each step.

Modeled on `isaacsimenvs/tests/test_cube_falling_scene.py` which uses the
`RigidObject` wrapper to do the PhysX rigid-body view init correctly —
spawning the prim with rigid-body props alone is not enough.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "videos" / "smoke_test_raycaster_video"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--focal-length", type=float, default=24.0)
    parser.add_argument("--horizontal-aperture", type=float, default=33.19737869997174)
    parser.add_argument("--horizontal-aperture-offset", type=float, default=0.0)
    parser.add_argument("--vertical-aperture-offset", type=float, default=0.0)
    # Side view of a table; cube drops onto the table top.
    parser.add_argument("--cam-pos", type=float, nargs=3, default=[1.4, -1.0, 0.9])
    parser.add_argument("--cam-look-at", type=float, nargs=3, default=[0.0, 0.0, 0.55])
    parser.add_argument("--cube-init-z", type=float, default=1.5)
    parser.add_argument("--cube-size", type=float, default=0.20)
    parser.add_argument("--table-size", type=float, nargs=3, default=[0.8, 0.8, 0.5],
                        metavar=("X", "Y", "Z"),
                        help="Table footprint XY + height (top surface at z=table_size.z).")
    parser.add_argument("--depth-min-m", type=float, default=0.0)
    parser.add_argument("--depth-max-m", type=float, default=2.0)
    parser.add_argument("--num-steps", type=int, default=180)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args).app

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
    from isaaclab.sensors.ray_caster import (
        MultiMeshRayCasterCamera,
        MultiMeshRayCasterCameraCfg,
        patterns,
    )

    PHYSICS_DT = 1.0 / 60.0
    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT))

    # --- Scene ---
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Static table: a MeshCuboid with collision (no rigid body) so the cube
    # lands on it. Centered at the world origin, top at z=table_size.z.
    tx, ty, tz = (float(v) for v in args.table_size)
    table_spawn = sim_utils.MeshCuboidCfg(
        size=(tx, ty, tz),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
    )
    table_spawn.func("/World/Table", table_spawn, translation=(0.0, 0.0, tz / 2.0))

    # Cube as a triangulated UsdGeom.Mesh with full rigid-body / collision /
    # mass props, wrapped in RigidObject so the PhysX view is set up
    # correctly. Non-kinematic — PhysX gravity will drop it.
    s = float(args.cube_size)
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(s, s, s),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, float(args.cube_init_z))),
    )
    cube = RigidObject(cfg=cube_cfg)

    # --- Camera prim parent for the raycaster (TiledCamera spawns its own) ---
    sim_utils.create_prim("/World/RayCasterCamera", "Xform")

    # --- TiledCamera (depth + RGB) ---
    tiled_cfg = TiledCameraCfg(
        prim_path="/World/TiledCamera",
        update_period=0,
        update_latest_camera_pose=True,
        height=int(args.height),
        width=int(args.width),
        data_types=["distance_to_image_plane", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args.focal_length),
            horizontal_aperture=float(args.horizontal_aperture),
            horizontal_aperture_offset=float(args.horizontal_aperture_offset),
            vertical_aperture_offset=float(args.vertical_aperture_offset),
            clipping_range=(0.1, 50.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )
    tiled = TiledCamera(cfg=tiled_cfg)

    # --- RayCaster: ground static, cube dynamic ---
    ray_pattern = patterns.PinholeCameraPatternCfg(
        focal_length=float(args.focal_length),
        horizontal_aperture=float(args.horizontal_aperture),
        horizontal_aperture_offset=float(args.horizontal_aperture_offset),
        vertical_aperture_offset=float(args.vertical_aperture_offset),
        width=int(args.width),
        height=int(args.height),
    )
    GroundTarget = MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
        prim_expr="/World/GroundPlane", track_mesh_transforms=False
    )
    TableTarget = MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
        prim_expr="/World/Table", track_mesh_transforms=False
    )
    CubeTarget = MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
        prim_expr="/World/Cube", track_mesh_transforms=True
    )
    multi_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/RayCasterCamera",
        update_period=0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"
        ),
        mesh_prim_paths=[GroundTarget, TableTarget, CubeTarget],
        pattern_cfg=ray_pattern,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=20.0,
    )
    ray_cam = MultiMeshRayCasterCamera(cfg=multi_cfg)

    sim.reset()

    # Bake camera pose after reset (set_world_poses_from_view + Fabric-sync
    # flag so the rasterizer reads it).
    dev = tiled.device
    eyes = torch.tensor([args.cam_pos], dtype=torch.float32, device=dev)
    targets = torch.tensor([args.cam_look_at], dtype=torch.float32, device=dev)
    for cam in (tiled, ray_cam):
        view = getattr(cam, "_view", None)
        if view is not None and hasattr(view, "_sync_usd_on_fabric_write"):
            view._sync_usd_on_fabric_write = True
    tiled.set_world_poses_from_view(eyes=eyes, targets=targets)
    ray_cam.set_world_poses_from_view(eyes=eyes, targets=targets)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_depth(d: np.ndarray) -> np.ndarray:
        finite = np.isfinite(d)
        out = d.copy()
        out[~finite] = args.depth_max_m
        out = np.clip(
            (out - args.depth_min_m) / max(args.depth_max_m - args.depth_min_m, 1e-6),
            0.0,
            1.0,
        )
        return (out * 255.0).astype(np.uint8)

    def _gray_to_rgb(g: np.ndarray) -> np.ndarray:
        return np.repeat(g[..., None], 3, axis=-1)

    def _label_panel(img: np.ndarray, title: str) -> np.ndarray:
        """Pad the top of `img` with a black bar containing the title text."""
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont
        h, w = img.shape[:2]
        bar = 18  # px tall label strip
        panel = np.zeros((h + bar, w, 3), dtype=np.uint8)
        panel[bar:, :, :] = img
        pil = _Image.fromarray(panel)
        draw = _ImageDraw.Draw(pil)
        try:
            font = _ImageFont.truetype("DejaVuSans.ttf", 12)
        except Exception:
            font = _ImageFont.load_default()
        tw = draw.textlength(title, font=font)
        draw.text(((w - tw) / 2, 2), title, fill=(255, 255, 255), font=font)
        return np.array(pil)

    rgb_frames: list[np.ndarray] = []
    tiled_frames: list[np.ndarray] = []
    raycaster_frames: list[np.ndarray] = []
    composite_frames: list[np.ndarray] = []

    dt = sim.get_physics_dt()
    capture_every = max(1, round((1.0 / args.video_fps) / dt))
    print(
        f"[smoke-video] steps={args.num_steps} physics_dt={dt:.4f}s "
        f"capture_every={capture_every} (video_fps={args.video_fps})",
        flush=True,
    )

    for step_i in range(int(args.num_steps)):
        sim.step(render=True)
        cube.update(dt)
        tiled.update(dt)
        ray_cam.update(dt)

        if step_i % capture_every == 0:
            t_d = tiled.data.output["distance_to_image_plane"][0].detach().float().cpu().numpy()
            if t_d.ndim == 3 and t_d.shape[-1] == 1:
                t_d = t_d[..., 0]
            r_d = ray_cam.data.output["distance_to_image_plane"][0].detach().float().cpu().numpy()
            if r_d.ndim == 3 and r_d.shape[-1] == 1:
                r_d = r_d[..., 0]
            rgb = tiled.data.output["rgb"][0].detach().cpu().numpy()
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            rgb = rgb.astype(np.uint8)

            t_norm = _normalize_depth(t_d)
            r_norm = _normalize_depth(r_d)
            rgb_frames.append(rgb)
            tiled_frames.append(t_norm)
            raycaster_frames.append(r_norm)

            # 3-column composite with titles above each panel
            rgb_panel = _label_panel(rgb, "RGB")
            tiled_panel = _label_panel(_gray_to_rgb(t_norm), "Tiled Depth")
            ray_panel = _label_panel(_gray_to_rgb(r_norm), "RayCaster Depth")
            composite_frames.append(np.concatenate([rgb_panel, tiled_panel, ray_panel], axis=1))

            if step_i % 10 == 0:
                cube_z = float(cube.data.root_pos_w[0, 2].item())
                cy, cx = t_d.shape[0] // 2, t_d.shape[1] // 2
                print(
                    f"[smoke-video] step {step_i:03d}  cube_z={cube_z:.3f}  "
                    f"tiled[min={np.nanmin(t_d):.3f} center={t_d[cy, cx]:.3f}]  "
                    f"raycaster[min={np.nanmin(r_d):.3f} center={r_d[cy, cx]:.3f}]",
                    flush=True,
                )

    tiled_mp4 = out_dir / "tiled.mp4"
    raycaster_mp4 = out_dir / "raycaster.mp4"
    rgb_mp4 = out_dir / "rgb.mp4"
    side_mp4 = out_dir / "side_by_side.mp4"
    imageio.mimwrite(str(rgb_mp4), rgb_frames, fps=int(args.video_fps))
    imageio.mimwrite(str(tiled_mp4), tiled_frames, fps=int(args.video_fps))
    imageio.mimwrite(str(raycaster_mp4), raycaster_frames, fps=int(args.video_fps))
    imageio.mimwrite(str(side_mp4), composite_frames, fps=int(args.video_fps))
    print(f"[smoke-video] wrote {rgb_mp4}", flush=True)
    print(f"[smoke-video] wrote {tiled_mp4}", flush=True)
    print(f"[smoke-video] wrote {raycaster_mp4}", flush=True)
    print(f"[smoke-video] wrote {side_mp4} (RGB | Tiled | RayCaster)", flush=True)

    final_z = float(cube.data.root_pos_w[0, 2].item())
    print(f"[smoke-video] cube final z: {final_z:.4f} (expected ~{args.cube_size / 2:.2f})", flush=True)

    sim._timeline.stop()
    sim.clear_all_callbacks()
    sim.clear_instance()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
