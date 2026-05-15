"""Step-5 smoke test: build PegInHoleDepthStudent env with backend=`tiled`
vs backend=`raycaster`, capture env 0's `get_student_obs()['image']` after
a deterministic reset, and save the raw .npy + a normalized PNG.

Run it twice (once per backend), then a numerical diff confirms the two
backends agree on geometry for the real env's scene.

Usage:
    .venv_isaacsim/bin/python isaacsimenvs/tests/smoke_test_pih_raycaster_backend.py \
        --backend tiled
    .venv_isaacsim/bin/python isaacsimenvs/tests/smoke_test_pih_raycaster_backend.py \
        --backend raycaster

Aperture offsets are forced to 0 so the two camera models see the same rays
(TiledCamera silently ignores offsets; RayCaster honors them — leaving the
ZED defaults shifts the RayCaster image vs the rasterizer).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "videos" / "smoke_test_pih_raycaster_backend"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tiled", "raycaster"), required=True)
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    parser.add_argument("--problem", default="Lpeg.tol0p5mm")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    spec = gym.spec(args.task)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})

    # --- Lock the scene ---
    cfg.scene.num_envs = int(args.num_envs)
    cfg.peg_in_hole.problem = args.problem
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"
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

    # --- Force aperture offsets to 0 so the two backends share the same rays.
    cfg.student_obs.horizontal_aperture_offset = 0.0
    cfg.student_obs.vertical_aperture_offset = 0.0
    cfg.student_obs.camera_backend = args.backend
    # Raycaster defaults in StudentObsCfg already point at the right
    # /visuals subgroups (Table/Hole/Object plus Robot/.*/visuals), so we
    # don't need to override here. Set explicitly if you want to ablate.
    cfg.student_obs.crop_enabled = False
    cfg.student_obs.image_input_width = int(cfg.student_obs.image_width)
    cfg.student_obs.image_input_height = int(cfg.student_obs.image_height)
    # depth_preprocess_mode keeps the same window_normalize → both backends
    # produce the same [0, 1] depth image as `get_student_obs()["image"]`.

    print(
        f"[smoke-pih] backend={args.backend} problem={args.problem} "
        f"n_envs={args.num_envs} crop=off offsets=0 "
        f"cam_size={cfg.student_obs.image_width}x{cfg.student_obs.image_height}",
        flush=True,
    )

    env = gym.make(args.task, cfg=cfg)
    inner = env.unwrapped
    actions = torch.zeros(
        (inner.num_envs, cfg.action_space), device=inner.device, dtype=torch.float32
    )
    env.reset()
    for _ in range(int(args.num_warmup_steps)):
        env.step(actions)

    obs = inner.get_student_obs()
    # image shape: (B, 1, H, W) after window_normalize. Take env 0, channel 0.
    image = obs["image"][0, 0].detach().float().cpu().numpy()
    print(
        f"[smoke-pih] image: shape={image.shape} "
        f"min={image.min():.4f} max={image.max():.4f} mean={image.mean():.4f}",
        flush=True,
    )

    # --- Raw (pre-window-normalize) depth from the underlying sensor ---
    cam = inner.student_camera
    raw_key = "distance_to_image_plane"
    raw = cam.data.output[raw_key][0].detach().float().cpu().numpy()
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    finite = np.isfinite(raw)
    print(
        f"[smoke-pih] raw distance_to_image_plane (m): finite_frac={finite.mean():.3f} "
        f"min={(np.nanmin(raw) if finite.any() else float('nan')):.3f} "
        f"max={(np.nanmax(raw) if finite.any() else float('nan')):.3f} "
        f"mean={(np.nanmean(raw) if finite.any() else float('nan')):.3f}",
        flush=True,
    )

    # --- Raycaster mesh diagnostics (only present on MultiMeshRayCaster) ---
    if hasattr(cam, "_mesh_views"):
        try:
            print(f"[smoke-pih] raycaster mesh views: {len(cam._mesh_views)}", flush=True)
            for idx, view in enumerate(cam._mesh_views):
                prim_path = getattr(view, "_regex_prim_paths", None) or getattr(view, "prim_paths", None) or "?"
                print(f"[smoke-pih]   view[{idx}]: prim_paths={prim_path}", flush=True)
        except Exception as exc:
            print(f"[smoke-pih] mesh-view dump failed: {exc}", flush=True)
    if hasattr(cam, "_num_meshes_per_env"):
        try:
            print(f"[smoke-pih] _num_meshes_per_env: {cam._num_meshes_per_env}", flush=True)
        except Exception as exc:
            print(f"[smoke-pih] num_meshes dump failed: {exc}", flush=True)

    # Dump the xformOp chain for each cube and the resolve_prim_pose result
    # used by the raycaster's parser.
    try:
        from pxr import UsdGeom, Usd
        from isaaclab.sim import utils as _sim_utils
        for target_path in [
            "/World/envs/env_0/Table",
            "/World/envs/env_0/Object",
        ]:
            target_prim = stage.GetPrimAtPath(target_path)
            print(f"\n[smoke-pih] xform chain under {target_path}:", flush=True)
            for prim in Usd.PrimRange(target_prim):
                if not prim.IsA(UsdGeom.Gprim) and prim != target_prim:
                    # Check if it has xformOps (intermediate Xform)
                    if prim.HasAPI(UsdGeom.XformCommonAPI) or UsdGeom.Xformable(prim).GetOrderedXformOps():
                        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                        if ops:
                            op_strs = [f"{op.GetOpName()}={op.Get()}" for op in ops]
                            print(f"[smoke-pih]   {prim.GetPath()}: {op_strs}", flush=True)
                    continue
                path = prim.GetPath().pathString
                if "/visuals/" not in path:
                    continue
                # cube prim: dump its xformOps + resolved-relative-to-target pose
                ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                op_strs = [f"{op.GetOpName()}={op.Get()}" for op in ops]
                try:
                    rel_pos, rel_quat = _sim_utils.resolve_prim_pose(prim, target_prim)
                    print(
                        f"[smoke-pih]   CUBE {path}: ops={op_strs}\n"
                        f"[smoke-pih]     resolve_prim_pose→target: pos={rel_pos} quat={rel_quat}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[smoke-pih]   CUBE {path}: resolve_prim_pose error: {exc}", flush=True)
    except Exception as exc:
        print(f"[smoke-pih] xform-chain dump failed: {exc}", flush=True)

    # Inspect the actual warp meshes in the BVH cache — what world-space
    # bounding box does the raycaster *think* each target occupies?
    try:
        from isaaclab.sensors.ray_caster import MultiMeshRayCaster
        import warp as wp
        for key, wp_mesh in MultiMeshRayCaster.meshes.items():
            verts = wp_mesh.points.numpy()
            tris = wp_mesh.indices.numpy()
            if verts.size == 0:
                print(f"[smoke-pih] wp_mesh {key}: EMPTY (0 vertices)", flush=True)
                continue
            n_tri = tris.size // 3
            bb_min = verts.min(axis=0)
            bb_max = verts.max(axis=0)
            print(
                f"[smoke-pih] wp_mesh {key}: verts={len(verts)} tris={n_tri} "
                f"bb=[{bb_min[0]:.3f},{bb_min[1]:.3f},{bb_min[2]:.3f}]→"
                f"[{bb_max[0]:.3f},{bb_max[1]:.3f},{bb_max[2]:.3f}]",
                flush=True,
            )
    except Exception as exc:
        print(f"[smoke-pih] wp mesh dump failed: {exc}", flush=True)

    # World poses of camera + each raycast target — useful to sanity-check
    # that the table/hole/peg are actually within the camera's FOV.
    try:
        cam_pos_w = cam.data.pos_w[0].detach().cpu().tolist()
        print(f"[smoke-pih] camera pos_w: {cam_pos_w}", flush=True)
    except Exception as exc:
        print(f"[smoke-pih] camera pos read failed: {exc}", flush=True)
    try:
        from isaaclab.sim.utils import find_matching_prims
        from pxr import UsdGeom, Usd
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for target_path in [
            "/World/envs/env_0/Table",
            "/World/envs/env_0/Hole",
            "/World/envs/env_0/Object",
        ]:
            prim = next(iter(find_matching_prims(target_path)), None)
            if prim is None:
                print(f"[smoke-pih] target {target_path}: NOT FOUND")
                continue
            t = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            print(f"[smoke-pih] target {target_path} world pos: ({float(t[0]):.3f}, {float(t[1]):.3f}, {float(t[2]):.3f})", flush=True)
    except Exception as exc:
        print(f"[smoke-pih] target pose dump failed: {exc}", flush=True)

    # Walk the USD tree under each target to see what mesh prims actually
    # exist and how many triangles they have.
    try:
        from pxr import UsdGeom
        stage = inner.scene.stage if hasattr(inner.scene, "stage") else None
        if stage is None:
            from isaacsim.core.utils.stage import get_current_stage
            stage = get_current_stage()
        for root_path in [
            "/World/ground",
            "/World/envs/env_0/Table",
            "/World/envs/env_0/Hole",
            "/World/envs/env_0/Object",
        ]:
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                print(f"[smoke-pih] usd_tree {root_path}: NOT VALID", flush=True)
                continue
            gprim_summaries: list[str] = []
            for prim in stage.Traverse():
                p = prim.GetPath().pathString
                if not p.startswith(root_path):
                    continue
                if prim.IsA(UsdGeom.Gprim):
                    type_name = prim.GetTypeName()
                    extra = ""
                    if prim.IsA(UsdGeom.Mesh):
                        mesh = UsdGeom.Mesh(prim)
                        n_faces = len(mesh.GetFaceVertexCountsAttr().Get() or [])
                        n_pts = len(mesh.GetPointsAttr().Get() or [])
                        extra = f" (faces={n_faces}, pts={n_pts})"
                    gprim_summaries.append(f"{type_name}: {p}{extra}")
            print(
                f"[smoke-pih] usd_tree {root_path}: {len(gprim_summaries)} Gprim descendants",
                flush=True,
            )
            for s in gprim_summaries[:8]:
                print(f"[smoke-pih]   {s}", flush=True)
    except Exception as exc:
        print(f"[smoke-pih] usd tree walk failed: {exc}", flush=True)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.backend}.npy", image)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
        out_dir / f"{args.backend}.png"
    )

    # Save the raw distance_to_image_plane (pre-window-normalize) and a
    # PNG auto-stretched to its actual finite range so the geometry beyond
    # the env's depth window (>1.10 m) is still visible.
    raw_cam = inner.student_camera.data.output["distance_to_image_plane"][0].detach().float().cpu().numpy()
    if raw_cam.ndim == 3 and raw_cam.shape[-1] == 1:
        raw_cam = raw_cam[..., 0]
    np.save(out_dir / f"{args.backend}_raw.npy", raw_cam)
    finite_mask = np.isfinite(raw_cam)
    if finite_mask.any():
        valid = raw_cam[finite_mask]
        lo, hi = float(valid.min()), float(valid.max())
        # Fill non-finite with `hi` so they go to white instead of NaN.
        d = np.where(finite_mask, raw_cam, hi)
        norm = np.clip((d - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        Image.fromarray((norm * 255.0).astype(np.uint8)).save(
            out_dir / f"{args.backend}_raw_autoscale.png"
        )
        print(
            f"[smoke-pih] raw autoscale PNG: range [{lo:.3f} m, {hi:.3f} m]",
            flush=True,
        )
    print(f"[smoke-pih] wrote {out_dir / f'{args.backend}.npy'}", flush=True)
    print(f"[smoke-pih] wrote {out_dir / f'{args.backend}.png'}", flush=True)

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
