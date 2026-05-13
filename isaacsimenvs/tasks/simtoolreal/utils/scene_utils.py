"""Scene construction, asset conversion, and runtime material setup."""

from __future__ import annotations

import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_angle_axis, quat_mul
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

from .generate_objects import generate_handle_head_urdfs


# ----------------------------------------------------------------------------
# Joint names / regexes / body names
# ----------------------------------------------------------------------------

ARM_JOINT_REGEX = "iiwa14_joint_.*"
HAND_JOINT_REGEX = "left_.*"

# Legacy policy order; Isaac Lab tensors are permuted at action/obs boundaries.
JOINT_NAMES_CANONICAL: tuple[str, ...] = (
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)
assert len(JOINT_NAMES_CANONICAL) == 29

PALM_BODY_NAME = "iiwa14_link_7"
# Merged fingertip bodies land on the DP links in both sims.
FINGERTIP_BODY_REGEX = "left_(index|middle|ring|thumb|pinky)_DP"
FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
)


# Per-joint PD gains and dynamics (verified with pretrained checkpoint).
ARM_JOINT_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}
ARM_JOINT_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}

HAND_JOINT_STIFFNESS: dict[str, float] = {
    "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
    "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
    "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9, "left_index_DIP": 0.9,
    "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
    "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
    "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
    "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
}
HAND_JOINT_DAMPING: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
    "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
    "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
    "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}
HAND_JOINT_ARMATURE: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
    "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
    "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
    "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
    "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
    "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
    "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
    "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
    "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
    "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
}
HAND_JOINT_FRICTION: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.132, "left_thumb_CMC_AA": 0.132,
    "left_thumb_MCP_FE": 0.07456, "left_thumb_MCP_AA": 0.07456, "left_thumb_IP": 0.01276,
    "left_2_index_MCP_FE": 0.07456, "left_index_MCP_AA": 0.07456,
    "left_index_PIP": 0.01276, "left_index_DIP": 0.00378738,
    "left_3_middle_MCP_FE": 0.07456, "left_middle_MCP_AA": 0.07456,
    "left_middle_PIP": 0.01276, "left_middle_DIP": 0.00378738,
    "left_4_ring_MCP_FE": 0.07456, "left_ring_MCP_AA": 0.07456,
    "left_ring_PIP": 0.01276, "left_ring_DIP": 0.00378738,
    "left_5_pinky_CMC": 0.012, "left_pinky_MCP_FE": 0.07456,
    "left_pinky_MCP_AA": 0.07456, "left_pinky_PIP": 0.01276, "left_pinky_DIP": 0.00378738,
}

assert len(ARM_JOINT_STIFFNESS) == 7 and len(ARM_JOINT_DAMPING) == 7
assert len(HAND_JOINT_STIFFNESS) == 22 and len(HAND_JOINT_DAMPING) == 22
assert len(HAND_JOINT_ARMATURE) == 22 and len(HAND_JOINT_FRICTION) == 22

# Proven-working default arm pose (isaacsim_conversion/isaacsim_env.py:101-109).
ARM_DEFAULT_JOINT_POS: dict[str, float] = {
    "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}

_CONTACT_OFFSET = 0.002
_REST_OFFSET = 0.0

# group: "rb" (RigidBodyAPI) or "art" (ArticulationRootAPI).
# attr_name: USD attribute path. vtype_str: matched against pxr.Sdf.ValueTypeNames.
_PHYSICS_SPECS: dict[str, tuple[str, str, str]] = {
    "kinematic_enabled": ("rb", "physics:kinematicEnabled", "Bool"),
    "disable_gravity": ("rb", "physxRigidBody:disableGravity", "Bool"),
    "max_depenetration_velocity": ("rb", "physxRigidBody:maxDepenetrationVelocity", "Float"),
    "rb_solver_position_iterations": ("rb", "physxRigidBody:solverPositionIterationCount", "Int"),
    "rb_solver_velocity_iterations": ("rb", "physxRigidBody:solverVelocityIterationCount", "Int"),
    "articulation_enabled": ("art", "physics:articulationEnabled", "Bool"),
    "enabled_self_collisions": ("art", "physxArticulation:enabledSelfCollisions", "Bool"),
    "solver_position_iterations": ("art", "physxArticulation:solverPositionIterationCount", "Int"),
    "solver_velocity_iterations": ("art", "physxArticulation:solverVelocityIterationCount", "Int"),
}


def build_robot_articulation_usd_cfg(usd_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.8, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                **ARM_DEFAULT_JOINT_POS,
                **{name: 0.0 for name in HAND_JOINT_STIFFNESS},
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[ARM_JOINT_REGEX],
                stiffness=ARM_JOINT_STIFFNESS,
                damping=ARM_JOINT_DAMPING,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[HAND_JOINT_REGEX],
                stiffness=HAND_JOINT_STIFFNESS,
                damping=HAND_JOINT_DAMPING,
                armature=HAND_JOINT_ARMATURE,
            ),
        },
    )


def build_rigid_object_cfg(prim_path: str, usd_paths: list[str]) -> RigidObjectCfg:
    """Spawn a RigidObject from one or more pre-baked USDs (round-robin)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=MultiUsdFileCfg(usd_path=list(usd_paths), random_choice=False),
    )


def _log_scene_step(start_time: float, message: str) -> None:
    print(f"[scene_utils][+{time.perf_counter() - start_time:.2f}s] {message}", flush=True)


def _student_camera_data_types(modality: str) -> list[str]:
    modality = str(modality).lower()
    if modality == "depth":
        return ["distance_to_image_plane"]
    if modality == "rgb":
        return ["rgb"]
    if modality == "rgbd":
        return ["rgb", "distance_to_image_plane"]
    raise ValueError(
        "cfg.student_obs.image_modality must be one of "
        f"('depth', 'rgb', 'rgbd'), got {modality!r}."
    )


def hide_goal_viz_for_student_camera(env) -> None:
    cfg = getattr(env.cfg, "student_obs", None)
    if cfg is None or not cfg.enabled or not cfg.hide_goal_viz:
        return

    from pxr import UsdGeom

    stage = get_current_stage()
    goal_viz_paths = find_matching_prim_paths("/World/envs/env_.*/GoalViz")
    for prim_path in goal_viz_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    _log_scene_step(
        time.perf_counter(),
        f"hid {len(goal_viz_paths)} GoalViz prims from render products",
    )


def setup_student_camera(env) -> None:
    """Create the optional per-env student camera sensor.

    The camera is registered with ``env.scene.sensors`` so DirectRLEnv owns its
    lifecycle. Student observations remain opt-in through
    ``env.unwrapped.get_student_obs()``; the normal teacher/critic observation
    path does not touch this sensor.
    """
    cfg = getattr(env.cfg, "student_obs", None)
    env.student_camera = None
    if cfg is None or not cfg.enabled or not cfg.image_enabled:
        return

    backend = str(cfg.camera_backend).lower()
    if backend not in ("tiled", "standard"):
        raise ValueError(
            "cfg.student_obs.camera_backend must be 'tiled' or 'standard', "
            f"got {backend!r}."
        )

    camera_mount = str(cfg.camera_mount).lower()
    if camera_mount != "world":
        raise NotImplementedError(
            "Only world-mounted student cameras are wired into DirectRLEnv right "
            f"now; got cfg.student_obs.camera_mount={camera_mount!r}."
        )

    t0 = time.perf_counter()
    from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg

    camera_cfg_cls = TiledCameraCfg if backend == "tiled" else CameraCfg
    camera_cls = TiledCamera if backend == "tiled" else Camera
    camera_cfg = camera_cfg_cls(
        prim_path="/World/envs/env_.*/StudentCamera",
        update_period=0,
        update_latest_camera_pose=True,
        height=int(cfg.image_height),
        width=int(cfg.image_width),
        data_types=_student_camera_data_types(cfg.image_modality),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(cfg.focal_length),
            focus_distance=float(cfg.focus_distance),
            horizontal_aperture=float(cfg.horizontal_aperture),
            clipping_range=tuple(float(x) for x in cfg.clipping_range),
        ),
        offset=camera_cfg_cls.OffsetCfg(
            pos=tuple(float(x) for x in cfg.camera_pos),
            rot=tuple(float(x) for x in cfg.camera_quat_wxyz),
            convention=str(cfg.camera_convention),
        ),
    )
    env.student_camera = camera_cls(cfg=camera_cfg)
    env.scene.sensors["student_camera"] = env.student_camera
    _log_scene_step(
        t0,
        f"registered student camera backend={backend} "
        f"modality={cfg.image_modality} "
        f"size={int(cfg.image_width)}x{int(cfg.image_height)}",
    )


def _apply_depth_noise(env, depth: torch.Tensor) -> torch.Tensor:
    """5-stage depth noise pipeline on raw-meters depth, shape (B, 1, H, W).

    Stages (all gated by their own σ/prob being > 0):
      1. additive Gaussian
      2. spatially-correlated Gaussian (k×k mean-blur of i.i.d. noise)
      3. per-pixel dropout to 0
      4. per-pixel random-uniform replacement in [randu_min, randu_max]
      5. stick artifacts (small random streaks)

    No-op when `cfg.use_depth_aug=False`. Defaults match the team's "medium"
    preset (see StudentObsCfg).
    """
    cfg = env.cfg.student_obs
    if not bool(getattr(cfg, "use_depth_aug", False)):
        return depth

    out = depth.float()
    device = out.device

    # 1. additive Gaussian
    gauss_std = float(cfg.depth_aug_gaussian_std_m)
    if gauss_std > 0.0:
        out = out + torch.randn_like(out) * gauss_std

    # 2. spatially-correlated Gaussian: i.i.d. noise blurred by mean k×k kernel.
    corr_std = float(cfg.depth_aug_correlated_std_m)
    k = int(cfg.depth_aug_correlated_kernel_size)
    if corr_std > 0.0 and k > 1:
        noise = torch.randn_like(out) * corr_std
        kernel = torch.ones(1, 1, k, k, device=device, dtype=out.dtype) / (k * k)
        out = out + F.conv2d(noise, kernel, padding=k // 2)

    # 3. per-pixel dropout to 0
    p_drop = float(cfg.depth_aug_dropout_prob)
    if p_drop > 0.0:
        keep = (torch.rand_like(out) >= p_drop).to(out.dtype)
        out = out * keep

    # 4. per-pixel random-uniform replacement
    p_randu = float(cfg.depth_aug_randu_prob)
    if p_randu > 0.0:
        lo = float(cfg.depth_aug_randu_min_m)
        hi = float(cfg.depth_aug_randu_max_m)
        mask = torch.rand_like(out) < p_randu
        randu = torch.rand_like(out) * (hi - lo) + lo
        out = torch.where(mask, randu, out)

    # 5. stick artifacts (Poisson-count per image, vectorized line rasterization).
    stick_prob = float(cfg.depth_aug_stick_prob)
    max_sticks = int(cfg.depth_aug_max_sticks_per_image)
    if stick_prob > 0.0 and max_sticks > 0:
        out = _draw_depth_sticks(out, cfg=cfg, stick_prob=stick_prob, max_sticks=max_sticks)

    return out


def _draw_depth_sticks(
    depth_b1hw: torch.Tensor,
    *,
    cfg,
    stick_prob: float,
    max_sticks: int,
) -> torch.Tensor:
    """Add random line streaks to a (B, 1, H, W) depth buffer.

    Fully vectorized: zero Python loops, zero per-step ``.item()``/``.tolist()``
    syncs. Per env we sample up to ``max_sticks`` candidate sticks, rasterize
    each as a length-``max_len_px`` line, expand by the stick's width, mask
    in-bounds + active steps, and scatter fill values in a single index
    write. Profile shows ~250 ms/step at B=512 with the previous Python loop;
    this implementation runs in <1 ms.

    Per-stick parameter distributions match the previous Python loop:
    length ~ U[1, max_len], width ~ U[1, max_w], angle ~ U[0, 2π),
    fill ~ U[randu_min, randu_max], origin ~ U(pixel grid). The per-image
    stick count is sampled per-candidate as Bernoulli(p) where
    ``p = expected_per_image / max_sticks``, giving the same expected count
    as the original ``Poisson(expected).clamp(max=max_sticks)`` (variance
    differs by O(1), which is irrelevant for this DR noise channel).
    """
    B, _, H, W = depth_b1hw.shape
    device = depth_b1hw.device
    dtype = depth_b1hw.dtype

    if max_sticks <= 0 or stick_prob <= 0.0:
        return depth_b1hw

    max_len = max(1, int(cfg.depth_aug_stick_max_len_px))
    max_w = max(1, int(cfg.depth_aug_stick_max_width_px))
    w_half_max = max_w // 2
    K = 2 * w_half_max + 1  # kernel side, matches the old `out[..., y±w_half, x±w_half]` slice
    lo = float(cfg.depth_aug_randu_min_m)
    hi = float(cfg.depth_aug_randu_max_m)

    # Per-candidate gate: expected_per_image = max_sticks * p_candidate.
    expected_per_image = stick_prob * float(H * W)
    p_candidate = min(1.0, expected_per_image / float(max_sticks))
    active = torch.rand(B, max_sticks, device=device) < p_candidate          # (B, S)

    # Per-stick parameters. All shapes (B, S).
    x0 = torch.randint(0, W, (B, max_sticks), device=device)
    y0 = torch.randint(0, H, (B, max_sticks), device=device)
    theta = torch.rand(B, max_sticks, device=device) * (2.0 * torch.pi)
    lengths = torch.randint(1, max_len + 1, (B, max_sticks), device=device)
    widths = torch.randint(1, max_w + 1, (B, max_sticks), device=device)
    fills = torch.rand(B, max_sticks, device=device) * (hi - lo) + lo
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Rasterize the line center across max_len pixels.
    s = torch.arange(max_len, device=device, dtype=torch.float32)            # (L,)
    xs = x0.float()[:, :, None] + cos_t[:, :, None] * s[None, None, :]       # (B, S, L)
    ys = y0.float()[:, :, None] + sin_t[:, :, None] * s[None, None, :]       # (B, S, L)
    s_active = s[None, None, :] < lengths.float()[:, :, None]                # (B, S, L)
    line_active = active[:, :, None] & s_active                              # (B, S, L)

    # Expand by stick width.
    offs = torch.arange(-w_half_max, w_half_max + 1, device=device)          # (K,)
    dy_g, dx_g = torch.meshgrid(offs, offs, indexing="ij")                   # (K, K) each
    w_half_b = (widths // 2)[:, :, None, None, None]                         # (B, S, 1, 1, 1)
    within_w = (dx_g[None, None, None] .abs() <= w_half_b) & \
               (dy_g[None, None, None].abs() <= w_half_b)                    # (B, S, 1, K, K)

    xi = xs[:, :, :, None, None].long() + dx_g[None, None, None]             # (B, S, L, K, K)
    yi = ys[:, :, :, None, None].long() + dy_g[None, None, None]
    in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    final_mask = line_active[:, :, :, None, None] & within_w & in_bounds     # (B, S, L, K, K)

    if not bool(final_mask.any()):
        return depth_b1hw

    # Flatten and scatter. One kernel launch.
    b_idx = torch.arange(B, device=device)[:, None, None, None, None].expand_as(xi)
    fill_full = fills[:, :, None, None, None].expand_as(xi).to(dtype)
    flat_mask = final_mask.reshape(-1)
    flat_b = b_idx.reshape(-1)[flat_mask]
    flat_y = yi.reshape(-1)[flat_mask]
    flat_x = xi.reshape(-1)[flat_mask]
    flat_fill = fill_full.reshape(-1)[flat_mask]

    out = depth_b1hw.clone()
    out[flat_b, 0, flat_y, flat_x] = flat_fill
    return out


def _apply_camera_pose_rand_at_reset(env, env_ids: torch.Tensor) -> None:
    """Sample per-env camera-pose noise and apply via student_camera.set_world_poses.

    Per-env at reset cadence: a fresh random offset is drawn each time
    `env_ids` reset and the camera is moved on the spot. No persistent
    per-env state; noise is regenerated every reset. No-op when
    `cfg.use_camera_pose_rand=False`.
    """
    cfg = getattr(env.cfg, "student_obs", None)
    camera = getattr(env, "student_camera", None)
    if cfg is None or camera is None:
        return
    if not bool(getattr(cfg, "use_camera_pose_rand", False)):
        return

    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = int(env_ids.numel())
    if n == 0:
        return

    device = env.device
    pos_range = torch.as_tensor(cfg.camera_pos_noise_m, device=device, dtype=torch.float32)
    rot_range_deg = torch.as_tensor(cfg.camera_rot_noise_deg, device=device, dtype=torch.float32)
    base_pos = torch.as_tensor(cfg.camera_pos, device=device, dtype=torch.float32)
    base_quat = torch.as_tensor(cfg.camera_quat_wxyz, device=device, dtype=torch.float32)

    pos_noise = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * pos_range
    rot_noise_rad = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * rot_range_deg * (torch.pi / 180.0)

    # RPY → wxyz quat: q = q_yaw * q_pitch * q_roll
    axes = torch.eye(3, device=device, dtype=torch.float32)
    q_roll = quat_from_angle_axis(rot_noise_rad[:, 0], axes[0].expand(n, -1))
    q_pitch = quat_from_angle_axis(rot_noise_rad[:, 1], axes[1].expand(n, -1))
    q_yaw = quat_from_angle_axis(rot_noise_rad[:, 2], axes[2].expand(n, -1))
    rot_noise_quat = quat_mul(q_yaw, quat_mul(q_pitch, q_roll))

    pos_w = env.scene.env_origins[env_ids] + base_pos + pos_noise
    quat = quat_mul(rot_noise_quat, base_quat.expand(n, -1))

    # Isaac Lab 5.1+: when Fabric is enabled, write through to USD so the
    # renderer actually picks up the new camera pose.
    view = getattr(camera, "_view", None)
    if view is not None and hasattr(view, "_sync_usd_on_fabric_write"):
        view._sync_usd_on_fabric_write = True

    camera.set_world_poses(
        positions=pos_w,
        orientations=quat,
        env_ids=env_ids,
        convention=str(cfg.camera_convention),
    )


def _preprocess_student_depth(env, depth: torch.Tensor) -> torch.Tensor:
    cfg = env.cfg.student_obs
    depth = depth.float()
    near = float(cfg.depth_min_m)
    far = float(cfg.depth_max_m)
    if far <= near:
        raise ValueError(
            "cfg.student_obs.depth_max_m must be greater than "
            "cfg.student_obs.depth_min_m."
        )

    mode = str(cfg.depth_preprocess_mode).lower()
    valid = torch.isfinite(depth) & (depth >= near) & (depth <= far)
    if mode == "clip_divide":
        clipped = torch.clamp(
            torch.nan_to_num(depth, nan=far, posinf=far, neginf=near),
            near,
            far,
        )
        return clipped / far
    if mode == "metric":
        return torch.where(valid, depth, torch.zeros_like(depth))
    if mode == "window_normalize":
        safe_depth = torch.nan_to_num(depth, nan=far, posinf=far, neginf=near)
        normalized = (safe_depth - near) / (far - near)
        return torch.clamp(normalized, 0.0, 1.0)
    raise ValueError(
        "cfg.student_obs.depth_preprocess_mode must be one of "
        f"('clip_divide', 'window_normalize', 'metric'), got {mode!r}."
    )


def _validate_student_image_shape(env, image: torch.Tensor) -> torch.Tensor:
    cfg = env.cfg.student_obs
    size = (int(cfg.image_input_height), int(cfg.image_input_width))
    if image.shape[-2:] == size:
        return image
    raise RuntimeError(
        "Student image shape does not match configured input shape. "
        "Adjust crop/image_input settings; this path does not resize images. "
        f"got HxW={tuple(image.shape[-2:])}, expected HxW={size}."
    )


def _crop_student_image(env, image: torch.Tensor) -> torch.Tensor:
    cfg = env.cfg.student_obs
    if not cfg.crop_enabled:
        return image

    height, width = image.shape[-2:]
    x0, y0 = (int(v) for v in cfg.crop_top_left)
    x1, y1 = (int(v) for v in cfg.crop_bottom_right)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            "Invalid student image crop coordinates: "
            f"top_left=({x0}, {y0}), bottom_right=({x1}, {y1}), "
            f"image={width}x{height}. Coordinates use x/y pixels with "
            "bottom_right exclusive."
        )
    return image[..., y0:y1, x0:x1]


def read_student_camera_image(env) -> torch.Tensor:
    """Return the configured student image as ``(num_envs, channels, H, W)``."""
    cfg = getattr(env.cfg, "student_obs", None)
    if cfg is None or not cfg.enabled:
        raise RuntimeError(
            "cfg.student_obs.enabled is false; no student image is available."
        )
    if not cfg.image_enabled:
        raise RuntimeError(
            "cfg.student_obs.image_enabled is false; no student image is available."
        )

    camera = getattr(env, "student_camera", None)
    if camera is None:
        raise RuntimeError(
            "Student camera was not created. Check cfg.student_obs.enabled and "
            "launch with cameras enabled."
        )

    env.sim.render()
    dt = float(getattr(env, "physics_dt", env.cfg.sim.dt))
    camera.update(dt, force_recompute=True)

    outputs = camera.data.output
    available = {key: value for key, value in outputs.items() if value is not None}
    if not available:
        raise RuntimeError("Student camera produced no outputs.")

    modality = str(cfg.image_modality).lower()
    image_parts: list[torch.Tensor] = []
    if modality in ("rgb", "rgbd"):
        rgb = available.get("rgb")
        if rgb is None:
            raise RuntimeError(
                f"RGB output missing. Available student camera outputs: {list(available.keys())}"
            )
        rgb = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
        image_parts.append(_crop_student_image(env, rgb))

    if modality in ("depth", "rgbd"):
        depth = available.get("distance_to_image_plane")
        if depth is None:
            raise RuntimeError(
                f"Depth output missing. Available student camera outputs: {list(available.keys())}"
            )
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)
        else:
            raise RuntimeError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")
        depth_raw = depth
        depth = _apply_depth_noise(env, depth)
        depth_noisy_m = depth
        # The policy view: preprocess + crop on the (maybe-noisy) depth.
        depth_policy = _crop_student_image(env, _preprocess_student_depth(env, depth_noisy_m))
        image_parts.append(depth_policy)
        # Stash the cropped, normalized policy view + the matching clean view
        # for the interactive viewer's clean/noisy A/B. When noise is off the
        # two are identical (cheap, and avoids branching in the viewer).
        env._last_student_image_noisy = depth_policy.detach()
        if bool(getattr(env.cfg.student_obs, "use_depth_aug", False)):
            depth_clean = _crop_student_image(env, _preprocess_student_depth(env, depth_raw))
            env._last_student_image_clean = depth_clean.detach()
        else:
            env._last_student_image_clean = depth_policy.detach()

    if not image_parts:
        raise ValueError(f"Unsupported student image modality: {modality!r}")
    return _validate_student_image_shape(env, torch.cat(image_parts, dim=1))


def _set_usd_attr(prim, name: str, value, value_type) -> None:
    # The URDF converter occasionally emits attributes with malformed type
    # names; in that case remove and recreate so the typed Set lands.
    attr = prim.GetAttribute(name)
    if attr and (not attr.GetTypeName() or not str(attr.GetTypeName())):
        prim.RemoveProperty(name)
        attr = None
    (attr or prim.CreateAttribute(name, value_type, False)).Set(value)


@dataclass(frozen=True)
class _UrdfSdfCollisionMarker:
    mesh_stem: str
    mesh_filename: str
    resolution: int | None = None
    margin: float | None = None
    narrow_band_thickness: float | None = None
    subgrid_resolution: int | None = None


def _usd_safe_identifier(name: str) -> str:
    """Mirror the conservative subset of USD identifier rules we need here."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"mesh_{safe}"
    return safe


def _resolve_urdf_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()


def _prepare_urdf_for_isaacsim(asset_path: str, usd_work_dir: Path) -> str:
    """Return a URDF path whose mesh stems are valid USD prim identifiers.

    Isaac's URDF importer names USD prims from mesh stems.  Meshes such as
    ``6_hole_patch.obj`` therefore fail conversion because USD identifiers
    cannot start with a digit.  When needed, write a temporary URDF with safe
    mesh aliases while leaving the source asset untouched.
    """
    urdf_path = Path(asset_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    changed = False
    alias_dir = usd_work_dir / "_mesh_aliases" / urdf_path.stem
    alias_by_source: dict[Path, Path] = {}
    used_alias_names: set[str] = set()

    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if not filename:
            continue
        source_mesh = _resolve_urdf_mesh_path(urdf_path, filename)
        original_stem = source_mesh.stem
        safe_stem = _usd_safe_identifier(original_stem)
        if safe_stem == original_stem:
            continue

        changed = True
        alias = alias_by_source.get(source_mesh)
        if alias is None:
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias_name = f"{safe_stem}{source_mesh.suffix}"
            if alias_name in used_alias_names:
                index = 1
                while f"{safe_stem}_{index}{source_mesh.suffix}" in used_alias_names:
                    index += 1
                alias_name = f"{safe_stem}_{index}{source_mesh.suffix}"
            used_alias_names.add(alias_name)
            alias = alias_dir / alias_name
            shutil.copy2(source_mesh, alias)
            alias_by_source[source_mesh] = alias

        mesh_tag.set("filename", str(alias))

    if not changed:
        return asset_path

    # The copied URDF lives in the converter work dir, so make every remaining
    # mesh path absolute to preserve source-relative references.
    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if filename:
            mesh_tag.set("filename", str(_resolve_urdf_mesh_path(urdf_path, filename)))

    out_dir = usd_work_dir / "_urdf_preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / urdf_path.name
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return str(out_path)


def _parse_optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _parse_urdf_sdf_collision_markers(asset_path: str) -> list[_UrdfSdfCollisionMarker]:
    urdf_path = Path(asset_path)
    root = ET.parse(urdf_path).getroot()
    markers: list[_UrdfSdfCollisionMarker] = []
    for collision in root.findall(".//collision"):
        sdf_tag = collision.find("sdf")
        if sdf_tag is None:
            continue
        mesh_tag = collision.find("geometry/mesh")
        if mesh_tag is None or not mesh_tag.get("filename"):
            continue
        mesh_filename = str(mesh_tag.get("filename"))
        mesh_path = _resolve_urdf_mesh_path(urdf_path, mesh_filename)
        markers.append(
            _UrdfSdfCollisionMarker(
                mesh_stem=_usd_safe_identifier(mesh_path.stem),
                mesh_filename=mesh_filename,
                resolution=_parse_optional_int(sdf_tag.get("resolution")),
                margin=_parse_optional_float(sdf_tag.get("margin")),
                narrow_band_thickness=_parse_optional_float(
                    sdf_tag.get("narrow_band_thickness")
                    or sdf_tag.get("narrowBandThickness")
                ),
                subgrid_resolution=_parse_optional_int(
                    sdf_tag.get("subgrid_resolution")
                    or sdf_tag.get("subgridResolution")
                ),
            )
        )
    return markers


def _apply_urdf_sdf_collision_markers(
    usd_path: str,
    source_asset_path: str,
    markers: list[_UrdfSdfCollisionMarker],
) -> None:
    if not markers:
        return

    from pxr import Usd, UsdPhysics

    from isaaclab.sim.schemas import SDFMeshPropertiesCfg, define_mesh_collision_properties

    raw_usd_path = Path(usd_path)
    physics_usd_path = raw_usd_path.parent / "configuration" / f"{raw_usd_path.stem}_physics.usd"
    edit_usd_path = physics_usd_path if physics_usd_path.exists() else raw_usd_path

    stage = Usd.Stage.Open(str(edit_usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying URDF SDF markers: {edit_usd_path}")
    stage.Load()

    marker_by_stem = {marker.mesh_stem: marker for marker in markers}
    matched: dict[str, int] = {marker.mesh_stem: 0 for marker in markers}

    fallback_matches = []
    collider_matches = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        path = prim.GetPath().pathString
        path_parts = [part for part in path.split("/") if part]
        marker = next((marker_by_stem[part] for part in path_parts if part in marker_by_stem), None)
        if marker is None:
            continue
        if path.startswith("/colliders/"):
            collider_matches.append((prim, marker))
        else:
            fallback_matches.append((prim, marker))

    for prim, marker in collider_matches or fallback_matches:
        define_mesh_collision_properties(
            str(prim.GetPath()),
            SDFMeshPropertiesCfg(
                sdf_margin=marker.margin,
                sdf_narrow_band_thickness=marker.narrow_band_thickness,
                sdf_resolution=marker.resolution,
                sdf_subgrid_resolution=marker.subgrid_resolution,
            ),
            stage=stage,
        )
        matched[marker.mesh_stem] += 1

    stage.GetRootLayer().Save()

    matched_count = sum(matched.values())
    missing = [stem for stem, count in matched.items() if count == 0]
    if missing:
        print(
            f"[scene_utils] warning: URDF SDF markers in {source_asset_path!r} did not match "
            f"USD collision prims for mesh stems {missing}",
            flush=True,
        )
    if matched_count:
        details = ", ".join(f"{stem}:{count}" for stem, count in matched.items() if count)
        print(
            f"[scene_utils] applied URDF SDF collision markers to {matched_count} prims "
            f"in {edit_usd_path.name} ({details})",
            flush=True,
        )


def _convert_urdf_to_usd(
    asset_path: str,
    usd_work_dir: Path,
    *,
    fix_base: bool,
    self_collision: bool | None = None,
    replace_cylinders_with_capsules: bool = False,
    joint_drive=None,
) -> str:
    converter_asset_path = _prepare_urdf_for_isaacsim(asset_path, usd_work_dir)
    cfg_kwargs = dict(
        asset_path=converter_asset_path,
        usd_dir=str(usd_work_dir / Path(asset_path).stem),
        force_usd_conversion=True,
        fix_base=fix_base,
        merge_fixed_joints=True,
        make_instanceable=False,
        replace_cylinders_with_capsules=replace_cylinders_with_capsules,
        joint_drive=joint_drive,
    )
    if self_collision is not None:
        cfg_kwargs["self_collision"] = self_collision
    usd_path = UrdfConverter(UrdfConverterCfg(**cfg_kwargs)).usd_path
    _apply_urdf_sdf_collision_markers(
        usd_path,
        converter_asset_path,
        _parse_urdf_sdf_collision_markers(converter_asset_path),
    )
    return usd_path



def _robot_joint_drive_cfg():
    # DriveAPI prims must exist for ImplicitActuator runtime gains to land.
    return UrdfConverterCfg.JointDriveCfg(
        drive_type="force", target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    )


def _bake_usd(
    raw_usd_path: str,
    bake_root: Path,
    baked_subdir: str,
    *,
    props: dict | None = None,
    apply_physx_articulation: bool = False,
    collision_enabled: bool | None = None,
) -> str:
    """Copy raw USD into bake_root/baked_subdir and pre-author physics props.

    ``props`` keys come from ``_PHYSICS_SPECS``; ``None`` values are skipped,
    and keys whose group doesn't match a prim's APIs are skipped per-prim.
    """
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    vtypes = {
        "Bool": Sdf.ValueTypeNames.Bool,
        "Float": Sdf.ValueTypeNames.Float,
        "Int": Sdf.ValueTypeNames.Int,
    }
    props = props or {}

    raw = Path(raw_usd_path)
    baked_usd_path = bake_root / baked_subdir / raw.parent.name / raw.name
    baked_usd_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, baked_usd_path)
    for child in raw.parent.iterdir():
        if child.name.startswith(".") or child.name in (raw.name, "config.yaml"):
            continue
        dst = baked_usd_path.parent / child.name
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child, dst)
        elif child.is_file():
            shutil.copy2(child, dst)

    stage = Usd.Stage.Open(str(baked_usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open baked USD: {baked_usd_path}")
    root = stage.GetDefaultPrim()
    if not (root and root.IsValid()):
        root = next((p for p in stage.GetPseudoRoot().GetChildren() if p.IsValid()), None)
    if root is None:
        raise RuntimeError(f"No root prim in USD: {baked_usd_path}")

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    for prim in Usd.PrimRange(root):
        is_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if is_rb:
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if is_art and apply_physx_articulation:
            PhysxSchema.PhysxArticulationAPI.Apply(prim)
        for key, val in props.items():
            if val is None:
                continue
            group, attr_name, vtype_str = _PHYSICS_SPECS[key]
            if group == "rb" and not is_rb:
                continue
            if group == "art" and not is_art:
                continue
            _set_usd_attr(prim, attr_name, val, vtypes[vtype_str])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            px = PhysxSchema.PhysxCollisionAPI(prim) or PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(_CONTACT_OFFSET)
            px.CreateRestOffsetAttr().Set(_REST_OFFSET)
            if collision_enabled is not None:
                ce = UsdPhysics.CollisionAPI(prim)
                (ce.GetCollisionEnabledAttr() or ce.CreateCollisionEnabledAttr()).Set(
                    collision_enabled
                )

    stage.GetRootLayer().Save()
    return str(baked_usd_path)


# ----------------------------------------------------------------------------
# Runtime material setup (post-launch, via PhysX views)
# ----------------------------------------------------------------------------

def apply_physx_material_properties(env) -> None:
    """Set contact materials through PhysX tensor views.

    Follows Isaac Lab's large-scale randomization path: avoid post-spawn USD
    relationship authoring and per-clone material prims. Must run after
    ``DirectRLEnv`` starts the simulator and ``root_physx_view`` exists.
    """
    assets_cfg = env.cfg.assets
    if not assets_cfg.modify_asset_frictions:
        return

    t0 = time.perf_counter()
    default = torch.tensor(
        [float(assets_cfg.robot_friction), float(assets_cfg.robot_friction), 0.0],
        dtype=torch.float32, device="cpu",
    )
    fingertip = torch.tensor(
        [float(assets_cfg.finger_tip_friction), float(assets_cfg.finger_tip_friction), 0.0],
        dtype=torch.float32, device="cpu",
    )
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")

    robot_view = env.robot.root_physx_view
    robot_materials = robot_view.get_material_properties()
    robot_materials[:] = default

    shape_start = 0
    for link_name, link_path in zip(robot_view.shared_metatype.link_names, robot_view.link_paths[0]):
        link_view = env.robot._physics_sim_view.create_rigid_body_view(link_path)
        shape_end = shape_start + link_view.max_shapes
        if link_name in FINGERTIP_LINK_NAMES:
            robot_materials[:, shape_start:shape_end] = fingertip
        shape_start = shape_end
    if shape_start != robot_view.max_shapes:
        raise RuntimeError(
            f"Robot shape count mismatch while assigning materials: "
            f"computed {shape_start}, view reports {robot_view.max_shapes}."
        )
    robot_view.set_material_properties(robot_materials, env_ids)

    for name in ("table", "object", "goal_viz", "hole"):
        if not hasattr(env, name):
            continue
        view = getattr(env, name).root_physx_view
        materials = view.get_material_properties()
        materials[:] = default
        view.set_material_properties(materials, env_ids)

    _log_scene_step(t0, "applied PhysX material properties")


# ----------------------------------------------------------------------------
# Scene assembly
# ----------------------------------------------------------------------------

def _materialize_env_prims(env) -> None:
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")


def _build_object_scale_tensor(env, object_scales_normalized, num_object_usds: int) -> None:
    num_envs = env.num_envs
    object_prim_paths = find_matching_prim_paths("/World/envs/env_.*/Object")
    if len(object_prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Object prims after MultiUsdFileCfg spawn, "
            f"got {len(object_prim_paths)}. Cloner-drop bug may have returned."
        )

    env._object_scale_per_env = torch.zeros(num_envs, 3, device=env.device, dtype=torch.float32)
    env._object_asset_index_per_env = torch.zeros(num_envs, device=env.device, dtype=torch.long)
    for source_idx, obj_path in enumerate(object_prim_paths):
        env_id = int(obj_path.rsplit("/", 2)[-2].removeprefix("env_"))
        asset_index = source_idx % num_object_usds
        env._object_scale_per_env[env_id] = torch.tensor(
            object_scales_normalized[asset_index], device=env.device, dtype=torch.float32,
        )
        env._object_asset_index_per_env[env_id] = asset_index


def setup_scene(env) -> None:
    """Build and register robot, table, object, goal, ground, and light."""
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(
        setup_t0,
        f"setup start num_envs={env.num_envs} "
        f"num_assets_per_type={assets_cfg.num_assets_per_type}",
    )

    # 1. Generate procedural URDFs in a per-launch temp dir.
    env._tmp_asset_dir = tempfile.mkdtemp(prefix="simtoolreal_assets_")
    urdf_paths, object_scales_normalized = generate_handle_head_urdfs(
        handle_head_types=tuple(assets_cfg.handle_head_types),
        num_per_type=assets_cfg.num_assets_per_type,
        out_dir=env._tmp_asset_dir,
        shuffle=assets_cfg.shuffle_assets,
    )
    if not urdf_paths:
        raise ValueError(
            "No procedural object URDFs were generated. "
            "Check cfg.assets.handle_head_types and num_assets_per_type."
        )
    env._object_urdf_paths = [str(path) for path in urdf_paths]
    _log_scene_step(setup_t0, f"generated {len(urdf_paths)} object URDFs")

    # 2. Convert URDFs -> raw USDs -> role-specific baked USDs.
    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    object_raw_usds = [
        _convert_urdf_to_usd(
            str(urdf), usd_work_dir, fix_base=False, replace_cylinders_with_capsules=True,
        )
        for urdf in urdf_paths
    ]
    object_usd_paths = [
        _bake_usd(usd, bake_root, "object", props=dict(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=1000.0, articulation_enabled=False,
        ))
        for usd in object_raw_usds
    ]
    goalviz_usd_paths = [
        _bake_usd(usd, bake_root, "goalviz", props=dict(
            kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
        ), collision_enabled=False)
        for usd in object_raw_usds
    ]

    robot_usd_path = _bake_usd(
        _convert_urdf_to_usd(
            assets_cfg.robot_urdf, usd_work_dir,
            fix_base=True, self_collision=False,
            joint_drive=_robot_joint_drive_cfg(),
        ),
        bake_root, "robot",
        props=dict(
            disable_gravity=True, max_depenetration_velocity=1000.0,
            enabled_self_collisions=False,
            solver_position_iterations=8, solver_velocity_iterations=0,
        ),
        apply_physx_articulation=True,
    )
    table_usd_path = _bake_usd(
        _convert_urdf_to_usd(assets_cfg.table_urdf, usd_work_dir, fix_base=False),
        bake_root, "table",
        props=dict(
            kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
        ),
    )
    _log_scene_step(setup_t0, "resolved baked USDs")

    # 3. Pre-create env roots so regex spawns resolve to every env.
    _materialize_env_prims(env)

    # 4. Spawn assets.
    env.robot = Articulation(build_robot_articulation_usd_cfg(robot_usd_path))
    env.table = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Table", [table_usd_path]))
    env.object = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_paths))
    env.goal_viz = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_paths))
    _log_scene_step(setup_t0, "spawned robot/table/object/goalviz")

    # 5. Ground plane + dome light (global, outside env_*).
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 6. Per-env scale tensor for spawned Objects.
    _build_object_scale_tensor(env, object_scales_normalized, len(object_usd_paths))

    # 7. Register with scene so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    hide_goal_viz_for_student_camera(env)
    setup_student_camera(env)
    _log_scene_step(setup_t0, "registered assets with scene")
