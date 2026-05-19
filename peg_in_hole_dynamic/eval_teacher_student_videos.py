#!/usr/bin/env python3
"""Compare teacher (state SAPG) and student (depth_cnn_mlp BC) on PegInHole,
saving side-by-side videos from matched initial states.

For each of `--num-envs` envs (default 10), runs:
  1. The state-based SAPG teacher policy
  2. The depth_cnn_mlp BC-only student policy
both on the SAME initial state (deterministic reset via shared seed). Captures
RGB frames per step from a per-env recording TiledCamera, stops appending
frames for an env once it succeeds, and writes one side-by-side mp4 per env
(teacher on the left, student on the right) plus a summary.json.

Both policies are loaded against the SimToolReal/PegInHole DEPTH-STUDENT env
(`Isaacsimenvs-PegInHoleDepthStudent-Direct-v0`) because that env exposes
BOTH the privileged teacher_obs (consumed by the state-MLP teacher) AND the
depth student_obs (consumed by the depth-CNN student) from the same world,
so the two passes see identical resets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Defaults pointing at the specific runs the user asked about
# --------------------------------------------------------------------------
DEFAULT_TEACHER = (
    "/share/portal/kk837/depthbasedRL/train_dir/teachers_final/"
    "play2win_peg_insertion/lpeg_tol0p5mm_finetune_rgf0_2026-05-15_03-38-51/"
    "0_lpeg_tol0p5mm_finetune_rgf0_2026-05-15_03-38-51/last/model.pth"
)
DEFAULT_STUDENT = (
    "/share/portal/kk837/depthbasedRL/train_dir/students_final/"
    "play2win_peg_insertion_distill_bc_only/"
    "lpeg_matchedmass_tol0p5mm_rgf0_distill_bc_only_2026-05-15_11-35-26/"
    "0_lpeg_tol0p5mm_ppo_dagger/last/model.pth"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg_matchedmass.tol0p5mm")
    p.add_argument("--goal-mode", default="preInsertAndFinal")
    p.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER)
    p.add_argument("--student-checkpoint", default=DEFAULT_STUDENT)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1200,
                   help="Cap per-env rollout length. Recording stops earlier for envs that succeed.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--hole-yaw-deg", type=float, default=10.0,
                   help="Per-episode hole yaw randomization range (deg). "
                        "Must match what teacher/student were trained with.")
    p.add_argument("--success-steps", type=int, default=1,
                   help="env.termination.success_steps. Training=1; default "
                        "PegInHoleDepthStudent.yaml=10. Use 10 to ablate the "
                        "'must hold steady at goal' requirement.")
    p.add_argument("--skip-videos", action="store_true",
                   help="Skip mp4 writes (just produce summary.json). Useful "
                        "for n>10 envs where 1 mp4 / env is unnecessary.")
    p.add_argument("--skip-teacher", action="store_true",
                   help="Skip the teacher pass entirely; only run + report "
                        "student. Use when evaluating the student on a "
                        "different env config than the teacher was trained on "
                        "(e.g. --goal-mode finalGoalOnly).")
    p.add_argument("--episode-length-s", type=float, default=None,
                   help="Override env.episode_length_s (default 10.0 = 600 "
                        "policy steps). Use 20.0 with finalGoalOnly to match "
                        "preInsertAndFinal's effective budget (which gets a "
                        "free timer reset on pre-insert hit, doubling its "
                        "step budget to ~1200).")
    p.add_argument("--student-agent-key", default="rl_games_dagger_ppo_cfg_entry_point",
                   help="Which dagger cfg to load for the student network. "
                        "Use rl_games_dagger_sapg_cfg_entry_point for older "
                        "SAPG-trained students (depth_cnn_lstm, has_block_id).")
    p.add_argument("--camera-backend", default="raycaster",
                   help="env.student_obs.camera_backend ('raycaster' or 'tiled').")
    p.add_argument("--horizontal-aperture", type=float, default=14.524,
                   help="Sim camera horizontal_aperture (cm). 14.524 = matched-cy "
                        "70x70 crop; 33.197 = full ZED 160x90 aperture.")
    p.add_argument("--h-aperture-offset", type=float, default=-0.389,
                   help="Sim camera horizontal_aperture_offset (dimensionless).")
    p.add_argument("--v-aperture-offset", type=float, default=0.105,
                   help="Sim camera vertical_aperture_offset (dimensionless).")
    p.add_argument("--image-width", type=int, default=70,
                   help="Render resolution width (env.student_obs.image_width). "
                        "Use 160 for 160x90+crop pipeline, 70 for direct.")
    p.add_argument("--image-height", type=int, default=70,
                   help="Render resolution height.")
    p.add_argument("--crop-enabled", type=lambda v: v.lower() in ("true", "1", "yes"),
                   default=False,
                   help="env.student_obs.crop_enabled. True for the 160x90+crop "
                        "pipeline (older students), False for direct 70x70.")
    p.add_argument("--crop-top-left", default=None,
                   help="env.student_obs.crop_top_left as 'x0,y0' (only when crop_enabled).")
    p.add_argument("--crop-bottom-right", default=None,
                   help="env.student_obs.crop_bottom_right as 'x1,y1' (exclusive).")
    p.add_argument("--image-input-width", type=int, default=70,
                   help="env.student_obs.image_input_width (policy input dims after crop).")
    p.add_argument("--image-input-height", type=int, default=70)
    p.add_argument("--fs-stereo-width", type=int, default=None,
                   help="env.student_obs.fs_stereo_width (FS render width, mult of 32).")
    p.add_argument("--fs-stereo-height", type=int, default=None,
                   help="env.student_obs.fs_stereo_height (FS render height, mult of 32).")
    p.add_argument("--dump-action-trace", action="store_true",
                   help="Save env-0's policy action + obs hash per step to "
                        "{out_dir}/action_trace.npz. Used to diff two runs.")
    p.add_argument("--override-json", type=str, default=None,
                   help="JSON dict of extra Hydra overrides to pass to "
                        "_apply_env_overrides. Use to set fixed-init params.")
    p.add_argument("--audit-tiled", action="store_true",
                   help="Add a SECOND camera (TiledCamera, 160x94 cy-matched + "
                        "crop[90:160,0:70]) alongside the policy's primary "
                        "camera. Used to render side-by-side [tiled | "
                        "primary-backend] videos with GUARANTEED matched init "
                        "states (single rollout, two cameras observing the "
                        "same world). Implies --skip-teacher.")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-height", type=int, default=210,
                   help="Per-policy frame height. Upscale factor = "
                        "round(video_height / image_input_height); frame width "
                        "follows from upscale * image_input_width. Side-by-side "
                        "video width is 2x.")
    p.add_argument("--video-quality", type=int, default=5,
                   help="imageio mp4 quality (0-10, lower=smaller file).")
    p.add_argument("--capture-every", type=int, default=2,
                   help="Capture one RGB frame every N sim steps. "
                        "N=2 means 30 Hz capture from a 60 Hz sim -> real-time when video_fps=30.")
    p.add_argument("--output-dir", default=None,
                   help="Output dir. Default: peg_in_hole_dynamic/eval_teacher_student_outputs/<datetime>/")
    return p.parse_args()


def _reseed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _build_student_from_cfg(student_agent_cfg: dict, env, action_dim: int,
                             student_checkpoint: str, device: str):
    """Build a depth-CNN student matching the trained checkpoint and load weights."""
    import torch
    from rl_games.algos_torch import model_builder
    from isaacsimenvs.dagger import networks as _net  # noqa: F401  registers depth_cnn_mlp

    net_params = dict(student_agent_cfg["params"]["network"])
    net_name = str(net_params["name"])

    # Refresh proprio_dim from the live env (the yaml default may be stale).
    student_obs0 = env.get_student_obs()
    actual_proprio_dim = int(student_obs0["proprio"].shape[-1])
    if actual_proprio_dim != int(net_params.get("proprio_dim", -1)):
        net_params["proprio_dim"] = actual_proprio_dim

    image_channels = int(net_params.get("image_channels", 1))
    image_hw = tuple(net_params["image_hw"])
    has_block_id = bool(net_params.get("has_block_id", False))
    obs_dim = image_channels * image_hw[0] * image_hw[1] + actual_proprio_dim + (1 if has_block_id else 0)

    builder = model_builder.NETWORK_REGISTRY[net_name]()
    builder.load(net_params)
    student = builder.build("student", actions_num=action_dim,
                            input_shape=(obs_dim,)).to(device)

    sd = torch.load(student_checkpoint, map_location=device, weights_only=False)
    # SAPG nested {block: {...}} vs flat {"model": {...}}.
    if isinstance(sd, dict) and "model" not in sd and any(isinstance(k, int) for k in sd):
        block_keys = [k for k in sd if isinstance(k, int)]
        sd = sd[min(block_keys)]
    model_sd = sd.get("model", sd)
    # Strip the rl_games wrapper prefix and drop normalizer keys we don't carry.
    prefix = "a2c_network."
    stripped: dict[str, torch.Tensor] = {}
    for k, v in model_sd.items():
        if k.startswith(prefix):
            stripped[k[len(prefix):]] = v
        elif k.startswith(("value_mean_std.", "running_mean_std.")):
            continue
        else:
            stripped[k] = v
    missing, unexpected = student.load_state_dict(stripped, strict=False)
    loaded = len(set(student.state_dict().keys()) & set(stripped.keys()))
    print(f"=> loaded {loaded}/{len(student.state_dict())} student weights "
          f"from '{student_checkpoint}' (network={net_name})", flush=True)
    if loaded == 0:
        raise RuntimeError(
            "Student state_dict load matched zero keys. Architecture mismatch?")
    student.eval()
    return student, image_channels, image_hw, has_block_id


def _student_action(env, student, image_channels: int, image_hw: tuple,
                    has_block_id: bool, rnn_state_ref: list):
    """Run one student forward; threads RNN state through `rnn_state_ref[0]`
    so recurrent networks (depth_cnn_lstm) actually accumulate state across
    timesteps. Non-RNN nets (depth_cnn_mlp) ignore rnn_state in their forward
    and return the same `None`, so this is a no-op for them."""
    import torch
    out = env.get_student_obs()
    image = out["image"]
    proprio = out["proprio"]
    if has_block_id:
        block_id = torch.zeros(image.shape[0], 1, device=image.device)
        flat = torch.cat([image.flatten(1), proprio, block_id], dim=-1)
    else:
        flat = torch.cat([image.flatten(1), proprio], dim=-1)
    with torch.no_grad():
        mu, _ls, _v, rnn_state = student(
            {"obs": flat, "rnn_states": rnn_state_ref[0], "seq_length": 1}
        )
    rnn_state_ref[0] = rnn_state
    return mu


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "peg_in_hole_dynamic" / "eval_teacher_student_outputs"
        / time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=> output dir: {out_dir}", flush=True)

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = True
    AppLauncher(launcher_args)

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
    import isaacsimenvs  # noqa: F401  triggers gym.register
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info
    from isaacsimenvs.dagger.teacher import Teacher
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
    )

    # ---- Env config matching the BC student's training overrides ----
    cfg = _load_env_cfg(args.task)
    import json as _json
    _extra = _json.loads(args.override_json) if args.override_json else {}
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
        extra_overrides=_extra,
    )
    # Camera backend / intrinsics configurable via CLI to match either the
    # "matched-cy raycaster" trained students (defaults) or the older
    # "tiled rasterizer + full ZED aperture" students.
    cfg.student_obs.camera_backend = str(args.camera_backend)
    cfg.student_obs.image_width = int(args.image_width)
    cfg.student_obs.image_height = int(args.image_height)
    cfg.student_obs.image_input_width = int(args.image_input_width)
    cfg.student_obs.image_input_height = int(args.image_input_height)
    cfg.student_obs.crop_enabled = bool(args.crop_enabled)
    if args.crop_top_left is not None:
        cfg.student_obs.crop_top_left = tuple(int(x) for x in args.crop_top_left.split(","))
    if args.crop_bottom_right is not None:
        cfg.student_obs.crop_bottom_right = tuple(int(x) for x in args.crop_bottom_right.split(","))
    cfg.student_obs.horizontal_aperture = float(args.horizontal_aperture)
    cfg.student_obs.horizontal_aperture_offset = float(args.h_aperture_offset)
    cfg.student_obs.vertical_aperture_offset = float(args.v_aperture_offset)
    if args.fs_stereo_width is not None:
        cfg.student_obs.fs_stereo_width = int(args.fs_stereo_width)
    if args.fs_stereo_height is not None:
        cfg.student_obs.fs_stereo_height = int(args.fs_stereo_height)
    cfg.student_obs.camera_update_period_s = 0.0
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    # Hole yaw randomization matches training.
    cfg.peg_in_hole.hole_yaw_range_deg = float(args.hole_yaw_deg)
    # All DR off (eval scenario).
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0
    cfg.reset.table_reset_xy_range_m = (0.0, 0.0)
    cfg.reset.table_reset_yaw_range_deg = 0.0
    # --- Critical: match the BC student's training env exactly. ---
    # success_steps=1 (default 10): student trained to count success after a
    # SINGLE near-goal step, not 10 consecutive ones; without this fix the
    # eval is much stricter than training, masking real student capability.
    cfg.termination.success_steps = int(args.success_steps)
    # goal_xy_obs_noise=0.0 (default 0.002): student saw a clean goal obs
    # during training; a 2 mm random offset on the observed goal pushes its
    # learned action distribution off the actual insertion target.
    cfg.peg_in_hole.goal_xy_obs_noise = 0.0
    # Restore the training-time reset randomization (eval_isaacsim's
    # EVAL_DR_OFF_OVERRIDES zeros these, but the student learned to bootstrap
    # from a noisy initial dof pose / velocity).
    cfg.reset.reset_dof_pos_random_interval_arm = 0.1
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.1
    cfg.reset.reset_dof_vel_random_interval = 0.5
    # Optional episode_length_s override (matches preInsertAndFinal's effective
    # step budget when running finalGoalOnly).
    if args.episode_length_s is not None:
        cfg.episode_length_s = float(args.episode_length_s)
        print(f"=> overriding episode_length_s={cfg.episode_length_s} "
              f"(default 10.0)", flush=True)

    env = _instantiate_env(args.task, cfg)
    print(f"=> env instantiated, num_envs={env.num_envs}, "
          f"action_dim={env.action_space.shape[-1]}", flush=True)

    # ---- Use the literal policy obs as the video frame ----
    # env.get_student_obs()["image"] returns the same window-normalized depth
    # the depth-CNN consumes (post the env's normalization / optional aug
    # pipeline; with all DR off here it's just window-normalized depth).
    # The actual obs shape is image_input_h x image_input_w (after any crop),
    # which is what matters for the video upscale factor.
    actual_h = int(cfg.student_obs.image_input_height)
    actual_w = int(cfg.student_obs.image_input_width)
    upscale = max(1, int(round(int(args.video_height) / max(actual_h, 1))))
    out_w = actual_w * upscale
    out_h = actual_h * upscale
    print(f"=> video frames: env.get_student_obs()[\"image\"] "
          f"({actual_w}x{actual_h}) upscaled {upscale}x -> {out_w}x{out_h} "
          f"(literal policy input, no extra rendering)",
          flush=True)

    # ---- Optional audit camera (second TiledCamera, cy-matched) ----
    # Spawns a second camera in the SAME env that renders depth with the tiled
    # rasterizer at 160x94+crop[90:160,0:70] -> 70x70 (matched-cy via crop).
    # Both cameras observe the SAME world state every step => init states are
    # guaranteed matched for the side-by-side comparison.
    env.audit_camera = None
    if args.audit_tiled:
        # The audit camera must NOT be the same prim_path as student_camera.
        audit_cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/AuditCamera",
            update_period=0.0,
            update_latest_camera_pose=True,
            height=94, width=160,
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=33.19737869997174,
                clipping_range=(0.1, 5.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(float(x) for x in cfg.student_obs.camera_pos),
                rot=tuple(float(x) for x in cfg.student_obs.camera_quat_wxyz),
                convention=str(cfg.student_obs.camera_convention),
            ),
        )
        env.audit_camera = TiledCamera(cfg=audit_cfg)
        env.scene.sensors["audit_camera"] = env.audit_camera
        # Single env.sim.reset() to activate the new sensor. NO warmup loop:
        # audit_camera_init_order.py verified that registering BEFORE the
        # wrapped.reset()s that the policy will see is sufficient — the env's
        # next reset cycle stabilizes both sensors together (0 contamination
        # vs single-camera baseline, 0/12439 pixels differ).
        env.sim.reset()
        env.audit_camera.update(0.0)
        print(f"=> audit camera (tiled 160x94 cy-matched via crop, no warmup, "
              f"construction-time-equivalent registration): spawned", flush=True)
        # --audit-tiled implies --skip-teacher (we use the teacher half of
        # the side-by-side video for the audit camera output).
        args.skip_teacher = True

    # ---- rl_games wrapping ----
    agent_cfg_teacher = _configure_agent(
        args.task, "rl_games_sapg_cfg_entry_point",
        rl_device=args.rl_device, num_envs=int(args.num_envs),
        deterministic=True, games=1, extra_overrides={},
    )
    clip_obs = float(agent_cfg_teacher["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg_teacher["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=args.rl_device,
                                    clip_obs=clip_obs, clip_actions=clip_actions)

    # ---- Teacher ----
    env_info_teacher = teacher_env_info(wrapped)
    teacher = Teacher(
        task_id="Isaacsimenvs-PegInHole-Direct-v0",
        agent_key="rl_games_sapg_cfg_entry_point",
        checkpoint_path=args.teacher_checkpoint,
        num_envs=int(args.num_envs),
        rl_device=args.rl_device,
        env_info=env_info_teacher,
    )
    print(f"=> teacher loaded from '{args.teacher_checkpoint}'", flush=True)

    # ---- Student ----
    student_agent_cfg = load_cfg_from_registry(
        args.task, args.student_agent_key,
    )
    action_dim = int(env.action_space.shape[-1])
    student, image_channels, image_hw, has_block_id = _build_student_from_cfg(
        student_agent_cfg, env, action_dim,
        args.student_checkpoint, args.rl_device,
    )

    # ---- Audit camera capture (returns frames matched to primary obs shape) ----
    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import (
        _preprocess_student_depth as _scene_preprocess_depth,
    )

    def _capture_audit_frames() -> np.ndarray:
        if env.audit_camera is None:
            return None
        env.audit_camera.update(env.sim.get_physics_dt())
        depth = env.audit_camera.data.output["distance_to_image_plane"]
        if depth is None:
            return None
        d = depth.detach().float()
        if d.ndim == 4 and d.shape[-1] == 1:
            d = d[..., 0]
        # crop matches student_obs.crop_top_left/bottom_right; hardcoded here
        # because audit camera renders at a different resolution than the env's
        # student_camera, so _crop_student_image (which reads cfg) doesn't fit.
        d = d[:, 0:70, 90:160]
        norm = _scene_preprocess_depth(env, d)  # window_normalize via env cfg
        u8 = (norm * 255.0).byte().cpu().numpy()
        if upscale > 1:
            u8 = np.repeat(np.repeat(u8, upscale, axis=1), upscale, axis=2)
        return np.stack([u8, u8, u8], axis=-1)

    # ---- Per-pass rollout ----
    def _capture_rgb_frames() -> np.ndarray:
        """Grab the student's window-normalized depth obs (the literal policy
        input) for all envs, upscale by `upscale`, return uint8 (N, H, W, 3).

        Mirrors `depth_compare_raycaster_capture.py:232`:
            image = obs["image"][0, 0].detach().float().cpu().numpy()
        """
        obs = env.get_student_obs()
        image = obs["image"][:, 0].detach().float().clamp_(0.0, 1.0).cpu().numpy()
        u8 = (image * 255.0).astype(np.uint8)  # (N, H, W)
        if upscale > 1:
            u8 = np.repeat(np.repeat(u8, upscale, axis=1), upscale, axis=2)
        return np.stack([u8, u8, u8], axis=-1)  # (N, H_up, W_up, 3)

    def _init_state_fingerprint() -> dict:
        """Snapshot of post-reset world state for cross-pass determinism check.
        Identical numbers across the two passes ⇒ deterministic reset works."""
        snap = {
            "robot_joint_pos": env.robot.data.joint_pos.detach().cpu().numpy().copy(),
            "object_root_pose": env.object.data.root_state_w[:, :7].detach().cpu().numpy().copy(),
            "hole_root_pose": env.hole.data.root_state_w[:, :7].detach().cpu().numpy().copy()
                if hasattr(env, "hole") else None,
            "goal_pos_obs_noise": env.goal_pos_obs_noise.detach().cpu().numpy().copy()
                if hasattr(env, "goal_pos_obs_noise") else None,
            "goal_yaw_obs_noise": env.goal_yaw_obs_noise.detach().cpu().numpy().copy()
                if hasattr(env, "goal_yaw_obs_noise") else None,
        }
        # Compact hash for one-line logging.
        hasher = []
        for k, v in snap.items():
            if v is None:
                hasher.append(f"{k}=None")
            else:
                hasher.append(f"{k}={float(np.abs(v).sum()):.6f}")
        snap["_summary"] = " ".join(hasher)
        return snap

    def _run_pass(actor_fn, pass_name: str) -> tuple[list[list[np.ndarray]], np.ndarray, np.ndarray, dict]:
        """Run one rollout pass with `actor_fn() -> action`.

        Returns:
          frames_per_env: list of length N, each a list of (H, W, 3) uint8 frames
                          (recording stops appending for an env after it succeeds).
          success_per_env: bool array, did this env hit at least one insertion goal.
          steps_to_success: int array per env (max_steps if never).
          fingerprint: dict of initial-state arrays for determinism verification.
        """
        _reseed(args.seed)
        teacher.reset()
        wrapped.reset()
        _reseed(args.seed)
        wrapped.reset()  # double reset for determinism (matches offline_eval_robustness)
        fingerprint = _init_state_fingerprint()
        print(f"   [{pass_name}] post-reset fingerprint: {fingerprint['_summary']}", flush=True)

        n = int(args.num_envs)
        frames_per_env: list[list[np.ndarray]] = [[] for _ in range(n)]
        # Parallel audit-camera frame buffer (same shape as primary frames).
        # Only populated when env.audit_camera is set (--audit-tiled).
        audit_frames_per_env: list[list[np.ndarray]] = [[] for _ in range(n)]
        recording_active = np.ones(n, dtype=bool)
        succeeded = np.zeros(n, dtype=bool)
        steps_to_success = np.full(n, int(args.max_steps), dtype=np.int32)
        # Running max of `_successes` ACROSS resets — `_successes` zeros at
        # episode reset (truncation at 600 steps), but `_prev_episode_successes`
        # captures the just-completed episode's max, so the elementwise max
        # of (running, _successes, _prev_episode_successes) tracks the lifetime
        # max of insertion goals hit per env.
        ever_max = torch.zeros(n, dtype=torch.long, device=args.rl_device)
        env_max_goals = env.env_max_goals.detach().clone()
        capture_every = max(1, int(getattr(args, "capture_every", 2)))

        # First-episode-only tracking: freeze each env's outcome when its FIRST
        # episode ends (detected via env._termination_reasons below).
        first_ep_max = torch.zeros(n, dtype=torch.long, device=args.rl_device)
        first_ep_done = torch.zeros(n, dtype=torch.bool, device=args.rl_device)

        # Capture frame at t=0 (post-reset, pre-action).
        first_frame = _capture_rgb_frames()
        if first_frame is not None:
            for i in range(n):
                frames_per_env[i].append(first_frame[i].copy())
        if env.audit_camera is not None:
            first_audit = _capture_audit_frames()
            if first_audit is not None:
                for i in range(n):
                    audit_frames_per_env[i].append(first_audit[i].copy())

        for step in range(int(args.max_steps)):
            action = actor_fn()
            wrapped.step(action)

            # Update lifetime max of successes (survives episode resets via
            # _prev_episode_successes).
            cur = env._successes.detach()
            prev = env._prev_episode_successes.detach() if hasattr(
                env, "_prev_episode_successes") else cur
            ever_max = torch.maximum(ever_max, torch.maximum(cur, prev))

            # First-episode tracking. We can't trust episode_length_buf drops
            # — peg_in_hole_env.py:484 also zeros it on intermediate SUBGOAL
            # successes (not just real resets). The reliable signal is
            # env._termination_reasons (set in _get_dones line 538) which is
            # True for any env that terminated/truncated this step. We OR all
            # four reasons together to get "episode ended this step", then
            # snapshot _successes (NOT _prev_episode_successes — that hasn't
            # been written yet at the moment _termination_reasons fires for
            # the just-completed _get_dones; it'll be updated in the next
            # call to _reset_idx) into first_ep_max.
            term_reasons = getattr(env, "_termination_reasons", None)
            if term_reasons is not None:
                ended_this_step = (
                    term_reasons["fall"] | term_reasons["max_successes"]
                    | term_reasons["hand_far"] | term_reasons["timeout"]
                ).detach()
                newly_done = ended_this_step & ~first_ep_done
                if newly_done.any():
                    # `prev` is _prev_episode_successes AFTER the reset has
                    # already run (rl_games wrapper invokes _reset_idx in step),
                    # so it holds the just-completed episode's max.
                    first_ep_max = torch.where(newly_done, prev, first_ep_max)
                    first_ep_done = first_ep_done | newly_done
            else:
                newly_done = torch.zeros(n, dtype=torch.bool, device=args.rl_device)

            # "Succeeded" = at any point reached env_max_goals (= full insertion).
            successes_now = (ever_max >= env_max_goals).cpu().numpy()
            newly = successes_now & (~succeeded)
            if newly.any():
                idxs = np.where(newly)[0]
                steps_to_success[idxs] = step + 1
                succeeded[idxs] = True
                recording_active[idxs] = False
                for i in idxs:
                    print(f"   [{pass_name}] env {int(i):02d} succeeded at step {step+1}", flush=True)

            # Also stop recording when an env's FIRST episode ends (success or
            # failure) — videos should show episode 1 only.
            if newly_done.any():
                idxs = np.where(newly_done.cpu().numpy())[0]
                recording_active[idxs] = False

            # Append RGB at the chosen cadence to each env still recording.
            if recording_active.any() and (step % capture_every == 0):
                frame_batch = _capture_rgb_frames()
                if frame_batch is not None:
                    for i in range(n):
                        if recording_active[i]:
                            frames_per_env[i].append(frame_batch[i].copy())
                # Audit camera (second-cam, observes same world state).
                if env.audit_camera is not None:
                    audit_batch = _capture_audit_frames()
                    if audit_batch is not None:
                        for i in range(n):
                            if recording_active[i]:
                                audit_frames_per_env[i].append(audit_batch[i].copy())

            if not recording_active.any():
                break

        # For envs whose first episode never reset (still in episode 1 at end
        # of rollout), use the current/lifetime max as their first-episode max.
        still_running = ~first_ep_done
        if still_running.any():
            cur = env._successes.detach()
            prev = env._prev_episode_successes.detach() if hasattr(
                env, "_prev_episode_successes") else cur
            first_ep_max = torch.where(
                still_running, torch.maximum(cur, prev), first_ep_max
            )

        # Per-env partial success ratio (for parity with offline_eval_robustness'
        # 85% metric).
        partial = (ever_max.float() / env_max_goals.float().clamp_min(1.0)).clamp(0.0, 1.0)
        first_partial = (first_ep_max.float() / env_max_goals.float().clamp_min(1.0)).clamp(0.0, 1.0)
        first_full = (first_ep_max >= env_max_goals).float()
        partial_np = partial.cpu().numpy()
        ever_max_np = ever_max.cpu().numpy()
        env_max_goals_np = env_max_goals.cpu().numpy()
        first_ep_max_np = first_ep_max.cpu().numpy()
        first_partial_np = first_partial.cpu().numpy()
        first_full_np = first_full.cpu().numpy()
        print(f"   [{pass_name}] ever_max per env: {ever_max_np.tolist()[:20]}"
              f"{'...' if len(ever_max_np) > 20 else ''}  "
              f"env_max_goals[0]={int(env_max_goals_np[0])}  "
              f"ever_partial={float(partial.mean().item()):.3f}  "
              f"first_ep_partial={float(first_partial.mean().item()):.3f}  "
              f"first_ep_full={float(first_full.mean().item()):.3f}",
              flush=True)
        return (frames_per_env, succeeded, steps_to_success, fingerprint,
                partial_np, ever_max_np, env_max_goals_np,
                first_ep_max_np, first_partial_np, first_full_np,
                audit_frames_per_env)

    def _teacher_action() -> torch.Tensor:
        # The depth-student env's _get_observations returns
        # {"policy", "critic", "teacher_obs"}; teacher consumes teacher_obs.
        obs = env._get_observations()
        return teacher.get_action(obs["teacher_obs"])

    # rnn_state for the student. depth_cnn_lstm needs it threaded across
    # steps; depth_cnn_mlp ignores it. We reset it (None -> fresh zero state)
    # at the start of each pass via _student_action_fn_reset().
    _student_rnn = [None]

    def _student_action_fn_reset():
        _student_rnn[0] = None
        if hasattr(student, "reset_default_state"):
            student.reset_default_state()

    # Action-trace dump: per-step env-0 (image_hash, proprio_hash, action_hash,
    # joint_pos_hash). Hashes are sum of abs values — float-precise enough to
    # detect any per-step divergence between runs.
    trace_step = [0]
    image_hashes: list[float] = []
    proprio_hashes: list[float] = []
    action_hashes: list[float] = []
    joint_hashes: list[float] = []
    object_pos_traces: list[np.ndarray] = []
    object_quat_traces: list[np.ndarray] = []
    hole_pos_traces: list[np.ndarray] = []
    hole_quat_traces: list[np.ndarray] = []

    def _student_action_fn() -> torch.Tensor:
        obs = env.get_student_obs()
        image = obs["image"]
        proprio = obs["proprio"]
        if has_block_id:
            block_id = torch.zeros(image.shape[0], 1, device=image.device)
            flat = torch.cat([image.flatten(1), proprio, block_id], dim=-1)
        else:
            flat = torch.cat([image.flatten(1), proprio], dim=-1)
        with torch.no_grad():
            mu, _ls, _v, rnn_state = student(
                {"obs": flat, "rnn_states": _student_rnn[0], "seq_length": 1}
            )
        _student_rnn[0] = rnn_state
        if args.dump_action_trace:
            image_hashes.append(float(image[0].abs().sum().item()))
            proprio_hashes.append(float(proprio[0].abs().sum().item()))
            action_hashes.append(float(mu[0].abs().sum().item()))
            joint_hashes.append(float(env.robot.data.joint_pos[0].abs().sum().item()))
            obj_pos = env.object.data.root_pos_w[0].detach().float().cpu().numpy().copy()
            obj_quat = env.object.data.root_quat_w[0].detach().float().cpu().numpy().copy()
            hole_pos = env.hole.data.root_pos_w[0].detach().float().cpu().numpy().copy()
            hole_quat = env.hole.data.root_quat_w[0].detach().float().cpu().numpy().copy()
            object_pos_traces.append(obj_pos)
            object_quat_traces.append(obj_quat)
            hole_pos_traces.append(hole_pos)
            hole_quat_traces.append(hole_quat)
            trace_step[0] += 1
        return mu

    # ---- Run teacher then student ----
    n_envs = int(args.num_envs)
    if args.skip_teacher:
        print("=> skipping TEACHER pass (--skip-teacher)", flush=True)
        teacher_frames = [[] for _ in range(n_envs)]
        teacher_ok = np.zeros(n_envs, dtype=bool)
        teacher_steps = np.full(n_envs, int(args.max_steps), dtype=np.int32)
        teacher_fp = {"_summary": "(skipped)"}
        teacher_partial = np.full(n_envs, float("nan"), dtype=np.float32)
        teacher_ever = np.zeros(n_envs, dtype=np.int64)
        teacher_mg = env.env_max_goals.detach().cpu().numpy()
        teacher_first_ep = np.zeros(n_envs, dtype=np.int64)
        teacher_first_partial = np.full(n_envs, float("nan"), dtype=np.float32)
        teacher_first_full = np.full(n_envs, float("nan"), dtype=np.float32)
    else:
        print(f"=> running TEACHER pass ({args.num_envs} envs, "
              f"up to {args.max_steps} steps)...", flush=True)
        t0 = time.time()
        (teacher_frames, teacher_ok, teacher_steps, teacher_fp, teacher_partial,
         teacher_ever, teacher_mg,
         teacher_first_ep, teacher_first_partial, teacher_first_full,
         _teacher_audit_unused) = _run_pass(_teacher_action, "teacher")
        print(f"   teacher done in {time.time() - t0:.1f}s  "
              f"full_success={teacher_ok.sum()}/{args.num_envs}  "
              f"mean_partial={float(teacher_partial.mean()):.3f}  "
              f"first_ep_full={float(teacher_first_full.mean()):.3f}  "
              f"first_ep_partial={float(teacher_first_partial.mean()):.3f}",
              flush=True)

    print(f"=> running STUDENT pass...", flush=True)
    _student_action_fn_reset()  # clear LSTM hidden state (no-op for non-RNN nets)
    t0 = time.time()
    (student_frames, student_ok, student_steps, student_fp, student_partial,
     student_ever, student_mg,
     student_first_ep, student_first_partial, student_first_full,
     student_audit_frames) = _run_pass(_student_action_fn, "student")
    # If --audit-tiled, hijack the (empty) teacher_frames with audit camera
    # frames captured during the student pass. The side-by-side video then
    # shows [audit (tiled) | primary (student backend)] with guaranteed
    # matched init states (same env, single rollout, two cameras).
    if args.audit_tiled and any(len(f) > 0 for f in student_audit_frames):
        teacher_frames = student_audit_frames
        print(f"=> using audit-camera frames as left half of side-by-side video",
              flush=True)
    print(f"   student done in {time.time() - t0:.1f}s  "
          f"full_success={student_ok.sum()}/{args.num_envs}  "
          f"mean_partial={float(student_partial.mean()):.3f}  "
          f"first_ep_full={float(student_first_full.mean()):.3f}  "
          f"first_ep_partial={float(student_first_partial.mean()):.3f}",
          flush=True)

    # ---- Verify the two passes started from IDENTICAL initial states ----
    det_report: dict = {"verified": True, "diffs": {}}
    if args.skip_teacher:
        det_report["verified"] = False
        det_report["diffs"] = {"skipped": "teacher pass skipped"}
        print("=> determinism check skipped (--skip-teacher)", flush=True)
    else:
      for key in ("robot_joint_pos", "object_root_pose", "hole_root_pose",
                  "goal_pos_obs_noise", "goal_yaw_obs_noise"):
        a = teacher_fp.get(key)
        b = student_fp.get(key)
        if a is None or b is None:
            det_report["diffs"][key] = "N/A (field missing)"
            continue
        diff = float(np.abs(a - b).max())
        det_report["diffs"][key] = diff
        if diff > 1e-5:
            det_report["verified"] = False
    if args.skip_teacher:
        pass  # determinism check N/A when only running student
    elif det_report["verified"]:
        print("=> ✓ deterministic reset verified: teacher and student passes "
              "started from identical initial states (max abs diff < 1e-5 "
              "across joint_pos / object / hole / goal-noise).", flush=True)
    else:
        print("=> ✗ WARNING: teacher and student passes did NOT start from "
              "identical initial states! Per-field max abs diffs:", flush=True)
        for k, v in det_report["diffs"].items():
            print(f"     {k}: {v}", flush=True)

    # ---- Compose side-by-side videos + write summary ----
    import imageio

    summary = {
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "student_checkpoint": str(args.student_checkpoint),
        "task": args.task,
        "problem": args.problem,
        "goal_mode": args.goal_mode,
        "hole_yaw_deg": float(args.hole_yaw_deg),
        "num_envs": int(args.num_envs),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "video_fps": int(args.video_fps),
        "video_size_per_policy": [int(out_w), int(args.video_height)],
        "teacher_full_success_rate": float(teacher_ok.mean()),
        "student_full_success_rate": float(student_ok.mean()),
        "teacher_partial_success_rate": float(teacher_partial.mean()),
        "student_partial_success_rate": float(student_partial.mean()),
        "teacher_first_episode_full_success_rate": float(teacher_first_full.mean()),
        "student_first_episode_full_success_rate": float(student_first_full.mean()),
        "teacher_first_episode_partial_success_rate": float(teacher_first_partial.mean()),
        "student_first_episode_partial_success_rate": float(student_first_partial.mean()),
        "teacher_ever_max_per_env": teacher_ever.tolist(),
        "student_ever_max_per_env": student_ever.tolist(),
        "teacher_first_episode_max_per_env": teacher_first_ep.tolist(),
        "student_first_episode_max_per_env": student_first_ep.tolist(),
        "env_max_goals_per_env": teacher_mg.tolist(),
        "deterministic_reset_verified": bool(det_report["verified"]),
        "deterministic_reset_max_abs_diff": {
            k: (v if isinstance(v, str) else float(v))
            for k, v in det_report["diffs"].items()
        },
        "per_env": [],
    }

    # Pre-render a label band (drawn once via PIL) for the --audit-tiled case.
    # Shows which side of the side-by-side video is which camera backend.
    label_band = None
    if args.audit_tiled:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        band_h = 22
        band_w = out_w * 2
        band_img = Image.new("RGB", (band_w, band_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(band_img)
        draw.text((6, 3), "tiled (audit cam)", fill=(255, 255, 255), font=font)
        draw.text((out_w + 6, 3), "raycaster (policy)", fill=(255, 255, 255), font=font)
        label_band = np.array(band_img)

    for i in range(int(args.num_envs)):
        tf = teacher_frames[i]
        sf = student_frames[i]
        # Pad shorter with last frame to match length (so both halves play in lockstep).
        n_t = len(tf)
        n_s = len(sf)
        n_max = max(n_t, n_s, 1)
        if n_t < n_max:
            tf = tf + [tf[-1]] * (n_max - n_t) if tf else [np.zeros((out_h, out_w, 3), dtype=np.uint8)] * n_max
        if n_s < n_max:
            sf = sf + [sf[-1]] * (n_max - n_s) if sf else [np.zeros((out_h, out_w, 3), dtype=np.uint8)] * n_max

        # Concat horizontally per frame: (H, 2W, 3).
        # Encode the first-episode outcome into the filename so videos are
        # self-labeling. For --audit-tiled: env_NN_RC{val}of{max}.mp4 (audit
        # cam is just visualization; the policy ran the primary backend, so
        # the success number reflects that). For default teacher|student
        # side-by-side: env_NN_T{val}of{max}_S{val}of{max}.mp4.
        t_first = int(teacher_first_ep[i])
        s_first = int(student_first_ep[i])
        t_mg = int(teacher_mg[i])
        s_mg = int(student_mg[i])
        if args.audit_tiled:
            out_path = out_dir / f"env_{i:02d}_RC{s_first}of{s_mg}.mp4"
        else:
            out_path = out_dir / (
                f"env_{i:02d}_T{t_first}of{t_mg}_S{s_first}of{s_mg}.mp4"
            )
        if not args.skip_videos:
            combined = [np.concatenate([t, s], axis=1) for t, s in zip(tf, sf)]
            # Prepend per-frame label band when --audit-tiled is set.
            if label_band is not None:
                combined = [np.concatenate([label_band, c], axis=0) for c in combined]
            imageio.mimwrite(str(out_path), combined, fps=int(args.video_fps),
                              quality=int(args.video_quality),
                              codec="libx264", pixelformat="yuv420p",
                              macro_block_size=1)
        summary["per_env"].append({
            "env_id": i,
            "teacher_success": bool(teacher_ok[i]),
            "teacher_steps_to_success": int(teacher_steps[i]),
            "teacher_video_frames": int(n_t),
            "student_success": bool(student_ok[i]),
            "student_steps_to_success": int(student_steps[i]),
            "student_video_frames": int(n_s),
            "video_path": str(out_path.name),
        })
        print(f"   env {i:02d}: "
              f"teacher={teacher_ever[i]}/{teacher_mg[i]} step={teacher_steps[i]:>4d}  "
              f"student={student_ever[i]}/{student_mg[i]} step={student_steps[i]:>4d}  "
              f"-> {out_path.name}",
              flush=True)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"=> wrote {summary_path}", flush=True)
    print(f"=> teacher: full_succ={summary['teacher_full_success_rate']:.2%}  "
          f"partial={summary['teacher_partial_success_rate']:.2%}  "
          f"first_ep_full={summary['teacher_first_episode_full_success_rate']:.2%}  "
          f"first_ep_partial={summary['teacher_first_episode_partial_success_rate']:.2%}",
          flush=True)
    print(f"=> student: full_succ={summary['student_full_success_rate']:.2%}  "
          f"partial={summary['student_partial_success_rate']:.2%}  "
          f"first_ep_full={summary['student_first_episode_full_success_rate']:.2%}  "
          f"first_ep_partial={summary['student_first_episode_partial_success_rate']:.2%}",
          flush=True)

    if args.dump_action_trace and image_hashes:
        np.savez_compressed(
            out_dir / "action_trace.npz",
            image_hash=np.asarray(image_hashes, dtype=np.float64),
            proprio_hash=np.asarray(proprio_hashes, dtype=np.float64),
            action_hash=np.asarray(action_hashes, dtype=np.float64),
            joint_hash=np.asarray(joint_hashes, dtype=np.float64),
            object_pos=np.stack(object_pos_traces, axis=0).astype(np.float64),
            object_quat=np.stack(object_quat_traces, axis=0).astype(np.float64),
            hole_pos=np.stack(hole_pos_traces, axis=0).astype(np.float64),
            hole_quat=np.stack(hole_quat_traces, axis=0).astype(np.float64),
        )
        print(f"=> dumped action trace: {len(image_hashes)} steps -> "
              f"{out_dir}/action_trace.npz", flush=True)

    try:
        env.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
