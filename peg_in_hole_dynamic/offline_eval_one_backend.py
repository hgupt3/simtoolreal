#!/usr/bin/env python3
"""Single-policy single-camera-backend offline eval.

Runs one batch of `num_envs` parallel rollouts for one
`cfg.student_obs.camera_backend` choice ({tiled, raycaster,
foundation_stereo}), with all DR knobs forced OFF to match the
no_delays_no_camnoise checkpoint's training distribution. Reports the
per-env max insertion success and writes a JSON result.

Driven by `offline_eval_3_backends.sh` which calls this 3 times (one per
backend) and aggregates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    p.add_argument("--problem", default="Lpeg.tol0p5mm")
    p.add_argument("--goal-mode", default="finalGoalOnly")
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--student-checkpoint", required=True)
    p.add_argument("--policy-name", required=True,
                   help="Short label (used in JSON + log).")
    p.add_argument(
        "--camera-backend",
        choices=("tiled", "raycaster", "foundation_stereo"),
        required=True,
    )
    p.add_argument(
        "--cy-match-real",
        type=int,
        default=1,
        choices=(0, 1),
        help="1: shift cy to 47.13 (real ZED HD1080 cal). 0: cy = H/2 = 45.",
    )
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--max-steps-per-episode", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    # FS-specific knobs (consumed only when --camera-backend foundation_stereo).
    p.add_argument(
        "--fs-model-dir",
        default="third_party/Fast-FoundationStereo/weights/23-36-37",
    )
    p.add_argument(
        "--fs-engine-dir",
        default="third_party/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4",
        help="If empty, falls back to PyTorch inference on the .pth.",
    )
    p.add_argument("--fs-valid-iters", type=int, default=4)
    p.add_argument("--fs-stereo-width", type=int, default=384)
    p.add_argument("--fs-stereo-height", type=int, default=224)
    p.add_argument("--output-json", required=True)
    p.add_argument(
        "--video-path",
        default=None,
        help="If set, write a mp4 of the last env's 70x70 policy-input depth "
        "across the rollout (one frame per env step).",
    )
    p.add_argument("--video-fps", type=int, default=60)
    p.add_argument("--video-upscale", type=int, default=4,
                   help="Nearest-neighbor upscale factor (70x70 -> 280x280 by "
                        "default) so the encoded mp4 has comfortable dims for "
                        "common IDE / browser players.")
    return p.parse_args()


def _reseed(seed: int) -> None:
    import numpy as np
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

    import torch
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info
    from isaacsimenvs.dagger.teacher import Teacher
    from isaacsimenvs.dagger import networks as _net  # noqa: F401

    from peg_in_hole_dynamic.eval_isaacsim import (
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
    )
    from peg_in_hole_dynamic.eval_student_isaacsim import _build_student

    cfg = _load_env_cfg(args.task)
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
        extra_overrides={},
    )

    # All DR off (matches no_delays_no_camnoise training distribution).
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.reset.table_reset_xy_range_m = (0.0, 0.0)
    cfg.reset.table_reset_yaw_range_deg = 0.0
    cfg.assets.table_scale_range_x = (1.0, 1.0)
    cfg.assets.table_scale_range_y = (1.0, 1.0)
    cfg.assets.table_scale_num_variants = 1

    # Per-backend cy offset units. See eval_student_isaacsim._resolve_v_aperture_offset:
    #   - tiled / foundation_stereo: TiledCamera consumes cm-at-sensor (real
    #     ZED HD1080 cal => 0.4418 cm shifts cy from 45 to 47.13 at 160x90).
    #   - raycaster: PinholeCameraPatternCfg consumes a dimensionless ratio
    #     ((cy - H/2) / fy_px = (47.13 - 45) / 115.67 = 0.01841).
    if bool(args.cy_match_real):
        if args.camera_backend == "raycaster":
            cfg.student_obs.vertical_aperture_offset = (47.13 - 45.0) / 115.67
        else:
            cfg.student_obs.vertical_aperture_offset = 0.4418
    else:
        cfg.student_obs.vertical_aperture_offset = 0.0
    cfg.student_obs.horizontal_aperture_offset = 0.0  # cx already at image center

    # The variable under test: camera backend.
    cfg.student_obs.camera_backend = args.camera_backend
    if args.camera_backend == "foundation_stereo":
        # Fast-FS was trained on Omniverse path-traced stereo at 32-128 SPP.
        # Our default sim render is RTX RealTime at 1 SPP with antialiasing OFF
        # (see peg_in_hole_env_cfg._default_peg_in_hole_sim_cfg). That gives
        # heavy speckle noise on the RGB output, which is uncorrelated between
        # left and right views -> destroys stereo matching. Upgrade to the
        # "quality" preset + DLAA + DL denoiser + 16 SPP so the rendered RGB
        # is in-distribution for FS without paying the full path-tracer cost.
        cfg.sim.render.rendering_mode = "quality"
        cfg.sim.render.antialiasing_mode = "DLAA"
        cfg.sim.render.enable_dl_denoiser = True
        cfg.sim.render.samples_per_pixel = 16
        cfg.sim.render.enable_direct_lighting = True
        cfg.sim.render.enable_reflections = True
        cfg.sim.render.enable_global_illumination = True
        cfg.sim.render.enable_shadows = True
        cfg.sim.render.enable_ambient_occlusion = True
        cfg.student_obs.fs_model_dir = str(args.fs_model_dir)
        cfg.student_obs.fs_engine_dir = str(args.fs_engine_dir) if args.fs_engine_dir else ""
        cfg.student_obs.fs_valid_iters = int(args.fs_valid_iters)
        cfg.student_obs.fs_stereo_width = int(args.fs_stereo_width)
        cfg.student_obs.fs_stereo_height = int(args.fs_stereo_height)

    print(
        f"=> backend={args.camera_backend}  cy_match_real={bool(args.cy_match_real)} "
        f"v_off={cfg.student_obs.vertical_aperture_offset:.5f}  "
        f"policy={args.policy_name}  problem={args.problem}  "
        f"num_envs={args.num_envs}  max_steps={args.max_steps_per_episode}  "
        f"seed={args.seed}",
        flush=True,
    )

    env = _instantiate_env(args.task, cfg)
    agent_cfg = _configure_agent(
        args.task, "rl_games_sapg_cfg_entry_point",
        rl_device=args.rl_device, num_envs=int(args.num_envs),
        deterministic=True, games=1, extra_overrides={},
    )
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=args.rl_device,
                                   clip_obs=clip_obs, clip_actions=clip_actions)

    env_info_teacher = teacher_env_info(wrapped)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    student_agent_cfg = load_cfg_from_registry(args.task, "rl_games_dagger_sapg_cfg_entry_point")
    dagger_block = student_agent_cfg["params"]["config"].get("dagger", {})
    teacher = Teacher(
        task_id=str(dagger_block.get("teacher_task_id", "Isaacsimenvs-PegInHole-Direct-v0")),
        agent_key=str(dagger_block.get("teacher_agent_key", "rl_games_sapg_cfg_entry_point")),
        checkpoint_path=args.teacher_checkpoint,
        num_envs=int(args.num_envs),
        rl_device=args.rl_device,
        env_info=env_info_teacher,
    )

    net_params = student_agent_cfg["params"]["network"]
    student_obs0 = env.get_student_obs()
    actual_proprio_dim = int(student_obs0["proprio"].shape[-1])
    image_channels = int(net_params.get("image_channels", 1))
    image_hw = tuple(net_params["image_hw"])
    has_block_id = bool(net_params.get("has_block_id", True))
    num_blocks = int(student_agent_cfg["params"]["config"].get("expl_coef_block_size", 1))
    if actual_proprio_dim != int(net_params.get("proprio_dim", -1)):
        net_params = dict(net_params)
        net_params["proprio_dim"] = actual_proprio_dim
    student = _build_student(
        net_params=net_params,
        action_dim=int(env.action_space.shape[-1]),
        image_channels=image_channels,
        image_hw=image_hw,
        proprio_dim=int(net_params["proprio_dim"]),
        has_block_id=has_block_id,
        num_blocks=max(num_blocks, 1),
        student_checkpoint=args.student_checkpoint,
        device=args.rl_device,
    )
    student.eval()

    def _student_action(rnn_state) -> tuple[torch.Tensor, tuple | None]:
        out = env.get_student_obs()
        image = out["image"]
        proprio = out["proprio"]
        if has_block_id:
            block_id = torch.zeros(image.shape[0], 1, device=image.device)
            flat = torch.cat([image.flatten(1), proprio, block_id], dim=-1)
        else:
            flat = torch.cat([image.flatten(1), proprio], dim=-1)
        with torch.no_grad():
            mu, _ls, _v, rnn_state = student({
                "obs": flat, "rnn_states": rnn_state, "seq_length": 1,
            })
        return mu, rnn_state

    # --- rollout ---
    _reseed(args.seed)
    if hasattr(student, "reset_default_state"):
        student.reset_default_state()
    teacher.reset()
    wrapped.reset()
    _reseed(args.seed)
    wrapped.reset()
    rnn_state = None
    max_succ = torch.zeros(int(args.num_envs), device=args.rl_device)
    last_env_idx = int(args.num_envs) - 1
    record_video = args.video_path is not None
    depth_frames: list = [] if record_video else None
    # The auto-reset happens inside wrapped.step(), so 600 raw steps span
    # multiple episodes per env. Cap the video recording at env_last's FIRST
    # episode (until env.reset_buf[last_env_idx] flips True the first time)
    # so the saved mp4 shows a single rollout for the user to inspect.
    env_last_first_ep_done = False
    t0 = time.time()
    for _step in range(int(args.max_steps_per_episode)):
        act, rnn_state = _student_action(rnn_state)
        wrapped.step(act)
        max_succ = torch.maximum(max_succ, env._successes.float())
        # Detect FIRST reset of the recording env. reset_buf was just
        # computed inside step(); True iff this step triggered a reset.
        if (record_video and not env_last_first_ep_done
                and bool(env.reset_buf[last_env_idx].item())):
            env_last_first_ep_done = True
        # Append the policy-input depth (70x70 in [0,1]) only while we're
        # still inside that env's first episode. Frames recorded after the
        # auto-reset would show a NEW episode's initial state.
        if record_video and not env_last_first_ep_done:
            frame_t = env.get_student_obs()["image"][last_env_idx, 0]
            depth_frames.append(frame_t.detach().float().cpu().numpy())
    dt = time.time() - t0

    max_goals = float(env.env_max_goals[0].item())
    succ_rate = (max_succ / max(max_goals, 1.0)).clamp(0.0, 1.0)
    result = {
        "policy_name": args.policy_name,
        "camera_backend": args.camera_backend,
        "cy_match_real": bool(args.cy_match_real),
        "vertical_aperture_offset": float(cfg.student_obs.vertical_aperture_offset),
        "student_checkpoint": str(args.student_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "problem": args.problem,
        "num_envs": int(args.num_envs),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "seed": int(args.seed),
        "max_goals": max_goals,
        "max_succ_per_env": max_succ.cpu().tolist(),
        "succ_rate_per_env": succ_rate.cpu().tolist(),
        "mean_succ": float(succ_rate.mean().item()),
        "rollout_seconds": float(dt),
    }
    print(
        f"=> backend={args.camera_backend}  mean_succ={result['mean_succ']:.4f}  "
        f"({dt:.1f}s)",
        flush=True,
    )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"=> wrote {args.output_json}", flush=True)

    if record_video and depth_frames:
        import imageio.v2 as imageio
        import numpy as np
        # Each frame: 70x70 in [0,1] -> uint8 grayscale -> nearest-neighbor
        # upscale -> 3-ch for h264 compatibility. The upscale gives a
        # comfortable encoded size (e.g. 280x280 at scale=4) that renders
        # reliably in VS Code, browsers, etc., without auto-padding by ffmpeg.
        u = max(1, int(args.video_upscale))
        frames_u8 = []
        for f in depth_frames:
            g = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
            g = np.repeat(np.repeat(g, u, axis=0), u, axis=1)
            frames_u8.append(np.repeat(g[..., None], 3, axis=-1))
        video_path = Path(args.video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        # macro_block_size=1 disables the auto-pad-to-16 behaviour; we already
        # upscaled to a comfortable size so the pixel-format converter works
        # cleanly. codec="libx264" + pixelformat="yuv420p" is the standard
        # h264 mp4 most players accept.
        imageio.mimwrite(
            str(video_path), frames_u8,
            fps=int(args.video_fps),
            codec="libx264", pixelformat="yuv420p", macro_block_size=1,
        )
        print(
            f"=> wrote {video_path}  ({len(frames_u8)} frames, "
            f"{int(args.video_fps)} fps, env_idx={last_env_idx}, "
            f"{frames_u8[0].shape[1]}x{frames_u8[0].shape[0]} after {u}x upscale)",
            flush=True,
        )

    try:
        env.close()
    except Exception:
        pass
    del app
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
