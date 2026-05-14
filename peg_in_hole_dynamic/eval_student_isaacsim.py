#!/usr/bin/env python3
"""Isaac Sim peg-in-hole eval with the same Viser UI as ``eval_isaacsim.py``,
plus an Action-Source dropdown that toggles between **teacher** (rl_games SAPG
checkpoint), **student** (DepthCNNLSTMBuilder + .pth or random-init), and
**zero** actions, and a depth panel that shows the student's policy-input
depth tensor live as the episode runs.

Architecture mirrors ``eval_isaacsim.py``:
  - parent process owns Viser (subclass of ``eval_isaacgym.PegDynamicDemo``)
  - child sim worker is launched via ``--worker`` and talks to the parent
    over a ``multiprocessing.connection`` Pipe
  - the worker loads BOTH the teacher (rl_games player) and the student
    (custom rl_games network), and per-step picks the action by looking at
    a shared ``action_source`` updated via a parent → child message.

Single-process is not viable because Isaac Sim's Kit init is one-shot per
process, so swapping problem/checkpoint requires spawning a fresh worker —
exactly the same constraint that drove ``eval_isaacsim.py``'s split.

Examples:
    .venv_isaacsim/bin/python peg_in_hole_dynamic/eval_student_isaacsim.py \\
        --teacher-checkpoint .../last/model.pth \\
        --student-checkpoint .../student.pth   # optional; else random-init
        --problem Lpeg.tol0p5mm --port 8081
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default home for the 5 cam-noise student checkpoints copied from train_dir
# by hardware_rollouts/2026-05-13_camera_noise_checkpoints/copy_from_train_dir.sh.
# Each subdir holds model.pth + metadata.json; metadata.json's
# training_toggles block populates the GUI checkbox defaults when the
# Student-Policy dropdown changes.
DEFAULT_STUDENT_POLICIES_DIR = (
    REPO_ROOT / "hardware_rollouts" / "2026-05-13_camera_noise_checkpoints"
)

# Dropdown order + fallback toggle defaults if a policy's metadata.json is
# missing or unreadable. Keys must match the subdir names produced by the
# copy script.
STUDENT_POLICY_ORDER: tuple[str, ...] = (
    "no_delays_no_camnoise",
    "camrand_off_depthaug_off",
    "camrand_on_depthaug_off",
    "camrand_off_depthaug_on",
    "camrand_on_depthaug_on",
)
STUDENT_POLICY_FALLBACK_TOGGLES: dict[str, dict[str, bool]] = {
    "no_delays_no_camnoise":    {"delays": False, "camera_pose_rand": False, "depth_aug": False},
    "camrand_off_depthaug_off": {"delays": True,  "camera_pose_rand": False, "depth_aug": False},
    "camrand_on_depthaug_off":  {"delays": True,  "camera_pose_rand": True,  "depth_aug": False},
    "camrand_off_depthaug_on":  {"delays": True,  "camera_pose_rand": False, "depth_aug": True},
    "camrand_on_depthaug_on":   {"delays": True,  "camera_pose_rand": True,  "depth_aug": True},
}

# Table domain-randomization dropdown choices. Values map labels -> (override
# value). xy is the per-axis half-width in meters; yaw is the half-width in
# degrees; scale is (range_x, range_y) feeding the MultiUsd variant baker.
TABLE_XY_CHOICES: dict[str, tuple[float, float]] = {
    "off":   (0.0, 0.0),
    "1 cm":  (0.01, 0.01),
    "3 cm":  (0.03, 0.03),
    "5 cm":  (0.05, 0.05),
}
TABLE_YAW_CHOICES: dict[str, float] = {
    "off":   0.0,
    "2 deg": 2.0,
    "5 deg": 5.0,
    "10 deg": 10.0,
}
TABLE_SCALE_CHOICES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "off":    ((1.0, 1.0), (1.0, 1.0)),
    "+/- 5%":  ((0.95, 1.05), (0.95, 1.05)),
    "+/- 15%": ((0.85, 1.15), (0.85, 1.15)),
    "+/- 25%": ((0.75, 1.25), (0.75, 1.25)),
}
# Number of pre-baked USD variants when scale is on. Round-robined per env.
TABLE_SCALE_N_VARIANTS_DEFAULT = 10

# Init-pose dropdown. Values mirror the training-side fixed-init subs in
# isaacsimenvs/final_experiments/play2win/dagger/fixed_init_ablations/*.sub
# (FIXED_START_POSE / HOLE_*_RANGE / RESET_* / TABLE_RESET_Z_RANGE).
#
#   - "random": task-yaml defaults => object xy + hole xy + robot joints +
#     table z all sampled per-env at reset.
#   - "fixed":  every reset-time randomness clamped to a single point so the
#     env reproduces the canonical scene the fixed-init policies trained on.
_INIT_RESET_NOISE_DEFAULTS = dict(
    reset_position_noise_x=0.1,
    reset_position_noise_y=0.1,
    reset_position_noise_z=0.02,
    reset_dof_pos_random_interval_arm=0.1,
    reset_dof_pos_random_interval_fingers=0.1,
    reset_dof_vel_random_interval=0.5,
    table_reset_z_range=0.01,
)
_INIT_RESET_NOISE_FIXED = dict(
    reset_position_noise_x=0.0,
    reset_position_noise_y=0.0,
    reset_position_noise_z=0.0,
    reset_dof_pos_random_interval_arm=0.0,
    reset_dof_pos_random_interval_fingers=0.0,
    reset_dof_vel_random_interval=0.0,
    table_reset_z_range=0.0,
)
INIT_CHOICES: dict[str, dict] = {
    "random": {
        "fixed_start_pose": None,
        "hole_x_range": [-0.1875, 0.1875],
        "hole_y_range": [-0.10, 0.10],
        "reset_noise": _INIT_RESET_NOISE_DEFAULTS,
    },
    "fixed": {
        "fixed_start_pose": [-0.10, 0.0, 0.63, 1.0, 0.0, 0.0, 0.0],
        "hole_x_range": [0.10, 0.10],
        "hole_y_range": [0.0, 0.0],
        "reset_noise": _INIT_RESET_NOISE_FIXED,
    },
}


# Reuse helpers from the teacher eval. They're already battle-tested for env
# bootstrap, hydra cfg loading, and rl_games player setup.
from peg_in_hole_dynamic.eval_isaacsim import (  # noqa: E402
    CONTROL_DT,
    DEFAULT_AGENT,
    DEFAULT_PROBLEM,
    DEFAULT_TASK,
    GOAL_MODES,
    PopenAdapter,
    _apply_env_overrides,
    _configure_agent,
    _coerce_override_value,
    _default_isaacsim_python,
    _done0,
    _instantiate_env,
    _load_env_cfg,
    _resolve_path,
    _sim_get_state,
    _tensor_bool,
    _tensor_int,
)


# =====================================================================
# CHILD PROCESS — sim worker
# =====================================================================


def _student_obs_flat(env, image_channels: int, image_hw: tuple,
                      proprio_dim: int, has_block_id: bool):
    """Build the flat obs tensor the student net expects from get_student_obs."""
    import torch
    out = env.unwrapped.get_student_obs()
    image = out["image"]                                                  # (N, C, H, W)
    proprio = out["proprio"]                                              # (N, P)
    if has_block_id:
        # No SAPG block-id signal in the eval env; assume block 0.
        block_id = torch.zeros(image.shape[0], 1, device=image.device)
        flat = torch.cat([image.flatten(1), proprio, block_id], dim=-1)
    else:
        flat = torch.cat([image.flatten(1), proprio], dim=-1)
    return flat, image, proprio


def _depth_to_rgb_uint8(depth_t, depth_min: float, depth_max: float):
    """Map a 2D depth tensor to a (H, W, 3) uint8 grayscale RGB image."""
    import numpy as np
    d = depth_t.detach().float().clamp(min=depth_min, max=depth_max)
    norm = (d - depth_min) / max(depth_max - depth_min, 1e-6)
    arr = (norm.cpu().numpy() * 255.0).astype("uint8")
    return np.stack([arr, arr, arr], axis=-1)


def _normalized_to_rgb_uint8(norm_t):
    """Map a 2D [0,1] normalized depth tensor to a (H, W, 3) uint8 grayscale image."""
    import numpy as np
    d = norm_t.detach().float().clamp(0.0, 1.0)
    arr = (d.cpu().numpy() * 255.0).astype("uint8")
    return np.stack([arr, arr, arr], axis=-1)


def _build_student(net_params: dict, action_dim: int, image_channels: int,
                   image_hw: tuple, proprio_dim: int, has_block_id: bool,
                   num_blocks: int, student_checkpoint: str | None, device: str):
    """Build the depth_cnn_lstm rl_games network and (optionally) load weights."""
    import torch
    from rl_games.algos_torch import model_builder
    # Importing isaacsimenvs.dagger.networks registers 'depth_cnn_lstm'.
    from isaacsimenvs.dagger import networks as _net  # noqa: F401

    builder = model_builder.NETWORK_REGISTRY["depth_cnn_lstm"]()
    builder.load(net_params)
    obs_dim = (
        image_channels * image_hw[0] * image_hw[1]
        + proprio_dim
        + (1 if has_block_id else 0)
    )
    build_kwargs = {"actions_num": action_dim, "input_shape": (obs_dim,)}
    if net_params.get("space", {}).get("continuous", {}).get("fixed_sigma") == "coef_cond":
        build_kwargs["coef_ids"] = torch.arange(num_blocks, dtype=torch.float32)
        build_kwargs["coef_id_idx"] = obs_dim - 1
    student = builder.build("student", **build_kwargs).to(device)
    if student_checkpoint:
        # weights_only=False because rl_games checkpoints stash a few
        # numpy scalars (e.g. running_mean_std counts) alongside the tensor
        # state_dict; PyTorch 2.6's default weights_only=True refuses to
        # unpickle those. The checkpoint comes from our own train_dir so
        # the loosened policy is acceptable.
        sd = torch.load(student_checkpoint, map_location=device, weights_only=False)
        # SAPG rl_games checkpoints are nested: {<block_id>: {"model": state_dict, ...}}.
        # Old non-SAPG checkpoints are flat: {"model": state_dict, ...}. Handle both.
        if isinstance(sd, dict) and "model" not in sd and any(isinstance(k, int) for k in sd):
            block_keys = [k for k in sd if isinstance(k, int)]
            sd = sd[min(block_keys)]  # SAPG always trains block 0 as the canonical actor
        model_sd = sd.get("model", sd)
        # rl_games saves the full model wrapper (`ModelA2CContinuousLogStd`),
        # which prefixes the inner Network with `a2c_network.` and stores the
        # value/obs normalizers under their own top-level prefixes. We built
        # the inner `DepthCNNLSTMBuilder.Network` directly, so strip the
        # `a2c_network.` prefix and drop the normalizer keys (we don't carry
        # them on the bare Network module).
        prefix = "a2c_network."
        stripped: dict[str, torch.Tensor] = {}
        skipped_norm = 0
        for k, v in model_sd.items():
            if k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            elif k.startswith(("value_mean_std.", "running_mean_std.")):
                skipped_norm += 1
            else:
                stripped[k] = v
        net_keys = set(student.state_dict().keys())
        ckpt_keys = set(stripped.keys())
        missing = sorted(net_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - net_keys)
        missing_report = (
            f"\n  missing ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}"
            if missing else ""
        )
        unexpected_report = (
            f"\n  unexpected ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}"
            if unexpected else ""
        )
        # strict=True so any future shape mismatch lands as a real error instead
        # of a silently random-init student.
        student.load_state_dict(stripped, strict=False)
        loaded = len(net_keys & ckpt_keys)
        print(
            f"=> loaded {loaded}/{len(net_keys)} student weights from "
            f"'{student_checkpoint}' "
            f"(skipped {skipped_norm} normalizer keys){missing_report}{unexpected_report}",
            flush=True,
        )
        if loaded == 0:
            raise RuntimeError(
                "Student state_dict load matched zero keys after stripping the "
                "`a2c_network.` prefix; the checkpoint architecture probably "
                "doesn't match the yaml's depth_cnn_lstm config. Refusing to "
                "run with a fully random-init student."
            )
    else:
        print("=> student is random-init", flush=True)
    student.eval()
    return student


def _extract_teacher_obs(obs):
    """Pull the 140-d teacher state out of the rl_games wrapper's dict obs.

    The DAgger-aware wrapper returns ``{"obs": student_4987d, "states": ...,
    "teacher": teacher_140d}`` from reset/step. Different rl_games / Isaac Lab
    versions wrap that in tuples; handle the common shapes.
    """
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if "teacher" in obs:
            return obs["teacher"]
        if "teacher_obs" in obs:
            return obs["teacher_obs"]
        # Some wrappers nest the dict under "obs"
        inner = obs.get("obs")
        if isinstance(inner, dict) and "teacher" in inner:
            return inner["teacher"]
    raise RuntimeError(
        "Could not find teacher obs in wrapped env output. Expected a dict "
        "with a 'teacher' or 'teacher_obs' key. Got keys: "
        f"{list(obs.keys()) if isinstance(obs, dict) else type(obs)}"
    )


def _student_episode(
    conn,
    env,
    wrapped,
    teacher,
    student,
    *,
    deterministic: bool,
    action_source: str,
    image_channels: int,
    image_hw: tuple,
    proprio_dim: int,
    has_block_id: bool,
    depth_min: float,
    depth_max: float,
    env_id: int,
    depth_send_every: int,
):
    """Episode loop: pick teacher / student / zero per step and send live
    state + a depth frame back to the launcher GUI.

    Teacher reads ``obs["teacher"]`` (140-d state) each step; student reads
    the image+proprio path through ``env.get_student_obs()`` like training.
    """
    import torch
    teacher.reset()
    reset_out = wrapped.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    rnn_state = None  # student LSTM state, lazily initialized via the network's defaults
    # Send the new env_id table pose so viser tracks the sampled xy/yaw at
    # every episode start. Cheap (one cpu transfer of 7 floats).
    try:
        origin = env.scene.env_origins[int(env_id)]
        pos = (env.table.data.root_pos_w[int(env_id)] - origin).cpu().tolist()
        quat = env.table.data.root_quat_w[int(env_id)].cpu().tolist()
        conn.send(("table_pose", *pos, *quat))
    except Exception as exc:
        print(f"[worker] table_pose report (per-episode) failed: {exc!r}", flush=True)

    paused = False
    step = 0
    done = False
    peak_successes = 0
    max_goals_seen = max(1, _tensor_int(env.env_max_goals[0]))
    retract_ok = False

    while not done:
        # Drain command messages — including action-source updates.
        while conn.poll(0):
            cmd = conn.recv()
            if isinstance(cmd, tuple) and len(cmd) >= 1 and cmd[0] == "set_action_source":
                action_source = str(cmd[1])
                print(f"[worker] action_source -> {action_source}", flush=True)
                continue
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return None
            elif cmd == "quit":
                return "quit"

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()

        # --- teacher action (always computed; powers the L2 overlay) ---
        teacher_obs = _extract_teacher_obs(obs)
        teacher_action = teacher.get_action(teacher_obs)

        # --- student action ---
        flat_obs, image_t, proprio_t = _student_obs_flat(
            env, image_channels, image_hw, proprio_dim, has_block_id,
        )
        with torch.no_grad():
            mu, _logstd, _value, rnn_state = student(
                {"obs": flat_obs, "rnn_states": rnn_state, "seq_length": 1}
            )
            student_action = mu

        if action_source == "teacher":
            act = teacher_action
        elif action_source == "student":
            act = student_action
        else:  # zero
            act = torch.zeros_like(teacher_action)

        step_out = wrapped.step(act)
        # rl_games envs return either (obs, rew, dones, info) or
        # (obs, rew, terminations, truncations, info). Handle both.
        if len(step_out) == 4:
            obs, _rew, dones, _infos = step_out
        else:
            obs, _rew, terms, truncs, _infos = step_out
            try:
                dones = terms | truncs
            except TypeError:
                dones = [bool(t) or bool(tr) for t, tr in zip(terms, truncs)]
        done = _done0(dones)
        step += 1

        cur_succ = _tensor_int(env._successes[0])
        cur_max = _tensor_int(env.env_max_goals[0])
        peak_successes = max(peak_successes, cur_succ)
        max_goals_seen = max(max_goals_seen, cur_max)
        retract_ok = retract_ok or _tensor_bool(env.retract_succeeded[0])
        state = _sim_get_state(env, done_pending=done)
        conn.send(("state", state, cur_succ, cur_max, step, retract_ok))

        if step % depth_send_every == 0:
            policy_rgb = _normalized_to_rgb_uint8(image_t[env_id, 0])
            l2 = float((student_action - teacher_action).pow(2).mean().sqrt().detach().cpu())
            conn.send(("depth", policy_rgb, l2, action_source))

        sleep = CONTROL_DT - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

    goal_pct = 100.0 * peak_successes / max(1, max_goals_seen)
    conn.send(("done", goal_pct, step, retract_ok))
    return None


def sim_worker(
    conn,
    *,
    teacher_checkpoint_path: str,
    student_checkpoint_path: str | None,
    task: str,
    agent: str,
    student_agent: str,
    problem: str,
    goal_mode: str,
    random_goal_fraction: float,
    insertion_success_tolerance: float,
    retract_success_tolerance: float,
    num_envs: int,
    games: int,
    rl_device: str,
    sim_device: str,
    deterministic: bool,
    headless: bool,
    sdf: bool,
    keep_dr: bool,
    extra_overrides: dict,
    initial_action_source: str,
    env_id: int,
    depth_send_every: int,
) -> None:
    app = None
    env = None
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

        from isaaclab.app import AppLauncher

        launcher_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(launcher_parser)
        launcher_args, _ = launcher_parser.parse_known_args([])
        launcher_args.headless = bool(headless)
        # Cameras must be enabled — the env's depth pipeline needs the TiledCamera.
        launcher_args.enable_cameras = True
        app = AppLauncher(launcher_args).app

        import isaacsimenvs  # noqa: F401  triggers gym.register
        from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
        from rl_games.torch_runner import Runner, _load_checkpoint_weights

        cfg = _load_env_cfg(task)
        _apply_env_overrides(
            cfg,
            problem=problem,
            goal_mode=goal_mode,
            random_goal_fraction=random_goal_fraction,
            insertion_success_tolerance=insertion_success_tolerance,
            retract_success_tolerance=retract_success_tolerance,
            num_envs=num_envs,
            sim_device=sim_device,
            sdf=sdf,
            keep_dr=keep_dr,
            extra_overrides=extra_overrides,
        )
        env = _instantiate_env(task, cfg)

        agent_cfg = _configure_agent(
            task,
            agent,
            rl_device=rl_device,
            num_envs=num_envs,
            deterministic=deterministic,
            games=games,
            extra_overrides=extra_overrides,
        )
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
        wrapped = register_rlgames_env(
            env, rl_device=rl_device, clip_obs=clip_obs, clip_actions=clip_actions
        )

        # ---- teacher player (rl_games SAPG) ----
        # Build the teacher against the env's *teacher* obs space (140-d state),
        # not the student env's full 4987-d obs. The student env exposes both
        # spaces via the DAgger-aware wrapper and ``teacher_env_info`` reads
        # ``wrapped.teacher_obs_space`` for the correct shape.
        from isaacsimenvs.dagger.teacher import Teacher
        from isaacsimenvs.utils.rlgames_utils import teacher_env_info

        if not hasattr(wrapped, "teacher_obs_space"):
            raise RuntimeError(
                "wrapped env has no `teacher_obs_space`; this eval requires a "
                "depth-student task whose env exposes both student and teacher "
                "obs (Isaacsimenvs-PegInHoleDepthStudent-Direct-v0)."
            )
        env_info_teacher = teacher_env_info(wrapped)
        # Read teacher_task_id / teacher_agent_key from the student yaml's
        # dagger block (matches what DAggerA2CAgent uses at training time).
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        student_agent_cfg_pre = load_cfg_from_registry(task, student_agent)
        dagger_block = student_agent_cfg_pre["params"]["config"].get("dagger", {})
        teacher_task_id = str(dagger_block.get("teacher_task_id",
                                               "Isaacsimenvs-PegInHole-Direct-v0"))
        teacher_agent_key = str(dagger_block.get("teacher_agent_key",
                                                 "rl_games_sapg_cfg_entry_point"))
        teacher = Teacher(
            task_id=teacher_task_id,
            agent_key=teacher_agent_key,
            checkpoint_path=teacher_checkpoint_path,
            num_envs=num_envs,
            rl_device=rl_device,
            env_info=env_info_teacher,
        )
        print(f"=> teacher loaded from '{teacher_checkpoint_path}'", flush=True)

        # ---- student (depth_cnn_lstm) ----
        # Build the student net from the dagger train yaml's network params.
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        student_agent_cfg = load_cfg_from_registry(task, student_agent)
        net_params = student_agent_cfg["params"]["network"]
        # Pull image / proprio shapes off the live env so we can sanity-check
        # the yaml against actual env output.
        student_obs0 = env.get_student_obs()
        actual_proprio_dim = int(student_obs0["proprio"].shape[-1])
        actual_image_shape = tuple(student_obs0["image"].shape[1:])           # (C, H, W)
        image_channels = int(net_params.get("image_channels", 1))
        image_hw = tuple(net_params["image_hw"])
        has_block_id = bool(net_params.get("has_block_id", True))
        num_blocks = int(student_agent_cfg["params"]["config"].get("expl_coef_block_size", 1))
        if actual_proprio_dim != int(net_params.get("proprio_dim", -1)):
            print(
                f"[worker] WARNING: yaml proprio_dim={net_params.get('proprio_dim')} but env "
                f"emits proprio_dim={actual_proprio_dim}. Overriding net_params.",
                flush=True,
            )
            net_params = dict(net_params)
            net_params["proprio_dim"] = actual_proprio_dim

        action_dim = int(env.action_space.shape[-1])
        student = _build_student(
            net_params=net_params,
            action_dim=action_dim,
            image_channels=image_channels,
            image_hw=image_hw,
            proprio_dim=int(net_params["proprio_dim"]),
            has_block_id=has_block_id,
            num_blocks=max(num_blocks, 1),
            student_checkpoint=student_checkpoint_path,
            device=rl_device,
        )

        depth_min = float(cfg.student_obs.depth_min_m)
        depth_max = float(cfg.student_obs.depth_max_m)

        teacher.reset()
        # Prime the env so subsequent calls return the dict obs.
        _ = wrapped.reset()
        conn.send(("ready", _sim_get_state(env)))
        # Report the actual (sx, sy) for the displayed env so viser can resize
        # its static table mesh to match. Round-robin assignment: env_id maps
        # to variant_idx = env_id % len(variants). Trivial scale -> (1, 1).
        try:
            variant_scales = getattr(env, "_table_variant_scales", [(1.0, 1.0)])
            sx, sy = variant_scales[int(env_id) % max(len(variant_scales), 1)]
            conn.send(("table_scale", float(sx), float(sy)))
        except Exception as exc:
            print(f"[worker] table_scale report failed: {exc!r}", flush=True)
        try:
            # Report env_id's actual table pose (env-local frame) so viser can
            # move its /table frame to the sampled xy + yaw. Updated again at
            # each reset (see "reset_done" handler in _student_episode).
            origin = env.scene.env_origins[int(env_id)]
            pos = (env.table.data.root_pos_w[int(env_id)] - origin).cpu().tolist()
            quat = env.table.data.root_quat_w[int(env_id)].cpu().tolist()
            conn.send(("table_pose", *pos, *quat))
        except Exception as exc:
            print(f"[worker] table_pose report failed: {exc!r}", flush=True)

        action_source = initial_action_source
        while True:
            cmd = conn.recv()
            if isinstance(cmd, tuple) and cmd[0] == "set_action_source":
                action_source = str(cmd[1])
                print(f"[worker] action_source -> {action_source}", flush=True)
                continue
            if cmd == "run":
                result = _student_episode(
                    conn,
                    env,
                    wrapped,
                    teacher,
                    student,
                    deterministic=deterministic,
                    action_source=action_source,
                    image_channels=image_channels,
                    image_hw=image_hw,
                    proprio_dim=int(net_params["proprio_dim"]),
                    has_block_id=has_block_id,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    env_id=env_id,
                    depth_send_every=depth_send_every,
                )
                if result == "quit":
                    break
            elif cmd == "quit":
                break
    except Exception as exc:
        try:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
        except Exception:
            pass
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            if app is not None:
                del app
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


# =====================================================================
# PARENT PROCESS — viser GUI (subclass of PegDynamicDemo)
# =====================================================================


class _SingletonDropdownStub:
    """Stand-in for self._dd_policy when we hide the base-class teacher dropdown.

    Some base-class methods read self._dd_policy.value to look up a teacher
    checkpoint by name. We always ship a single teacher here, so this stub
    keeps `.value` pinned to that one name and lets the existing code paths
    keep working without a real GUI widget.
    """

    def __init__(self, value: str) -> None:
        self.value = value


def _load_student_policies(policies_dir: Path) -> dict[str, dict]:
    """Discover the on-disk subset of STUDENT_POLICY_ORDER under policies_dir.

    Returns a name -> {ckpt: Path, toggles: dict, metadata: dict | None}
    mapping in STUDENT_POLICY_ORDER order. Skips entries with no model.pth
    so the dropdown only shows runnable policies. metadata.json is optional;
    when absent or unparseable, falls back to STUDENT_POLICY_FALLBACK_TOGGLES.
    """
    available: dict[str, dict] = {}
    for name in STUDENT_POLICY_ORDER:
        sub = policies_dir / name
        ckpt = sub / "model.pth"
        if not ckpt.is_file():
            continue
        meta_path = sub / "metadata.json"
        toggles = dict(STUDENT_POLICY_FALLBACK_TOGGLES.get(name, {}))
        metadata: dict | None = None
        if meta_path.is_file():
            try:
                with meta_path.open() as fp:
                    metadata = json.load(fp)
                meta_toggles = metadata.get("training_toggles") if isinstance(metadata, dict) else None
                if isinstance(meta_toggles, dict):
                    for key in ("delays", "camera_pose_rand", "depth_aug"):
                        if key in meta_toggles:
                            toggles[key] = bool(meta_toggles[key])
            except Exception as exc:
                print(f"[eval_student_isaacsim] warning: failed to parse {meta_path}: {exc}")
        # Final fallback for required keys
        for key in ("delays", "camera_pose_rand", "depth_aug"):
            toggles.setdefault(key, False)
        available[name] = {"ckpt": ckpt, "toggles": toggles, "metadata": metadata}
    return available


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0",
                        help="Must be a depth-student task (env exposes get_student_obs() + has dagger entry points).")
    parser.add_argument("--agent", default=DEFAULT_AGENT,
                        help="Hydra entry point for the *teacher* train yaml.")
    parser.add_argument("--student-agent", default="rl_games_dagger_sapg_cfg_entry_point",
                        help="Hydra entry point for the *student* train yaml. The network "
                             "block is read from here.")
    parser.add_argument("--python", default=None, help="Python executable for the Isaac Sim worker.")
    parser.add_argument("--teacher-checkpoint", required=True,
                        help="Frozen teacher .pth (rl_games format).")
    parser.add_argument("--student-checkpoint", default=None,
                        help="Optional explicit student .pth. If omitted, the Student-Policy "
                             "dropdown picks from --student-policies-dir.")
    parser.add_argument(
        "--student-policies-dir",
        default=None,
        help=(
            "Directory containing per-policy subdirs (model.pth + metadata.json). "
            f"Defaults to {DEFAULT_STUDENT_POLICIES_DIR}. Populates the Student-Policy "
            "dropdown; subdir order follows STUDENT_POLICY_ORDER."
        ),
    )
    parser.add_argument(
        "--initial-student-policy",
        default=None,
        help="Pre-select this Student-Policy dropdown entry. Defaults to the first "
             "STUDENT_POLICY_ORDER entry that has a model.pth in the policies dir.",
    )
    parser.add_argument("--problem", default=None)
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="preInsertAndFinal")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--random-goal-fraction", type=float, default=0.0)
    parser.add_argument("--insertion-success-tolerance", type=float, default=0.005)
    parser.add_argument("--retract-success-tolerance", type=float, default=0.005)
    parser.add_argument("--eval-success-tolerance", type=float, default=None)
    parser.add_argument("--sdf", action="store_true")
    parser.add_argument("--keep-dr", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--initial-action-source", choices=("teacher", "student", "zero"),
                        default="student")
    parser.add_argument("--env-id", type=int, default=0,
                        help="Which env's depth obs to display in the viser image panel.")
    parser.add_argument("--depth-send-every", type=int, default=4,
                        help="Send the depth panel update every N env steps.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--override", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
        help="Extra env/agent override, e.g. --override env.reward.lifting_bonus 0.0",
    )

    # Hidden worker mode args.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--worker-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-authkey", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--override-json", default="{}", help=argparse.SUPPRESS)
    return parser


def _worker_main(args) -> int:
    from multiprocessing.connection import Client

    conn = Client(
        (args.worker_host, int(args.worker_port)),
        authkey=bytes.fromhex(args.worker_authkey),
    )
    extra_overrides = json.loads(args.override_json or "{}")
    sim_worker(
        conn,
        teacher_checkpoint_path=args.teacher_checkpoint,
        student_checkpoint_path=args.student_checkpoint,
        task=args.task,
        agent=args.agent,
        student_agent=args.student_agent,
        problem=args.problem,
        goal_mode=args.goal_mode,
        random_goal_fraction=args.random_goal_fraction,
        insertion_success_tolerance=args.insertion_success_tolerance,
        retract_success_tolerance=args.retract_success_tolerance,
        num_envs=args.num_envs,
        games=args.games,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        deterministic=args.deterministic,
        headless=args.worker_headless,
        sdf=args.sdf,
        keep_dr=args.keep_dr,
        extra_overrides=extra_overrides,
        initial_action_source=args.initial_action_source,
        env_id=args.env_id,
        depth_send_every=args.depth_send_every,
    )
    return 0


def _run_viewer(args) -> int:
    from multiprocessing.connection import Listener

    import numpy as np
    import peg_in_hole_dynamic.eval_isaacgym as gym_eval

    teacher_ckpt = _resolve_path(args.teacher_checkpoint)
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"--teacher-checkpoint not found: {teacher_ckpt}")

    # Discover the on-disk student-policy registry.
    policies_dir = (
        _resolve_path(args.student_policies_dir)
        if args.student_policies_dir
        else DEFAULT_STUDENT_POLICIES_DIR
    )
    student_policies = _load_student_policies(policies_dir) if policies_dir.is_dir() else {}
    if student_policies:
        print(
            f"[eval_student_isaacsim] {len(student_policies)} student policies discovered "
            f"under {policies_dir}: {list(student_policies)}"
        )
    else:
        print(
            f"[eval_student_isaacsim] no student policies found under {policies_dir} "
            "(run hardware_rollouts/2026-05-13_camera_noise_checkpoints/copy_from_train_dir.sh "
            "to populate)"
        )

    # Resolve initial student checkpoint: explicit --student-checkpoint wins,
    # else --initial-student-policy, else first available, else None (random-init).
    initial_student_name: str | None = None
    student_ckpt: Path | None = None
    if args.student_checkpoint:
        student_ckpt = _resolve_path(args.student_checkpoint)
        if not student_ckpt.is_file():
            raise FileNotFoundError(f"--student-checkpoint not found: {student_ckpt}")
    elif args.initial_student_policy:
        if args.initial_student_policy not in student_policies:
            raise FileNotFoundError(
                f"--initial-student-policy={args.initial_student_policy!r} "
                f"not in discovered set {list(student_policies)}"
            )
        initial_student_name = args.initial_student_policy
        student_ckpt = student_policies[initial_student_name]["ckpt"]
    elif student_policies:
        initial_student_name = next(iter(student_policies))
        student_ckpt = student_policies[initial_student_name]["ckpt"]

    initial_problem = args.problem or DEFAULT_PROBLEM
    extra_overrides = {key: _coerce_override_value(value) for key, value in args.override}
    if args.eval_success_tolerance is not None:
        extra_overrides["env.termination.eval_success_tolerance"] = float(
            args.eval_success_tolerance
        )

    # The base-class `policies` dict expects (config_path, checkpoint_path) tuples.
    # We register one entry under the teacher checkpoint name; the student/teacher/zero
    # selection happens via our extra dropdown rather than the base "Policy" dropdown.
    policy_name = teacher_ckpt.parent.name or "teacher"
    viewer_policies = {policy_name: ("", str(teacher_ckpt))}

    class StudentEvalDemo(gym_eval.PegDynamicDemo):
        def __init__(self, *demo_args, **demo_kwargs):
            self.task = args.task
            self.agent = args.agent
            self.student_agent = args.student_agent
            self.worker_python = args.python or _default_isaacsim_python()
            self.num_envs = int(args.num_envs)
            self.games = int(args.games)
            self.rl_device = args.rl_device
            self.sim_device = args.sim_device
            self.deterministic = bool(args.deterministic)
            self.sdf = bool(args.sdf)
            self.keep_dr = bool(args.keep_dr)
            self.teacher_ckpt = teacher_ckpt
            self.student_ckpt = student_ckpt
            self.student_policies = student_policies
            # Initial dropdown selection (None when no policies discovered):
            self.student_policy_name = initial_student_name
            # Initial toggle states — driven by the active policy's metadata
            # (if any). When no policy is selected, default to all-off so the
            # env matches the random-init case.
            init_toggles = (
                student_policies[initial_student_name]["toggles"]
                if initial_student_name is not None
                else {"delays": False, "camera_pose_rand": False, "depth_aug": False}
            )
            self.toggle_delays = bool(init_toggles["delays"])
            self.toggle_camera_pose_rand = bool(init_toggles["camera_pose_rand"])
            self.toggle_depth_aug = bool(init_toggles["depth_aug"])
            # Table-DR dropdown choices — default to off, user picks via GUI.
            # Not auto-populated from policy metadata (no training_toggles yet).
            self.table_xy_choice = "off"
            self.table_yaw_choice = "off"
            self.table_scale_choice = "off"
            self.init_choice = "random"
            self.action_source = args.initial_action_source
            self.env_id = int(args.env_id)
            self.depth_send_every = int(args.depth_send_every)
            super().__init__(*demo_args, **demo_kwargs)
            self._sl_insertion_tol.value = float(args.insertion_success_tolerance)
            self._sl_retract_tol.value = float(args.retract_success_tolerance)
            # Note: _build_gui is called from super().__init__; the override
            # below adds my top sections first then defers to the base class.

        def _build_gui(self):
            # One Controls panel up top with EVERYTHING the user picks before
            # clicking Load env (action source, student policy, task selection,
            # env toggles), the Load button, and the live status markdowns
            # right below it. Then the depth obs panel, the play/pause/stop
            # controls, and the display checkboxes follow underneath.
            with self.server.gui.add_folder("Controls", expand_by_default=True):
                # === policy + action selection ===
                self._dd_action_source = self.server.gui.add_dropdown(
                    "Action source",
                    ("teacher", "student", "zero"),
                    initial_value=self.action_source,
                )
                self._dd_action_source.on_update(lambda _: self._cmd_set_action_source())
                if self.student_policies:
                    self._dd_student_policy = self.server.gui.add_dropdown(
                        "Student policy",
                        tuple(self.student_policies),
                        initial_value=self.student_policy_name,
                    )
                    self._dd_student_policy.on_update(lambda _: self._cmd_set_student_policy())
                else:
                    self._dd_student_policy = None

                # === task selection (moved here from base class) ===
                self._dd_problem = self.server.gui.add_dropdown(
                    "Problem", options=self.problem_names, initial_value=self.problem_name,
                )
                # Base class also creates self._dd_policy; we ship only one
                # teacher checkpoint here, so skip the redundant dropdown.
                # Provide a stub so any base-class code that consults
                # self._dd_policy.value still works.
                self._dd_policy = _SingletonDropdownStub(
                    next(iter(self.policies)) if self.policies else "teacher"
                )
                self._dd_goal_mode = self.server.gui.add_dropdown(
                    "Goal mode", options=GOAL_MODES, initial_value=self.goal_mode,
                )
                self._sl_rgf = self.server.gui.add_slider(
                    "Random goal frac", min=0.0, max=1.0, step=0.1,
                    initial_value=self.random_goal_fraction,
                )
                self._sl_insertion_tol = self.server.gui.add_slider(
                    "Insertion tol (m)", min=0.001, max=0.02, step=0.001,
                    initial_value=0.01,
                )
                self._sl_retract_tol = self.server.gui.add_slider(
                    "Retract tol (m)", min=0.001, max=0.01, step=0.001,
                    initial_value=0.005,
                )

                # === env toggles ===
                self._cb_delays = self.server.gui.add_checkbox(
                    "Obs/action/camera delays (max=3)",
                    initial_value=self.toggle_delays,
                )
                self._cb_camera_pose_rand = self.server.gui.add_checkbox(
                    "Camera pose randomization",
                    initial_value=self.toggle_camera_pose_rand,
                )
                self._cb_depth_aug = self.server.gui.add_checkbox(
                    "Depth-image noise",
                    initial_value=self.toggle_depth_aug,
                )

                # === table DR ===
                self._dd_table_xy = self.server.gui.add_dropdown(
                    "Table xy noise",
                    options=tuple(TABLE_XY_CHOICES.keys()),
                    initial_value=self.table_xy_choice,
                )
                self._dd_table_yaw = self.server.gui.add_dropdown(
                    "Table yaw noise",
                    options=tuple(TABLE_YAW_CHOICES.keys()),
                    initial_value=self.table_yaw_choice,
                )
                self._dd_table_scale = self.server.gui.add_dropdown(
                    "Table size scale",
                    options=tuple(TABLE_SCALE_CHOICES.keys()),
                    initial_value=self.table_scale_choice,
                )
                self._dd_init = self.server.gui.add_dropdown(
                    "Init pose",
                    options=tuple(INIT_CHOICES.keys()),
                    initial_value=self.init_choice,
                )

                # === load env ===
                self._btn_load_top = self.server.gui.add_button("Load / reload env")
                self._btn_load_top.on_click(lambda _: self._load_env())
                # Alias so any base-class code that pokes self._btn_load works.
                self._btn_load = self._btn_load_top

                # === status (was its own folder in the base class) ===
                self._md_status = self.server.gui.add_markdown("**Status:** Ready")
                self._md_action_l2 = self.server.gui.add_markdown(
                    "**L2(student-teacher):** --"
                )
                self._md_student_ckpt = self.server.gui.add_markdown(
                    self._student_ckpt_md()
                )
                self._md_task = self.server.gui.add_markdown("**Task:** --")
                self._md_hole = self.server.gui.add_markdown("**Hole pos:** --")
                self._md_object_pose = self.server.gui.add_markdown("**Object pose:** --")
                self._md_goal_pose = self.server.gui.add_markdown("**Goal pose:** --")
                self._md_pose_delta = self.server.gui.add_markdown("**Object-goal z dist:** --")
                self._md_prog = self.server.gui.add_markdown("**Progress:** --")
                self._md_diag = self.server.gui.add_markdown("**Goal dist:** --")
                self._md_retract = self.server.gui.add_markdown("**Retract:** --")
                self._md_force = self.server.gui.add_markdown("**Table force:** --")
                self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")

            # Live depth obs panel between Controls and Episode Controls.
            with self.server.gui.add_folder(
                "Depth obs (env_id={})".format(self.env_id), expand_by_default=True
            ):
                import numpy as _np
                self._img_policy = self.server.gui.add_image(
                    _np.zeros((10, 10, 3), dtype=_np.uint8),
                    label="policy-input depth (normalized)",
                )

            # Episode + display controls inlined from the base class (cheaper
            # than monkey-patching super()._build_gui's folder layout).
            with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
                self._btn_run = self.server.gui.add_button("Run Episode")
                self._btn_run.on_click(lambda _: self._cmd_run())
                self._btn_pause = self.server.gui.add_button("Pause")
                self._btn_pause.on_click(lambda _: self._cmd_pause())
                self._btn_stop = self.server.gui.add_button("Stop")
                self._btn_stop.on_click(lambda _: self._cmd_stop())

            with self.server.gui.add_folder("Display", expand_by_default=False):
                self._cb_keypoints = self.server.gui.add_checkbox(
                    "Show keypoints", initial_value=True
                )
                self._cb_keypoints.on_update(lambda _: self._apply_keypoint_visibility())
                self._cb_goal = self.server.gui.add_checkbox(
                    "Show goal", initial_value=True
                )
                self._cb_goal.on_update(lambda _: self._apply_goal_visibility())
                self._sl_goal_opacity = self.server.gui.add_slider(
                    "Goal opacity", min=0.0, max=1.0, step=0.05, initial_value=0.5,
                )
                self._sl_goal_opacity.on_update(lambda _: self._apply_goal_visibility())
                self._sl_fixture_opacity = self.server.gui.add_slider(
                    "Fixture opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0,
                )
                self._sl_fixture_opacity.on_update(lambda _: self._apply_fixture_opacity())
                self._sl_object_opacity = self.server.gui.add_slider(
                    "Object opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0,
                )
                self._sl_object_opacity.on_update(lambda _: self._apply_object_opacity())
                self._cb_target_vol = self.server.gui.add_checkbox(
                    "Show target volume", initial_value=False
                )
                self._cb_target_vol.on_update(lambda _: self._toggle_target_volume())

        def _cmd_set_action_source(self):
            self.action_source = self._dd_action_source.value
            if self._conn is not None:
                try:
                    self._conn.send(("set_action_source", self.action_source))
                except Exception as exc:
                    print(f"[launcher] failed to send action_source: {exc}")

        def _student_ckpt_md(self) -> str:
            label = str(self.student_ckpt) if self.student_ckpt is not None else "random-init"
            return f"**Student checkpoint:** `{label}`"

        def _cmd_set_student_policy(self):
            """Dropdown updated -> swap student_ckpt + auto-set training toggles.

            The change is staged; Load Env must be clicked for the worker to
            be re-spawned with the new checkpoint + toggle overrides.
            """
            name = self._dd_student_policy.value
            entry = self.student_policies.get(name)
            if entry is None:
                print(f"[launcher] unknown student policy {name!r}; ignoring")
                return
            self.student_policy_name = name
            self.student_ckpt = entry["ckpt"]
            toggles = entry["toggles"]
            self.toggle_delays = bool(toggles["delays"])
            self.toggle_camera_pose_rand = bool(toggles["camera_pose_rand"])
            self.toggle_depth_aug = bool(toggles["depth_aug"])
            # Reflect the new defaults in the GUI without firing on_update.
            self._cb_delays.value = self.toggle_delays
            self._cb_camera_pose_rand.value = self.toggle_camera_pose_rand
            self._cb_depth_aug.value = self.toggle_depth_aug
            self._md_student_ckpt.content = self._student_ckpt_md()
            print(
                f"[launcher] student policy -> {name} "
                f"(toggles: delays={self.toggle_delays}, "
                f"cam_pose_rand={self.toggle_camera_pose_rand}, "
                f"depth_aug={self.toggle_depth_aug}). Click Load Env to apply."
            )

        def _handle(self, msg):
            tag = msg[0]
            if tag == "depth":
                policy_rgb = msg[1]
                l2 = float(msg[2])
                src = msg[3] if len(msg) > 3 else self.action_source
                self._img_policy.image = policy_rgb
                self._md_action_l2.content = (
                    f"**L2(student−teacher):** {l2:.4f} &nbsp;|&nbsp; "
                    f"**Active source:** {src}"
                )
                return
            super()._handle(msg)

        def _load_env(self):
            problem_name = self._dd_problem.value
            goal_mode = self._dd_goal_mode.value
            rgf = self._sl_rgf.value
            insertion_tol = float(self._sl_insertion_tol.value)
            retract_tol = float(self._sl_retract_tol.value)

            try:
                self._set_problem_assets(problem_name)
                self._reload_problem_meshes()
            except Exception as exc:
                self._md_status.content = (
                    f"**Status:** Problem load error -- {str(exc)[:200]}"
                )
                print(
                    f"[student launcher] Problem load error for {problem_name!r}:\n"
                    f"{traceback.format_exc()}"
                )
                return

            self._kill_subprocess()

            policy_label = self.student_policy_name or "(custom)"
            toggles_label = (
                f"delays={self._cb_delays.value} "
                f"camposerand={self._cb_camera_pose_rand.value} "
                f"depthaug={self._cb_depth_aug.value}"
            )
            label = (
                f"{problem_name} | source: {self.action_source} | "
                f"policy: {policy_label} | {toggles_label} | "
                f"goals: {goal_mode} | rgf: {rgf:.1f} | "
                f"ins: {insertion_tol * 1000:.1f}mm | ret: {retract_tol * 1000:.1f}mm"
            )
            self._md_status.content = f"**Status:** Loading Isaac Sim *{label}* ..."
            self._md_task.content = f"**Task:** Isaac Sim Student | {label}"
            self._md_retract.content = "**Retract:** --"
            self._md_hole.content = "**Hole pos:** --"
            self._md_object_pose.content = "**Object pose:** --"
            self._md_goal_pose.content = "**Goal pose:** --"
            self._md_pose_delta.content = "**Object-goal z dist:** --"
            self.ep_count = 0
            self._peak_force = 0.0
            self._md_stats.content = "**Stats:** No episodes yet"

            self.robot.update_cfg(gym_eval.DEFAULT_DOF_POS)
            self._setup_scene_objects()

            # Read the current checkbox states (user may have toggled them
            # after the policy auto-population) and stage them as worker overrides.
            self.toggle_delays = bool(self._cb_delays.value)
            self.toggle_camera_pose_rand = bool(self._cb_camera_pose_rand.value)
            self.toggle_depth_aug = bool(self._cb_depth_aug.value)
            self.table_xy_choice = str(self._dd_table_xy.value)
            self.table_yaw_choice = str(self._dd_table_yaw.value)
            self.table_scale_choice = str(self._dd_table_scale.value)
            self.init_choice = str(self._dd_init.value)

            worker_overrides = dict(self.extra_overrides)
            # Map the 3 GUI toggles to the underlying StudentObsCfg /
            # DomainRandomizationCfg keys. Delays gate three independent flags
            # but are surfaced as one switch in the GUI; their max values stay
            # at the training defaults (max=3) when on.
            worker_overrides["env.domain_randomization.use_obs_delay"] = self.toggle_delays
            worker_overrides["env.domain_randomization.use_action_delay"] = self.toggle_delays
            worker_overrides["env.student_obs.use_camera_delay"] = self.toggle_delays
            worker_overrides["env.student_obs.use_camera_pose_rand"] = self.toggle_camera_pose_rand
            worker_overrides["env.student_obs.use_depth_aug"] = self.toggle_depth_aug
            # Table-DR knobs. xy/yaw are reset-time; scale is USD-time (baked
            # at scene init by the worker -- the Load env spawn picks them up).
            xy_range = TABLE_XY_CHOICES[self.table_xy_choice]
            yaw_deg = TABLE_YAW_CHOICES[self.table_yaw_choice]
            scale_x, scale_y = TABLE_SCALE_CHOICES[self.table_scale_choice]
            n_variants = (TABLE_SCALE_N_VARIANTS_DEFAULT
                          if self.table_scale_choice != "off" else 1)
            worker_overrides["env.reset.table_reset_xy_range_m"] = list(xy_range)
            worker_overrides["env.reset.table_reset_yaw_range_deg"] = float(yaw_deg)
            worker_overrides["env.assets.table_scale_range_x"] = list(scale_x)
            worker_overrides["env.assets.table_scale_range_y"] = list(scale_y)
            worker_overrides["env.assets.table_scale_num_variants"] = int(n_variants)
            # Init pose: "random" leaves all reset-time randomness at task-yaml
            # defaults; "fixed" pins peg, hole, robot joints, and table z to
            # the canonical fixed-init point used by the training subs.
            init_params = INIT_CHOICES[self.init_choice]
            worker_overrides["env.reset.fixed_start_pose"] = init_params["fixed_start_pose"]
            worker_overrides["env.peg_in_hole.hole_x_range"] = list(init_params["hole_x_range"])
            worker_overrides["env.peg_in_hole.hole_y_range"] = list(init_params["hole_y_range"])
            for noise_key, noise_val in init_params["reset_noise"].items():
                worker_overrides[f"env.reset.{noise_key}"] = float(noise_val)
            # Resize the viser /table/wood mesh to the max-extent envelope so the
            # static viz roughly matches the largest scaled variant in the sim.
            self._update_table_viz(scale_x_range=scale_x, scale_y_range=scale_y)

            authkey = os.urandom(16)
            listener = Listener(("127.0.0.1", 0), authkey=authkey)
            host, port = listener.address
            listener._listener._socket.settimeout(60.0)

            cmd = [
                self.worker_python,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-host", str(host),
                "--worker-port", str(port),
                "--worker-authkey", authkey.hex(),
                "--task", self.task,
                "--agent", self.agent,
                "--student-agent", self.student_agent,
                "--teacher-checkpoint", str(self.teacher_ckpt),
                "--problem", problem_name,
                "--goal-mode", goal_mode,
                "--num-envs", str(self.num_envs),
                "--games", str(self.games),
                "--random-goal-fraction", str(float(rgf)),
                "--insertion-success-tolerance", str(insertion_tol),
                "--retract-success-tolerance", str(retract_tol),
                "--rl-device", self.rl_device,
                "--sim-device", self.sim_device,
                "--initial-action-source", self.action_source,
                "--env-id", str(self.env_id),
                "--depth-send-every", str(self.depth_send_every),
                "--override-json", json.dumps(worker_overrides),
            ]
            if self.student_ckpt is not None:
                cmd += ["--student-checkpoint", str(self.student_ckpt)]
            if self.deterministic:
                cmd.append("--deterministic")
            if self.sdf:
                cmd.append("--sdf")
            if self.keep_dr:
                cmd.append("--keep-dr")
            if self.headless:
                cmd.append("--worker-headless")

            env_vars = os.environ.copy()
            env_vars.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            try:
                popen = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env_vars)
                self._proc = PopenAdapter(popen)
                self._conn = listener.accept()
            except Exception as exc:
                listener.close()
                if self._proc is not None:
                    self._proc.kill()
                    self._proc.join(timeout=2)
                    self._proc = None
                self._md_status.content = (
                    f"**Status:** Worker launch error -- {str(exc)[:200]}"
                )
                print(
                    "[student launcher] Worker launch error:\n"
                    f"{traceback.format_exc()}"
                )
                return
            finally:
                try:
                    listener.close()
                except Exception:
                    pass

            print(
                f"[student launcher] Spawned pid={self._proc.pid} problem={problem_name} "
                f"goal_mode={goal_mode} action_source={self.action_source}"
            )

        def run(self):
            print()
            print(f"  Peg-in-Hole Student/Teacher Eval   http://localhost:{self.port}")
            print()
            try:
                while True:
                    self._poll()
                    time.sleep(1.0 / 120.0)
            except KeyboardInterrupt:
                print("\n[student launcher] Shutting down...")
                self._kill_subprocess()

    if args.dry_run:
        print(f"[eval_student_isaacsim] worker python: {args.python or _default_isaacsim_python()}")
        print(f"[eval_student_isaacsim] teacher: {teacher_ckpt}")
        print(f"[eval_student_isaacsim] student: {student_ckpt or '(random init)'}")
        print(f"[eval_student_isaacsim] problem: {initial_problem}")
        print(f"[eval_student_isaacsim] port: {args.port}")
        return 0

    StudentEvalDemo(
        policies=viewer_policies,
        port=args.port,
        headless=not args.no_headless,
        goal_mode=args.goal_mode,
        random_goal_fraction=args.random_goal_fraction,
        initial_policy=policy_name,
        extra_overrides=extra_overrides,
        initial_problem=initial_problem,
    ).run()
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.worker:
        return _worker_main(args)
    return _run_viewer(args)


if __name__ == "__main__":
    sys.exit(main())
