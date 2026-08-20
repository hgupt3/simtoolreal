"""Warm-resume env state capture/restore for SimToolReal.

``get_env_state`` snapshots every cross-step evolving buffer that shapes
dynamics, observations, rewards, or logging, plus sim dof/root states and
process RNG, into a CPU tensor dict that rides inside rl_games checkpoints
(``A2CBase.get_full_state_weights`` -> ``vec_env.get_env_state``).
``set_env_state`` writes it all back so a resumed run continues with no mass
environment reset.

Deliberately excluded: within-step caches recomputed before use each step
(``_keypoints_max_dist``, ``_curr_fingertip_distances``, ``_near_goal``,
``_is_success``, ``_termination_reasons``, ``_reward_terms``, ``reward_buf``,
``reset_terminated``/``reset_time_outs``) and PhysX solver-internal
transients (contact caches, warm-start impulses), which cannot be serialized
through Isaac Lab and decay within a step.
"""

from __future__ import annotations

import torch


ENV_STATE_SCHEMA = "simtoolreal_env_state_v1"

# Cross-step per-env buffers captured verbatim (allocate_state_buffers).
_BUFFER_NAMES = (
    "_cur_targets",
    "_prev_targets",
    "_object_scale_multiplier",
    "_lifted_object",
    "_closest_keypoint_max_dist",
    "_closest_fingertip_dist",
    "_successes",
    "_near_goal_steps",
    "_prev_episode_successes",
    "_object_init_z",
    "_table_z_per_env",
    "_object_state_queue",
    "_obs_queue",
    "_action_queue",
    "_random_force_prob",
    "_random_torque_prob",
    "_object_forces",
    "_object_torques",
    "episode_length_buf",
)

# Buffers that only exist for some configurations / task variants.
_OPTIONAL_BUFFER_NAMES = (
    "_slow_hand_targets",
    "_traj_id",
    "_traj_step",
    "_student_camera_queue",
    "_student_obs_queue",
    "goal_yaw_obs_noise",
)


def get_env_state(env) -> dict | None:
    """Snapshot warm-resume state; None when disabled by config."""
    if not env.cfg.checkpoint_env_state:
        return None

    buffers = {}
    for name in _BUFFER_NAMES:
        buffers[name] = getattr(env, name).detach().cpu().clone()
    for name in _OPTIONAL_BUFFER_NAMES:
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor):
            buffers[name] = value.detach().cpu().clone()

    sim = {
        "robot_joint_pos": env.robot.data.joint_pos.detach().cpu().clone(),
        "robot_joint_vel": env.robot.data.joint_vel.detach().cpu().clone(),
        "object_root_state": env.object.data.root_state_w.detach().cpu().clone(),
        "table_root_state": env.table.data.root_state_w.detach().cpu().clone(),
        "goal_root_state": env.goal_viz.data.root_state_w.detach().cpu().clone(),
    }

    rng = {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(env.device)
            if torch.device(env.device).type == "cuda"
            else None
        ),
    }

    export_fn = env.cfg.action.hand_state_export_fn
    return {
        "schema": ENV_STATE_SCHEMA,
        "num_envs": int(env.num_envs),
        "buffers": buffers,
        "scalars": {
            "_current_success_tolerance": float(env._current_success_tolerance),
            "_frame_counter": int(env._frame_counter),
            "_last_curriculum_update": int(env._last_curriculum_update),
            "common_step_counter": int(env.common_step_counter),
            "_sim_step_counter": int(env._sim_step_counter),
        },
        "sim": sim,
        "rng": rng,
        "hand_transform": None if export_fn is None else export_fn(),
    }


def set_env_state(env, state: dict | None) -> None:
    """Restore warm-resume state written by :func:`get_env_state`.

    ``None`` (pre-warm-resume checkpoint) keeps the legacy resume behavior —
    envs continue from their current (initial) states — with a warning. An
    env-count mismatch also degrades to that path, mirroring the rl_games
    shape-mismatch skip for the algo-side per-env tensors. Anything else
    malformed fails closed.
    """
    if state is None:
        print(
            "[simtoolreal] WARNING: checkpoint carries no env_state; resuming "
            "without warm world state (expect a mass env reset transient)"
        )
        return
    if not isinstance(state, dict) or state.get("schema") != ENV_STATE_SCHEMA:
        raise ValueError(
            f"unrecognized env_state schema: {state.get('schema') if isinstance(state, dict) else type(state)!r}"
        )
    if int(state["num_envs"]) != env.num_envs:
        print(
            "[simtoolreal] WARNING: checkpoint env_state has "
            f"{state['num_envs']} envs but this run has {env.num_envs}; "
            "skipping warm env state restore"
        )
        return

    device = env.device
    buffers = state["buffers"]
    for name in _BUFFER_NAMES:
        getattr(env, name).copy_(buffers[name].to(device))
    for name in _OPTIONAL_BUFFER_NAMES:
        live = getattr(env, name, None)
        saved = buffers.get(name)
        if isinstance(live, torch.Tensor) != (saved is not None):
            raise ValueError(
                f"checkpoint/env mismatch for optional state buffer {name!r}"
            )
        if saved is not None:
            if isinstance(live, torch.Tensor):
                live.copy_(saved.to(device))
            else:
                setattr(env, name, saved.to(device))

    scalars = state["scalars"]
    env._current_success_tolerance = float(scalars["_current_success_tolerance"])
    env._frame_counter = int(scalars["_frame_counter"])
    env._last_curriculum_update = int(scalars["_last_curriculum_update"])
    env.common_step_counter = int(scalars["common_step_counter"])
    env._sim_step_counter = int(scalars["_sim_step_counter"])

    sim = state["sim"]
    env.robot.write_joint_state_to_sim(
        sim["robot_joint_pos"].to(device), sim["robot_joint_vel"].to(device)
    )
    for asset, key in (
        (env.object, "object_root_state"),
        (env.table, "table_root_state"),
        (env.goal_viz, "goal_root_state"),
    ):
        root_state = sim[key].to(device)
        asset.write_root_pose_to_sim(root_state[:, 0:7])
        asset.write_root_velocity_to_sim(root_state[:, 7:13])
    env.scene.write_data_to_sim()
    env.sim.forward()

    rng = state["rng"]
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda"] is not None:
        torch.cuda.set_rng_state(rng["torch_cuda"], device)

    import_fn = env.cfg.action.hand_state_import_fn
    if import_fn is not None:
        if "hand_transform" not in state:
            raise ValueError("env_state is missing the hand_transform payload")
        import_fn(state["hand_transform"])

    # Post-resume diagnostics window: the env prints per-step reset counts and
    # commanded-target deltas for the first few steps after a warm restore.
    env._warm_resume_log_steps = 3
    print(
        "[simtoolreal] warm env state restored: "
        f"num_envs={env.num_envs} "
        f"frame_counter={env._frame_counter} "
        f"success_tolerance={env._current_success_tolerance}"
    )


__all__ = ["ENV_STATE_SCHEMA", "get_env_state", "set_env_state"]
