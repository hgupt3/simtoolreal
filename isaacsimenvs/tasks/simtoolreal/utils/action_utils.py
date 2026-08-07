"""Action processing and random object wrench helpers."""

from __future__ import annotations

import math

import torch


NUM_HAND_JOINTS = 22


def compute_hand_raw_targets(
    hand_actions: torch.Tensor,
    prev_hand_targets: torch.Tensor,
    hand_lower: torch.Tensor,
    hand_upper: torch.Tensor,
    hand_action_transform=None,
) -> torch.Tensor:
    """Decode policy hand actions into absolute Lab-order hand joint targets.

    A custom transform receives the raw policy-order ``(N, K)`` hand slice and
    Lab-order ``(N, 22)`` previous hand targets. The default path instead
    receives an already canonical-to-Lab-permuted ``(N, 22)`` hand slice.
    """
    if hand_action_transform is None:
        assert hand_actions.shape[-1] == NUM_HAND_JOINTS, (
            "hand_action_dim must be 22 when hand_action_transform is None"
        )
        return hand_lower + 0.5 * (hand_actions + 1.0) * (
            hand_upper - hand_lower
        )

    hand_targets = hand_action_transform(hand_actions, prev_hand_targets)
    if hand_targets.shape != prev_hand_targets.shape:
        raise ValueError(
            "hand_action_transform must return shape "
            f"{tuple(prev_hand_targets.shape)}, got {tuple(hand_targets.shape)}"
        )
    return hand_targets


def sample_log_uniform(lo_hi: tuple[float, float], n: int) -> torch.Tensor:
    """Log-uniform sample on CPU (caller moves to device)."""
    lo, hi = lo_hi
    return torch.exp(
        torch.empty(n).uniform_(math.log(lo + 1e-12), math.log(hi + 1e-12))
    )


def apply_action_pipeline(env, actions: torch.Tensor) -> None:
    """Apply delay, canonical-to-Lab mapping, smoothing, and target clamps."""
    # Debug replay accepts an already Lab-order target and intentionally bypasses
    # action delay, hand transformation, smoothing, and clamping.
    replay_target = getattr(env, "_replay_target_lab_order", None)
    if replay_target is not None:
        env._cur_targets[:] = replay_target
        env._prev_targets = env._cur_targets.clone()
        if env._slow_hand_targets is not None:
            env._slow_hand_targets.copy_(
                replay_target[:, env._hand_joint_ids]
            )
        return

    dr = env.cfg.domain_randomization
    act_cfg = env.cfg.action
    dt = env.step_dt  # policy step (= decimation * sim.dt = 1/60 s)

    actions = actions.clone().to(env.device)

    # Action delay.
    if dr.use_action_delay and dr.action_delay_max > 0:
        episode_start = (env.episode_length_buf == 0) & (env._successes == 0)
        if episode_start.any():
            env._action_queue[episode_start] = actions[episode_start].unsqueeze(1)
        env._action_queue = torch.roll(env._action_queue, shifts=1, dims=1)
        env._action_queue[:, 0, :] = actions
        delay_idx = torch.randint(
            0, env._action_queue.shape[1], (env.num_envs,), device=env.device
        )
        actions = env._action_queue[
            torch.arange(env.num_envs, device=env.device), delay_idx
        ]

    # Arm: velocity-delta accumulator.
    arm_action = actions[:, env._arm_action_ids]
    prev_arm_targets = env._prev_targets[:, env._arm_joint_ids]
    arm_raw = prev_arm_targets + act_cfg.dof_speed_scale * dt * arm_action
    arm_raw = torch.clamp(arm_raw, env._arm_lower, env._arm_upper)
    arm_smoothed = (
        act_cfg.arm_moving_average * arm_raw
        + (1.0 - act_cfg.arm_moving_average) * prev_arm_targets
    )
    arm_smoothed = torch.clamp(arm_smoothed, env._arm_lower, env._arm_upper)

    # Hand: configurable policy slice -> absolute Lab-order targets.
    hand_action = actions[:, 7:]
    prev_hand_targets = env._prev_targets[:, env._hand_joint_ids]
    if act_cfg.hand_action_transform is None:
        hand_action = hand_action[:, env._hand_action_ids]
    hand_raw = compute_hand_raw_targets(
        hand_action,
        prev_hand_targets,
        env._hand_lower,
        env._hand_upper,
        act_cfg.hand_action_transform,
    )
    previous_slow_hand_targets = prev_hand_targets
    if env._slow_hand_targets is not None:
        previous_slow_hand_targets = env._slow_hand_targets
    hand_smoothed = (
        act_cfg.hand_moving_average * hand_raw
        + (1.0 - act_cfg.hand_moving_average) * previous_slow_hand_targets
    )
    if act_cfg.hand_fast_offset_transform is not None:
        if env._slow_hand_targets is None:
            raise RuntimeError("fast-offset control requires slow-target state")
        fast_offset = act_cfg.hand_fast_offset_transform(hand_action)
        if fast_offset.shape != hand_smoothed.shape:
            raise ValueError(
                "hand_fast_offset_transform must return shape "
                f"{tuple(hand_smoothed.shape)}, got {tuple(fast_offset.shape)}"
            )
        if (
            fast_offset.device != hand_smoothed.device
            or fast_offset.dtype != hand_smoothed.dtype
        ):
            raise ValueError(
                "hand_fast_offset_transform output device/dtype must match "
                "the hand target"
            )
        env._slow_hand_targets.copy_(hand_smoothed)
        hand_smoothed = hand_smoothed + fast_offset
    hand_smoothed = torch.clamp(hand_smoothed, env._hand_lower, env._hand_upper)

    # Write Lab-order targets and cache them for the next step.
    env._cur_targets[:, env._arm_joint_ids] = arm_smoothed
    env._cur_targets[:, env._hand_joint_ids] = hand_smoothed
    env._prev_targets = env._cur_targets.clone()


def apply_wrench_dr(env) -> None:
    """Apply decayed random force/torque impulses to the object."""
    dr = env.cfg.domain_randomization
    dt_pol = env.step_dt

    # Decay previous wrench.
    if dr.force_decay > 0.0:
        env._object_forces *= dr.force_decay ** (dt_pol / dr.force_decay_interval)
    else:
        env._object_forces.zero_()
    if dr.torque_decay > 0.0:
        env._object_torques *= dr.torque_decay ** (dt_pol / dr.torque_decay_interval)
    else:
        env._object_torques.zero_()

    # Sample new wrench with per-env probability.
    force_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_force_prob
    ).view(-1, 1, 1)
    torque_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_torque_prob
    ).view(-1, 1, 1)
    mass = env._object_mass.unsqueeze(-1)  # (N, 1, 1)
    new_force = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.force_scale
    )
    new_torque = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.torque_scale
    )
    env._object_forces = torch.where(force_fire, new_force, env._object_forces)
    env._object_torques = torch.where(torque_fire, new_torque, env._object_torques)

    # _lifted_object is from the previous step because rewards update later.
    if dr.force_only_when_lifted:
        env._object_forces *= env._lifted_object.float().view(-1, 1, 1)
    if dr.torque_only_when_lifted:
        env._object_torques *= env._lifted_object.float().view(-1, 1, 1)

    # Optional task-level gate: subclasses can disable wrench DR per env (e.g.
    # peg_in_hole zeros impulses once the final insert is achieved so the part
    # doesn't fly out of the hole during retract). Same opt-in convention as
    # _curriculum_eligible_mask in termination_utils.
    if hasattr(env, "_wrench_dr_active_mask"):
        active = env._wrench_dr_active_mask()
        if active is not None:
            active_f = active.float().view(-1, 1, 1)
            env._object_forces *= active_f
            env._object_torques *= active_f

    env.object.set_external_force_and_torque(
        forces=env._object_forces,
        torques=env._object_torques,
        is_global=True,
    )


__all__ = [
    "apply_action_pipeline",
    "apply_wrench_dr",
    "compute_hand_raw_targets",
    "sample_log_uniform",
]
