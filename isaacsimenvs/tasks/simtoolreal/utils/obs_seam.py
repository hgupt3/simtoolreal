"""Kit-free observation sizing and external latent-term helpers."""

from __future__ import annotations

from collections.abc import Callable

import torch


NUM_JOINTS: int = 29
NUM_HAND_JOINTS: int = 22
NUM_FINGERTIPS: int = 5
NUM_KEYPOINTS: int = 4

OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,  # 15
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,  # 12
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,  # 12
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,  # 5
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
    # Dynamic width supplied to compute_obs_dim from obs.num_latent_obs.
    "latent_state": 0,
}


def validate_latent_obs_config(
    num_latent_obs: int,
    latent_obs_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
) -> int:
    """Validate the adapter seam and return its normalized width."""
    width = int(num_latent_obs)
    if width < 0:
        raise ValueError("obs.num_latent_obs must be non-negative")
    if latent_obs_fn is None:
        if width > 0:
            raise ValueError(
                "obs.latent_obs_fn is required when obs.num_latent_obs > 0"
            )
    elif not callable(latent_obs_fn):
        raise TypeError("obs.latent_obs_fn must be callable or None")
    return width


def compute_obs_dim(field_list, num_latent_obs: int = 0) -> int:
    """Return the configured width of an ordered observation field list."""
    width = int(num_latent_obs)
    if width < 0:
        raise ValueError("obs.num_latent_obs must be non-negative")
    return sum(
        width if field == "latent_state" else OBS_FIELD_SIZES[field]
        for field in field_list
    )


def compute_latent_state(
    hand_joint_pos: torch.Tensor,
    hand_prev_targets: torch.Tensor,
    num_latent_obs: int,
    latent_obs_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Call the adapter once and validate its ``(N, L)`` result."""
    width = validate_latent_obs_config(num_latent_obs, latent_obs_fn)
    expected_input_shape = (hand_joint_pos.shape[0], NUM_HAND_JOINTS)
    if tuple(hand_joint_pos.shape) != expected_input_shape:
        raise ValueError(
            "latent_obs_fn input must have shape "
            f"{expected_input_shape}, got {tuple(hand_joint_pos.shape)}"
        )
    if tuple(hand_prev_targets.shape) != expected_input_shape:
        raise ValueError(
            "latent_obs_fn previous-target input must have shape "
            f"{expected_input_shape}, got {tuple(hand_prev_targets.shape)}"
        )
    if hand_prev_targets.device != hand_joint_pos.device:
        raise ValueError(
            "latent_obs_fn previous-target input must be on the "
            "measured-position device"
        )
    if hand_prev_targets.dtype != hand_joint_pos.dtype:
        raise ValueError(
            "latent_obs_fn previous-target input must have the "
            "measured-position dtype"
        )
    if width == 0:
        return hand_joint_pos[:, :0]

    latent_state = latent_obs_fn(hand_joint_pos, hand_prev_targets)
    if not isinstance(latent_state, torch.Tensor):
        raise TypeError("obs.latent_obs_fn must return a torch.Tensor")
    expected_output_shape = (hand_joint_pos.shape[0], width)
    if tuple(latent_state.shape) != expected_output_shape:
        raise ValueError(
            "obs.latent_obs_fn must return shape "
            f"{expected_output_shape}, got {tuple(latent_state.shape)}"
        )
    if latent_state.device != hand_joint_pos.device:
        raise ValueError(
            "obs.latent_obs_fn must return a tensor on the input device"
        )
    if latent_state.dtype != hand_joint_pos.dtype:
        raise ValueError(
            "obs.latent_obs_fn must return a tensor with the input dtype"
        )
    return latent_state


__all__ = [
    "NUM_FINGERTIPS",
    "NUM_HAND_JOINTS",
    "NUM_JOINTS",
    "NUM_KEYPOINTS",
    "OBS_FIELD_SIZES",
    "compute_latent_state",
    "compute_obs_dim",
    "validate_latent_obs_config",
]
