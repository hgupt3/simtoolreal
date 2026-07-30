"""Thin DirectRLEnv wrapper for SimToolReal.

The env owns Isaac Lab hook wiring and state buffers. Task math lives in the
utility modules called from each hook.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from .simtoolreal_env_cfg import SimToolRealEnvCfg
from .utils.action_utils import apply_action_pipeline, apply_wrench_dr
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import (
    build_observations,
    build_student_observations,
    compute_intermediate_values,
    compute_obs_dim,
)
from .utils.obs_seam import validate_latent_obs_config
from .utils.reset_utils import allocate_state_buffers, reset_env_state
from .utils.reward_utils import compute_rewards
from .utils.scene_utils import apply_physx_material_properties, setup_scene
from .utils.termination_utils import compute_terminations, update_tolerance_curriculum


__all__ = ["SimToolRealEnv", "SimToolRealEnvCfg"]


class SimToolRealEnv(DirectRLEnv):
    cfg: SimToolRealEnvCfg

    def __init__(
        self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        # Override spaces from configured field lists before
        # DirectRLEnv / rl_games observes the configclass.
        hand_action_dim = int(cfg.action.hand_action_dim)
        if hand_action_dim < 1:
            raise ValueError("action.hand_action_dim must be positive")
        if cfg.action.hand_action_transform is None:
            assert hand_action_dim == 22, (
                "action.hand_action_dim must be 22 when "
                "action.hand_action_transform is None"
            )
        elif not callable(cfg.action.hand_action_transform):
            raise TypeError("action.hand_action_transform must be callable or None")
        num_latent_obs = validate_latent_obs_config(
            cfg.obs.num_latent_obs, cfg.obs.latent_obs_fn
        )
        cfg.obs.num_latent_obs = num_latent_obs
        cfg.action_space = 7 + hand_action_dim
        cfg.observation_space = compute_obs_dim(
            cfg.obs.obs_list, num_latent_obs
        )
        cfg.state_space = compute_obs_dim(cfg.obs.state_list, num_latent_obs)

        super().__init__(cfg, render_mode, **kwargs)  # runs _setup_scene
        apply_physx_material_properties(self)
        allocate_state_buffers(self)

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self,
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        # Called decimation times per policy step; idempotent.
        self.robot.set_joint_position_target(self._cur_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)

    def get_student_obs(self) -> dict[str, torch.Tensor]:
        """Return opt-in student observations for distillation code."""
        return build_student_observations(self)
