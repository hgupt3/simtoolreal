"""Thin wrapper that adds get_env_info() to the raw isaacgym VecTask env.

The raw isaacgym env already has:
  - observation_space (gym.Space)
  - action_space (gym.Space)
  - num_envs (int)
  - set_train_info(frame, *args, **kwargs)
  - get_env_state() / set_env_state(state)
  - reset() / step(actions)

simple_rl's Agent and Player call env.get_env_info() to discover
observation/action shapes. This wrapper adds that method.
"""

from __future__ import annotations

from typing import Any


class SimpleRLEnvWrapper:
    """Wrap an isaacgym VecTask env so simple_rl Agent/Player can use it."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def get_env_info(self) -> dict:
        env_info = {
            "observation_space": self._env.observation_space,
            "action_space": self._env.action_space,
            "agents": 1,
            "value_size": 1,
        }
        if getattr(self._env, "num_states", 0) > 0:
            env_info["state_space"] = self._env.state_space
        return env_info

    # ── forward all other attribute access to the underlying env ──────────

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails — so 'get_env_info' above wins.
        return getattr(self._env, name)

    def __repr__(self) -> str:
        return f"SimpleRLEnvWrapper({self._env!r})"
