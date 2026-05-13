"""Peg-in-hole task registration."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .peg_in_hole_env import PegInHoleEnv
from .peg_in_hole_env_cfg import PegInHoleEnvCfg


__all__ = ["PegInHoleEnv", "PegInHoleEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-PegInHole-Direct-v0",
    entry_point="isaacsimenvs.tasks.peg_in_hole.peg_in_hole_env:PegInHoleEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.peg_in_hole.peg_in_hole_env_cfg:PegInHoleEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PegInHole.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
    },
)

gym.register(
    id="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0",
    entry_point="isaacsimenvs.tasks.peg_in_hole.peg_in_hole_env:PegInHoleEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.peg_in_hole.peg_in_hole_env_cfg:PegInHoleEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PegInHoleDepthStudent.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
        "rl_games_dagger_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PegInHoleDepthStudentSAPG.yaml"),
        "rl_games_dagger_ppo_cfg_entry_point": str(_CFG_DIR / "train" / "PegInHoleDepthStudentPPO.yaml"),
        "distill_cfg_entry_point": str(_CFG_DIR / "train" / "PegInHoleDepthDistill.yaml"),
    },
)
