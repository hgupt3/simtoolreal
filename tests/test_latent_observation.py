"""Kit-free tests for the adapter-supplied latent observation term."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OBS_SEAM_PATH = (
    REPO_ROOT
    / "isaacsimenvs"
    / "tasks"
    / "simtoolreal"
    / "utils"
    / "obs_seam.py"
)
SPEC = importlib.util.spec_from_file_location("simtoolreal_obs_seam", OBS_SEAM_PATH)
assert SPEC is not None and SPEC.loader is not None
OBS_SEAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OBS_SEAM)


def _task_obs_config() -> dict:
    task_yaml = REPO_ROOT / "isaacsimenvs" / "cfg" / "task" / "SimToolReal.yaml"
    with task_yaml.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["obs"]


def test_default_and_four_wide_observation_dims() -> None:
    obs_cfg = _task_obs_config()

    assert obs_cfg["num_latent_obs"] == 0
    assert obs_cfg["obs_list"][-1] == "latent_state"
    assert obs_cfg["state_list"][-1] == "latent_state"
    assert OBS_SEAM.compute_obs_dim(obs_cfg["obs_list"], 0) == 140
    assert OBS_SEAM.compute_obs_dim(obs_cfg["state_list"], 0) == 162
    assert OBS_SEAM.compute_obs_dim(obs_cfg["obs_list"], 4) == 144
    assert OBS_SEAM.compute_obs_dim(obs_cfg["state_list"], 4) == 166


def test_zero_width_append_is_bit_exact() -> None:
    torch.manual_seed(29)
    existing_obs = torch.randn(8, 140, dtype=torch.float32)
    latent_state = existing_obs[:, :0]

    assert torch.equal(
        torch.cat([existing_obs], dim=-1),
        torch.cat([existing_obs, latent_state], dim=-1),
    )


def test_latent_fn_receives_lab_order_hand_positions_once() -> None:
    hand_joint_pos = torch.arange(3 * 22, dtype=torch.float32).reshape(3, 22)
    received = []

    def latent_obs_fn(positions: torch.Tensor) -> torch.Tensor:
        received.append(positions)
        return positions[:, :4]

    latent_state = OBS_SEAM.compute_latent_state(
        hand_joint_pos, 4, latent_obs_fn
    )

    assert len(received) == 1
    assert received[0] is hand_joint_pos
    assert torch.equal(latent_state, hand_joint_pos[:, :4])


def test_positive_latent_width_requires_callable() -> None:
    with pytest.raises(
        ValueError,
        match="latent_obs_fn is required when obs.num_latent_obs > 0",
    ):
        OBS_SEAM.validate_latent_obs_config(4, None)


def test_latent_fn_result_shape_is_checked() -> None:
    hand_joint_pos = torch.zeros(2, 22)

    with pytest.raises(ValueError, match=r"must return shape \(2, 4\)"):
        OBS_SEAM.compute_latent_state(
            hand_joint_pos, 4, lambda positions: positions[:, :3]
        )
