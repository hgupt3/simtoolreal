"""Kit-free tensor tests for the pluggable hand-action target mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ACTION_UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaacsimenvs"
    / "tasks"
    / "simtoolreal"
    / "utils"
    / "action_utils.py"
)
SPEC = importlib.util.spec_from_file_location("simtoolreal_action_utils", ACTION_UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
ACTION_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTION_UTILS)


def test_default_hand_mapping_is_bit_exact() -> None:
    torch.manual_seed(17)
    actions = torch.rand(64, 22, dtype=torch.float32) * 2.0 - 1.0
    previous = torch.randn(64, 22, dtype=torch.float32)
    lower = -torch.rand(64, 22, dtype=torch.float32) * 2.0
    upper = torch.rand(64, 22, dtype=torch.float32) * 2.0

    original = lower + 0.5 * (actions + 1.0) * (upper - lower)
    actual = ACTION_UTILS.compute_hand_raw_targets(
        actions, previous, lower, upper
    )

    assert torch.equal(actual, original)


def test_default_hand_mapping_rejects_non_joint_action_width() -> None:
    actions = torch.zeros(4, 8)
    joint_targets = torch.zeros(4, 22)

    try:
        ACTION_UTILS.compute_hand_raw_targets(
            actions, joint_targets, joint_targets, joint_targets
        )
    except AssertionError as exc:
        assert "hand_action_dim must be 22" in str(exc)
    else:
        raise AssertionError("default hand mapping accepted K != 22")


def test_replay_target_bypasses_hand_transform_path() -> None:
    class ReplayEnv:
        pass

    env = ReplayEnv()
    env._replay_target_lab_order = torch.randn(3, 29)
    env._cur_targets = torch.zeros_like(env._replay_target_lab_order)
    env._prev_targets = torch.zeros_like(env._replay_target_lab_order)

    # The deliberately incompatible K=8 action reaches no config or hook access.
    ACTION_UTILS.apply_action_pipeline(env, torch.zeros(3, 15))

    assert torch.equal(env._cur_targets, env._replay_target_lab_order)
    assert torch.equal(env._prev_targets, env._replay_target_lab_order)
