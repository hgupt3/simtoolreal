"""Isaac Sim smoke tests for configurable hand-action transforms."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_action_dim", type=int, choices=(8, 22), required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  (registers gym envs)
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    class TestHandTransform:
        def __init__(self, hand_action_dim: int):
            self.hand_action_dim = hand_action_dim
            self.lower = None
            self.upper = None
            self.default_action_ids = None
            self.matrix = None
            self.last_raw = None

        def bind(self, inner) -> None:
            self.lower = inner._hand_lower
            self.upper = inner._hand_upper
            self.default_action_ids = inner._hand_action_ids
            self.matrix = torch.linspace(
                -0.5,
                0.5,
                self.hand_action_dim * 22,
                device=inner.device,
            ).reshape(self.hand_action_dim, 22)

        def __call__(
            self, hand_actions: torch.Tensor, prev_hand_targets: torch.Tensor
        ) -> torch.Tensor:
            assert self.lower is not None and self.upper is not None
            assert hand_actions.shape == (
                prev_hand_targets.shape[0],
                self.hand_action_dim,
            )
            if self.hand_action_dim == 22:
                normalized = hand_actions[:, self.default_action_ids]
            else:
                normalized = (hand_actions @ self.matrix).clamp(-1.0, 1.0)
            self.last_raw = self.lower + 0.5 * (normalized + 1.0) * (
                self.upper - self.lower
            )
            return self.last_raw

    transform = TestHandTransform(args.hand_action_dim)
    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.domain_randomization.use_action_delay = False
    cfg.action.hand_action_dim = args.hand_action_dim
    cfg.action.hand_action_transform = transform

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    transform.bind(inner)

    expected_action_dim = 7 + args.hand_action_dim
    assert cfg.action_space == expected_action_dim
    assert inner.single_action_space.shape == (expected_action_dim,)
    assert inner._action_queue.shape[-1] == expected_action_dim
    assert inner._cur_targets.shape[-1] == 29
    assert inner._prev_targets.shape[-1] == 29

    obs, _ = env.reset()
    assert obs["policy"].shape == (args.num_envs, 140)
    assert obs["critic"].shape == (args.num_envs, 162)

    generator = torch.Generator(device=inner.device).manual_seed(2026)
    for step in range(args.steps):
        actions = (
            torch.rand(
                args.num_envs,
                expected_action_dim,
                generator=generator,
                device=inner.device,
            )
            * 2.0
            - 1.0
        )
        prev_hand_targets = inner._prev_targets[:, inner._hand_joint_ids].clone()
        obs, reward, _, _, _ = env.step(actions)

        expected_hand_targets = (
            cfg.action.hand_moving_average * transform.last_raw
            + (1.0 - cfg.action.hand_moving_average) * prev_hand_targets
        ).clamp(inner._hand_lower, inner._hand_upper)
        actual_hand_targets = inner._cur_targets[:, inner._hand_joint_ids]
        assert torch.equal(actual_hand_targets, expected_hand_targets), (
            f"hand targets differ at step {step}"
        )
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(obs["critic"]).all()
        assert torch.isfinite(reward).all()

    mode = "identity" if args.hand_action_dim == 22 else "linear-8-to-22"
    print(
        f"[hook-smoke] OK — {mode}, action_dim={expected_action_dim}, "
        f"{args.steps} finite steps with exact target checks"
    )

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
