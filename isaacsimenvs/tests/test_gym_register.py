"""Smoke test Phase 1: verify gym.register + YAML overlay compose correctly.

Confirms:
  1. `import isaacsimenvs` triggers `gym.register("Isaacsimenvs-Cartpole-Direct-v0", ...)`.
  2. `load_cfg_from_registry` resolves each entry_point to the expected type.
  3. `env_cfg.from_dict(task_yaml)` applies overrides onto the configclass cleanly
     (the merge Phase 2's trainer will rely on).

Does NOT build the sim — no AppLauncher, no `gym.make`. Pure registry introspection
so this stays under a second.

    /share/portal/kk837/depthbasedRL/.venv_isaacsim/bin/python \
        isaacsimenvs/tests/test_gym_register.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # AppLauncher not needed — this test is config-registry-only.
    # But `isaaclab.envs.utils.spaces` pulls in `isaaclab.envs` which has a
    # module-level `ensure_app_launcher`. Short-circuit by booting a minimal
    # headless launcher.
    from isaaclab.app import AppLauncher
    import argparse

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])
    app = AppLauncher(args).app

    import gymnasium as gym
    import yaml

    import isaacsimenvs  # noqa: F401  triggers gym.register side effect
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaacsimenvs.tasks.cartpole.cartpole_env import CartpoleEnvCfg
    from isaacsimenvs.tasks.peg_in_hole.peg_in_hole_env_cfg import PegInHoleEnvCfg

    task_id = "Isaacsimenvs-Cartpole-Direct-v0"

    # 1. Registration succeeded.
    spec = gym.spec(task_id)
    print(f"[test] gym.spec({task_id!r}) → {spec.entry_point}")
    for key in (
        "env_cfg_entry_point",
        "env_cfg_yaml_entry_point",
        "rl_games_cfg_entry_point",
        "rl_games_sapg_cfg_entry_point",
    ):
        assert key in spec.kwargs, f"missing {key} in gym.register kwargs"
        print(f"  {key}: {spec.kwargs[key]}")

    # 2. env_cfg entry point resolves to the configclass.
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert isinstance(env_cfg, CartpoleEnvCfg), f"expected CartpoleEnvCfg, got {type(env_cfg)}"
    print(f"[test] configclass default num_envs = {env_cfg.scene.num_envs}")
    print(f"[test] configclass default sim.dt   = {env_cfg.sim.dt}")

    # 3. rl_games entry point is a YAML → dict.
    agent_cfg = load_cfg_from_registry(task_id, "rl_games_cfg_entry_point")
    assert isinstance(agent_cfg, dict), f"expected dict from rl_games YAML, got {type(agent_cfg)}"
    assert "params" in agent_cfg and "config" in agent_cfg["params"]
    print(f"[test] agent_cfg['params']['config']['name'] = {agent_cfg['params']['config']['name']}")

    # 4. Task YAML overlay applies cleanly onto the configclass.
    yaml_path = Path(spec.kwargs["env_cfg_yaml_entry_point"])
    with open(yaml_path) as f:
        overlay = yaml.safe_load(f) or {}

    # Phase 2 work: `clip_observations` / `clip_actions` are currently in the
    # task YAML (read directly by the old `train.py`). They move to the rl_games
    # agent YAML once the hydra trainer lands; until then, pop them so
    # `from_dict` (which is strict) doesn't reject them.
    overlay.pop("clip_observations", None)
    overlay.pop("clip_actions", None)

    # Pick a known overlay key and confirm round-trip.
    pre = env_cfg.scene.num_envs
    overlay["scene"]["num_envs"] = 77  # sentinel value unlikely to match default
    env_cfg.from_dict(overlay)
    assert env_cfg.scene.num_envs == 77, f"overlay did not apply: got {env_cfg.scene.num_envs}"
    print(f"[test] overlay applied: scene.num_envs {pre} → {env_cfg.scene.num_envs}")

    # Confirm the new record_camera_* fields round-trip through from_dict.
    overlay["record_camera_eye"] = [-5.0, 1.0, 3.0]
    env_cfg.from_dict(overlay)
    assert tuple(env_cfg.record_camera_eye) == (-5.0, 1.0, 3.0), env_cfg.record_camera_eye
    print(f"[test] overlay applied: record_camera_eye → {env_cfg.record_camera_eye}")

    # PegInHole teacher/student registrations share the dynamic problem config.
    for pih_task_id in (
        "Isaacsimenvs-PegInHole-Direct-v0",
        "Isaacsimenvs-PegInHoleDepthStudent-Direct-v0",
    ):
        pih_spec = gym.spec(pih_task_id)
        pih_cfg = load_cfg_from_registry(pih_task_id, "env_cfg_entry_point")
        assert isinstance(pih_cfg, PegInHoleEnvCfg), type(pih_cfg)
        with Path(pih_spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
            pih_overlay = yaml.safe_load(f) or {}
        pih_cfg.from_dict(pih_overlay)
        assert pih_cfg.peg_in_hole.problem == "peg.tol0p5mm"
        assert pih_cfg.peg_in_hole.goal_mode == "preInsertAndFinal"
        print(
            f"[test] {pih_task_id}: problem={pih_cfg.peg_in_hole.problem} "
            f"goal_mode={pih_cfg.peg_in_hole.goal_mode}"
        )

    print("[test] Phase 1 registration smoke test OK")
    import os
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
