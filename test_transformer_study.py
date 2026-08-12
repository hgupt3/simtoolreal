from pathlib import Path
from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from isaacgymenvs.utils.study_objects import (
    allegro_kuka_small_cuboid_scales,
)
from isaacgymenvs.utils.simple_rl_env_wrapper import SimpleRLEnvWrapper


def test_parity_configs_resolve_all_hidden_optimizer_defaults() -> None:
    cfg_dir = str((Path(__file__).parent / "isaacgymenvs" / "cfg").resolve())
    with initialize_config_dir(version_base="1.1", config_dir=cfg_dir):
        for config_name, actor_value_key in (
            ("SimToolRealStudyCurrentLSTMSAPGParity", True),
            ("SimToolRealStudyLegacyLSTMSAPGParity", None),
        ):
            cfg = compose(
                config_name="config",
                overrides=[
                    "task=SimToolReal",
                    f"train={config_name}",
                    "num_envs=6144",
                ],
            )
            ppo = OmegaConf.to_container(cfg.train.ppo, resolve=True)
            assert ppo["num_actors"] == 6144
            assert ppo["lr_schedule"] == "adaptive"
            assert ppo["schedule_type"] == "standard"
            assert ppo["kl_threshold"] == 0.016
            assert ppo["bounds_loss_coef"] == 0.0001
            assert ppo["asymmetric_critic"]["e_clip"] == 0.2
            if actor_value_key is not None:
                assert ppo["train_actor_value_with_asymmetric_critic"] is True


def test_current_actor_value_loss_defaults_to_historical_parity() -> None:
    from simple_rl.agent import Agent, PpoConfig

    assert PpoConfig.__dataclass_fields__[
        "train_actor_value_with_asymmetric_critic"
    ].default is True
    agent = Agent.__new__(Agent)
    agent.has_asymmetric_critic = True
    agent.cfg = SimpleNamespace(train_actor_value_with_asymmetric_critic=True)
    assert agent.trains_actor_value
    agent.cfg.train_actor_value_with_asymmetric_critic = False
    assert not agent.trains_actor_value


def test_easy_cuboid_pool_matches_allegro_kuka_bounds() -> None:
    scales = allegro_kuka_small_cuboid_scales()
    volumes = np.prod(np.asarray(scales), axis=1)
    masses = volumes * 400.0

    assert len(scales) == 654
    assert len(set(scales)) == 654
    assert np.isclose(masses.min(), 0.05)
    assert np.isclose(masses.max(), 0.125)
    assert np.isclose(np.min(scales), 0.025)
    assert np.isclose(np.max(scales), 0.15)


def test_simple_rl_wrapper_exposes_privileged_state_space() -> None:
    state_space = object()
    env = type(
        "Env",
        (),
        {
            "observation_space": object(),
            "action_space": object(),
            "state_space": state_space,
            "num_states": 10,
        },
    )()
    assert SimpleRLEnvWrapper(env).get_env_info()["state_space"] is state_space


def test_legacy_batch_patch_applies_to_april_snapshot(tmp_path: Path) -> None:
    # The launcher's extraction/apply path is exercised without importing Isaac Gym.
    import subprocess

    repo = Path(__file__).resolve().parent
    archive = subprocess.Popen(
        ["git", "archive", "5e831aaa", "simple_rl"], cwd=repo, stdout=subprocess.PIPE
    )
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], stdin=archive.stdout, check=True)
    assert archive.stdout is not None
    archive.stdout.close()
    assert archive.wait() == 0
    subprocess.run(
        [
            "git",
            "apply",
            "--check",
            str(repo / "study" / "legacy_simple_rl_batch_compat.patch"),
        ],
        cwd=tmp_path,
        check=True,
    )
