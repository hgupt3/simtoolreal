from pathlib import Path

import numpy as np

from isaacgymenvs.utils.study_objects import (
    allegro_kuka_small_cuboid_scales,
)
from isaacgymenvs.utils.simple_rl_env_wrapper import SimpleRLEnvWrapper


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
