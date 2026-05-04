#!/usr/bin/env python3
"""Direct smoke test for FabricaEnv (no rl_games).

Loads the env with a small number of envs, runs a forced reset, steps zero
actions a few times, and asserts the single-insertion invariants:
  - per-env (part_idx, scene_idx) follows the round-robin assignment,
  - per-env scene URDF path matches scenes.npz,
  - reset_idx resamples start_idx in [0, M),
  - object_init_state[xy, quat] matches _si_start_poses_t[part, scene, start],
  - env_max_goals matches _si_traj_lengths_t[part, scene, start],
  - retract state and rewards are initialized,
  - per-env object asset name matches `<assembly>_<insertion_part>_coacd`,
  - goal-XY observation noise is sampled in ±goalXyObsNoise and the obs_buf
    keypoints_rel_goal slice differs from a clean rebuild by exactly -noise
    (states_buf / critic stays clean),
  - step() with zero actions does not crash and produces finite rewards.

Run:
    python fabrica/smoke_test_env.py task.env.assemblyName=beam_2x
    python fabrica/smoke_test_env.py task.env.assemblyName=beam_2x \\
        task.env.goalMode=preInsertAndFinal
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# isaacgym must be imported before torch — pulled in transitively.
from isaacgymenvs.tasks import isaacgym_task_map  # noqa: E402,F401

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaacgymenvs.utils.reformat import omegaconf_to_dict  # noqa: E402


@hydra.main(
    config_name="config",
    config_path=str(REPO / "isaacgymenvs" / "cfg"),
    version_base="1.1",
)
def main(cfg):
    # Force small, headless settings.
    cfg.task.name = "FabricaEnv"
    cfg.task.env.numEnvs = 8
    cfg.task.env.viserViz = False
    cfg.task.env.enableCameraSensors = False
    cfg.task.env.capture_video = False
    cfg.headless = True
    cfg.capture_video = False

    # Disable other obs randomization so the goal-noise signal is clean.
    cfg.task.env.useObsDelay = False
    cfg.task.env.useObjectStateDelayNoise = False
    cfg.task.env.jointVelocityObsNoiseStd = 0.0

    # Use the small scenes_smoke.npz so this works even while a full
    # `scenes.npz` regeneration is in progress.
    assembly = cfg.task.env.assemblyName
    smoke_path = REPO / "assets" / "urdf" / "fabrica" / assembly / "scenes_smoke.npz"
    if not smoke_path.exists():
        raise FileNotFoundError(
            f"{smoke_path} not found. Generate it with:\n  "
            f"python -m fabrica.scene_generation.generate_scenes "
            f"--assembly {assembly} --num-scenes-per-part 5 "
            f"--num-starts-per-scene 5 --output-name scenes_smoke.npz --force"
        )
    cfg.task.env.scenesFilename = "scenes_smoke.npz"
    print(f"[smoke] using scenesFilename=scenes_smoke.npz (from {smoke_path})")

    cfg_dict = omegaconf_to_dict(cfg.task)
    env_cls = isaacgym_task_map[cfg.task.name]
    env = env_cls(
        cfg=cfg_dict,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    num_envs = env.num_envs
    num_acts = env.num_actions
    P = env._mp_num_parts
    N = env._si_n_scenes
    M = env._si_m_starts
    print(f"\n[smoke] Env created: num_envs={num_envs}, num_actions={num_acts}, "
          f"goal_mode={env.goal_mode}, P={P}, N={N}, M={M}, "
          f"max_consecutive_successes={env.max_consecutive_successes}")
    print(f"[smoke]   env_part_idx  = {env._si_env_part_idx_t.cpu().tolist()}")
    print(f"[smoke]   env_scene_idx = {env._si_env_scene_idx_t.cpu().tolist()}")
    print(f"[smoke]   env_max_goals = {env.env_max_goals.cpu().tolist()}")

    # ── (1) Round-robin (part, scene) assignment ──
    combo = np.arange(num_envs) % (P * N)
    exp_part = combo // N
    exp_scene = combo % N
    got_part = env._si_env_part_idx_t.cpu().numpy()
    got_scene = env._si_env_scene_idx_t.cpu().numpy()
    assert np.array_equal(got_part, exp_part), (got_part, exp_part)
    assert np.array_equal(got_scene, exp_scene), (got_scene, exp_scene)
    print("[smoke] PASS: round-robin (part_idx, scene_idx) assignment")

    # ── (2) Per-env URDF path matches scenes.npz ──
    scene_urdf_paths = env._si_scene_urdf_paths
    for i in range(num_envs):
        expected = str(scene_urdf_paths[got_part[i], got_scene[i]])
        got = env._mp_env_table_urdfs[i]
        assert got == expected, (i, got, expected)
    print("[smoke] PASS: per-env table URDF path matches scenes.npz")

    # ── (3) Forced reset resamples start_idx ──
    all_envs = torch.arange(num_envs, device="cuda:0")
    env.reset_idx(all_envs, tensor_reset=True)
    s1 = env._si_env_start_idx_t.clone()
    print(f"\n[smoke] After 1st forced reset: start_idx = {s1.cpu().tolist()}")
    assert s1.shape == (num_envs,), s1.shape
    assert s1.dtype == torch.long, s1.dtype
    assert int(s1.min()) >= 0 and int(s1.max()) < M, (s1.min(), s1.max())

    env.reset_idx(all_envs, tensor_reset=True)
    s2 = env._si_env_start_idx_t.clone()
    print(f"[smoke] After 2nd forced reset: start_idx = {s2.cpu().tolist()}")
    # Probabilistic — over 8 envs at M >= 5, P(all unchanged) is tiny but not 0.
    if torch.equal(s1, s2):
        env.reset_idx(all_envs, tensor_reset=True)
        s2 = env._si_env_start_idx_t.clone()
    assert not torch.equal(s1, s2), "start_idx didn't change across two resets"
    print("[smoke] PASS: reset_idx resamples start_idx in [0, M)")

    # ── (4) object_init_state[xy, quat] matches start_poses ──
    parts = env._si_env_part_idx_t
    scenes = env._si_env_scene_idx_t
    starts = env._si_env_start_idx_t
    expected_poses = env._si_start_poses_t[parts, scenes, starts]            # (num_envs, 7)
    diff_xy = (env.object_init_state[:, 0:2] - expected_poses[:, 0:2]).abs().max().item()
    diff_quat = (env.object_init_state[:, 3:7] - expected_poses[:, 3:7]).abs().max().item()
    print(f"[smoke]   object_init vs expected: max |dxy|={diff_xy:.6e}  "
          f"max |dquat|={diff_quat:.6e}")
    assert diff_xy < 1e-5, (env.object_init_state[:, 0:2], expected_poses[:, 0:2])
    assert diff_quat < 1e-5
    print("[smoke] PASS: object_init_state[xy, quat] matches _si_start_poses_t")

    # ── (5) env_max_goals matches traj_lengths ──
    expected_max = env._si_traj_lengths_t[parts, scenes, starts]
    assert torch.equal(env.env_max_goals, expected_max), (
        env.env_max_goals, expected_max
    )
    print("[smoke] PASS: env_max_goals = _si_traj_lengths_t[part, scene, start]")

    # ── (6) Retract state initialized ──
    assert env.retract_phase.dtype == torch.bool and env.retract_phase.shape == (num_envs,)
    assert env.retract_succeeded.dtype == torch.bool
    assert "retract_rew" in env.rewards_episode
    print("[smoke] PASS: retract_phase / retract_succeeded / retract_rew initialized")

    # ── (7) Per-env object asset matches part_idx ──
    assembly = env.cfg["env"]["assemblyName"]
    insertion_parts = env._si_insertion_parts
    for p_idx in range(P):
        env_for_p = (got_part == p_idx).nonzero()[0]
        if len(env_for_p) == 0:
            continue
        expected_name = f"{assembly}_{insertion_parts[p_idx]}_coacd"
        got_name = env._mp_object_names[p_idx]
        assert got_name == expected_name, (p_idx, got_name, expected_name)
    print("[smoke] PASS: per-env object asset name matches part_idx")

    # ── (8) Goal-XY noise sampled in ±goalXyObsNoise; Z=0 ──
    goal_xy_bound = float(env.cfg["env"]["goalXyObsNoise"])
    assert env.goal_pos_obs_noise.shape == (num_envs, 3)
    noise = env.goal_pos_obs_noise.clone()
    assert noise[:, 0:2].abs().max().item() <= goal_xy_bound + 1e-9, (
        f"XY noise out of bound: {noise[:, 0:2].abs().max().item()} > {goal_xy_bound}"
    )
    assert noise[:, 2].abs().max().item() == 0.0, "Z noise should stay 0"
    print(f"[smoke] PASS: goal_pos_obs_noise in [-{goal_xy_bound},+{goal_xy_bound}] XY; Z=0")

    # ── (9) obs_buf keypoints_rel_goal differs from clean by exactly -noise ──
    actions = torch.zeros(num_envs, num_acts, device="cuda:0")
    env.step(actions)
    saved_noise = env.goal_pos_obs_noise.clone()
    kp_slice = env._goal_kp_obs_slice
    K = env.num_keypoints

    env.goal_pos_obs_noise.zero_()
    env.populate_obs_and_states_buffers()
    clean_obs = env.obs_buf.clone()
    clean_states = env.states_buf.clone()

    env.goal_pos_obs_noise.copy_(saved_noise)
    env.populate_obs_and_states_buffers()
    noisy_obs = env.obs_buf.clone()
    noisy_states = env.states_buf.clone()

    assert torch.allclose(clean_states, noisy_states, atol=1e-6), (
        "states_buf differs between noise-on/off — critic got noised!"
    )
    print("[smoke] PASS: states_buf identical between noise-on/off (critic clean)")

    diff = noisy_obs - clean_obs
    outside_mask = torch.ones(diff.shape[-1], dtype=torch.bool, device=diff.device)
    outside_mask[kp_slice] = False
    assert diff[:, outside_mask].abs().max().item() == 0.0, (
        "obs_buf changed outside keypoints_rel_goal slice"
    )
    expected = (-saved_noise).unsqueeze(1).expand(num_envs, K, 3).reshape(num_envs, -1)
    assert torch.allclose(diff[:, kp_slice], expected, atol=1e-6), (
        "obs_buf kp slice doesn't match expected -noise broadcast"
    )
    print("[smoke] PASS: obs_buf delta == -noise, only in keypoints_rel_goal slice")

    # ── (10) Step with zero actions × 3 ──
    for i in range(3):
        actions = torch.zeros(num_envs, num_acts, device="cuda:0")
        obs, rew, done, info = env.step(actions)
        assert torch.isfinite(rew).all(), f"non-finite reward at step {i}: {rew}"
        assert done.dtype in (torch.bool, torch.int64), done.dtype
        if isinstance(obs, dict):
            for k, v in obs.items():
                assert torch.isfinite(v).all(), f"non-finite obs[{k}] at step {i}"
        else:
            assert torch.isfinite(obs).all(), f"non-finite obs at step {i}"
        print(f"[smoke] step {i}: rew_mean={rew.mean().item():+.3f}  "
              f"done={int(done.sum())}  retract={int(env.retract_phase.sum())}")

    print("\n[smoke] ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
