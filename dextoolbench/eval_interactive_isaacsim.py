"""DexToolBench Interactive Policy Demo — Isaac Sim (Isaac Lab) backend.

Same viser UI as ``eval_interactive.py`` (it is reused directly); only the
simulator child process differs: instead of Isaac Gym, the worker boots Kit
via AppLauncher and runs ``Isaacsimenvs-SimToolReal-Direct-v0``.

Run inside the Isaac Sim venv:

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python dextoolbench/eval_interactive_isaacsim.py \\
        --config-path pretrained_policy/config.yaml \\
        --checkpoint-path pretrained_policy/model.pth

Notes:
- Each "Load Environment" click spawns a fresh subprocess (Kit cannot cleanly
  tear down in-process, same constraint as Isaac Gym). Kit boot takes ~1 min.
- The worker exits via os._exit(0) to dodge the Kit shutdown hang.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from dextoolbench.eval_interactive import (
    REPO_ROOT,
    TABLE_Z,
    Z_OFFSET,
    InteractiveDemo,
)

CONTROL_DT = 1.0 / 60.0


def _write_isaac_trajectory(goals_xyzw, out_dir: Path) -> str:
    """gym-format goals ([x,y,z,qx,qy,qz,qw] list) -> fixed_trajectory_file
    format (pos (1,K,3), quat wxyz (1,K,4))."""
    pos = [[g[:3] for g in goals_xyzw]]
    quat_wxyz = [[[g[6], g[3], g[4], g[5]] for g in goals_xyzw]]
    path = out_dir / "trajectory_isaac_format.json"
    with open(path, "w") as f:
        json.dump({"pos": pos, "quat_wxyz": quat_wxyz}, f)
    return str(path)


def _sim_get_state(inner, obs):
    """Extract visualisation state (joint_pos, object pose, goal pose) in the
    same format the gym worker sends: poses are env-local, quats xyzw."""
    import numpy as np

    obs_np = obs["policy"][0].detach().cpu().numpy()
    lower = inner._joint_lower_canon.detach().cpu().numpy()
    upper = inner._joint_upper_canon.detach().cpu().numpy()
    joint_pos = 0.5 * (obs_np[:29] + 1.0) * (upper - lower) + lower

    origin = inner.scene.env_origins[0].detach().cpu().numpy()

    def _pose7(rigid):
        pos = rigid.data.root_pos_w[0].detach().cpu().numpy() - origin
        wxyz = rigid.data.root_quat_w[0].detach().cpu().numpy()
        xyzw = wxyz[[1, 2, 3, 0]]
        return np.concatenate([pos, xyzw]).astype(np.float32)

    return joint_pos, _pose7(inner.object), _pose7(inner.goal_viz)


def _sim_episode(conn, env, inner, player, n_act, n_goals, rl_device):
    """Run one episode, streaming state to the parent via *conn*."""
    import time

    import torch

    player.player.init_rnn()
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(torch.zeros((1, n_act), device=inner.device))

    step, done, paused = 0, False, False
    goals_reached = 0

    while not done:
        while conn.poll(0):
            cmd = conn.recv()
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()

        state = _sim_get_state(inner, obs)
        policy_obs = obs["policy"].to(rl_device)
        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        obs, _, terminated, truncated, _ = env.step(action.to(inner.device))
        done = bool(terminated[0].item() or truncated[0].item())
        if done:
            goals_reached = max(goals_reached, int(inner._prev_episode_successes[0].item()))
        else:
            goals_reached = max(goals_reached, int(inner._successes[0].item()))
        step += 1

        conn.send(("state", state, goals_reached, n_goals, step))

        elapsed = time.time() - t0
        if (sleep := CONTROL_DT - elapsed) > 0:
            time.sleep(sleep)

    goal_pct = 100 * goals_reached / n_goals
    conn.send(("done", goal_pct, step))


def sim_worker_isaacsim(conn, category, object_name, task_name, table_urdf,
                        config_path, checkpoint_path):
    """Child process entry-point: boots Kit, creates the env, waits for commands."""
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        launcher_args = parser.parse_args([])
        launcher_args.headless = True
        app = AppLauncher(launcher_args).app  # noqa: F841

        import gymnasium as gym
        import torch

        import isaacsimenvs  # noqa: F401  registers gym envs
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
        from deployment.rl_player import RlPlayer
        from dextoolbench.objects import NAME_TO_OBJECT

        rl_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Trajectory (gym format) -> fixed_trajectory_file (isaac format).
        traj_path = (
            REPO_ROOT / "dextoolbench" / "trajectories"
            / category / object_name / f"{task_name}.json"
        )
        with open(traj_path) as f:
            traj_data = json.load(f)
        traj_data["start_pose"][2] += Z_OFFSET
        n_goals = len(traj_data["goals"])
        tmp_dir = Path(tempfile.mkdtemp(prefix="dextoolbench_interactive_"))
        traj_file = _write_isaac_trajectory(traj_data["goals"], tmp_dir)

        obj = NAME_TO_OBJECT[object_name]

        cfg = SimToolRealEnvCfg()
        cfg.scene.num_envs = 1
        cfg.assets.object_urdf = str(obj.urdf_path)
        cfg.assets.object_scale = tuple(obj.scale)
        # table_urdf arrives relative to assets/ (gym convention).
        cfg.assets.table_urdf = str(REPO_ROOT / "assets" / table_urdf)

        rs = cfg.reset
        rs.reset_position_noise_x = 0.0
        rs.reset_position_noise_y = 0.0
        rs.reset_position_noise_z = 0.0
        rs.reset_dof_pos_random_interval_arm = 0.0
        rs.reset_dof_pos_random_interval_fingers = 0.0
        rs.reset_dof_vel_random_interval = 0.0
        rs.table_reset_z = TABLE_Z
        rs.table_reset_z_range = 0.0
        rs.start_arm_higher = True
        sp = traj_data["start_pose"]
        rs.fixed_start_pose = (sp[0], sp[1], sp[2], sp[6], sp[3], sp[4], sp[5])
        rs.fixed_trajectory_file = traj_file

        dr = cfg.domain_randomization
        dr.use_obs_delay = False
        dr.use_action_delay = False
        dr.use_object_state_delay_noise = False
        dr.object_scale_noise_multiplier_range = (1.0, 1.0)
        dr.joint_velocity_obs_noise_std = 0.0
        dr.force_scale = 0.0
        dr.torque_scale = 0.0
        dr.force_prob_range = (0.0001, 0.0001)
        dr.torque_prob_range = (0.0001, 0.0001)

        term = cfg.termination
        term.eval_success_tolerance = 0.01
        term.success_steps = 1
        term.max_consecutive_successes = n_goals

        env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        inner._replay_target_lab_order = None

        n_act = cfg.action_space
        player = RlPlayer(
            num_observations=inner.cfg.observation_space,
            num_actions=n_act,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=rl_device,
            num_envs=1,
        )

        player.player.init_rnn()
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(torch.zeros((1, n_act), device=inner.device))
        init_state = _sim_get_state(inner, obs)

        conn.send(("ready", init_state))

        while True:
            cmd = conn.recv()
            if cmd == "run":
                _sim_episode(conn, env, inner, player, n_act, n_goals, rl_device)
            elif cmd == "quit":
                break

    except Exception as exc:
        try:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
        except (BrokenPipeError, OSError):
            pass

    try:
        conn.close()
    except OSError:
        pass
    # Skip Kit teardown (it hangs); the parent already has everything it needs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DexToolBench Interactive Policy Demo (Isaac Sim backend)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--config-path", type=str, default="pretrained_policy/config.yaml",
        help="Path to the policy config YAML",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="pretrained_policy/model.pth",
        help="Path to the policy checkpoint",
    )
    args = parser.parse_args()
    InteractiveDemo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        port=args.port,
        worker_target=sim_worker_isaacsim,
    ).run()
