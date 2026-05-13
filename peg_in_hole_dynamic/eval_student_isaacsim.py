#!/usr/bin/env python3
"""Isaac Sim peg-in-hole eval with the same Viser UI as ``eval_isaacsim.py``,
plus an Action-Source dropdown that toggles between **teacher** (rl_games SAPG
checkpoint), **student** (DepthCNNLSTMBuilder + .pth or random-init), and
**zero** actions, and a depth panel that shows the student's policy-input
depth tensor live as the episode runs.

Architecture mirrors ``eval_isaacsim.py``:
  - parent process owns Viser (subclass of ``eval_isaacgym.PegDynamicDemo``)
  - child sim worker is launched via ``--worker`` and talks to the parent
    over a ``multiprocessing.connection`` Pipe
  - the worker loads BOTH the teacher (rl_games player) and the student
    (custom rl_games network), and per-step picks the action by looking at
    a shared ``action_source`` updated via a parent → child message.

Single-process is not viable because Isaac Sim's Kit init is one-shot per
process, so swapping problem/checkpoint requires spawning a fresh worker —
exactly the same constraint that drove ``eval_isaacsim.py``'s split.

Examples:
    .venv_isaacsim/bin/python peg_in_hole_dynamic/eval_student_isaacsim.py \\
        --teacher-checkpoint .../last/model.pth \\
        --student-checkpoint .../student.pth   # optional; else random-init
        --problem Lpeg.tol0p5mm --port 8081
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers from the teacher eval. They're already battle-tested for env
# bootstrap, hydra cfg loading, and rl_games player setup.
from peg_in_hole_dynamic.eval_isaacsim import (  # noqa: E402
    CONTROL_DT,
    DEFAULT_AGENT,
    DEFAULT_PROBLEM,
    DEFAULT_TASK,
    GOAL_MODES,
    PopenAdapter,
    _apply_env_overrides,
    _configure_agent,
    _coerce_override_value,
    _default_isaacsim_python,
    _done0,
    _instantiate_env,
    _load_env_cfg,
    _resolve_path,
    _sim_get_state,
    _tensor_bool,
    _tensor_int,
)


# =====================================================================
# CHILD PROCESS — sim worker
# =====================================================================


def _student_obs_flat(env, image_channels: int, image_hw: tuple,
                      proprio_dim: int, has_block_id: bool):
    """Build the flat obs tensor the student net expects from get_student_obs."""
    import torch
    out = env.unwrapped.get_student_obs()
    image = out["image"]                                                  # (N, C, H, W)
    proprio = out["proprio"]                                              # (N, P)
    if has_block_id:
        # No SAPG block-id signal in the eval env; assume block 0.
        block_id = torch.zeros(image.shape[0], 1, device=image.device)
        flat = torch.cat([image.flatten(1), proprio, block_id], dim=-1)
    else:
        flat = torch.cat([image.flatten(1), proprio], dim=-1)
    return flat, image, proprio


def _depth_to_rgb_uint8(depth_t, depth_min: float, depth_max: float):
    """Map a 2D depth tensor to a (H, W, 3) uint8 grayscale RGB image."""
    import numpy as np
    d = depth_t.detach().float().clamp(min=depth_min, max=depth_max)
    norm = (d - depth_min) / max(depth_max - depth_min, 1e-6)
    arr = (norm.cpu().numpy() * 255.0).astype("uint8")
    return np.stack([arr, arr, arr], axis=-1)


def _normalized_to_rgb_uint8(norm_t):
    """Map a 2D [0,1] normalized depth tensor to a (H, W, 3) uint8 grayscale image."""
    import numpy as np
    d = norm_t.detach().float().clamp(0.0, 1.0)
    arr = (d.cpu().numpy() * 255.0).astype("uint8")
    return np.stack([arr, arr, arr], axis=-1)


def _build_student(net_params: dict, action_dim: int, image_channels: int,
                   image_hw: tuple, proprio_dim: int, has_block_id: bool,
                   num_blocks: int, student_checkpoint: str | None, device: str):
    """Build the depth_cnn_lstm rl_games network and (optionally) load weights."""
    import torch
    from rl_games.algos_torch import model_builder
    # Importing isaacsimenvs.dagger.networks registers 'depth_cnn_lstm'.
    from isaacsimenvs.dagger import networks as _net  # noqa: F401

    builder = model_builder.NETWORK_REGISTRY["depth_cnn_lstm"]()
    builder.load(net_params)
    obs_dim = (
        image_channels * image_hw[0] * image_hw[1]
        + proprio_dim
        + (1 if has_block_id else 0)
    )
    build_kwargs = {"actions_num": action_dim, "input_shape": (obs_dim,)}
    if net_params.get("space", {}).get("continuous", {}).get("fixed_sigma") == "coef_cond":
        build_kwargs["coef_ids"] = torch.arange(num_blocks, dtype=torch.float32)
        build_kwargs["coef_id_idx"] = obs_dim - 1
    student = builder.build("student", **build_kwargs).to(device)
    if student_checkpoint:
        sd = torch.load(student_checkpoint, map_location=device)
        student.load_state_dict(sd.get("model", sd), strict=False)
        print(f"=> loaded student weights from '{student_checkpoint}'", flush=True)
    else:
        print("=> student is random-init", flush=True)
    student.eval()
    return student


def _student_episode(
    conn,
    env,
    wrapped,
    teacher_player,
    student,
    *,
    deterministic: bool,
    action_source: str,
    image_channels: int,
    image_hw: tuple,
    proprio_dim: int,
    has_block_id: bool,
    depth_min: float,
    depth_max: float,
    env_id: int,
    depth_send_every: int,
):
    """Mirror ``eval_isaacsim._sim_episode`` but supports a runtime
    teacher/student/zero toggle and emits a 'depth' message every
    ``depth_send_every`` steps with the env_id'th policy-input image."""
    import torch
    teacher_player.reset()
    obs = teacher_player.env_reset(wrapped)
    rnn_state = None  # student LSTM state, lazily initialized via the network's defaults

    paused = False
    step = 0
    done = False
    peak_successes = 0
    max_goals_seen = max(1, _tensor_int(env.env_max_goals[0]))
    retract_ok = False

    while not done:
        # Drain command messages — including action-source updates.
        while conn.poll(0):
            cmd = conn.recv()
            if isinstance(cmd, tuple) and len(cmd) >= 1 and cmd[0] == "set_action_source":
                action_source = str(cmd[1])
                print(f"[worker] action_source -> {action_source}", flush=True)
                continue
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return None
            elif cmd == "quit":
                return "quit"

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()

        # --- teacher action (always computed; powers the L2 overlay) ---
        teacher_action = teacher_player.get_action(obs, is_deterministic=True)

        # --- student action ---
        flat_obs, image_t, proprio_t = _student_obs_flat(
            env, image_channels, image_hw, proprio_dim, has_block_id,
        )
        with torch.no_grad():
            mu, _logstd, _value, rnn_state = student(
                {"obs": flat_obs, "rnn_states": rnn_state, "seq_length": 1}
            )
            student_action = mu

        if action_source == "teacher":
            act = teacher_action
        elif action_source == "student":
            act = student_action
        else:  # zero
            act = torch.zeros_like(teacher_action)

        obs, _rew, dones, _infos = teacher_player.env_step(wrapped, act)
        done = _done0(dones)
        step += 1

        cur_succ = _tensor_int(env._successes[0])
        cur_max = _tensor_int(env.env_max_goals[0])
        peak_successes = max(peak_successes, cur_succ)
        max_goals_seen = max(max_goals_seen, cur_max)
        retract_ok = retract_ok or _tensor_bool(env.retract_succeeded[0])
        state = _sim_get_state(env, done_pending=done)
        conn.send(("state", state, cur_succ, cur_max, step, retract_ok))

        if step % depth_send_every == 0:
            policy_rgb = _normalized_to_rgb_uint8(image_t[env_id, 0])
            l2 = float((student_action - teacher_action).pow(2).mean().sqrt().detach().cpu())
            conn.send(("depth", policy_rgb, l2, action_source))

        sleep = CONTROL_DT - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

    goal_pct = 100.0 * peak_successes / max(1, max_goals_seen)
    conn.send(("done", goal_pct, step, retract_ok))
    return None


def sim_worker(
    conn,
    *,
    teacher_checkpoint_path: str,
    student_checkpoint_path: str | None,
    task: str,
    agent: str,
    student_agent: str,
    problem: str,
    goal_mode: str,
    random_goal_fraction: float,
    insertion_success_tolerance: float,
    retract_success_tolerance: float,
    num_envs: int,
    games: int,
    rl_device: str,
    sim_device: str,
    deterministic: bool,
    headless: bool,
    sdf: bool,
    keep_dr: bool,
    extra_overrides: dict,
    initial_action_source: str,
    env_id: int,
    depth_send_every: int,
) -> None:
    app = None
    env = None
    try:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

        from isaaclab.app import AppLauncher

        launcher_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(launcher_parser)
        launcher_args, _ = launcher_parser.parse_known_args([])
        launcher_args.headless = bool(headless)
        # Cameras must be enabled — the env's depth pipeline needs the TiledCamera.
        launcher_args.enable_cameras = True
        app = AppLauncher(launcher_args).app

        import isaacsimenvs  # noqa: F401  triggers gym.register
        from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
        from rl_games.torch_runner import Runner, _load_checkpoint_weights

        cfg = _load_env_cfg(task)
        _apply_env_overrides(
            cfg,
            problem=problem,
            goal_mode=goal_mode,
            random_goal_fraction=random_goal_fraction,
            insertion_success_tolerance=insertion_success_tolerance,
            retract_success_tolerance=retract_success_tolerance,
            num_envs=num_envs,
            sim_device=sim_device,
            sdf=sdf,
            keep_dr=keep_dr,
            extra_overrides=extra_overrides,
        )
        env = _instantiate_env(task, cfg)

        agent_cfg = _configure_agent(
            task,
            agent,
            rl_device=rl_device,
            num_envs=num_envs,
            deterministic=deterministic,
            games=games,
            extra_overrides=extra_overrides,
        )
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
        wrapped = register_rlgames_env(
            env, rl_device=rl_device, clip_obs=clip_obs, clip_actions=clip_actions
        )

        # ---- teacher player (rl_games SAPG) ----
        runner = Runner()
        runner.load(agent_cfg)
        runner.reset()
        teacher_player = runner.create_player()
        weights = _load_checkpoint_weights(teacher_player, str(teacher_checkpoint_path))
        teacher_player.set_weights(weights)
        teacher_player.has_batch_dimension = True
        teacher_player.reset()
        print(f"=> teacher loaded from '{teacher_checkpoint_path}'", flush=True)

        # ---- student (depth_cnn_lstm) ----
        # Build the student net from the dagger train yaml's network params.
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        student_agent_cfg = load_cfg_from_registry(task, student_agent)
        net_params = student_agent_cfg["params"]["network"]
        # Pull image / proprio shapes off the live env so we can sanity-check
        # the yaml against actual env output.
        student_obs0 = env.get_student_obs()
        actual_proprio_dim = int(student_obs0["proprio"].shape[-1])
        actual_image_shape = tuple(student_obs0["image"].shape[1:])           # (C, H, W)
        image_channels = int(net_params.get("image_channels", 1))
        image_hw = tuple(net_params["image_hw"])
        has_block_id = bool(net_params.get("has_block_id", True))
        num_blocks = int(student_agent_cfg["params"]["config"].get("expl_coef_block_size", 1))
        if actual_proprio_dim != int(net_params.get("proprio_dim", -1)):
            print(
                f"[worker] WARNING: yaml proprio_dim={net_params.get('proprio_dim')} but env "
                f"emits proprio_dim={actual_proprio_dim}. Overriding net_params.",
                flush=True,
            )
            net_params = dict(net_params)
            net_params["proprio_dim"] = actual_proprio_dim

        action_dim = int(env.action_space.shape[-1])
        student = _build_student(
            net_params=net_params,
            action_dim=action_dim,
            image_channels=image_channels,
            image_hw=image_hw,
            proprio_dim=int(net_params["proprio_dim"]),
            has_block_id=has_block_id,
            num_blocks=max(num_blocks, 1),
            student_checkpoint=student_checkpoint_path,
            device=rl_device,
        )

        depth_min = float(cfg.student_obs.depth_min_m)
        depth_max = float(cfg.student_obs.depth_max_m)

        teacher_player.reset()
        obs = teacher_player.env_reset(wrapped)
        del obs
        conn.send(("ready", _sim_get_state(env)))

        action_source = initial_action_source
        while True:
            cmd = conn.recv()
            if isinstance(cmd, tuple) and cmd[0] == "set_action_source":
                action_source = str(cmd[1])
                print(f"[worker] action_source -> {action_source}", flush=True)
                continue
            if cmd == "run":
                result = _student_episode(
                    conn,
                    env,
                    wrapped,
                    teacher_player,
                    student,
                    deterministic=deterministic,
                    action_source=action_source,
                    image_channels=image_channels,
                    image_hw=image_hw,
                    proprio_dim=int(net_params["proprio_dim"]),
                    has_block_id=has_block_id,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    env_id=env_id,
                    depth_send_every=depth_send_every,
                )
                if result == "quit":
                    break
            elif cmd == "quit":
                break
    except Exception as exc:
        try:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
        except Exception:
            pass
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            if app is not None:
                del app
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


# =====================================================================
# PARENT PROCESS — viser GUI (subclass of PegDynamicDemo)
# =====================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0",
                        help="Must be a depth-student task (env exposes get_student_obs() + has dagger entry points).")
    parser.add_argument("--agent", default=DEFAULT_AGENT,
                        help="Hydra entry point for the *teacher* train yaml.")
    parser.add_argument("--student-agent", default="rl_games_dagger_sapg_cfg_entry_point",
                        help="Hydra entry point for the *student* train yaml. The network "
                             "block is read from here.")
    parser.add_argument("--python", default=None, help="Python executable for the Isaac Sim worker.")
    parser.add_argument("--teacher-checkpoint", required=True,
                        help="Frozen teacher .pth (rl_games format).")
    parser.add_argument("--student-checkpoint", default=None,
                        help="Optional student .pth. If omitted, student is random-init.")
    parser.add_argument("--problem", default=None)
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="preInsertAndFinal")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--random-goal-fraction", type=float, default=0.0)
    parser.add_argument("--insertion-success-tolerance", type=float, default=0.005)
    parser.add_argument("--retract-success-tolerance", type=float, default=0.005)
    parser.add_argument("--eval-success-tolerance", type=float, default=None)
    parser.add_argument("--sdf", action="store_true")
    parser.add_argument("--keep-dr", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--initial-action-source", choices=("teacher", "student", "zero"),
                        default="teacher")
    parser.add_argument("--env-id", type=int, default=0,
                        help="Which env's depth obs to display in the viser image panel.")
    parser.add_argument("--depth-send-every", type=int, default=4,
                        help="Send the depth panel update every N env steps.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--override", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
        help="Extra env/agent override, e.g. --override env.reward.lifting_bonus 0.0",
    )

    # Hidden worker mode args.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--worker-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-authkey", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--override-json", default="{}", help=argparse.SUPPRESS)
    return parser


def _worker_main(args) -> int:
    from multiprocessing.connection import Client

    conn = Client(
        (args.worker_host, int(args.worker_port)),
        authkey=bytes.fromhex(args.worker_authkey),
    )
    extra_overrides = json.loads(args.override_json or "{}")
    sim_worker(
        conn,
        teacher_checkpoint_path=args.teacher_checkpoint,
        student_checkpoint_path=args.student_checkpoint,
        task=args.task,
        agent=args.agent,
        student_agent=args.student_agent,
        problem=args.problem,
        goal_mode=args.goal_mode,
        random_goal_fraction=args.random_goal_fraction,
        insertion_success_tolerance=args.insertion_success_tolerance,
        retract_success_tolerance=args.retract_success_tolerance,
        num_envs=args.num_envs,
        games=args.games,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        deterministic=args.deterministic,
        headless=args.worker_headless,
        sdf=args.sdf,
        keep_dr=args.keep_dr,
        extra_overrides=extra_overrides,
        initial_action_source=args.initial_action_source,
        env_id=args.env_id,
        depth_send_every=args.depth_send_every,
    )
    return 0


def _run_viewer(args) -> int:
    from multiprocessing.connection import Listener

    import numpy as np
    import peg_in_hole_dynamic.eval_isaacgym as gym_eval

    teacher_ckpt = _resolve_path(args.teacher_checkpoint)
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"--teacher-checkpoint not found: {teacher_ckpt}")
    student_ckpt = (
        _resolve_path(args.student_checkpoint) if args.student_checkpoint else None
    )
    if student_ckpt is not None and not student_ckpt.is_file():
        raise FileNotFoundError(f"--student-checkpoint not found: {student_ckpt}")

    initial_problem = args.problem or DEFAULT_PROBLEM
    extra_overrides = {key: _coerce_override_value(value) for key, value in args.override}
    if args.eval_success_tolerance is not None:
        extra_overrides["env.termination.eval_success_tolerance"] = float(
            args.eval_success_tolerance
        )

    # The base-class `policies` dict expects (config_path, checkpoint_path) tuples.
    # We register one entry under the teacher checkpoint name; the student/teacher/zero
    # selection happens via our extra dropdown rather than the base "Policy" dropdown.
    policy_name = teacher_ckpt.parent.name or "teacher"
    viewer_policies = {policy_name: ("", str(teacher_ckpt))}

    class StudentEvalDemo(gym_eval.PegDynamicDemo):
        def __init__(self, *demo_args, **demo_kwargs):
            self.task = args.task
            self.agent = args.agent
            self.student_agent = args.student_agent
            self.worker_python = args.python or _default_isaacsim_python()
            self.num_envs = int(args.num_envs)
            self.games = int(args.games)
            self.rl_device = args.rl_device
            self.sim_device = args.sim_device
            self.deterministic = bool(args.deterministic)
            self.sdf = bool(args.sdf)
            self.keep_dr = bool(args.keep_dr)
            self.teacher_ckpt = teacher_ckpt
            self.student_ckpt = student_ckpt
            self.action_source = args.initial_action_source
            self.env_id = int(args.env_id)
            self.depth_send_every = int(args.depth_send_every)
            super().__init__(*demo_args, **demo_kwargs)
            self._sl_insertion_tol.value = float(args.insertion_success_tolerance)
            self._sl_retract_tol.value = float(args.retract_success_tolerance)
            self._add_student_gui()

        def _add_student_gui(self):
            with self.server.gui.add_folder("Student Eval", expand_by_default=True):
                self._dd_action_source = self.server.gui.add_dropdown(
                    "Action source",
                    ("teacher", "student", "zero"),
                    initial_value=self.action_source,
                )
                self._dd_action_source.on_update(lambda _: self._cmd_set_action_source())
                self._md_action_l2 = self.server.gui.add_markdown(
                    "**L2(student-teacher):** --"
                )
                ckpt_label = (
                    str(self.student_ckpt) if self.student_ckpt is not None else "random-init"
                )
                self.server.gui.add_markdown(f"**Student checkpoint:** `{ckpt_label}`")
                self.server.gui.add_markdown(
                    f"**Teacher checkpoint:** `{self.teacher_ckpt}`"
                )
            with self.server.gui.add_folder("Depth obs (env_id={})".format(self.env_id),
                                            expand_by_default=True):
                import numpy as _np
                self._img_policy = self.server.gui.add_image(
                    _np.zeros((10, 10, 3), dtype=_np.uint8),
                    label="policy-input depth (normalized)",
                )

        def _cmd_set_action_source(self):
            self.action_source = self._dd_action_source.value
            if self._conn is not None:
                try:
                    self._conn.send(("set_action_source", self.action_source))
                except Exception as exc:
                    print(f"[launcher] failed to send action_source: {exc}")

        def _handle(self, msg):
            tag = msg[0]
            if tag == "depth":
                policy_rgb = msg[1]
                l2 = float(msg[2])
                src = msg[3] if len(msg) > 3 else self.action_source
                self._img_policy.image = policy_rgb
                self._md_action_l2.content = (
                    f"**L2(student−teacher):** {l2:.4f} &nbsp;|&nbsp; "
                    f"**Active source:** {src}"
                )
                return
            super()._handle(msg)

        def _load_env(self):
            problem_name = self._dd_problem.value
            goal_mode = self._dd_goal_mode.value
            rgf = self._sl_rgf.value
            insertion_tol = float(self._sl_insertion_tol.value)
            retract_tol = float(self._sl_retract_tol.value)

            try:
                self._set_problem_assets(problem_name)
                self._reload_problem_meshes()
            except Exception as exc:
                self._md_status.content = (
                    f"**Status:** Problem load error -- {str(exc)[:200]}"
                )
                print(
                    f"[student launcher] Problem load error for {problem_name!r}:\n"
                    f"{traceback.format_exc()}"
                )
                return

            self._kill_subprocess()

            label = (
                f"{problem_name} | source: {self.action_source} | "
                f"goals: {goal_mode} | rgf: {rgf:.1f} | "
                f"ins: {insertion_tol * 1000:.1f}mm | ret: {retract_tol * 1000:.1f}mm"
            )
            self._md_status.content = f"**Status:** Loading Isaac Sim *{label}* ..."
            self._md_task.content = f"**Task:** Isaac Sim Student | {label}"
            self._md_retract.content = "**Retract:** --"
            self._md_hole.content = "**Hole pos:** --"
            self._md_object_pose.content = "**Object pose:** --"
            self._md_goal_pose.content = "**Goal pose:** --"
            self._md_pose_delta.content = "**Object-goal z dist:** --"
            self.ep_count = 0
            self._peak_force = 0.0
            self._md_stats.content = "**Stats:** No episodes yet"

            self.robot.update_cfg(gym_eval.DEFAULT_DOF_POS)
            self._setup_scene_objects()

            worker_overrides = dict(self.extra_overrides)
            authkey = os.urandom(16)
            listener = Listener(("127.0.0.1", 0), authkey=authkey)
            host, port = listener.address
            listener._listener._socket.settimeout(60.0)

            cmd = [
                self.worker_python,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-host", str(host),
                "--worker-port", str(port),
                "--worker-authkey", authkey.hex(),
                "--task", self.task,
                "--agent", self.agent,
                "--student-agent", self.student_agent,
                "--teacher-checkpoint", str(self.teacher_ckpt),
                "--problem", problem_name,
                "--goal-mode", goal_mode,
                "--num-envs", str(self.num_envs),
                "--games", str(self.games),
                "--random-goal-fraction", str(float(rgf)),
                "--insertion-success-tolerance", str(insertion_tol),
                "--retract-success-tolerance", str(retract_tol),
                "--rl-device", self.rl_device,
                "--sim-device", self.sim_device,
                "--initial-action-source", self.action_source,
                "--env-id", str(self.env_id),
                "--depth-send-every", str(self.depth_send_every),
                "--override-json", json.dumps(worker_overrides),
            ]
            if self.student_ckpt is not None:
                cmd += ["--student-checkpoint", str(self.student_ckpt)]
            if self.deterministic:
                cmd.append("--deterministic")
            if self.sdf:
                cmd.append("--sdf")
            if self.keep_dr:
                cmd.append("--keep-dr")
            if self.headless:
                cmd.append("--worker-headless")

            env_vars = os.environ.copy()
            env_vars.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            try:
                popen = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env_vars)
                self._proc = PopenAdapter(popen)
                self._conn = listener.accept()
            except Exception as exc:
                listener.close()
                if self._proc is not None:
                    self._proc.kill()
                    self._proc.join(timeout=2)
                    self._proc = None
                self._md_status.content = (
                    f"**Status:** Worker launch error -- {str(exc)[:200]}"
                )
                print(
                    "[student launcher] Worker launch error:\n"
                    f"{traceback.format_exc()}"
                )
                return
            finally:
                try:
                    listener.close()
                except Exception:
                    pass

            print(
                f"[student launcher] Spawned pid={self._proc.pid} problem={problem_name} "
                f"goal_mode={goal_mode} action_source={self.action_source}"
            )

        def run(self):
            print()
            print(f"  Peg-in-Hole Student/Teacher Eval   http://localhost:{self.port}")
            print()
            try:
                while True:
                    self._poll()
                    time.sleep(1.0 / 120.0)
            except KeyboardInterrupt:
                print("\n[student launcher] Shutting down...")
                self._kill_subprocess()

    if args.dry_run:
        print(f"[eval_student_isaacsim] worker python: {args.python or _default_isaacsim_python()}")
        print(f"[eval_student_isaacsim] teacher: {teacher_ckpt}")
        print(f"[eval_student_isaacsim] student: {student_ckpt or '(random init)'}")
        print(f"[eval_student_isaacsim] problem: {initial_problem}")
        print(f"[eval_student_isaacsim] port: {args.port}")
        return 0

    StudentEvalDemo(
        policies=viewer_policies,
        port=args.port,
        headless=not args.no_headless,
        goal_mode=args.goal_mode,
        random_goal_fraction=args.random_goal_fraction,
        initial_policy=policy_name,
        extra_overrides=extra_overrides,
        initial_problem=initial_problem,
    ).run()
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.worker:
        return _worker_main(args)
    return _run_viewer(args)


if __name__ == "__main__":
    sys.exit(main())
