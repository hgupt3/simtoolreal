#!/usr/bin/env python
"""rospy node deploying the depth-CNN-LSTM student policy.

Sister of ``deployment/rl_policy_node.py`` (which serves the legacy 140-d
state-obs policy). This one:

  1. Subscribes to ``/iiwa/joint_states`` (7 DoF) and ``/sharpa/joint_states``
     (22 DoF) to build the 87-d proprio vector
     ``[joint_pos | joint_vel | prev_action_targets]``.
  2. Reads the latest preprocessed depth frame from a child ``ZedAsyncReader``
     process (window-normalized + cropped to 70x70 to match training).
  3. Runs the ``depth_cnn_lstm`` student at 60 Hz; appends a 0-valued
     block-id channel (the SAPG ``intr_reward_coef_embd`` slot) before
     forwarding through rl_games.
  4. Maps normalized actions to joint targets via the same recipe IsaacSim
     training used (delta-policy + moving avg for the 7 arm joints, scale-
     to-limits + moving avg for the 22 hand joints), then publishes
     ``/iiwa/joint_cmd`` and ``/sharpa/joint_cmd`` (sensor_msgs/JointState).

Examples:
    # Live deployment
    python deployment/depth_policy_node.py \\
        --checkpoint /path/to/best/model.pth \\
        --config     /path/to/train/.hydra/config.yaml

    # Inference latency sanity check (no ROS, no ZED)
    python deployment/depth_policy_node.py --benchmark \\
        --checkpoint /path/to/best/model.pth \\
        --config     /path/to/agent.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


N_ARM = 7
N_HAND = 22
N_ACTIONS = N_ARM + N_HAND  # 29
PROPRIO_DIM = 3 * N_ACTIONS  # joint_pos + joint_vel + prev_action_targets = 87


# ============================================================
# Joint limits — copied verbatim from
# isaacgymenvs/utils/observation_action_utils_sharpa.py so this ROS-only
# process doesn't have to import IsaacGym / Hydra / IsaacLab.
# Keep in sync with that file if the lab updates hardware.
# ============================================================

Q_LOWER_LIMITS_np = np.array(
    [
        -2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543,
        -0.1745, -0.3491, -0.5236, -0.3491,  0.0000, -0.1745, -0.0349,
         0.0000,  0.0000, -0.1745, -0.0349,  0.0000,  0.0000, -0.1745,
        -0.0349,  0.0000,  0.0000,  0.0000, -0.1745, -0.0349,  0.0000,
         0.0000,
    ],
    dtype=np.float64,
)
Q_UPPER_LIMITS_np = np.array(
    [
         2.9671,  2.0944,  2.9671,  2.0944,  2.9671,  2.0944,  3.0543,
         1.9199,  0.1309,  1.3963,  0.3491,  1.7453,  1.5708,  0.0349,
         1.7453,  1.3963,  1.5708,  0.0349,  1.7453,  1.3963,  1.5708,
         0.0349,  1.7453,  1.3963,  0.2618,  1.5708,  0.0349,  1.7453,
         1.3963,
    ],
    dtype=np.float64,
)
assert Q_LOWER_LIMITS_np.shape == (N_ACTIONS,)
assert Q_UPPER_LIMITS_np.shape == (N_ACTIONS,)

IIWA_JOINT_NAMES = [f"iiwa_joint_{i + 1}" for i in range(N_ARM)]
SHARPA_JOINT_NAMES = [f"joint_{i}.0" for i in range(N_HAND)]


# ============================================================
# Action mapping helpers (mirrors compute_joint_pos_targets in
# isaacgymenvs/utils/observation_action_utils_sharpa.py).
# ============================================================


def _scale_to_limits(
    actions: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Map ``actions`` in [-1, 1] to [lower, upper] elementwise."""
    return lower + 0.5 * (actions + 1.0) * (upper - lower)


def compute_joint_pos_targets(
    actions: np.ndarray,
    prev_targets: np.ndarray,
    hand_moving_average: float,
    arm_moving_average: float,
    hand_dof_speed_scale: float,
    dt: float,
) -> np.ndarray:
    """(N, 29) normalized actions -> (N, 29) clamped joint targets.

    - Hand joints 7..28: scale [-1,1] to limits + EMA blend + clamp.
    - Arm  joints 0..6:  delta-policy (prev + speed*dt*action) + clamp + EMA.
    Mirrors the training-time helper of the same name; reproduce inline to
    keep this module IsaacGym-import-free.
    """
    assert actions.shape == prev_targets.shape == (actions.shape[0], N_ACTIONS)
    assert 0.0 <= hand_moving_average <= 1.0
    assert 0.0 <= arm_moving_average <= 1.0

    cur_targets = prev_targets.copy()
    # Hand
    hand_lo = Q_LOWER_LIMITS_np[N_ARM:]
    hand_hi = Q_UPPER_LIMITS_np[N_ARM:]
    cur_targets[:, N_ARM:] = _scale_to_limits(
        actions[:, N_ARM:], hand_lo, hand_hi
    )
    cur_targets[:, N_ARM:] = (
        hand_moving_average * cur_targets[:, N_ARM:]
        + (1.0 - hand_moving_average) * prev_targets[:, N_ARM:]
    )
    cur_targets[:, N_ARM:] = np.clip(cur_targets[:, N_ARM:], hand_lo, hand_hi)

    # Arm
    arm_lo = Q_LOWER_LIMITS_np[:N_ARM]
    arm_hi = Q_UPPER_LIMITS_np[:N_ARM]
    cur_targets[:, :N_ARM] = (
        prev_targets[:, :N_ARM]
        + hand_dof_speed_scale * dt * actions[:, :N_ARM]
    )
    cur_targets[:, :N_ARM] = np.clip(cur_targets[:, :N_ARM], arm_lo, arm_hi)
    cur_targets[:, :N_ARM] = (
        arm_moving_average * cur_targets[:, :N_ARM]
        + (1.0 - arm_moving_average) * prev_targets[:, :N_ARM]
    )
    return cur_targets


# ============================================================
# Student policy loading. Mirrors `_build_student` in
# peg_in_hole_dynamic/eval_student_isaacsim.py — direct network build via the
# rl_games model_builder registry, weights loaded by hand. Avoids the
# 50-constant block-id quirk baked into deployment/rl_player.py.
# ============================================================


def _load_agent_yaml(path: Path) -> dict:
    with open(path) as fp:
        cfg = yaml.safe_load(fp)
    # rl_games yamls sometimes wrap the config under ``train`` / ``params`` /
    # ``config`` depending on how they were saved. Normalize to a flat dict
    # rooted at ``params``.
    if "train" in cfg and "params" in cfg.get("train", {}):
        return cfg["train"]
    if "params" in cfg:
        return cfg
    raise ValueError(
        f"agent yaml {path} has no `params` block; got top-level keys "
        f"{list(cfg.keys())}"
    )


def _register_depth_cnn_lstm_builder() -> None:
    """Register ``depth_cnn_lstm`` with rl_games' NETWORK_REGISTRY.

    Loaded via importlib so we don't trigger ``isaacsimenvs/__init__.py``,
    which transitively pulls in ``isaacsimenvs.tasks`` -> isaaclab. That keeps
    the deployment venv lean (only torch + rl_games + the leaf builder file)
    so it can coexist with the ROS / ZED conda env in the lab.
    """
    from rl_games.algos_torch import model_builder

    if "depth_cnn_lstm" in model_builder.NETWORK_REGISTRY:
        return
    import importlib.util as _ilu

    module_path = (
        REPO_ROOT / "isaacsimenvs" / "dagger" / "networks" / "depth_cnn_lstm.py"
    )
    spec = _ilu.spec_from_file_location("depth_cnn_lstm", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to load depth_cnn_lstm builder from {module_path}"
        )
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_builder.register_network("depth_cnn_lstm", module.DepthCNNLSTMBuilder)


def _build_student(
    net_params: dict,
    obs_dim: int,
    actions_num: int,
    num_blocks: int,
    device: torch.device,
):
    """Construct + load a depth_cnn_lstm student. Returns the net (eval mode)."""
    _register_depth_cnn_lstm_builder()
    from rl_games.algos_torch import model_builder

    builder = model_builder.NETWORK_REGISTRY["depth_cnn_lstm"]()
    builder.load(net_params)
    build_kwargs = {"actions_num": int(actions_num), "input_shape": (int(obs_dim),)}
    space = net_params.get("space", {}).get("continuous", {})
    if space.get("fixed_sigma") == "coef_cond":
        build_kwargs["coef_ids"] = torch.arange(num_blocks, dtype=torch.float32)
        build_kwargs["coef_id_idx"] = int(obs_dim) - 1
    net = builder.build("student", **build_kwargs).to(device)
    net.eval()
    return net


def _load_checkpoint_into(net: torch.nn.Module, ckpt_path: Path) -> None:
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # SAPG nests checkpoints under integer block-ids: ``{0: {"model": ...}}``.
    if isinstance(sd, dict) and "model" not in sd:
        int_keys = [k for k in sd.keys() if isinstance(k, int)]
        if int_keys:
            sd = sd[min(int_keys)]
    model_sd = sd.get("model", sd)
    prefix = "a2c_network."
    stripped: dict[str, torch.Tensor] = {}
    for k, v in model_sd.items():
        if k.startswith(prefix):
            stripped[k[len(prefix):]] = v
        elif not k.startswith(("value_mean_std.", "running_mean_std.")):
            stripped[k] = v
    missing, unexpected = net.load_state_dict(stripped, strict=False)
    n_loaded = sum(1 for k in net.state_dict().keys() if k not in missing)
    if n_loaded == 0:
        raise RuntimeError(
            f"Loaded 0 weights from {ckpt_path}; missing={len(missing)} "
            f"unexpected={len(unexpected)}. Check key prefixes."
        )
    print(
        f"[depth-policy] loaded {n_loaded}/{len(net.state_dict())} weights "
        f"from {ckpt_path} (missing={len(missing)} "
        f"unexpected={len(unexpected)})",
        flush=True,
    )


class DepthStudent:
    """Thin wrapper that maintains the LSTM state across calls."""

    def __init__(
        self,
        net: torch.nn.Module,
        obs_dim: int,
        actions_num: int,
        image_channels: int,
        image_hw: tuple[int, int],
        proprio_dim: int,
        has_block_id: bool,
        device: torch.device,
    ) -> None:
        self.net = net
        self.obs_dim = obs_dim
        self.actions_num = actions_num
        self.image_channels = image_channels
        self.image_hw = image_hw
        self.proprio_dim = proprio_dim
        self.has_block_id = has_block_id
        self.device = device
        self._rnn_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def reset(self) -> None:
        self._rnn_state = None

    @torch.no_grad()
    def act(
        self, image: np.ndarray, proprio: np.ndarray
    ) -> np.ndarray:
        """``image``: (H, W) float32 in [0,1]. ``proprio``: (P,) float32.

        Returns ``(N_ACTIONS,)`` numpy array of mu in [-1, 1].
        """
        if image.shape != self.image_hw:
            raise ValueError(
                f"image shape {image.shape} != expected {self.image_hw}"
            )
        if proprio.shape != (self.proprio_dim,):
            raise ValueError(
                f"proprio shape {proprio.shape} != expected ({self.proprio_dim},)"
            )
        img_t = torch.from_numpy(image).to(self.device, dtype=torch.float32)
        img_flat = img_t.reshape(1, self.image_channels * image.size)
        # Match the training shape: (1, C*H*W). image_channels==1 in our setup.
        prop_t = torch.from_numpy(proprio).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0)
        if self.has_block_id:
            block_id = torch.zeros(1, 1, device=self.device, dtype=torch.float32)
            flat = torch.cat([img_flat, prop_t, block_id], dim=-1)
        else:
            flat = torch.cat([img_flat, prop_t], dim=-1)
        if flat.shape != (1, self.obs_dim):
            raise RuntimeError(
                f"built obs shape {tuple(flat.shape)} != expected (1, {self.obs_dim})"
            )
        mu, _ls, _v, self._rnn_state = self.net(
            {"obs": flat, "rnn_states": self._rnn_state, "seq_length": 1}
        )
        return mu.squeeze(0).cpu().numpy()


# ============================================================
# Task-yaml lookup so we use the same depth window / image dims as training.
# ============================================================


def _load_task_yaml_defaults() -> dict:
    """Return the student_obs block from isaacsimenvs/cfg/task/PegInHoleDepthStudent.yaml.

    Falls back to hard-coded defaults if the file is missing or not yet
    populated (e.g. running this module before the env package is on disk).
    """
    yaml_path = (
        REPO_ROOT
        / "isaacsimenvs"
        / "cfg"
        / "task"
        / "PegInHoleDepthStudent.yaml"
    )
    if not yaml_path.exists():
        return {
            "depth_min_m": 0.70,
            "depth_max_m": 1.10,
            "image_input_height": 70,
            "image_input_width": 70,
        }
    with open(yaml_path) as fp:
        data = yaml.safe_load(fp) or {}
    return data.get("student_obs", data) if isinstance(data, dict) else {}


# ============================================================
# Inference benchmark (CLI helper, no ROS, no ZED).
# ============================================================


def _benchmark(args: argparse.Namespace) -> int:
    import statistics

    agent_cfg = _load_agent_yaml(Path(args.config))
    net_params = agent_cfg["params"]["network"]
    image_hw = tuple(net_params["image_hw"])
    image_channels = int(net_params.get("image_channels", 1))
    proprio_dim = int(net_params.get("proprio_dim", PROPRIO_DIM))
    has_block_id = bool(net_params.get("has_block_id", True))
    num_blocks = int(
        agent_cfg["params"]["config"].get("expl_coef_block_size", 1)
    )
    obs_dim = image_channels * image_hw[0] * image_hw[1] + proprio_dim + (
        1 if has_block_id else 0
    )

    device = torch.device(args.device)
    net = _build_student(
        net_params, obs_dim, N_ACTIONS, max(num_blocks, 1), device
    )
    _load_checkpoint_into(net, Path(args.checkpoint))
    student = DepthStudent(
        net=net,
        obs_dim=obs_dim,
        actions_num=N_ACTIONS,
        image_channels=image_channels,
        image_hw=image_hw,
        proprio_dim=proprio_dim,
        has_block_id=has_block_id,
        device=device,
    )

    image = np.random.rand(*image_hw).astype(np.float32)
    proprio = np.random.randn(proprio_dim).astype(np.float32)

    # Warmup
    for _ in range(20):
        _ = student.act(image, proprio)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms: list[float] = []
    n = int(args.bench_iters)
    for _ in range(n):
        t0 = time.perf_counter()
        _ = student.act(image, proprio)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.mean(times_ms)
    med = statistics.median(times_ms)
    p99 = sorted(times_ms)[int(n * 0.99)]
    print(
        f"[bench] device={args.device} obs_dim={obs_dim} N={n}  "
        f"mean={mean:.3f} ms  median={med:.3f} ms  p99={p99:.3f} ms  "
        f"=> {1000.0 / mean:.1f} Hz mean / {1000.0 / p99:.1f} Hz p99"
    )
    return 0


# ============================================================
# ROS node
# ============================================================


class DepthPolicyNode:
    def __init__(
        self,
        checkpoint: Path,
        config: Path,
        device: str,
        control_hz: float,
        zed_kwargs: dict,
        hand_moving_average: float,
        arm_moving_average: float,
        hand_dof_speed_scale: float,
        warmup_steps: int,
    ) -> None:
        import rospy  # local import so --benchmark can run without ROS
        from sensor_msgs.msg import JointState

        self._JointState = JointState
        self._rospy = rospy

        rospy.init_node("depth_policy_node")

        # --- policy load ---
        agent_cfg = _load_agent_yaml(config)
        net_params = agent_cfg["params"]["network"]
        self.image_hw = tuple(net_params["image_hw"])
        self.image_channels = int(net_params.get("image_channels", 1))
        self.proprio_dim = int(net_params.get("proprio_dim", PROPRIO_DIM))
        self.has_block_id = bool(net_params.get("has_block_id", True))
        num_blocks = int(
            agent_cfg["params"]["config"].get("expl_coef_block_size", 1)
        )
        if self.proprio_dim != PROPRIO_DIM:
            rospy.logwarn(
                f"agent yaml proprio_dim={self.proprio_dim} != node-side "
                f"PROPRIO_DIM={PROPRIO_DIM}. Joint layout assumes 29 DoF total."
            )
        obs_dim = (
            self.image_channels * self.image_hw[0] * self.image_hw[1]
            + self.proprio_dim
            + (1 if self.has_block_id else 0)
        )
        self.device = torch.device(device)
        net = _build_student(
            net_params, obs_dim, N_ACTIONS, max(num_blocks, 1), self.device
        )
        _load_checkpoint_into(net, checkpoint)
        self.student = DepthStudent(
            net=net,
            obs_dim=obs_dim,
            actions_num=N_ACTIONS,
            image_channels=self.image_channels,
            image_hw=self.image_hw,
            proprio_dim=self.proprio_dim,
            has_block_id=self.has_block_id,
            device=self.device,
        )

        # --- ZED reader ---
        from deployment.zed_async_reader import (
            ZedAsyncReader,
            ZedReaderConfig,
        )

        task_defaults = _load_task_yaml_defaults()
        zed_cfg = ZedReaderConfig(
            depth_near_m=float(task_defaults.get("depth_min_m", 0.70)),
            depth_far_m=float(task_defaults.get("depth_max_m", 1.10)),
            **zed_kwargs,
        )
        self.zed = ZedAsyncReader(
            zed_cfg,
            policy_height=int(self.image_hw[0]),
            policy_width=int(self.image_hw[1]),
        )
        self.zed.start()

        # --- ROS pubs/subs ---
        self.iiwa_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=1
        )
        self.sharpa_cmd_pub = rospy.Publisher(
            "/sharpa/joint_cmd", JointState, queue_size=1
        )
        self._lock = Lock()
        self._iiwa_msg: Optional[JointState] = None
        self._sharpa_msg: Optional[JointState] = None
        rospy.Subscriber(
            "/iiwa/joint_states",
            JointState,
            self._iiwa_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/sharpa/joint_states",
            JointState,
            self._sharpa_cb,
            queue_size=1,
        )

        # --- per-step state ---
        self.control_dt = 1.0 / float(control_hz)
        self.warmup_steps = int(max(0, warmup_steps))
        self.hand_moving_average = float(hand_moving_average)
        self.arm_moving_average = float(arm_moving_average)
        self.hand_dof_speed_scale = float(hand_dof_speed_scale)
        self.prev_targets: Optional[np.ndarray] = None

        rospy.on_shutdown(self._on_shutdown)

    # --- callbacks ---

    def _iiwa_cb(self, msg) -> None:
        with self._lock:
            self._iiwa_msg = msg

    def _sharpa_cb(self, msg) -> None:
        with self._lock:
            self._sharpa_msg = msg

    def _on_shutdown(self) -> None:
        try:
            self.zed.stop()
        except Exception:
            pass

    # --- main loop ---

    def _read_joint_state(
        self,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            iiwa = self._iiwa_msg
            sharpa = self._sharpa_msg
        if iiwa is None or sharpa is None:
            return None
        if len(iiwa.position) != N_ARM or len(sharpa.position) != N_HAND:
            self._rospy.logwarn_throttle(
                1.0,
                f"unexpected joint counts iiwa.pos={len(iiwa.position)} "
                f"sharpa.pos={len(sharpa.position)} "
                f"(expected {N_ARM} and {N_HAND})",
            )
            return None
        q = np.concatenate(
            [np.asarray(iiwa.position, dtype=np.float64),
             np.asarray(sharpa.position, dtype=np.float64)]
        )
        qd = np.concatenate(
            [np.asarray(iiwa.velocity, dtype=np.float64) if iiwa.velocity else np.zeros(N_ARM),
             np.asarray(sharpa.velocity, dtype=np.float64) if sharpa.velocity else np.zeros(N_HAND)]
        )
        return q, qd

    def _publish_targets(self, targets: np.ndarray) -> None:
        assert targets.shape == (N_ACTIONS,)
        now = self._rospy.Time.now()

        iiwa_msg = self._JointState()
        iiwa_msg.header.stamp = now
        iiwa_msg.name = list(IIWA_JOINT_NAMES)
        iiwa_msg.position = targets[:N_ARM].tolist()
        self.iiwa_cmd_pub.publish(iiwa_msg)

        sharpa_msg = self._JointState()
        sharpa_msg.header.stamp = now
        sharpa_msg.name = list(SHARPA_JOINT_NAMES)
        sharpa_msg.position = targets[N_ARM:].tolist()
        self.sharpa_cmd_pub.publish(sharpa_msg)

    def run(self) -> None:
        rospy = self._rospy
        rate = rospy.Rate(1.0 / self.control_dt)

        # Wait for first joint state + first depth frame.
        rospy.loginfo("[depth-policy] waiting for /iiwa/joint_states + /sharpa/joint_states ...")
        while not rospy.is_shutdown():
            js = self._read_joint_state()
            if js is not None:
                break
            rate.sleep()
        if rospy.is_shutdown():
            return
        q_now, _ = js
        self.prev_targets = q_now.copy()  # start from current pose
        rospy.loginfo("[depth-policy] joint states OK, waiting for first ZED frame ...")
        try:
            _ = self.zed.get_latest(timeout_s=2.0)
        except RuntimeError as exc:
            rospy.logerr(f"[depth-policy] ZED never produced a frame: {exc}")
            return

        # ---------------------------------------------------------------
        # Warmup: run the policy on real obs N times to warm the LSTM /
        # CUDA kernels / publisher path, but publish the CURRENT joint
        # state as the target so the robot doesn't move. Reset student
        # LSTM state at the end so the first "real" episode starts
        # from h0.
        # Mirrors deployment/rl_policy_node.py:_wait_and_warmup().
        # ---------------------------------------------------------------
        rospy.loginfo(
            f"[depth-policy] warmup: {self.warmup_steps} steps @ "
            f"{1.0 / self.control_dt:.0f} Hz (publishing current q, no motion)."
        )
        for step in range(int(self.warmup_steps)):
            if rospy.is_shutdown():
                return
            js = self._read_joint_state()
            if js is None:
                rate.sleep()
                continue
            q, qd = js
            try:
                depth, _frame_id, _age_s = self.zed.get_latest(timeout_s=0.05)
            except RuntimeError as exc:
                rospy.logwarn_throttle(
                    0.5, f"[depth-policy][warmup] zed stale: {exc}"
                )
                rate.sleep()
                continue
            proprio = np.concatenate(
                [q.astype(np.float32),
                 qd.astype(np.float32),
                 self.prev_targets.astype(np.float32)]
            )
            # Forward pass (warms LSTM + CUDA kernels). Discard the action.
            _ = self.student.act(depth, proprio)
            # Hold position: publish the current joint state, clipped.
            hold = np.clip(q, Q_LOWER_LIMITS_np, Q_UPPER_LIMITS_np)
            self._publish_targets(hold)
            self.prev_targets = hold
            rate.sleep()
        # Reset RNN so episode start is from a clean LSTM state.
        if hasattr(self.student, "reset"):
            self.student.reset()
        rospy.loginfo("[depth-policy] warmup complete; streaming policy "
                      "actions -- press Ctrl-C to stop.")

        n_steps = 0
        t_last_report = time.perf_counter()
        while not rospy.is_shutdown():
            t_start = time.perf_counter()

            js = self._read_joint_state()
            if js is None:
                rospy.logwarn_throttle(1.0, "[depth-policy] joint state went stale.")
                rate.sleep()
                continue
            q, qd = js

            try:
                depth, frame_id, age_s = self.zed.get_latest(timeout_s=0.02)
            except RuntimeError as exc:
                rospy.logwarn_throttle(0.5, f"[depth-policy] zed stale: {exc}")
                rate.sleep()
                continue

            proprio = np.concatenate(
                [q.astype(np.float32),
                 qd.astype(np.float32),
                 self.prev_targets.astype(np.float32)]
            )
            if proprio.shape[0] != self.proprio_dim:
                rospy.logerr_throttle(
                    1.0,
                    f"[depth-policy] proprio shape {proprio.shape[0]} != yaml "
                    f"proprio_dim {self.proprio_dim}; skipping step.",
                )
                rate.sleep()
                continue

            mu = self.student.act(depth, proprio)  # (29,) in [-1, 1]
            targets = compute_joint_pos_targets(
                actions=mu[None],
                prev_targets=self.prev_targets[None],
                hand_moving_average=self.hand_moving_average,
                arm_moving_average=self.arm_moving_average,
                hand_dof_speed_scale=self.hand_dof_speed_scale,
                dt=self.control_dt,
            )[0]
            targets = np.clip(targets, Q_LOWER_LIMITS_np, Q_UPPER_LIMITS_np)
            self._publish_targets(targets)
            self.prev_targets = targets

            n_steps += 1
            now = time.perf_counter()
            if now - t_last_report >= 2.0:
                step_ms = (now - t_start) * 1000.0
                rospy.loginfo(
                    f"[depth-policy] step={n_steps} zed_age_ms={age_s * 1000:.1f} "
                    f"loop_ms={step_ms:.1f} (target={self.control_dt * 1000:.1f}ms) "
                    f"frame_id={frame_id}"
                )
                t_last_report = now

            rate.sleep()


# ============================================================
# CLI
# ============================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--config", type=str,
        default=str(REPO_ROOT / "isaacsimenvs" / "cfg" / "train"
                    / "PegInHoleDepthStudentSAPG.yaml"),
        help="rl_games agent yaml. Defaults to the in-repo training yaml; "
             "override if deploying a checkpoint trained with a different "
             "config (e.g. a .hydra/config.yaml from a specific run).",
    )
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Inference device. Use cpu if CUDA isn't available.")
    p.add_argument("--control-hz", type=float, default=60.0)
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="At startup, run this many policy forward passes "
                        "while publishing the current joint pose (no motion) "
                        "to warm CUDA kernels + LSTM + publisher. The student "
                        "LSTM state is reset after warmup so the live episode "
                        "begins from h0. Set to 0 to skip.")

    p.add_argument("--hand-moving-average", type=float, default=0.1)
    p.add_argument("--arm-moving-average", type=float, default=0.1)
    p.add_argument("--hand-dof-speed-scale", type=float, default=1.5)

    p.add_argument("--zed-serial", type=str, default=None)
    p.add_argument("--zed-resolution", type=str, default=None)
    p.add_argument("--zed-depth-mode", type=str, default=None)
    p.add_argument("--zed-camera-fps", type=int, default=None)
    p.add_argument("--zed-grab-hz", type=float, default=None)
    p.add_argument("--zed-exposure", type=int, default=None)
    p.add_argument("--zed-gain", type=int, default=None)
    p.add_argument("--zed-upsidedown", action="store_true")

    p.add_argument("--benchmark", action="store_true",
                   help="Run a no-ROS, no-ZED inference latency benchmark.")
    p.add_argument("--bench-iters", type=int, default=200)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.benchmark:
        return _benchmark(args)

    zed_kwargs: dict = {}
    if args.zed_serial is not None:
        zed_kwargs["serial_number"] = args.zed_serial
    if args.zed_resolution is not None:
        zed_kwargs["resolution"] = args.zed_resolution
    if args.zed_depth_mode is not None:
        zed_kwargs["depth_mode"] = args.zed_depth_mode
    if args.zed_camera_fps is not None:
        zed_kwargs["camera_fps"] = args.zed_camera_fps
    if args.zed_grab_hz is not None:
        zed_kwargs["grab_hz"] = args.zed_grab_hz
    if args.zed_exposure is not None:
        zed_kwargs["exposure"] = args.zed_exposure
    if args.zed_gain is not None:
        zed_kwargs["gain"] = args.zed_gain
    if args.zed_upsidedown:
        zed_kwargs["camera_upsidedown"] = True

    node = DepthPolicyNode(
        checkpoint=Path(args.checkpoint),
        config=Path(args.config),
        device=args.device,
        control_hz=args.control_hz,
        zed_kwargs=zed_kwargs,
        hand_moving_average=args.hand_moving_average,
        arm_moving_average=args.arm_moving_average,
        hand_dof_speed_scale=args.hand_dof_speed_scale,
        warmup_steps=args.warmup_steps,
    )
    node.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
