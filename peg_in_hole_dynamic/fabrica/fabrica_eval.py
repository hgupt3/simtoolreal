"""Fabrica Assembly Evaluation with Retract Support.

Like eval_assembly.py but always uses FabricaEnv (with retract) regardless
of config task_name. This allows comparing pretrained (SimToolReal) and
finetuned (FabricaEnv) policies on the same environment.

Architecture:
  Main process  -- viser GUI + scene rendering (no isaacgym)
  Subprocess    -- IsaacGym FabricaEnv + policy (sends state back via pipe)

Usage:
    python fabrica/fabrica_eval.py \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth \
        --collision coacd
"""

import argparse
import json
import math
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from fabrica.viser_utils import COLORS

sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

# Goal modes map to config overrides read by FabricaEnv.
# Mirrored in fabrica_multi_init_eval.py to avoid a circular import.
GOAL_MODES = ["dense", "final_only", "pre_insert_and_final"]
_GOAL_MODE_OVERRIDE_KEY = {
    "final_only": "task.env.finalGoalOnly",
    "pre_insert_and_final": "task.env.preInsertAndFinal",
}

# ===================================================================
# Constants (lightweight -- no isaacgym imports)
# ===================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
Z_OFFSET = 0.03
N_ACT = 29
OBS_DIM = 140
CONTROL_DT = 1.0 / 60.0

ALL_ASSEMBLIES = [
    "beam", "car", "cooling_manifold", "duct",
    "gamepad", "plumbers_block", "stool_circular",
]

_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
DEFAULT_DOF_POS = np.zeros(N_ACT)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT

BASE_OVERRIDES = {
    "task.env.resetPositionNoiseX": 0.0,
    "task.env.resetPositionNoiseY": 0.0,
    "task.env.resetPositionNoiseZ": 0.0,
    "task.env.randomizeObjectRotation": False,
    "task.env.resetDofPosRandomIntervalFingers": 0.0,
    "task.env.resetDofPosRandomIntervalArm": 0.0,
    "task.env.resetDofVelRandomInterval": 0.0,
    "task.env.tableResetZRange": 0.0,
    "task.env.numEnvs": 1,
    "task.env.envSpacing": 0.4,
    "task.env.capture_video": False,
    "task.env.useActionDelay": False,
    "task.env.useObsDelay": False,
    "task.env.useObjectStateDelayNoise": False,
    "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
    "task.env.resetWhenDropped": False,
    "task.env.armMovingAverage": 0.1,
    "task.env.evalSuccessTolerance": 0.01,
    "task.env.forceConsecutiveNearGoalSteps": True,
    "task.env.fixedSizeKeypointReward": True,
    "task.env.useFixedInitObjectPose": True,
    "task.env.startArmHigher": True,
    "task.env.forceScale": 0.0,
    "task.env.torqueScale": 0.0,
    "task.env.linVelImpulseScale": 0.0,
    "task.env.angVelImpulseScale": 0.0,
    "task.env.forceOnlyWhenLifted": True,
    "task.env.torqueOnlyWhenLifted": True,
    "task.env.linVelImpulseOnlyWhenLifted": True,
    "task.env.angVelImpulseOnlyWhenLifted": True,
    "task.env.forceProbRange": [0.0001, 0.0001],
    "task.env.torqueProbRange": [0.0001, 0.0001],
    "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
    "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
}

ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"


# ===================================================================
# Helpers
# ===================================================================

def _load_assembly_order(assembly: str) -> List[str]:
    path = ASSETS_DIR / assembly / "assembly_order.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["steps"]


def _get_available_parts(assembly: str) -> List[str]:
    """Return parts that have both a trajectory and environment URDF."""
    order = _load_assembly_order(assembly)
    available = []
    for pid in order:
        name = f"{assembly}_{pid}"
        traj = ASSETS_DIR / assembly / "trajectories" / pid / "pick_place.json"
        urdf = ASSETS_DIR / assembly / "environments" / pid / "scene.urdf"
        if traj.exists() and urdf.exists():
            available.append(pid)
    return available


def _load_trajectory(assembly: str, pid: str) -> Optional[dict]:
    path = ASSETS_DIR / assembly / "trajectories" / pid / "pick_place.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


def _quat_xyzw_to_rpy(q) -> Tuple[float, float, float]:
    """Convert quaternion [x, y, z, w] to (roll, pitch, yaw)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def _table_urdf_rel(assembly: str, active_pid: str, collision_method: str) -> str:
    """Return the relative URDF path for the precomputed table scene."""
    suffix_map = {"vhacd": "", "sdf": "_sdf", "coacd": "_coacd"}
    suffix = suffix_map[collision_method]
    return f"urdf/fabrica/{assembly}/environments/{active_pid}/scene{suffix}.urdf"


# ===================================================================
# SUBPROCESS -- IsaacGym simulation (FabricaEnv forced)
# ===================================================================

def _create_fabrica_env(config_path, headless, device, overrides):
    """Like create_env but forces FabricaEnv task class.

    Injects FabricaEnv-specific config keys (enableRetract, etc.) so that
    even a SimToolReal config can be used with the FabricaEnv task class.
    """
    from deployment.rl_player_utils import read_cfg_omegaconf
    from deployment.isaac.isaac_env import merge_cfg_with_default_config, create_env_from_cfg
    from omegaconf import OmegaConf

    cfg = read_cfg_omegaconf(config_path=config_path, device=device)
    cfg = merge_cfg_with_default_config(cfg)

    # Force FabricaEnv and inject its config keys (single-insertion schema).
    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "FabricaEnv"
    cfg.task_name = "FabricaEnv"
    fabrica_defaults = {
        # Retract
        "enableRetract": True,
        "retractDistanceThreshold": 0.1,
        "retractRewardScale": 1.0,
        "retractSuccessBonus": 0.0,
        "retractSuccessTolerance": 0.005,
        # Single-insertion scene source
        "assemblyName": "beam_2x",
        "scenesFilename": "scenes.npz",
        "goalMode": "dense",
        # Eval: pin to specific (part, scene, start) combo
        "forcePartIdx": -1,
        "forceSceneIdx": -1,
        "forceStartIdx": -1,
        # Force sensor + goal-XY obs noise
        "withTableForceSensor": False,
        "tableForceResetThreshold": 100.0,
        "goalXyObsNoise": 0.002,
    }
    for k, v in fabrica_defaults.items():
        OmegaConf.update(cfg, f"task.env.{k}", v, force_add=True)
    print(f"[_create_fabrica_env] fabrica_defaults injected; "
          f"assemblyName={cfg.task.env.assemblyName!r}, "
          f"forcePartIdx={cfg.task.env.forcePartIdx}, "
          f"struct(task.env)={OmegaConf.is_struct(cfg.task.env)}")

    return create_env_from_cfg(
        cfg=cfg, headless=headless, overrides=overrides,
    )


def _sim_get_state(env, obs, joint_lower, joint_upper):
    obs_np = obs[0].cpu().numpy()
    joint_pos = 0.5 * (obs_np[:N_ACT] + 1.0) * (joint_upper - joint_lower) + joint_lower
    return (
        joint_pos,
        env.object_state[0, :7].cpu().numpy(),
        env.goal_pose[0].cpu().numpy(),
        env.obj_keypoint_pos[0].cpu().numpy(),
        env.goal_keypoint_pos[0].cpu().numpy(),
        bool(env.retract_phase[0].item()),
        bool(env.retract_succeeded[0].item()),
        float(env.curr_fingertip_distances[0].mean().item()),
    )


def _sim_reset(env, device):
    import torch
    obs, _, _, _ = env.step(torch.zeros((env.num_envs, N_ACT), device=device))
    return obs["obs"]


def _setup_collision_camera(env):
    """Create a second camera sensor that renders collision geometry."""
    from isaacgym import gymapi
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720
    cam_props.use_collision_geometry = True
    env._collision_cam = env.gym.create_camera_sensor(env.envs[0], cam_props)
    env._collision_cam_props = cam_props
    # Match position of the existing camera
    cam_target = gymapi.Vec3(0.0, 0.0, 0.53)
    cam_pos = cam_target + gymapi.Vec3(0.0, -1.0, 0.5)
    env.gym.set_camera_location(env._collision_cam, env.envs[0], cam_pos, cam_target)


def _capture_frame(env):
    """Capture a single RGBA frame from the collision camera."""
    from isaacgym import gymapi
    env.gym.render_all_camera_sensors(env.sim)
    cam = getattr(env, '_collision_cam', env.camera_handle)
    cam_props = getattr(env, '_collision_cam_props', env.camera_properties)
    color_image = env.gym.get_camera_image(
        env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR,
    )
    if color_image.size == 0:
        return None
    return color_image.reshape(cam_props.height, cam_props.width, 4)


def _save_video(frames, video_dir, assembly, part_id, ep_num, collision_method="vhacd"):
    """Save captured frames as an mp4."""
    import imageio
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / f"{assembly}_{part_id}_{collision_method}_ep{ep_num:03d}.mp4"
    imageio.mimsave(str(path), frames, fps=int(1.0 / CONTROL_DT))
    print(f"[sim_worker] Saved video: {path}")


def _sim_episode(conn, env, policy, joint_lower, joint_upper, device,
                 record_video=False, video_dir=None, assembly=None, part_id=None, ep_num=0,
                 collision_method="vhacd"):
    import time, torch  # noqa: E401

    policy.reset()
    obs = _sim_reset(env, device)
    frames = []

    if record_video:
        env.enable_viewer_sync = True

    step, done, paused = 0, False, False
    retract_ok = False  # Track across the episode (reset clears the tensor)
    last_arm_pos = None
    cum_total = 0.0
    cum_breakdown: dict = {}

    while not done:
        while conn.poll(0):
            cmd = conn.recv()
            if cmd == "pause":
                paused = True
            elif cmd == "resume":
                paused = False
            elif cmd == "stop":
                conn.send(("stopped",))
                return obs

        if paused:
            time.sleep(0.05)
            continue

        t0 = time.time()
        state = _sim_get_state(env, obs, joint_lower, joint_upper)
        # Save arm pos before the step that may trigger reset
        last_arm_pos = env.arm_hand_dof_pos[0, :N_ACT].cpu().numpy().tolist()

        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done_tensor, _ = env.step(action)
        obs = obs_dict["obs"]
        done = done_tensor[0].item()
        step += 1

        # retract_succeeded is set then cleared within compute_reward,
        # but extras["retract_success_ratio"] is logged before the clear
        if env.extras.get("retract_success_ratio", 0) > 0.5:
            retract_ok = True

        # Accumulate per-step reward + per-term breakdown from env.extras
        cum_total += float(env.rew_buf[0].item())
        step_ec = env.extras.get("episode_cumulative", {}) or {}
        for _k, _v in step_ec.items():
            try:
                _val = float(_v[0].item()) if hasattr(_v, "__len__") else float(_v.item())
            except Exception:
                try:
                    _val = float(_v)
                except Exception:
                    continue
            cum_breakdown[_k] = cum_breakdown.get(_k, 0.0) + _val

        if record_video:
            frame = _capture_frame(env)
            if frame is not None:
                frames.append(frame)

        # Override retract_succeeded in state with our latched value
        # (the tensor gets cleared within compute_reward before we can read it)
        state_with_retract = state[:6] + (retract_ok,) + state[7:]
        conn.send((
            "state", state_with_retract,
            int(env.successes[0].item()),
            env.max_consecutive_successes,
            step,
            float(env.keypoints_max_dist[0].item()),
            cum_total,
            dict(cum_breakdown),
        ))

        elapsed = time.time() - t0
        if (sleep := CONTROL_DT - elapsed) > 0:
            time.sleep(sleep)

    if record_video and frames:
        _save_video(frames, video_dir, assembly, part_id, ep_num, collision_method)

    goal_pct = 100 * int(env.successes[0].item()) / env.max_consecutive_successes
    # retract_ok was captured during the loop before reset cleared the tensor
    final_arm_pos = last_arm_pos or env.arm_hand_dof_pos[0, :N_ACT].cpu().numpy().tolist()
    conn.send(("done", goal_pct, step, retract_ok, final_arm_pos))
    return obs


def sim_worker(conn, assembly, part_id, config_path, checkpoint_path, table_urdf_rel,
               final_goal_tolerance=None, collision_method="vhacd", extra_overrides=None,
               headless=True, record_video=False, video_dir=None, initial_arm_pos=None):
    """Child process entry-point. Always uses FabricaEnv."""
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
        import json, torch  # noqa: E401
        from deployment.rl_player import RlPlayer
        import fabrica.objects  # noqa: F401

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_name = f"{assembly}_{part_id}"
        obj_suffix = {"vhacd": "", "sdf": "_sdf", "coacd": "_coacd"}[collision_method]
        object_name = base_name + obj_suffix
        # Legacy fabrica path used ``trajectories/{pid}/pick_place.json`` to
        # seed start_pose + fixedGoalStates via overrides. The new
        # single-insertion FabricaEnv loads everything from ``scenes.npz`` via
        # forcePartIdx/SceneIdx/StartIdx, so pick_place.json is no longer
        # required. Load it only if present (to keep the old path working).
        traj_path = ASSETS_DIR / assembly / "trajectories" / part_id / "pick_place.json"
        traj_overrides: dict = {}
        if traj_path.exists():
            with open(traj_path) as f:
                traj = json.load(f)
            print(f"[sim_worker] start_pose (from pick_place.json): {traj['start_pose']}")
            traj_overrides = {
                "task.env.useFixedGoalStates": True,
                "task.env.fixedGoalStates": traj["goals"],
                "task.env.objectStartPose": traj["start_pose"],
            }
        else:
            print(f"[sim_worker] no pick_place.json at {traj_path}; "
                  f"start/goals will come from scenes.npz via FabricaEnv")

        print(f"[sim_worker] table_urdf: {table_urdf_rel}")
        print(f"[sim_worker] object_name: {object_name}")
        print(f"[sim_worker] task: FabricaEnv (forced)")
        if initial_arm_pos is not None:
            print(f"[sim_worker] initial_arm_pos: provided (chained from previous part)")

        env = _create_fabrica_env(
            config_path=str(config_path), headless=headless, device=device,
            overrides={
                **BASE_OVERRIDES,
                "task.env.objectName": object_name,
                "task.env.asset.table": table_urdf_rel,
                "task.env.tableResetZ": TABLE_Z,
                **traj_overrides,
                **({"task.env.finalGoalSuccessTolerance": final_goal_tolerance}
                   if final_goal_tolerance is not None else {}),
                **({"task.env.useSDF": True} if collision_method == "sdf" else {}),
                **(extra_overrides or {}),
            },
        )

        joint_lower = env.arm_hand_dof_lower_limits[:N_ACT].cpu().numpy()
        joint_upper = env.arm_hand_dof_upper_limits[:N_ACT].cpu().numpy()

        env.set_env_state(torch.load(checkpoint_path, map_location=device)[0]["env_state"])

        # Override default arm pose for chained assembly sequence
        if initial_arm_pos is not None:
            env.hand_arm_default_dof_pos = torch.tensor(
                initial_arm_pos, dtype=torch.float32, device=device,
            )

        policy = RlPlayer(OBS_DIM, N_ACT, config_path, checkpoint_path, device, env.num_envs)

        if record_video:
            _setup_collision_camera(env)

        obs = _sim_reset(env, device)
        init_state = _sim_get_state(env, obs, joint_lower, joint_upper)
        conn.send(("ready", init_state))

        ep_num = 0
        while True:
            cmd = conn.recv()
            if cmd == "run":
                obs = _sim_episode(conn, env, policy, joint_lower, joint_upper, device,
                                   record_video=record_video, video_dir=video_dir,
                                   assembly=assembly, part_id=part_id, ep_num=ep_num,
                                   collision_method=collision_method)
                ep_num += 1
            elif cmd == "quit":
                break

    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))

    conn.close()


# ===================================================================
# MAIN PROCESS -- viser GUI
# ===================================================================

class AssemblyDemo:

    def __init__(self, policies: dict, port: int = 8082,
                 final_goal_tolerance: float = None, collision_method: str = "vhacd",
                 extra_overrides: dict = None, headless: bool = True,
                 record_video: bool = False, video_dir: str = None,
                 goal_mode: str = "dense",
                 initial_policy: str = None):
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}, got {goal_mode}")
        if not policies:
            raise ValueError("policies dict must be non-empty")
        self.policies = policies
        if initial_policy is None or initial_policy not in policies:
            initial_policy = next(iter(policies))
        self.initial_policy = initial_policy
        self.port = port
        self.final_goal_tolerance = final_goal_tolerance
        self.collision_method = collision_method
        self.record_video = record_video
        self.video_dir = video_dir or str(REPO_ROOT / "eval_videos")
        self.extra_overrides = extra_overrides or {}
        self.headless = headless
        self.goal_mode = goal_mode
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        self._proc = None  # type: Optional[multiprocessing.Process]
        self._conn = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False
        self._pending_assembly = ""
        self._pending_part = ""

        self.ep_count = 0
        self.ep_goals = []  # type: List[float]
        self.ep_lengths = []  # type: List[int]

        self.robot = None
        self._dyn = []  # type: list
        self._obj_frame = None
        self._goal_frame = None
        self._obj_keypoints = []  # type: list
        self._goal_keypoints = []  # type: list

        # Assembly context
        self._assembly_order = []  # type: List[str]
        self._trajectories = {}  # type: Dict[str, dict]  # pid -> trajectory

        # Auto-sequence state
        self._auto_seq_active = False
        self._auto_seq_step_idx = 0
        self._auto_seq_results = []  # type: List[Tuple[str, float, int, bool]]
        self._last_arm_pos = None  # type: Optional[List[float]]

        # Pre-load available parts per assembly
        self._assembly_parts = {}  # type: Dict[str, List[str]]
        for a in ALL_ASSEMBLIES:
            parts = _get_available_parts(a)
            if parts:
                self._assembly_parts[a] = parts

        self._build_gui()
        self._setup_static_scene()

    def _build_gui(self):
        self.server.gui.add_markdown("# Fabrica Assembly\n### Evaluation with Retract")

        _PH = "-- Select --"
        with self.server.gui.add_folder("Assembly Selection", expand_by_default=True):
            self._dd_policy = self.server.gui.add_dropdown(
                "Policy", options=list(self.policies.keys()),
                initial_value=self.initial_policy,
            )
            assemblies = [_PH] + list(self._assembly_parts.keys())
            self._dd_assembly = self.server.gui.add_dropdown(
                "Assembly", options=assemblies, initial_value=_PH,
            )
            self._dd_part = self.server.gui.add_dropdown(
                "Part", options=[_PH], initial_value=_PH,
            )
            self._dd_goal_mode = self.server.gui.add_dropdown(
                "Goal Mode", options=GOAL_MODES, initial_value=self.goal_mode,
            )
            self._btn_load = self.server.gui.add_button("Load Environment")
            self._btn_load.on_click(lambda _: self._load_env())
            self._md_status = self.server.gui.add_markdown("**Status:** Ready")
            self._dd_assembly.on_update(lambda _: self._on_assembly_change())

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop Episode")
            self._btn_stop.on_click(lambda _: self._cmd_stop())
            self._btn_run_all = self.server.gui.add_button("Run All Steps")
            self._btn_run_all.on_click(lambda _: self._cmd_run_all())

        with self.server.gui.add_folder("Display", expand_by_default=True):
            self._cb_keypoints = self.server.gui.add_checkbox(
                "Show Keypoints", initial_value=True,
            )
            self._cb_keypoints.on_update(lambda _: self._toggle_keypoints())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_retract = self.server.gui.add_markdown("**Retract:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self._md_obj = self.server.gui.add_markdown("**Object Pos:** --")
            self._md_dist = self.server.gui.add_markdown("**Dist to Goal:** --")
            self._md_reward = self.server.gui.add_markdown("**Cum Reward:** --")

    def _on_assembly_change(self):
        assembly = self._dd_assembly.value
        if assembly in self._assembly_parts:
            parts = [f"Part {p}" for p in self._assembly_parts[assembly]]
            self._dd_part.options = parts
            self._dd_part.value = parts[0]
            # Load assembly context
            self._assembly_order = _load_assembly_order(assembly)
            self._trajectories.clear()
            for pid in self._assembly_order:
                traj = _load_trajectory(assembly, pid)
                if traj is not None:
                    self._trajectories[pid] = traj
        else:
            self._dd_part.options = ["-- Select --"]
            self._dd_part.value = "-- Select --"
            self._assembly_order = []
            self._trajectories.clear()

    def _get_step_index(self, assembly: str, part_id: str) -> int:
        """Get the index of part_id in the assembly order (available parts only)."""
        available = self._assembly_parts.get(assembly, [])
        if part_id in available:
            return available.index(part_id)
        return 0

    def _get_completed_future_pids(self, assembly: str, part_id: str):
        """Return (completed_pids, future_pids) based on assembly order."""
        available = self._assembly_parts.get(assembly, [])
        if part_id not in available:
            return [], []
        si = available.index(part_id)
        completed = available[:si]
        future = available[si + 1:]
        return completed, future

    def _setup_static_scene(self):
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.robot = ViserUrdf(
            self.server,
            REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf",
            root_node_name="/robot",
        )
        self.robot.update_cfg(DEFAULT_DOF_POS)

        self.server.scene.add_frame(
            "/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self.server.scene.add_box(
            "/table/wood", color=(180, 130, 70),
            dimensions=(0.475, 0.4, 0.3), position=(0, 0, 0),
            side="double", opacity=0.9,
        )

    def _clear_dynamic(self):
        for h in self._dyn:
            try:
                h.remove()
            except Exception:
                pass
        self._dyn.clear()
        self._obj_frame = self._goal_frame = None
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()

    def _setup_object_goal(self, assembly, part_id):
        """Set up the active part, goal ghost, static context parts in viser."""
        self._clear_dynamic()
        from fabrica.objects import FABRICA_NAME_TO_OBJECT

        object_name = f"{assembly}_{part_id}"
        obj_urdf = FABRICA_NAME_TO_OBJECT[object_name].urdf_path
        available = self._assembly_parts.get(assembly, [])
        part_id_to_idx = {pid: i for i, pid in enumerate(available)}

        completed_pids, future_pids = self._get_completed_future_pids(assembly, part_id)

        # Completed parts -- at goal pose, full color
        for cpid in completed_pids:
            traj = self._trajectories.get(cpid)
            if traj is None:
                continue
            mesh_path = ASSETS_DIR / assembly / cpid / f"{cpid}_canonical.obj"
            if not mesh_path.exists():
                continue
            mesh = trimesh.load(str(mesh_path), force="mesh")
            gp = traj["goals"][-1]
            idx = part_id_to_idx.get(cpid, 0)
            color = COLORS[idx % len(COLORS)]
            frame_name = f"/completed_{cpid}"
            frame = self.server.scene.add_frame(
                frame_name,
                position=(gp[0], gp[1], gp[2]),
                wxyz=quat_xyzw_to_wxyz(gp[3:7]),
                show_axes=False,
            )
            self._dyn.append(frame)
            verts = np.array(mesh.vertices, dtype=np.float32)
            h = self.server.scene.add_mesh_simple(
                f"{frame_name}/mesh",
                vertices=verts,
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=color,
            )
            self._dyn.append(h)

        # Future parts -- at start pose, dim transparent
        for fpid in future_pids:
            traj = self._trajectories.get(fpid)
            if traj is None:
                continue
            mesh_path = ASSETS_DIR / assembly / fpid / f"{fpid}_canonical.obj"
            if not mesh_path.exists():
                continue
            mesh = trimesh.load(str(mesh_path), force="mesh")
            sp = traj["start_pose"]
            idx = part_id_to_idx.get(fpid, 0)
            color = COLORS[idx % len(COLORS)]
            frame_name = f"/future_{fpid}"
            frame = self.server.scene.add_frame(
                frame_name,
                position=(sp[0], sp[1], sp[2]),
                wxyz=quat_xyzw_to_wxyz(sp[3:7]),
                show_axes=False,
            )
            self._dyn.append(frame)
            verts = np.array(mesh.vertices, dtype=np.float32)
            h = self.server.scene.add_mesh_simple(
                f"{frame_name}/mesh",
                vertices=verts,
                faces=np.array(mesh.faces, dtype=np.uint32),
                color=color,
                opacity=0.3,
            )
            self._dyn.append(h)

        # Active part
        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)
        idx = part_id_to_idx.get(part_id, 0)
        color = COLORS[idx % len(COLORS)]
        color_override = tuple(int(c * 255) for c in color)
        ViserUrdf(self.server, obj_urdf, root_node_name="/object",
                  mesh_color_override=color_override)

        # Goal ghost
        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)
        ViserUrdf(self.server, obj_urdf, root_node_name="/goal",
                  mesh_color_override=(0, 255, 0, 0.5))

    def _setup_keypoints(self, num_keypoints):
        for kp in self._obj_keypoints + self._goal_keypoints:
            try:
                kp.remove()
            except Exception:
                pass
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()

        for i in range(num_keypoints):
            self._obj_keypoints.append(
                self.server.scene.add_icosphere(f"/obj_kp/{i}", radius=0.005, color=(255, 0, 0))
            )
            self._goal_keypoints.append(
                self.server.scene.add_icosphere(f"/goal_kp/{i}", radius=0.005, color=(0, 255, 0), opacity=0.5)
            )
        # Apply current visibility
        self._apply_keypoint_visibility()

    def _toggle_keypoints(self):
        self._apply_keypoint_visibility()

    def _apply_keypoint_visibility(self):
        visible = self._cb_keypoints.value
        for kp in self._obj_keypoints + self._goal_keypoints:
            kp.visible = visible

    # -- Subprocess management ------------------------------------------

    def _kill_subprocess(self):
        if self._conn is not None:
            try:
                self._conn.send("quit")
            except (BrokenPipeError, OSError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
            self._proc = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

    def _load_env(self, assembly=None, part_id=None):
        if assembly is None:
            assembly = self._dd_assembly.value
        if part_id is None:
            part_display = self._dd_part.value
            if not part_display.startswith("Part "):
                self._md_status.content = "**Status:** Please select an assembly and part."
                return
            part_id = part_display.split("Part ")[1]

        if assembly not in self._assembly_parts:
            self._md_status.content = "**Status:** Please select an assembly and part."
            return

        self._kill_subprocess()

        self._pending_assembly = assembly
        self._pending_part = part_id

        goal_mode = self._dd_goal_mode.value
        policy_name = self._dd_policy.value
        config_path, checkpoint_path = self.policies[policy_name]
        label = (
            f"{policy_name} | {assembly} / Part {part_id} / goals: {goal_mode}"
        )
        self._md_status.content = f"**Status:** Loading *{label}* ..."
        self._md_task.content = f"**Task:** {label}"
        self._md_retract.content = "**Retract:** --"

        if not self._auto_seq_active:
            self.ep_count = 0
            self.ep_goals.clear()
            self.ep_lengths.clear()
            self._md_stats.content = "**Stats:** No episodes yet"

        self.robot.update_cfg(DEFAULT_DOF_POS)

        # Ensure assembly context is loaded
        if not self._assembly_order or self._assembly_order != _load_assembly_order(assembly):
            self._assembly_order = _load_assembly_order(assembly)
            self._trajectories.clear()
            for pid in self._assembly_order:
                traj = _load_trajectory(assembly, pid)
                if traj is not None:
                    self._trajectories[pid] = traj

        # Use precomputed table URDF
        table_urdf_rel = _table_urdf_rel(assembly, part_id, self.collision_method)

        # Chain arm state from previous part in auto-sequence
        initial_arm_pos = None
        if self._auto_seq_active and self._auto_seq_step_idx > 0 and self._last_arm_pos is not None:
            initial_arm_pos = self._last_arm_pos

        scene_overrides = dict(self.extra_overrides)
        mode_key = _GOAL_MODE_OVERRIDE_KEY.get(goal_mode)
        if mode_key is not None:
            scene_overrides[mode_key] = True

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sim_worker,
            args=(child_conn, assembly, part_id,
                  config_path, checkpoint_path, table_urdf_rel,
                  self.final_goal_tolerance, self.collision_method, scene_overrides,
                  self.headless, self.record_video, self.video_dir, initial_arm_pos),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned subprocess pid={self._proc.pid} for {assembly}_{part_id}")

    # -- Commands -------------------------------------------------------

    def _send(self, msg):
        if self._conn is not None:
            try:
                self._conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    def _cmd_run(self):
        if not self._env_ready:
            self._md_status.content = "**Status:** Load an environment first."
            return
        if self._episode_running:
            return
        self._episode_running = True
        self._is_paused = False
        self._btn_pause.name = "Pause"
        self._md_status.content = "**Status:** Running episode..."
        self._md_retract.content = "**Retract:** --"
        self._send("run")

    def _cmd_pause(self):
        if not self._episode_running:
            return
        self._is_paused = not self._is_paused
        self._send("pause" if self._is_paused else "resume")
        self._btn_pause.name = "Resume" if self._is_paused else "Pause"

    def _cmd_stop(self):
        if self._episode_running:
            self._send("stop")
        # Also cancel auto-sequence
        if self._auto_seq_active:
            self._auto_seq_active = False
            self._md_status.content = "**Status:** Auto-sequence cancelled."

    def _cmd_run_all(self):
        assembly = self._dd_assembly.value
        if assembly not in self._assembly_parts:
            self._md_status.content = "**Status:** Please select an assembly first."
            return
        if self._episode_running:
            return

        available = self._assembly_parts[assembly]
        if not available:
            self._md_status.content = "**Status:** No available parts for this assembly."
            return

        self._auto_seq_active = True
        self._auto_seq_step_idx = 0
        self._auto_seq_results = []
        self._last_arm_pos = None
        self.ep_count = 0
        self.ep_goals.clear()
        self.ep_lengths.clear()

        pid = available[0]
        self._md_status.content = f"**Status:** Auto-sequence: step 1/{len(available)}"
        self._load_env(assembly, pid)

    # -- Scene update ---------------------------------------------------

    def _update_viz(self, state_tuple):
        joint_pos, obj_pose, goal_pose = state_tuple[0], state_tuple[1], state_tuple[2]
        self.robot.update_cfg(joint_pos)

        if self._obj_frame is not None:
            self._obj_frame.position = tuple(obj_pose[:3])
            self._obj_frame.wxyz = quat_xyzw_to_wxyz(obj_pose[3:7])
        if self._goal_frame is not None:
            self._goal_frame.position = tuple(goal_pose[:3])
            self._goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])

        if len(state_tuple) > 3:
            obj_kps, goal_kps = state_tuple[3], state_tuple[4]
            for handle, pos in zip(self._obj_keypoints, obj_kps):
                handle.position = tuple(pos)
            for handle, pos in zip(self._goal_keypoints, goal_kps):
                handle.position = tuple(pos)

    # -- Message handling -----------------------------------------------

    def _handle(self, msg):
        tag = msg[0]

        if tag == "ready":
            init_state = msg[1]
            self._setup_object_goal(self._pending_assembly, self._pending_part)
            if len(init_state) > 3:
                self._setup_keypoints(init_state[3].shape[0])
            self._update_viz(init_state)
            self._env_ready = True

            if self._auto_seq_active:
                # Auto-start the episode
                available = self._assembly_parts.get(self._pending_assembly, [])
                step_num = self._auto_seq_step_idx + 1
                self._md_status.content = (
                    f"**Status:** Auto-sequence: running step {step_num}/{len(available)} "
                    f"(Part {self._pending_part})"
                )
                self._cmd_run()
            else:
                self._md_status.content = "**Status:** Ready -- click **Run Episode**"
            print("[launcher] Environment ready")

        elif tag == "state":
            state, successes, max_succ, step = msg[1], msg[2], msg[3], msg[4]
            dist_to_goal = msg[5] if len(msg) > 5 else None
            cum_total = msg[6] if len(msg) > 6 else None
            cum_breakdown = msg[7] if len(msg) > 7 else None
            self._update_viz(state)
            pct = 100 * successes / max_succ if max_succ > 0 else 0
            self._md_prog.content = (
                f"**Time:** {step / 60.0:.1f}s &nbsp;|&nbsp; "
                f"**Goal:** {successes}/{max_succ} ({pct:.0f}%)"
            )
            obj_pos = state[1][:3]
            self._md_obj.content = (
                f"**Object Pos:** {obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}"
            )
            if dist_to_goal is not None:
                self._md_dist.content = f"**Dist to Goal:** {dist_to_goal:.4f}"
            if cum_total is not None:
                lines = [f"**Cum Reward:** {cum_total:.1f}"]
                if cum_breakdown:
                    order = [
                        "lifting_rew", "lift_bonus_rew",
                        "keypoint_rew", "bonus_rew",
                        "fingertip_delta_rew",
                        "kuka_actions_penalty", "hand_actions_penalty",
                        "retract_rew",
                    ]
                    for _k in order:
                        if _k in cum_breakdown:
                            lines.append(f"&nbsp;&nbsp;{_k}: {cum_breakdown[_k]:.1f}")
                self._md_reward.content = "\n\n".join(lines)

            # Retract status from state tuple
            if len(state) >= 8:
                retract_phase, retract_succeeded, mean_ft_dist = state[5], state[6], state[7]
                if retract_succeeded:
                    self._md_retract.content = f"**Retract:** SUCCESS (hand dist: {mean_ft_dist:.3f}m)"
                elif retract_phase:
                    self._md_retract.content = f"**Retract:** IN PROGRESS (hand dist: {mean_ft_dist:.3f}m)"

        elif tag == "done":
            goal_pct, steps = msg[1], msg[2]
            retract_ok = msg[3] if len(msg) > 3 else False
            final_arm_pos = msg[4] if len(msg) > 4 else None
            self._last_arm_pos = final_arm_pos
            self._episode_running = False
            self.ep_goals.append(goal_pct)
            self.ep_lengths.append(steps)
            self.ep_count += 1
            avg_g = np.mean(self.ep_goals)
            avg_t = np.mean(self.ep_lengths) / 60.0

            retract_str = f" | Retract: {'OK' if retract_ok else 'FAIL'}"
            self._md_retract.content = f"**Retract:** {'SUCCESS' if retract_ok else 'FAILED'}"
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Avg Goal:** {avg_g:.1f}% &nbsp;|&nbsp; "
                f"**Avg Time:** {avg_t:.1f}s"
            )
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals in {steps / 60.0:.1f}s{retract_str}")

            if self._auto_seq_active:
                self._auto_seq_results.append(
                    (self._pending_part, goal_pct, steps, retract_ok)
                )
                assembly = self._pending_assembly
                available = self._assembly_parts.get(assembly, [])
                self._auto_seq_step_idx += 1

                if self._auto_seq_step_idx < len(available):
                    next_pid = available[self._auto_seq_step_idx]
                    step_num = self._auto_seq_step_idx + 1
                    self._md_status.content = (
                        f"**Status:** Auto-sequence: loading step "
                        f"{step_num}/{len(available)} (Part {next_pid})..."
                    )
                    self._load_env(assembly, next_pid)
                else:
                    # All steps done -- show summary
                    self._auto_seq_active = False
                    summary_lines = ["**Auto-sequence complete:**\n"]
                    for pid, gpct, st, r_ok in self._auto_seq_results:
                        r_str = "OK" if r_ok else "FAIL"
                        summary_lines.append(
                            f"- Part {pid}: {gpct:.0f}% goals, {st / 60.0:.1f}s, retract: {r_str}"
                        )
                    overall_avg = np.mean([r[1] for r in self._auto_seq_results])
                    retract_results = [r[3] for r in self._auto_seq_results]
                    retract_rate = 100 * sum(retract_results) / len(retract_results)
                    summary_lines.append(
                        f"\n**Overall avg:** {overall_avg:.1f}% goals"
                        f" | **Retract:** {retract_rate:.0f}%"
                    )
                    self._md_status.content = '\n'.join(summary_lines)
                    print("[launcher] Auto-sequence complete")
                    for pid, gpct, st, r_ok in self._auto_seq_results:
                        print(f"  Part {pid}: {gpct:.0f}% goals, {st / 60.0:.1f}s, retract: {'OK' if r_ok else 'FAIL'}")
            else:
                self._md_status.content = (
                    f"**Status:** Done -- {steps / 60.0:.1f}s, {goal_pct:.0f}% goals{retract_str}"
                )

        elif tag == "stopped":
            self._episode_running = False
            self._md_status.content = "**Status:** Episode stopped."

        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
            self._auto_seq_active = False
            self._md_status.content = f"**Status:** Error -- {msg[1][:200]}"
            print(f"[launcher] Subprocess error:\n{msg[1]}")

    def _poll(self):
        if self._conn is None:
            return
        try:
            while self._conn.poll(0):
                self._handle(self._conn.recv())
        except (EOFError, ConnectionResetError, OSError):
            self._conn = None
            if self._proc is not None and not self._proc.is_alive():
                self._md_status.content = "**Status:** Subprocess exited unexpectedly."
                self._proc = None
                self._env_ready = False
                self._episode_running = False

    def run(self):
        print()
        print("  +-------------------------------------------------+")
        print("  |  Fabrica Assembly Evaluation (with Retract)      |")
        print(f"  |     http://localhost:{self.port:<26}|")
        print("  +-------------------------------------------------+")
        print()

        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_subprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabrica Assembly Evaluation with Retract")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policies-dir", type=str, default=None,
                        help="Directory of policy subfolders (each with "
                             "config.yaml + model.pth). Populates the Policy "
                             "dropdown. Takes precedence over --config-path; "
                             "combine with --policies to filter a subset.")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Policy names (filters --policies-dir).")
    parser.add_argument("--initial-policy", type=str, default=None,
                        help="Policy name to pre-select in the dropdown.")
    parser.add_argument("--config-path", type=str, default="pretrained_policy/config.yaml",
                        help="Single-policy fallback when --policies-dir is not used.")
    parser.add_argument("--checkpoint-path", type=str, default="pretrained_policy/model.pth",
                        help="Single-policy fallback when --policies-dir is not used.")
    parser.add_argument("--final-goal-tolerance", type=float, default=None,
                        help="Tighter tolerance for the last subgoal in fixedGoalStates")
    parser.add_argument("--collision", choices=["vhacd", "coacd", "sdf"],
                        default="vhacd",
                        help="Collision method: vhacd (default), coacd, or sdf")
    parser.add_argument("--no-headless", action="store_true",
                        help="Show IsaacGym viewer window")
    parser.add_argument("--record-video", action="store_true",
                        help="Record video of each episode from IsaacGym camera")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory for recorded videos (default: eval_videos/)")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="dense",
                        help="Initial goal-trajectory mode "
                             "(runtime-changeable via the viser dropdown)")
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"),
                        help="Extra config overrides, e.g. --override task.sim.physx.num_position_iterations 32")
    args = parser.parse_args()

    def _resolve(p):
        path = Path(p)
        if path.exists():
            return str(path)
        path = REPO_ROOT / p
        if path.exists():
            return str(path)
        raise FileNotFoundError(p)

    # Parse --override key value pairs, auto-cast numeric types
    extra_overrides = {}
    for key, val in args.override:
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        if val == "True":
            val = True
        elif val == "False":
            val = False
        extra_overrides[key] = val

    policies = {}
    if args.policies_dir is not None:
        pdir = Path(args.policies_dir)
        if not pdir.is_absolute():
            pdir = REPO_ROOT / pdir
        if not pdir.exists():
            raise FileNotFoundError(f"--policies-dir not found: {pdir}")
        name_filter = set(args.policies) if args.policies else None
        for sub in sorted(pdir.iterdir()):
            if name_filter is not None and sub.name not in name_filter:
                continue
            cfg = sub / "config.yaml"
            ckpt = sub / "model.pth"
            if cfg.exists() and ckpt.exists():
                policies[sub.name] = (str(cfg), str(ckpt))
        if not policies:
            raise FileNotFoundError(
                f"No policy subfolders in {pdir} "
                f"(each needs config.yaml + model.pth)"
            )
    else:
        name = Path(args.config_path).parent.name or "policy"
        policies[name] = (_resolve(args.config_path), _resolve(args.checkpoint_path))

    AssemblyDemo(
        policies=policies,
        port=args.port,
        final_goal_tolerance=args.final_goal_tolerance,
        collision_method=args.collision,
        extra_overrides=extra_overrides,
        headless=not args.no_headless,
        record_video=args.record_video,
        video_dir=args.video_dir,
        goal_mode=args.goal_mode,
        initial_policy=args.initial_policy,
    ).run()
