#!/usr/bin/env python3
"""Dynamic peg-in-hole evaluation with pretrained policy (viser).

Same architecture as ``peg_multi_init_eval.py`` (main-process viser GUI +
IsaacGym subprocess) but for ``PegInHoleDynamicEnv``: no scenes.npz, hole
position randomized each episode.

UI controls:
  - Policy      — pick from --policies-dir / --config-path subfolders
  - Goal mode   — preInsertAndFinal / finalGoalOnly
  - Random goal fraction — slider [0, 1] for co-training mix

Usage:
    python peg_in_hole/peg_dynamic_eval.py \\
        --config-path pretrained_policy/config.yaml \\
        --checkpoint-path pretrained_policy/model.pth \\
        --port 8043
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "peg_in_hole"

TABLE_Z = 0.38
N_ACT = 29
OBS_DIM = 140
CONTROL_DT = 1.0 / 60.0

_ARM_DEFAULT = np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308])
_ARM_DEFAULT[1] -= np.deg2rad(10)
_ARM_DEFAULT[3] += np.deg2rad(10)
DEFAULT_DOF_POS = np.zeros(N_ACT)
DEFAULT_DOF_POS[:7] = _ARM_DEFAULT

GOAL_MODES = ["preInsertAndFinal", "finalGoalOnly"]

HOLE_SCENE_Z = 0.15  # hole base Z in scene-local frame (= table top)

# Target volume for random-goal sampling (matches PegInHoleDynamicEnv.yaml)
TARGET_VOLUME_MINS = [-0.35, -0.1, 0.6]
TARGET_VOLUME_MAXS = [0.35, 0.2, 0.95]

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
    "task.env.viserViz": False,
    "task.env.withTableForceSensor": True,
    "task.env.tableForceResetThreshold": 1.0e6,
    "task.env.retractDistanceThreshold": 0.5,
}


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


def _load_mesh_for_viz(asset_path: Path) -> trimesh.Trimesh:
    """Load a mesh for viser visualization. Handles URDF (box + mesh geometry)
    and plain mesh files (OBJ, STL)."""
    if asset_path.suffix in (".obj", ".stl", ".ply"):
        return trimesh.load(str(asset_path), force="mesh")

    # Parse URDF: extract boxes and mesh references from collision geometry
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(asset_path))
    meshes = []
    for col in tree.iter("collision"):
        origin = col.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        geom = col.find("geometry")
        if geom is None:
            continue
        box = geom.find("box")
        mesh_elem = geom.find("mesh")
        if box is not None:
            size = [float(v) for v in box.get("size").split()]
            m = trimesh.creation.box(extents=size)
        elif mesh_elem is not None:
            mesh_file = asset_path.parent / mesh_elem.get("filename")
            if not mesh_file.exists():
                continue
            m = trimesh.load(str(mesh_file), force="mesh")
        else:
            continue
        # Apply origin transform
        T = np.eye(4)
        T[:3, 3] = xyz
        if any(v != 0 for v in rpy):
            from scipy.spatial.transform import Rotation
            T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        m.apply_transform(T)
        meshes.append(m)
    if not meshes:
        # Fallback: try loading the URDF's visual meshes
        for vis in tree.iter("visual"):
            geom = vis.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is not None:
                mesh_file = asset_path.parent / mesh_elem.get("filename")
                if mesh_file.exists():
                    meshes.append(trimesh.load(str(mesh_file), force="mesh"))
    if not meshes:
        return trimesh.creation.box(extents=(0.08, 0.08, 0.01))
    return trimesh.util.concatenate(meshes)


# ===================================================================
# SUBPROCESS -- IsaacGym simulation
# ===================================================================

def _create_env(config_path, headless, device, overrides):
    from deployment.rl_player_utils import read_cfg_omegaconf
    from deployment.isaac.isaac_env import merge_cfg_with_default_config, create_env_from_cfg
    from omegaconf import OmegaConf

    cfg = read_cfg_omegaconf(config_path=config_path, device=device)
    cfg = merge_cfg_with_default_config(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "PegInHoleDynamicEnv"
    cfg.task_name = "PegInHoleDynamicEnv"
    pih_defaults = {
        "objectName": "peg",
        "useFixedGoalStates": True,
        "useFixedInitObjectPose": True,
        "enableRetract": True,
        "retractRewardScale": 1.0,
        "retractDistanceThreshold": 0.1,
        "retractSuccessBonus": 1000.0,
        "retractSuccessTolerance": 0.005,
        "goalXyObsNoise": 0.0,
        "tableForceResetThreshold": 100.0,
        "holeUrdf": "urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf",
        "holeXRange": [-0.1875, 0.1875],
        "holeYRange": [-0.1, 0.2],
        "holeZOffset": 0.0,
        "insertPoseRelHole": [0.0, 0.0, 0.136, 0.0, -0.70710678, 0.0, 0.70710678],
        "insertionDirection": [0.0, 0.0, -1.0],
        "preInsertOffset": 0.05,
        "goalMode": "preInsertAndFinal",
        "randomGoalFraction": 0.0,
        "randomGoalMaxSuccesses": 50,
    }
    for k, v in pih_defaults.items():
        OmegaConf.update(cfg, f"task.env.{k}", v, force_add=True)
    return create_env_from_cfg(cfg=cfg, headless=headless, overrides=overrides)


def _sim_get_state(env, obs, joint_lower, joint_upper):
    obs_np = obs[0].cpu().numpy()
    joint_pos = 0.5 * (obs_np[:N_ACT] + 1.0) * (joint_upper - joint_lower) + joint_lower
    hole_pos = env.hole_pos[0].cpu().numpy() if hasattr(env, "hole_pos") else np.zeros(3)
    is_rg = bool(env.is_random_goal_env[0].item()) if hasattr(env, "is_random_goal_env") else False
    return (
        joint_pos,                                           # 0
        env.object_state[0, :7].cpu().numpy(),               # 1
        env.goal_pose[0].cpu().numpy(),                      # 2
        env.obj_keypoint_pos[0].cpu().numpy(),               # 3
        env.goal_keypoint_pos[0].cpu().numpy(),              # 4
        bool(env.retract_phase[0].item()),                   # 5
        bool(env.retract_succeeded[0].item()),               # 6
        float(env.curr_fingertip_distances[0].mean().item()),# 7
        float(env.keypoints_max_dist[0].item()),             # 8
        float(env.success_tolerance * env.keypoint_scale),   # 9
        int(env.near_goal_steps[0].item()),                  # 10
        int(env.progress_buf[0].item()),                     # 11
        int(env.max_episode_length),                         # 12
        bool(env.reset_buf[0].item()),                       # 13
        env.table_sensor_forces_smoothed[0, :3].cpu().numpy()
        if hasattr(env, "table_sensor_forces_smoothed")
        else np.zeros(3, dtype=np.float32),                  # 14
        hole_pos,                                            # 15
        is_rg,                                               # 16
    )


def _sim_reset(env, device):
    import torch
    obs, _, _, _ = env.step(torch.zeros((env.num_envs, N_ACT), device=device))
    return obs["obs"]


def _sim_episode(conn, env, policy, joint_lower, joint_upper, device):
    import time as _time
    import torch  # noqa: F401
    policy.reset()
    obs = _sim_reset(env, device)

    retract_ok = False
    peak_successes = 0
    max_goals_seen = max(1, int(
        env.env_max_goals[0].item() if hasattr(env, "env_max_goals")
        else env.max_consecutive_successes
    ))

    step, done, paused = 0, False, False
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
            _time.sleep(0.05)
            continue

        t0 = _time.time()
        state = _sim_get_state(env, obs, joint_lower, joint_upper)
        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, _, done_tensor, _ = env.step(action)
        obs = obs_dict["obs"]
        done = done_tensor[0].item()
        step += 1

        cur_succ = int(env.successes[0].item())
        cur_max = int(env.env_max_goals[0].item())
        if cur_succ > peak_successes:
            peak_successes = cur_succ
        if cur_max > max_goals_seen:
            max_goals_seen = cur_max
        if env.extras.get("retract_success_ratio", 0.0) > 0.5:
            retract_ok = True

        conn.send(("state", state, cur_succ, cur_max, step, retract_ok))

        elapsed = _time.time() - t0
        if (sleep := CONTROL_DT - elapsed) > 0:
            _time.sleep(sleep)

    goal_pct = 100 * peak_successes / max(1, max_goals_seen)
    conn.send(("done", goal_pct, step, retract_ok))
    return obs


def sim_worker(conn, config_path, checkpoint_path, goal_mode,
               random_goal_fraction, extra_overrides=None, headless=True):
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
        import torch
        from deployment.rl_player import RlPlayer
        import peg_in_hole.objects  # noqa: F401  (registers peg)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        overrides = {
            **BASE_OVERRIDES,
            "task.env.goalMode": goal_mode,
            "task.env.randomGoalFraction": float(random_goal_fraction),
            **(extra_overrides or {}),
        }
        env = _create_env(config_path=str(config_path), headless=headless,
                          device=device, overrides=overrides)

        # Debug: print keypoint offsets
        print(f"[diag] object_scales[0]: {env.object_scales[0].cpu().tolist()}", flush=True)
        print(f"[diag] object_keypoint_offsets[0]: {env.object_keypoint_offsets[0].cpu().tolist()}", flush=True)
        print(f"[diag] object_keypoint_offsets_fixed_size[0]: {env.object_keypoint_offsets_fixed_size[0].cpu().tolist()}", flush=True)
        print(f"[diag] fixedSize config: {env.cfg['env'].get('fixedSize')}", flush=True)
        print(f"[diag] keypointScale: {env.keypoint_scale}", flush=True)
        print(f"[diag] objectBaseSize: {env.object_base_size}", flush=True)

        joint_lower = env.arm_hand_dof_lower_limits[:N_ACT].cpu().numpy()
        joint_upper = env.arm_hand_dof_upper_limits[:N_ACT].cpu().numpy()
        env.set_env_state(torch.load(checkpoint_path, map_location=device)[0]["env_state"])
        policy = RlPlayer(OBS_DIM, N_ACT, config_path, checkpoint_path, device, env.num_envs)

        obs = _sim_reset(env, device)
        init_state = _sim_get_state(env, obs, joint_lower, joint_upper)
        conn.send(("ready", init_state))

        while True:
            cmd = conn.recv()
            if cmd == "run":
                obs = _sim_episode(conn, env, policy, joint_lower, joint_upper, device)
            elif cmd == "quit":
                break
    except Exception as exc:
        conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
    conn.close()


# ===================================================================
# MAIN PROCESS -- viser GUI
# ===================================================================

class PegDynamicDemo:
    def __init__(self, policies: Dict[str, Tuple[str, str]],
                 port: int = 8043, headless: bool = True,
                 goal_mode: str = "preInsertAndFinal",
                 random_goal_fraction: float = 0.0,
                 initial_policy: Optional[str] = None,
                 extra_overrides: Optional[dict] = None,
                 object_urdf: Optional[str] = None,
                 hole_urdf: Optional[str] = None):
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}")
        if not policies:
            raise ValueError("policies dict must be non-empty")
        self.policies = policies
        self.initial_policy = initial_policy if initial_policy in policies else next(iter(policies))
        self.port = port
        self.headless = headless
        self.goal_mode = goal_mode
        self.random_goal_fraction = random_goal_fraction
        self.extra_overrides = extra_overrides or {}
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        # Overridable asset paths (relative to assets/ root)
        self._object_urdf_rel = object_urdf or "urdf/peg_in_hole/peg/peg.urdf"
        self._hole_urdf_rel = hole_urdf or "urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf"
        self._object_urdf_abs = REPO_ROOT / "assets" / self._object_urdf_rel
        self._hole_urdf_abs = REPO_ROOT / "assets" / self._hole_urdf_rel

        self._proc = None
        self._conn = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

        self.ep_count = 0
        self._peak_force = 0.0

        self.robot = None
        self._dyn = []
        self._obj_frame = None
        self._goal_frame = None
        self._hole_frame = None
        self._obj_keypoints = []
        self._goal_keypoints = []

        self._hole_mesh = _load_mesh_for_viz(self._hole_urdf_abs)
        self._object_mesh = _load_mesh_for_viz(self._object_urdf_abs)
        self._target_vol_box = None

        self._build_gui()
        self._setup_static_scene()

    def _build_gui(self):
        self.server.gui.add_markdown(
            "# Peg-in-Hole Dynamic Eval\n"
            "### Pretrained policy with dynamic hole placement"
        )

        with self.server.gui.add_folder("Task Selection", expand_by_default=True):
            self._dd_policy = self.server.gui.add_dropdown(
                "Policy", options=list(self.policies.keys()),
                initial_value=self.initial_policy,
            )
            self._dd_goal_mode = self.server.gui.add_dropdown(
                "Goal mode", options=GOAL_MODES, initial_value=self.goal_mode,
            )
            self._sl_rgf = self.server.gui.add_slider(
                "Random goal frac", min=0.0, max=1.0, step=0.1,
                initial_value=self.random_goal_fraction,
            )
            self._btn_load = self.server.gui.add_button("Load / reload env")
            self._btn_load.on_click(lambda _: self._load_env())
            self._md_status = self.server.gui.add_markdown("**Status:** Ready")

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop")
            self._btn_stop.on_click(lambda _: self._cmd_stop())

        with self.server.gui.add_folder("Display", expand_by_default=True):
            self._cb_keypoints = self.server.gui.add_checkbox("Show keypoints", initial_value=True)
            self._cb_keypoints.on_update(lambda _: self._apply_keypoint_visibility())
            self._cb_target_vol = self.server.gui.add_checkbox("Show target volume", initial_value=False)
            self._cb_target_vol.on_update(lambda _: self._toggle_target_volume())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_hole = self.server.gui.add_markdown("**Hole pos:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_diag = self.server.gui.add_markdown("**Goal dist:** --")
            self._md_retract = self.server.gui.add_markdown("**Retract:** --")
            self._md_force = self.server.gui.add_markdown("**Table force:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")

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
            REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description"
            / "iiwa14_left_sharpa_adjusted_restricted.urdf",
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
        self._obj_frame = self._goal_frame = self._hole_frame = None
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()

    def _add_object_viz(self, node_name: str, color, opacity=1.0):
        """Add object mesh to scene. Uses ViserUrdf for URDFs, mesh_simple for OBJ/STL."""
        if self._object_urdf_abs.suffix == ".urdf":
            try:
                ViserUrdf(self.server, self._object_urdf_abs,
                          root_node_name=node_name, mesh_color_override=color)
                return
            except Exception:
                pass
        verts = np.array(self._object_mesh.vertices, dtype=np.float32)
        faces = np.array(self._object_mesh.faces, dtype=np.uint32)
        rgb = color[:3] if len(color) >= 3 else color
        self._dyn.append(self.server.scene.add_mesh_simple(
            f"{node_name}/mesh", vertices=verts, faces=faces,
            color=rgb, opacity=opacity,
        ))

    def _setup_scene_objects(self):
        self._clear_dynamic()

        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.05, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)
        self._add_object_viz("/object", (204, 40, 40))

        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.05, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)
        self._add_object_viz("/goal", (0, 255, 0), opacity=0.5)

        self._hole_frame = self.server.scene.add_frame(
            "/hole", position=(0, 0, TABLE_Z + HOLE_SCENE_Z),
            wxyz=(1, 0, 0, 0), show_axes=False,
        )
        self._dyn.append(self._hole_frame)
        self._dyn.append(self.server.scene.add_mesh_simple(
            "/hole/mesh",
            vertices=np.array(self._hole_mesh.vertices, dtype=np.float32),
            faces=np.array(self._hole_mesh.faces, dtype=np.uint32),
            color=(120, 120, 120),
        ))

    def _update_hole_viz(self, hole_pos):
        if self._hole_frame is not None:
            if hole_pos[2] < 0:
                self._hole_frame.visible = False
            else:
                self._hole_frame.visible = True
                self._hole_frame.position = (
                    float(hole_pos[0]), float(hole_pos[1]), float(hole_pos[2]),
                )

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
        self._apply_keypoint_visibility()

    def _apply_keypoint_visibility(self):
        visible = self._cb_keypoints.value
        for kp in self._obj_keypoints + self._goal_keypoints:
            kp.visible = visible

    def _toggle_target_volume(self):
        show = self._cb_target_vol.value
        if show and self._target_vol_box is None:
            tv_min = np.array(TARGET_VOLUME_MINS)
            tv_max = np.array(TARGET_VOLUME_MAXS)
            center = (tv_min + tv_max) / 2
            dims = tv_max - tv_min
            self._target_vol_box = self.server.scene.add_box(
                "/target_volume",
                color=(100, 255, 100),
                dimensions=tuple(dims.tolist()),
                position=tuple(center.tolist()),
                side="double",
                opacity=0.08,
            )
        if self._target_vol_box is not None:
            self._target_vol_box.visible = show

    # ── Subprocess management ────────────────────────────────────

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

    def _load_env(self):
        self._kill_subprocess()

        goal_mode = self._dd_goal_mode.value
        rgf = self._sl_rgf.value
        policy_name = self._dd_policy.value
        config_path, checkpoint_path = self.policies[policy_name]

        label = f"{policy_name} | goals: {goal_mode} | rgf: {rgf:.1f}"
        self._md_status.content = f"**Status:** Loading *{label}* ..."
        self._md_task.content = f"**Task:** {label}"
        self._md_retract.content = "**Retract:** --"
        self._md_hole.content = "**Hole pos:** --"
        self.ep_count = 0
        self._peak_force = 0.0
        self._md_stats.content = "**Stats:** No episodes yet"

        self.robot.update_cfg(DEFAULT_DOF_POS)
        self._setup_scene_objects()

        # Merge asset overrides into extra_overrides for the subprocess
        sim_overrides = dict(self.extra_overrides)
        sim_overrides["task.env.holeUrdf"] = self._hole_urdf_rel
        sim_overrides["task.env.targetVolumeMins"] = TARGET_VOLUME_MINS
        sim_overrides["task.env.targetVolumeMaxs"] = TARGET_VOLUME_MAXS

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sim_worker,
            args=(child_conn, config_path, checkpoint_path,
                  goal_mode, rgf, sim_overrides, self.headless),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned pid={self._proc.pid} goal_mode={goal_mode} rgf={rgf} "
              f"hole={self._hole_urdf_rel}")

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
        self._peak_force = 0.0
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

        if len(state_tuple) > 15:
            hole_pos = state_tuple[15]
            self._update_hole_viz(hole_pos)
            is_rg = state_tuple[16] if len(state_tuple) > 16 else False
            mode_str = "RANDOM GOAL" if is_rg else "INSERTION"
            if hole_pos[2] < 0:
                self._md_hole.content = f"**Hole pos:** hidden (random-goal mode) | **Mode:** {mode_str}"
            else:
                self._md_hole.content = (
                    f"**Hole pos:** ({hole_pos[0]:.3f}, {hole_pos[1]:.3f}, {hole_pos[2]:.3f})"
                    f" | **Mode:** {mode_str}"
                )

    def _handle(self, msg):
        tag = msg[0]
        if tag == "ready":
            init_state = msg[1]
            if len(init_state) > 3:
                self._setup_keypoints(init_state[3].shape[0])
            self._update_viz(init_state)
            self._env_ready = True
            self._md_status.content = "**Status:** Ready -- click **Run Episode**"
            print("[launcher] Environment ready")

        elif tag == "state":
            state, successes, max_succ, step = msg[1], msg[2], msg[3], msg[4]
            latched_retract = msg[5] if len(msg) > 5 else False
            self._update_viz(state)
            pct = 100 * successes / max_succ if max_succ > 0 else 0
            self._md_prog.content = (
                f"**Time:** {step / 60.0:.1f}s &nbsp;|&nbsp; "
                f"**Goal:** {successes}/{max_succ} ({pct:.0f}%)"
            )
            if len(state) >= 8:
                retract_phase, retract_ok, mean_ft_dist = state[5], state[6], state[7]
                retract_ok = retract_ok or latched_retract
                if retract_ok:
                    self._md_retract.content = f"**Retract:** SUCCESS (hand dist {mean_ft_dist:.3f}m)"
                elif retract_phase:
                    self._md_retract.content = f"**Retract:** IN PROGRESS (hand dist {mean_ft_dist:.3f}m)"
                else:
                    self._md_retract.content = f"**Retract:** not yet (hand dist {mean_ft_dist:.3f}m)"
            if len(state) >= 14:
                kp_max_dist = state[8]
                tol_m = state[9]
                near_steps = state[10]
                progress = state[11]
                max_ep_len = state[12]
                reset_pending = state[13]
                in_tol = "Y" if kp_max_dist <= tol_m else "N"
                self._md_diag.content = (
                    f"**Goal dist:** {kp_max_dist*1000:.1f} mm {in_tol}  "
                    f"&nbsp;(tol {tol_m*1000:.1f} mm)  "
                    f"&nbsp;near-goal-steps: **{near_steps}**  \n"
                    f"**progress_buf:** {progress}/{max_ep_len}  "
                    f"&nbsp;reset_buf: **{reset_pending}**"
                )
            if len(state) >= 15:
                force_vec = np.asarray(state[14], dtype=np.float32)
                force_mag = float(np.linalg.norm(force_vec))
                if force_mag > self._peak_force:
                    self._peak_force = force_mag
                self._md_force.content = (
                    f"**Table force:** {force_mag:.2f} N  "
                    f"&nbsp;(peak: **{self._peak_force:.2f} N**)  \n"
                    f"&nbsp;[fx, fy, fz] = "
                    f"[{force_vec[0]:+.2f}, {force_vec[1]:+.2f}, {force_vec[2]:+.2f}] N"
                )

        elif tag == "done":
            goal_pct, steps, retract_ok = msg[1], msg[2], msg[3]
            self._episode_running = False
            self.ep_count += 1
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Last goal:** {goal_pct:.0f}% &nbsp;|&nbsp; "
                f"**Last time:** {steps / 60.0:.1f}s"
            )
            self._md_status.content = (
                f"**Status:** Done -- {steps / 60.0:.1f}s, {goal_pct:.0f}% goals"
                f" | Retract {'OK' if retract_ok else 'FAIL'}"
            )
            self._md_retract.content = f"**Retract:** {'SUCCESS' if retract_ok else 'FAILED'}"
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals, {steps / 60.0:.1f}s")

        elif tag == "stopped":
            self._episode_running = False
            self._md_status.content = "**Status:** Episode stopped."

        elif tag == "error":
            self._env_ready = False
            self._episode_running = False
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
        print(f"  Peg-in-Hole Dynamic Eval   http://localhost:{self.port}")
        print()
        try:
            while True:
                self._poll()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[launcher] Shutting down...")
            self._kill_subprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8043)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--policies-dir", type=str, default=None,
                        help="Directory of policy subfolders (each with config.yaml + model.pth).")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Policy names (filters --policies-dir).")
    parser.add_argument("--initial-policy", type=str, default=None)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="preInsertAndFinal")
    parser.add_argument("--random-goal-fraction", type=float, default=0.0)
    parser.add_argument("--object-urdf", type=str, default=None,
                        help="Object URDF path relative to assets/ (default: urdf/peg_in_hole/peg/peg.urdf)")
    parser.add_argument("--hole-urdf", type=str, default=None,
                        help="Hole URDF path relative to assets/ (default: urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf)")
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"),
                        help="Extra task-env overrides (repeated).")
    args = parser.parse_args()

    def _resolve(p):
        path = Path(p)
        if path.exists():
            return str(path)
        path = REPO_ROOT / p
        if path.exists():
            return str(path)
        raise FileNotFoundError(p)

    extra_overrides = {}
    for key, val in args.override:
        # Try parsing as JSON first (handles lists like [-0.005,0.1,0.2])
        import json
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, ValueError):
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

    policies: Dict[str, Tuple[str, str]] = {}
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
            raise FileNotFoundError(f"No policy subfolders in {pdir}")
    else:
        if args.config_path is None or args.checkpoint_path is None:
            raise ValueError(
                "Provide either --policies-dir or both --config-path and --checkpoint-path"
            )
        name = Path(args.config_path).parent.name or "policy"
        policies[name] = (_resolve(args.config_path), _resolve(args.checkpoint_path))

    PegDynamicDemo(
        policies=policies,
        port=args.port,
        headless=not args.no_headless,
        goal_mode=args.goal_mode,
        random_goal_fraction=args.random_goal_fraction,
        initial_policy=args.initial_policy,
        extra_overrides=extra_overrides,
        object_urdf=args.object_urdf,
        hole_urdf=args.hole_urdf,
    ).run()
