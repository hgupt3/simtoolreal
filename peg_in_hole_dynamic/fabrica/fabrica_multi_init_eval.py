"""Fabrica Multi-Init-State Evaluation (interactive viser).

Like fabrica_eval.py but the per-part start pose + goals come from
``scenes.npz`` / ``scenes_val.npz`` instead of from the fixed
``trajectories/<pid>/pick_place.json``. This lets us evaluate multi-init-
trained policies on the same scenes they trained on (train split) or on a
held-out set of scenes (val split).

Architecture is identical to fabrica_eval.py:
  Main process  -- viser GUI + scene rendering (no isaacgym)
  Subprocess    -- IsaacGym FabricaEnv + policy

Usage:
    python fabrica/fabrica_multi_init_eval.py \\
        --config-path train_dir/.../config.yaml \\
        --checkpoint-path train_dir/.../model.pth \\
        --collision coacd
"""

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from fabrica.viser_utils import COLORS
from fabrica.fabrica_eval import (
    ASSETS_DIR,
    DEFAULT_DOF_POS,
    N_ACT,
    REPO_ROOT,
    TABLE_Z,
    _table_urdf_rel,
    quat_xyzw_to_wxyz,
    sim_worker,
)

sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

def _discover_fabrica_assemblies() -> List[str]:
    """All directories under assets/urdf/fabrica/ that carry
    canonical_transforms.json (picks up both `beam` and user-generated
    clones like `beam_2x`)."""
    return sorted(
        d.name for d in ASSETS_DIR.iterdir()
        if d.is_dir() and (d / "canonical_transforms.json").exists()
    )


ALL_ASSEMBLIES = _discover_fabrica_assemblies()

SCENES_FILENAME = "scenes.npz"  # single-insertion schema has no train/val split

# Goal modes map to config overrides read by FabricaEnv:
#   dense                 -> full dense trajectory
#   final_only            -> finalGoalOnly     (single insertion goal)
#   pre_insert_and_final  -> preInsertAndFinal (pre-insert + insert, 2 goals)
# The client-side names are kept kebab/snake for backwards compat with existing
# UI / CLI; the values pushed to task.env.goalMode are camelCase.
GOAL_MODES = ["dense", "final_only", "pre_insert_and_final"]

_GOAL_MODE_CAMEL = {
    "dense": "dense",
    "final_only": "finalGoalOnly",
    "pre_insert_and_final": "preInsertAndFinal",
}


def apply_goal_mode(goals: List[List[float]], mode: str) -> List[List[float]]:
    """Slice a dense goal trajectory for a given goal mode (client-side).

    Used by the reuse-worker eval path so goal-mode switching doesn't
    require re-instantiating the env.
    """
    if mode == "dense" or len(goals) <= 1:
        return goals
    if mode == "final_only":
        return [goals[-1]]
    if mode == "pre_insert_and_final":
        return [goals[-2], goals[-1]]
    raise ValueError(f"Unknown goal mode: {mode}")


# ===================================================================
# Scene data loading
# ===================================================================

def _assemblies_with_scenes() -> List[str]:
    """Return assemblies that have a scenes.npz on disk."""
    return [a for a in ALL_ASSEMBLIES if (ASSETS_DIR / a / SCENES_FILENAME).exists()]


def _load_scenes(assembly: str) -> Optional[dict]:
    """Load scenes.npz / scenes_val.npz for an assembly (single-insertion schema).

    Returns a dict with:
      - insertion_parts          : list[str]             — trainable parts (length P)
      - start_poses              : (P, N, M, 7)          — xyz + xyzw quat
      - goals                    : (P, N, M, T, 7)
      - traj_lengths             : (P, N, M)
      - partial_assembly_offsets : (P, N, 3)
      - scene_urdf_paths         : (P, N) of str         — relative to assets/
      - assembly_order           : list[str]             — full steps from JSON
                                                            (used for fixture rendering)
      - inserts_into             : dict[str, str]        — receiver per insertion part
    """
    path = ASSETS_DIR / assembly / SCENES_FILENAME
    if not path.exists():
        return None

    data = np.load(str(path), allow_pickle=True)
    order_path = ASSETS_DIR / assembly / "assembly_order.json"
    order_json = json.loads(order_path.read_text())

    return {
        "insertion_parts": [str(p) for p in data["insertion_parts"].tolist()],
        "start_poses": data["start_poses"],                             # (P, N, M, 7)
        "goals": data["goals"],                                          # (P, N, M, T, 7)
        "traj_lengths": data["traj_lengths"],                            # (P, N, M)
        "partial_assembly_offsets": data["partial_assembly_offsets"],    # (P, N, 3)
        "scene_urdf_paths": data["scene_urdf_paths"],                    # (P, N)
        "assembly_order": order_json["steps"],                           # full chain
        "inserts_into": order_json.get("inserts_into", {}),
    }


def _scene_start_and_goals(scenes: dict, scene_idx: int, part_id: str,
                           start_idx: int = 0
                           ) -> Tuple[List[float], List[List[float]]]:
    """Extract (start_pose, goals) for a specific (scene_idx, part_id, start_idx).

    Slices ``goals`` to the valid per-(p, n, m) trajectory length.
    """
    ins_parts = scenes["insertion_parts"]
    p_idx = ins_parts.index(part_id)
    start = scenes["start_poses"][p_idx, scene_idx, start_idx].tolist()
    L = int(scenes["traj_lengths"][p_idx, scene_idx, start_idx])
    goals = scenes["goals"][p_idx, scene_idx, start_idx, :L].tolist()
    return start, goals


# ===================================================================
# Main process -- viser GUI
# ===================================================================

class MultiInitAssemblyDemo:
    """Interactive viser-based eval using scenes.npz / scenes_val.npz."""

    _PH = "-- Select --"

    def __init__(self, policies: Dict[str, Tuple[str, str]],
                 port: int = 8080,
                 final_goal_tolerance: Optional[float] = None,
                 collision_method: str = "coacd",
                 extra_overrides: Optional[dict] = None,
                 headless: bool = True,
                 goal_mode: str = "dense",
                 initial_policy: Optional[str] = None):
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
        self.extra_overrides = extra_overrides or {}
        self.headless = headless
        self.goal_mode = goal_mode
        self.server = viser.ViserServer(host="0.0.0.0", port=port)

        self._proc: Optional[multiprocessing.Process] = None
        self._conn = None
        self._env_ready = False
        self._episode_running = False
        self._is_paused = False

        # Pending context (set at load time)
        self._pending_assembly = ""
        self._pending_part = ""
        self._pending_scene_idx = 0
        self._pending_start_idx = 0

        # Per-assembly scene caches
        self._scene_cache: Dict[str, dict] = {}

        # Auto-sequence state (run all parts in a scene)
        self._auto_seq_active = False
        self._auto_seq_step_idx = 0
        self._auto_seq_results: List[Tuple[str, float, int, bool]] = []
        self._last_arm_pos: Optional[List[float]] = None

        # Episode stats
        self.ep_count = 0
        self.ep_goals: List[float] = []
        self.ep_lengths: List[int] = []

        # Dynamic viser handles
        self.robot = None
        self._dyn: list = []
        self._obj_frame = None
        self._goal_frame = None
        self._obj_viser_urdf: Optional[ViserUrdf] = None
        self._goal_viser_urdf: Optional[ViserUrdf] = None
        self._obj_keypoints: list = []
        self._goal_keypoints: list = []

        self._build_gui()
        self._setup_static_scene()

    # -- GUI ------------------------------------------------------------

    def _build_gui(self):
        self.server.gui.add_markdown(
            "# Fabrica Multi-Init Eval\n### Train / Val scene trajectories"
        )

        with self.server.gui.add_folder("Scene Selection", expand_by_default=True):
            self._dd_policy = self.server.gui.add_dropdown(
                "Policy", options=list(self.policies.keys()),
                initial_value=self.initial_policy,
            )
            assemblies = [self._PH] + _assemblies_with_scenes()
            self._dd_assembly = self.server.gui.add_dropdown(
                "Assembly", options=assemblies, initial_value=self._PH,
            )
            self._dd_part = self.server.gui.add_dropdown(
                "Insertion Part", options=[self._PH], initial_value=self._PH,
            )
            self._dd_scene = self.server.gui.add_dropdown(
                "Scene Index", options=[self._PH], initial_value=self._PH,
            )
            self._dd_start = self.server.gui.add_dropdown(
                "Start Index", options=[self._PH], initial_value=self._PH,
            )
            self._dd_goal_mode = self.server.gui.add_dropdown(
                "Goal Mode", options=GOAL_MODES, initial_value=self.goal_mode,
            )
            self._btn_load = self.server.gui.add_button("Load Environment")
            self._btn_load.on_click(lambda _: self._load_env())
            self._md_status = self.server.gui.add_markdown("**Status:** Ready")
            self._dd_assembly.on_update(lambda _: self._on_assembly_change())
            self._dd_scene.on_update(lambda _: self._on_scene_change())

        with self.server.gui.add_folder("Episode Controls", expand_by_default=True):
            self._btn_run = self.server.gui.add_button("Run Episode")
            self._btn_run.on_click(lambda _: self._cmd_run())
            self._btn_pause = self.server.gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self._cmd_pause())
            self._btn_stop = self.server.gui.add_button("Stop Episode")
            self._btn_stop.on_click(lambda _: self._cmd_stop())
            self._btn_run_all = self.server.gui.add_button("Run All Parts in Scene")
            self._btn_run_all.on_click(lambda _: self._cmd_run_all())

        with self.server.gui.add_folder("Display", expand_by_default=True):
            self._cb_keypoints = self.server.gui.add_checkbox(
                "Show Keypoints", initial_value=True,
            )
            self._cb_keypoints.on_update(lambda _: self._apply_keypoint_visibility())

        with self.server.gui.add_folder("Status", expand_by_default=True):
            self._md_task = self.server.gui.add_markdown("**Task:** --")
            self._md_prog = self.server.gui.add_markdown("**Progress:** --")
            self._md_retract = self.server.gui.add_markdown("**Retract dist:** --")
            self._md_stats = self.server.gui.add_markdown("**Stats:** No episodes yet")
            self._md_obj = self.server.gui.add_markdown("**Object Pos:** --")
            self._md_dist = self.server.gui.add_markdown("**Dist to Goal:** --")
            self._md_reward = self.server.gui.add_markdown("**Cum Reward:** --")

    def _get_scenes(self, assembly: str) -> Optional[dict]:
        if assembly not in self._scene_cache:
            scenes = _load_scenes(assembly)
            if scenes is None:
                return None
            self._scene_cache[assembly] = scenes
        return self._scene_cache[assembly]

    def _on_assembly_change(self):
        assembly = self._dd_assembly.value
        if assembly == self._PH:
            self._dd_scene.options = [self._PH]
            self._dd_scene.value = self._PH
            self._dd_part.options = [self._PH]
            self._dd_part.value = self._PH
            self._dd_start.options = [self._PH]
            self._dd_start.value = self._PH
            return
        scenes = self._get_scenes(assembly)
        if scenes is None:
            self._md_status.content = (
                f"**Status:** No {SCENES_FILENAME} for assembly '{assembly}'."
            )
            return
        # New (P, N, M) schema: P = insertion parts, N = scenes per part,
        # M = start poses per (part, scene).
        P, N, M = scenes["start_poses"].shape[:3]
        # Insertion part options (4 parts for beam; `insertion_parts` excludes base).
        self._dd_part.options = [f"Part {p}" for p in scenes["insertion_parts"]]
        self._dd_part.value = self._dd_part.options[0]
        self._dd_scene.options = [str(i) for i in range(N)]
        self._dd_scene.value = "0"
        self._dd_start.options = [str(i) for i in range(M)]
        self._dd_start.value = "0"

    def _on_scene_change(self):
        # No-op: scene change doesn't cascade; user re-clicks Load.
        pass

    # -- Static viser scene --------------------------------------------

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
        # ViserUrdf instances own a whole subtree of mesh nodes and must be
        # removed explicitly — otherwise meshes from a previous scene linger
        # under /object and /goal even after their parent frame is gone.
        for vu in (self._obj_viser_urdf, self._goal_viser_urdf):
            if vu is not None:
                try:
                    vu.remove()
                except Exception:
                    pass
        self._obj_viser_urdf = None
        self._goal_viser_urdf = None

        # Keypoints: remove before clearing the list so their scene nodes
        # actually go away (previously we only dropped the Python refs).
        for kp in self._obj_keypoints + self._goal_keypoints:
            try:
                kp.remove()
            except Exception:
                pass
        self._obj_keypoints.clear()
        self._goal_keypoints.clear()

        for h in self._dyn:
            try:
                h.remove()
            except Exception:
                pass
        self._dyn.clear()
        self._obj_frame = self._goal_frame = None

    def _setup_scene_objects(self, assembly: str, scene_idx: int, part_id: str,
                             start_idx: int = 0):
        """Populate viser with the partial-assembly fixture + active inserting part.

        Single-insertion schema: the "fixture" is the chain of parts strictly
        before `part_id` in the full assembly_order, placed at
        partial_assembly_offsets[p_idx, scene_idx]. The receiver (per
        inserts_into) is highlighted. There are no "future parts" — only one
        dynamic inserting part per scene.
        """
        self._clear_dynamic()
        from fabrica.objects import FABRICA_NAME_TO_OBJECT

        scenes = self._get_scenes(assembly)
        assembly_order = scenes["assembly_order"]
        inserts_into = scenes["inserts_into"]
        receiver_pid = inserts_into.get(part_id, "")
        ins_parts = scenes["insertion_parts"]
        p_idx = ins_parts.index(part_id)
        full_part_idx = {pid: i for i, pid in enumerate(assembly_order)}
        active_i = full_part_idx[part_id]

        object_name = f"{assembly}_{part_id}"
        obj_urdf = FABRICA_NAME_TO_OBJECT[object_name].urdf_path

        # Fixture parts (everything strictly before `part_id` in assembly_order),
        # positioned at the scene's partial_assembly_offset via world_assembled_pose.
        from fabrica.benchmark_processing.step3_generate_trajectories import (
            quat_inverse_wxyz,
        )
        ct_path = ASSETS_DIR / assembly / "canonical_transforms.json"
        transforms = json.loads(ct_path.read_text())
        table_offset = scenes["partial_assembly_offsets"][p_idx, scene_idx]
        for fpid in assembly_order[:active_i]:
            centroid = np.array(transforms[fpid]["original_centroid"])
            pos = (centroid + table_offset).tolist()
            a2c = transforms[fpid]["assembled_to_canonical_wxyz"]
            quat_wxyz = quat_inverse_wxyz(a2c)
            # Pack into an xyzw pose (the format _add_context_part expects).
            pose = [pos[0], pos[1], pos[2],
                    quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            opacity = 1.0 if fpid == receiver_pid else 0.85
            self._add_context_part(
                assembly, fpid, full_part_idx,
                pose=pose,
                opacity=opacity,
                frame_name=f"/fixture_{fpid}",
            )

        # Active (inserting) part — live-updated by sim state.
        self._obj_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._obj_frame)
        active_color = COLORS[active_i % len(COLORS)]
        color_override = tuple(int(c * 255) for c in active_color)
        self._obj_viser_urdf = ViserUrdf(
            self.server, obj_urdf, root_node_name="/object",
            mesh_color_override=color_override,
        )

        # Goal ghost — live-updated by sim state.
        self._goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001,
        )
        self._dyn.append(self._goal_frame)
        self._goal_viser_urdf = ViserUrdf(
            self.server, obj_urdf, root_node_name="/goal",
            mesh_color_override=(0, 255, 0, 0.5),
        )

    def _final_goal(self, scenes: dict, scene_idx: int,
                    part_id: str, start_idx: int = 0) -> List[float]:
        p_idx = scenes["insertion_parts"].index(part_id)
        L = int(scenes["traj_lengths"][p_idx, scene_idx, start_idx])
        return scenes["goals"][p_idx, scene_idx, start_idx, L - 1].tolist()

    def _add_context_part(self, assembly: str, pid: str,
                          part_id_to_idx: Dict[str, int],
                          pose: List[float], opacity: float,
                          frame_name: str):
        mesh_path = ASSETS_DIR / assembly / pid / f"{pid}_canonical.obj"
        if not mesh_path.exists():
            return
        mesh = trimesh.load(str(mesh_path), force="mesh")
        frame = self.server.scene.add_frame(
            frame_name,
            position=(pose[0], pose[1], pose[2]),
            wxyz=quat_xyzw_to_wxyz(pose[3:7]),
            show_axes=False,
        )
        self._dyn.append(frame)
        color = COLORS[part_id_to_idx[pid] % len(COLORS)]
        verts = np.array(mesh.vertices, dtype=np.float32)
        h = self.server.scene.add_mesh_simple(
            f"{frame_name}/mesh",
            vertices=verts,
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=color,
            opacity=opacity,
        )
        self._dyn.append(h)

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
                self.server.scene.add_icosphere(
                    f"/obj_kp/{i}", radius=0.005, color=(255, 0, 0))
            )
            self._goal_keypoints.append(
                self.server.scene.add_icosphere(
                    f"/goal_kp/{i}", radius=0.005, color=(0, 255, 0), opacity=0.5)
            )
        self._apply_keypoint_visibility()

    def _apply_keypoint_visibility(self):
        visible = self._cb_keypoints.value
        for kp in self._obj_keypoints + self._goal_keypoints:
            kp.visible = visible

    # -- Subprocess management -----------------------------------------

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

    def _load_env(self, assembly=None, part_id=None, scene_idx=None,
                  start_idx=None):
        assembly = assembly if assembly is not None else self._dd_assembly.value

        if part_id is None:
            part_display = self._dd_part.value
            if not part_display.startswith("Part "):
                self._md_status.content = "**Status:** Pick assembly/scene/part first."
                return
            part_id = part_display.split("Part ")[1]

        if scene_idx is None:
            if self._dd_scene.value == self._PH:
                self._md_status.content = "**Status:** Pick a scene index first."
                return
            scene_idx = int(self._dd_scene.value)

        if start_idx is None:
            if self._dd_start.value == self._PH:
                start_idx = 0
            else:
                start_idx = int(self._dd_start.value)

        scenes = self._get_scenes(assembly)
        if scenes is None:
            self._md_status.content = (
                f"**Status:** No {SCENES_FILENAME} for '{assembly}'."
            )
            return
        P, N, M = scenes["start_poses"].shape[:3]
        if scene_idx < 0 or scene_idx >= N:
            self._md_status.content = (
                f"**Status:** Scene index {scene_idx} out of range [0, {N})."
            )
            return
        if start_idx < 0 or start_idx >= M:
            self._md_status.content = (
                f"**Status:** Start index {start_idx} out of range [0, {M})."
            )
            return
        if part_id not in scenes["insertion_parts"]:
            self._md_status.content = (
                f"**Status:** Part {part_id} is not in insertion_parts "
                f"({scenes['insertion_parts']})."
            )
            return
        p_idx = scenes["insertion_parts"].index(part_id)

        self._kill_subprocess()

        self._pending_assembly = assembly
        self._pending_part = part_id
        self._pending_scene_idx = scene_idx
        self._pending_start_idx = start_idx

        goal_mode = self._dd_goal_mode.value
        label = (
            f"{self._dd_policy.value} | "
            f"{assembly} / scene {scene_idx} / "
            f"Part {part_id} / start {start_idx} / goals: {goal_mode}"
        )
        self._md_status.content = f"**Status:** Loading *{label}* ..."
        self._md_task.content = f"**Task:** {label}"
        self._md_retract.content = "**Retract dist:** --"

        if not self._auto_seq_active:
            self.ep_count = 0
            self.ep_goals.clear()
            self.ep_lengths.clear()
            self._md_stats.content = "**Stats:** No episodes yet"

        self.robot.update_cfg(DEFAULT_DOF_POS)

        # Per-scene fixture URDF (pre-baked by generate_scenes.py, one per (p, n)).
        table_urdf_rel = str(scenes["scene_urdf_paths"][p_idx, scene_idx])

        # The new FabricaEnv loads scenes.npz itself and picks (part, scene, start)
        # via forcePartIdx / forceSceneIdx / forceStartIdx. Pass those through as
        # hydra overrides (and goalMode + assemblyName).
        scene_overrides = {
            **self.extra_overrides,
            "task.env.assemblyName": assembly,
            "task.env.scenesFilename": SCENES_FILENAME,
            "task.env.forcePartIdx": p_idx,
            "task.env.forceSceneIdx": scene_idx,
            "task.env.forceStartIdx": start_idx,
            "task.env.goalMode": _GOAL_MODE_CAMEL[goal_mode],
        }

        # Chain arm state between parts in auto-sequence
        initial_arm_pos = None
        if (self._auto_seq_active
                and self._auto_seq_step_idx > 0
                and self._last_arm_pos is not None):
            initial_arm_pos = self._last_arm_pos

        policy_name = self._dd_policy.value
        config_path, checkpoint_path = self.policies[policy_name]

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=sim_worker,
            args=(child_conn, assembly, part_id,
                  config_path, checkpoint_path, table_urdf_rel,
                  self.final_goal_tolerance, self.collision_method,
                  scene_overrides, self.headless, False, None, initial_arm_pos),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        print(f"[launcher] Spawned subprocess pid={self._proc.pid} "
              f"for {policy_name} on {assembly}/scene{scene_idx}/{part_id}")

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
        self._md_retract.content = "**Retract dist:** --"
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
        if self._auto_seq_active:
            self._auto_seq_active = False
            self._md_status.content = "**Status:** Auto-sequence cancelled."

    def _cmd_run_all(self):
        """Iterate every insertion part (4 for beam) at the selected scene,
        using the selected start_idx throughout."""
        assembly = self._dd_assembly.value
        if assembly == self._PH or self._dd_scene.value == self._PH:
            self._md_status.content = "**Status:** Select assembly and scene first."
            return
        if self._episode_running:
            return

        scenes = self._get_scenes(assembly)
        if scenes is None:
            return

        self._auto_seq_active = True
        self._auto_seq_step_idx = 0
        self._auto_seq_results = []
        self._last_arm_pos = None
        self.ep_count = 0
        self.ep_goals.clear()
        self.ep_lengths.clear()

        scene_idx = int(self._dd_scene.value)
        start_idx = int(self._dd_start.value) if self._dd_start.value != self._PH else 0
        insertion_parts = scenes["insertion_parts"]
        pid = insertion_parts[0]
        self._md_status.content = (
            f"**Status:** Auto-sequence: step 1/{len(insertion_parts)} (Part {pid})"
        )
        self._load_env(assembly=assembly, part_id=pid,
                       scene_idx=scene_idx, start_idx=start_idx)

    # -- Viz updates ----------------------------------------------------

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
            self._setup_scene_objects(
                self._pending_assembly, self._pending_scene_idx, self._pending_part,
                start_idx=getattr(self, "_pending_start_idx", 0),
            )
            if len(init_state) > 3:
                self._setup_keypoints(init_state[3].shape[0])
            self._update_viz(init_state)
            self._env_ready = True

            if self._auto_seq_active:
                scenes = self._get_scenes(self._pending_assembly)
                ins_parts = scenes["insertion_parts"] if scenes else []
                step_num = self._auto_seq_step_idx + 1
                self._md_status.content = (
                    f"**Status:** Auto-seq: running step {step_num}/{len(ins_parts)} "
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
            if len(state) >= 8:
                retract_phase, retract_succeeded, mean_ft_dist = state[5], state[6], state[7]
                if retract_succeeded:
                    phase_str = "SUCCESS"
                elif retract_phase:
                    phase_str = "IN PROGRESS"
                else:
                    phase_str = "pre-retract"
                self._md_retract.content = (
                    f"**Retract dist:** {mean_ft_dist:.3f} m ({phase_str})"
                )

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
            self._md_retract.content = (
                f"**Retract:** {'SUCCESS' if retract_ok else 'FAILED'}"
            )
            self._md_stats.content = (
                f"**Episodes:** {self.ep_count} &nbsp;|&nbsp; "
                f"**Avg Goal:** {avg_g:.1f}% &nbsp;|&nbsp; "
                f"**Avg Time:** {avg_t:.1f}s"
            )
            print(f"[launcher] Episode done: {goal_pct:.0f}% goals in "
                  f"{steps / 60.0:.1f}s{retract_str}")

            if self._auto_seq_active:
                self._auto_seq_results.append(
                    (self._pending_part, goal_pct, steps, retract_ok)
                )
                scenes = self._get_scenes(self._pending_assembly)
                ins_parts = scenes["insertion_parts"] if scenes else []
                self._auto_seq_step_idx += 1

                if self._auto_seq_step_idx < len(ins_parts):
                    next_pid = ins_parts[self._auto_seq_step_idx]
                    step_num = self._auto_seq_step_idx + 1
                    self._md_status.content = (
                        f"**Status:** Auto-seq: loading step "
                        f"{step_num}/{len(ins_parts)} (Part {next_pid})..."
                    )
                    self._load_env(
                        assembly=self._pending_assembly,
                        part_id=next_pid,
                        scene_idx=self._pending_scene_idx,
                        start_idx=getattr(self, "_pending_start_idx", 0),
                    )
                else:
                    self._auto_seq_active = False
                    summary_lines = ["**Auto-sequence complete:**\n"]
                    for pid, gpct, st, r_ok in self._auto_seq_results:
                        r_str = "OK" if r_ok else "FAIL"
                        summary_lines.append(
                            f"- Part {pid}: {gpct:.0f}% goals, "
                            f"{st / 60.0:.1f}s, retract: {r_str}"
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
                        print(f"  Part {pid}: {gpct:.0f}% goals, "
                              f"{st / 60.0:.1f}s, "
                              f"retract: {'OK' if r_ok else 'FAIL'}")
            else:
                self._md_status.content = (
                    f"**Status:** Done -- {steps / 60.0:.1f}s, "
                    f"{goal_pct:.0f}% goals{retract_str}"
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
        print("  |  Fabrica Multi-Init Eval (train/val scenes)      |")
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
    parser = argparse.ArgumentParser(
        description="Fabrica Multi-Init-State Evaluation (interactive viser)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policies-dir", type=str, default=None,
                        help="Directory of policy subfolders, each containing "
                             "config.yaml + model.pth. One subfolder = one "
                             "option in the viser Policy dropdown.")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Single-policy fallback when --policies-dir is "
                             "not used.")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Single-policy fallback when --policies-dir is "
                             "not used.")
    parser.add_argument("--initial-policy", type=str, default=None,
                        help="Name of policy subfolder to pre-select in the "
                             "dropdown (only meaningful with --policies-dir).")
    parser.add_argument("--final-goal-tolerance", type=float, default=None)
    parser.add_argument("--collision", choices=["vhacd", "coacd", "sdf"],
                        default="coacd")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="dense",
                        help="Initial goal-trajectory mode "
                             "(runtime-changeable via the viser dropdown)")
    parser.add_argument("--no-headless", action="store_true",
                        help="Show IsaacGym viewer window")
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"),
                        help="Extra config overrides")
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
        pdir = Path(_resolve(args.policies_dir))
        for sub in sorted(pdir.iterdir()):
            cfg = sub / "config.yaml"
            ckpt = sub / "model.pth"
            if cfg.exists() and ckpt.exists():
                policies[sub.name] = (str(cfg), str(ckpt))
        if not policies:
            raise SystemExit(
                f"No policy subfolders (with config.yaml + model.pth) in {pdir}"
            )
    if args.config_path and args.checkpoint_path:
        name = Path(args.config_path).parent.name or "policy"
        policies[name] = (_resolve(args.config_path), _resolve(args.checkpoint_path))
    if not policies:
        raise SystemExit(
            "Provide --policies-dir or (--config-path and --checkpoint-path)."
        )

    MultiInitAssemblyDemo(
        policies=policies,
        port=args.port,
        final_goal_tolerance=args.final_goal_tolerance,
        collision_method=args.collision,
        extra_overrides=extra_overrides,
        headless=not args.no_headless,
        goal_mode=args.goal_mode,
        initial_policy=args.initial_policy,
    ).run()
