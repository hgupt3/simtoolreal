"""
Isaac Sim environment for policy rollout, using Isaac Lab APIs.
Handles URDF→USD conversion, scene setup, physics config, state extraction.

Uses isaaclab.sim.converters.UrdfConverter for URDF import with per-joint PD gains,
and isaaclab.sim.SimulationContext for physics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from isaacgymenvs.utils.observation_action_utils_sharpa import JOINT_NAMES_ISAACGYM




# DOF drive properties from isaacgymenvs/tasks/simtoolreal/utils.py
# These are the exact values used during training in Isaac Gym.
JOINT_STIFFNESSES = {
    # Arm (7 DOF)
    "iiwa14_joint_1": 600, "iiwa14_joint_2": 600, "iiwa14_joint_3": 500,
    "iiwa14_joint_4": 400, "iiwa14_joint_5": 200, "iiwa14_joint_6": 200,
    "iiwa14_joint_7": 200,
    # Hand — thumb (5 DOF)
    "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2,
    "left_thumb_MCP_FE": 4.76, "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
    # Hand — index (4 DOF)
    "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9, "left_index_DIP": 0.9,
    # Hand — middle (4 DOF)
    "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
    # Hand — ring (4 DOF)
    "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
    # Hand — pinky (5 DOF)
    "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76,
    "left_pinky_MCP_AA": 6.62, "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
}

JOINT_DAMPINGS = {
    # Arm
    "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
    # Hand — thumb
    "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    # Hand — index
    "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
    # Hand — middle
    "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
    # Hand — ring
    "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
    # Hand — pinky
    "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}

assert len(JOINT_STIFFNESSES) == 29
assert len(JOINT_DAMPINGS) == 29

# Isaac Lab's UrdfConverter multiplies revolute joint stiffness/damping by pi/180
# (rad→deg conversion). But our values are already in PhysX units (same as Isaac Gym).
# So we pre-compensate by multiplying by 180/pi to cancel out the conversion.
import math
RAD_TO_DEG_COMPENSATION = 180.0 / math.pi
JOINT_STIFFNESSES_COMPENSATED = {k: v * RAD_TO_DEG_COMPENSATION for k, v in JOINT_STIFFNESSES.items()}
JOINT_DAMPINGS_COMPENSATED = {k: v * RAD_TO_DEG_COMPENSATION for k, v in JOINT_DAMPINGS.items()}

# Match the Isaac Gym training config in isaacgymenvs/cfg/task/SimToolReal.yaml:
# sim.dt=1/60, sim.substeps=2. In Isaac Lab, dt is the physics step, so we emulate the
# Gym control step by running 2 physics steps at 1/120 while holding the same action/targets.
CONTROL_DT = 1.0 / 60.0
PHYSICS_SUBSTEPS = 2
PHYSICS_DT = CONTROL_DT / PHYSICS_SUBSTEPS

# Match Isaac Gym asset friction overrides from SimToolReal.
DEFAULT_ASSET_FRICTION = 0.5
FINGERTIP_FRICTION = 1.5
FINGERTIP_LINK_NAMES = (
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
)

# Match the trained default arm pose from Isaac Gym rollout/eval.
DEFAULT_ARM_JOINT_POS = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}
DEFAULT_JOINT_POS = {name: 0.0 for name in JOINT_NAMES_ISAACGYM}
DEFAULT_JOINT_POS.update(DEFAULT_ARM_JOINT_POS)


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert Isaac Sim quaternion (w,x,y,z) to Isaac Gym convention (x,y,z,w)."""
    return quat_wxyz[[1, 2, 3, 0]]


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert Isaac Gym quaternion (x,y,z,w) to Isaac Sim convention (w,x,y,z)."""
    return quat_xyzw[[3, 0, 1, 2]]


def _log(msg: str):
    # Isaac Sim's Kit runtime captures stdout. Use stderr + carb logging to ensure visibility.
    import sys
    sys.stderr.write(f"[IsaacSimEnv] {msg}\n")
    sys.stderr.flush()


class IsaacSimEnv:
    def __init__(
        self,
        robot_urdf: str,
        table_urdf: str,
        object_urdf: str,
        headless: bool = True,
        usd_cache_dir: str | None = None,
        app=None,
    ):
        """Create Isaac Sim environment with robot, table, and object.

        Args:
            robot_urdf: Path to robot URDF.
            table_urdf: Path to table/scene URDF.
            object_urdf: Path to manipulation object URDF.
            headless: Run without display.
            usd_cache_dir: Directory to cache converted USD files. If None, uses /tmp.
            app: Existing SimulationApp instance. If None, creates a new one.
                 IMPORTANT: SimulationApp must be created BEFORE any isaaclab imports.
        """
        if app is None:
            from isaacsim import SimulationApp
            app = SimulationApp({"headless": headless})
        self.app = app
        _log("SimulationApp ready")

        # Isaac Lab imports (must be after SimulationApp/AppLauncher)
        import isaaclab.sim as sim_utils
        from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        class PatchedUrdfConverter(UrdfConverter):
            """Forward make_instanceable into the Isaac Sim URDF importer config.

            Isaac Lab exposes make_instanceable in UrdfConverterCfg, but the current
            UrdfConverter implementation does not pass it through to the importer.
            """

            def _get_urdf_import_config(self):
                import_config = super()._get_urdf_import_config()
                if hasattr(import_config, "set_make_instanceable"):
                    import_config.set_make_instanceable(self.cfg.make_instanceable)
                return import_config

        self._usd_cache_dir = usd_cache_dir or "/tmp/isaaclab_usd_cache"

        # Create simulation context with physics params matching SimToolReal.yaml.
        # Isaac Gym uses sim.dt=1/60 with substeps=2; Isaac Lab's dt is the physics dt.
        sim_cfg = SimulationCfg(
            dt=PHYSICS_DT,
            render_interval=PHYSICS_SUBSTEPS,
            gravity=(0.0, 0.0, -9.81),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=DEFAULT_ASSET_FRICTION,
                dynamic_friction=DEFAULT_ASSET_FRICTION,
                restitution=0.0,
            ),
            physx=PhysxCfg(
                solver_type=1,  # TGS
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.04,
                friction_correlation_distance=0.025,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        _log(
            "SimulationContext created "
            f"(control_dt={CONTROL_DT:.6f}, physics_dt={PHYSICS_DT:.6f}, substeps={PHYSICS_SUBSTEPS})"
        )

        # Convert and spawn assets
        self._robot_usd = self._convert_robot_urdf(robot_urdf, UrdfConverterCfg, PatchedUrdfConverter)
        self._table_usd = self._convert_table_urdf(table_urdf, UrdfConverterCfg, PatchedUrdfConverter)
        self._object_usd = self._convert_object_urdf(object_urdf, UrdfConverterCfg, PatchedUrdfConverter)

        # Spawn into scene
        self._spawn_scene(sim_utils)
        self._apply_physics_material_overrides(sim_utils)

        # Set up articulation (includes sim.reset())
        self._setup_articulations()

        # Validate joint ordering
        self.permutation = self._validate_joint_ordering()

    def _convert_robot_urdf(self, urdf_path, UrdfConverterCfg, UrdfConverter):
        """Convert robot URDF to USD with per-joint PD gains."""
        _log(f"Converting robot URDF: {urdf_path}")
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=f"{self._usd_cache_dir}/robot",
            force_usd_conversion=True,  # Regenerate USD with updated settings
            make_instanceable=False,
            fix_base=True,
            merge_fixed_joints=True,  # Match Isaac Gym's collapse_fixed_joints=True
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=JOINT_STIFFNESSES_COMPENSATED,
                    damping=JOINT_DAMPINGS_COMPENSATED,
                ),
            ),
        )
        converter = UrdfConverter(cfg)
        _log(f"Robot USD: {converter.usd_path}")
        return converter.usd_path

    def _convert_table_urdf(self, urdf_path, UrdfConverterCfg, UrdfConverter):
        """Convert table URDF to USD (static, fixed base)."""
        _log(f"Converting table URDF: {urdf_path}")
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=f"{self._usd_cache_dir}/table",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=True,
            merge_fixed_joints=True,
            joint_drive=None,
        )
        converter = UrdfConverter(cfg)
        _log(f"Table USD: {converter.usd_path}")
        return converter.usd_path

    def _convert_object_urdf(self, urdf_path, UrdfConverterCfg, UrdfConverter):
        """Convert object URDF to USD (dynamic, not fixed)."""
        _log(f"Converting object URDF: {urdf_path}")
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=f"{self._usd_cache_dir}/object",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
        )
        converter = UrdfConverter(cfg)
        _log(f"Object USD: {converter.usd_path}")
        return converter.usd_path

    def _spawn_scene(self, sim_utils):
        """Spawn robot, table, and object into the USD stage."""
        # Add ground plane and lighting for better rendering
        sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)).func(
            "/World/DomeLight", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        )
        # Spawn robot at (0, 0.8, 0) with gravity disabled and self-collision off
        robot_prim_path = "/World/Robot"
        cfg = sim_utils.UsdFileCfg(
            usd_path=self._robot_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        )
        cfg.func(robot_prim_path, cfg, translation=(0.0, 0.8, 0.0))
        self.robot_prim_path = robot_prim_path
        _log(f"Robot spawned (gravity=off, contact_offset=0.002, self_collision=off)")

        # Spawn table at (0, 0, 0.38)
        table_prim_path = "/World/Table"
        cfg = sim_utils.UsdFileCfg(
            usd_path=self._table_usd,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        )
        cfg.func(table_prim_path, cfg, translation=(0.0, 0.0, 0.38))
        self.table_prim_path = table_prim_path
        _log(f"Table spawned at {table_prim_path}")

        # Spawn object (position set later via set_object_pose)
        object_prim_path = "/World/Object"
        cfg = sim_utils.UsdFileCfg(
            usd_path=self._object_usd,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        )
        cfg.func(object_prim_path, cfg)
        self.object_prim_path = object_prim_path
        _log(f"Object spawned at {object_prim_path}")

    def _iter_collision_prim_paths(self, root_prim_path: str) -> list[str]:
        """Return all collision prim paths under a root prim, including instance proxies."""
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            return []

        collision_paths: list[str] = []
        prims_to_visit = [root_prim]
        while prims_to_visit:
            prim = prims_to_visit.pop()
            if prim.GetName() == "collisions":
                collision_paths.append(str(prim.GetPath()))
            prims_to_visit.extend(list(prim.GetChildren()))
        return collision_paths

    def _set_collision_offsets(self, collision_prim_paths: list[str], contact_offset: float, rest_offset: float):
        """Author collision offsets directly onto collider prims."""
        from pxr import PhysxSchema
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        for prim_path in collision_prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not physx_collision_api:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision_api.CreateContactOffsetAttr().Set(contact_offset)
            physx_collision_api.CreateRestOffsetAttr().Set(rest_offset)

    def _apply_material_to_collision_prims(self, material_path: str, collision_prim_paths: list[str]):
        """Bind a physics material to a list of collision prims."""
        from isaaclab.sim.utils import bind_physics_material

        raw_bind_physics_material = getattr(bind_physics_material, "__wrapped__", bind_physics_material)
        for prim_path in collision_prim_paths:
            raw_bind_physics_material(prim_path, material_path)

    def _apply_physics_material_overrides(self, sim_utils):
        """Mirror Isaac Gym friction overrides on robot/table/object assets."""
        robot_collision_paths = self._iter_collision_prim_paths(self.robot_prim_path)
        table_collision_paths = self._iter_collision_prim_paths(self.table_prim_path)
        object_collision_paths = self._iter_collision_prim_paths(self.object_prim_path)

        all_collision_paths = robot_collision_paths + table_collision_paths + object_collision_paths
        self._set_collision_offsets(all_collision_paths, contact_offset=0.002, rest_offset=0.0)

        default_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=DEFAULT_ASSET_FRICTION,
            dynamic_friction=DEFAULT_ASSET_FRICTION,
            restitution=0.0,
        )
        default_material_path = "/World/PhysicsMaterials/default_asset_material"
        default_material_cfg.func(default_material_path, default_material_cfg)
        self._apply_material_to_collision_prims(default_material_path, all_collision_paths)

        fingertip_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=FINGERTIP_FRICTION,
            dynamic_friction=FINGERTIP_FRICTION,
            restitution=0.0,
        )
        fingertip_material_path = "/World/PhysicsMaterials/fingertip_material"
        fingertip_material_cfg.func(fingertip_material_path, fingertip_material_cfg)
        fingertip_collision_paths = [
            f"{self.robot_prim_path}/{link_name}/collisions" for link_name in FINGERTIP_LINK_NAMES
        ]
        self._apply_material_to_collision_prims(fingertip_material_path, fingertip_collision_paths)
        _log(
            "Physics materials applied "
            f"(default_friction={DEFAULT_ASSET_FRICTION}, fingertip_friction={FINGERTIP_FRICTION}, "
            f"colliders={len(all_collision_paths)})"
        )

    def _setup_articulations(self):
        """Set up articulation and rigid object using Isaac Lab APIs."""
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg

        # Robot articulation with implicit PD actuators
        robot_cfg = ArticulationCfg(
            prim_path=self.robot_prim_path,
            spawn=None,  # Already spawned in _spawn_scene
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.8, 0.0),
                joint_pos=DEFAULT_JOINT_POS,
                joint_vel={name: 0.0 for name in JOINT_NAMES_ISAACGYM},
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["iiwa14_joint_.*"],
                    stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("iiwa14")},
                    damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("iiwa14")},
                ),
                "hand": ImplicitActuatorCfg(
                    joint_names_expr=["left_.*"],
                    stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("left")},
                    damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("left")},
                ),
            },
        )
        self.robot = Articulation(cfg=robot_cfg)

        # Object as rigid body
        object_cfg = RigidObjectCfg(
            prim_path=self.object_prim_path,
            spawn=None,  # Already spawned
        )
        self.object_rigid = RigidObject(cfg=object_cfg)

        # Initialize after sim reset
        self.sim.reset()
        _log("Simulation reset for articulation init")

        # Apply the default joint state explicitly so the robot starts in the trained pose
        # instead of spawning at zeros and snapping to the first commanded target.
        self.reset_robot_to_default_pose(render=False)

        num_dofs = self.robot.num_joints
        _log(f"Robot has {num_dofs} DOFs")
        _log(f"Robot joint names: {self.robot.joint_names}")
        assert num_dofs == 29, f"Expected 29 DOFs, got {num_dofs}"

    def _validate_joint_ordering(self) -> Optional[np.ndarray]:
        """Compare Isaac Sim joint names to JOINT_NAMES_ISAACGYM. Return permutation if needed."""
        sim_names = list(self.robot.joint_names)
        _log(f"Joint ordering validation:")
        _log(f"  Isaac Sim:  {sim_names}")
        _log(f"  Isaac Gym:  {JOINT_NAMES_ISAACGYM}")

        if sim_names == JOINT_NAMES_ISAACGYM:
            _log("  -> MATCH! No permutation needed.")
            return None
        else:
            _log("  -> MISMATCH! Computing permutation...")
            permutation = np.array(
                [sim_names.index(name) for name in JOINT_NAMES_ISAACGYM]
            )
            _log(f"  -> Permutation: {permutation}")
            return permutation

    def set_object_pose(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray):
        """Set object world pose. Takes xyzw quaternion (Isaac Gym convention)."""
        import torch
        quat_wxyz = xyzw_to_wxyz(quat_xyzw)
        # Isaac Lab uses (N, 7) tensors in wxyz format
        pose = torch.tensor(
            [list(pos_xyz) + list(quat_wxyz)], dtype=torch.float32, device=self.sim.device
        )
        self.object_rigid.write_root_pose_to_sim(pose)

    def get_robot_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get joint positions and velocities in JOINT_NAMES_ISAACGYM order."""
        q = self.robot.data.joint_pos[0].cpu().numpy()   # (num_joints,)
        qd = self.robot.data.joint_vel[0].cpu().numpy()  # (num_joints,)
        if self.permutation is not None:
            q = q[self.permutation]
            qd = qd[self.permutation]
        return q, qd

    def get_object_pose_xyzw(self) -> np.ndarray:
        """Get object world pose as (7,) array: [x, y, z, qx, qy, qz, qw]."""
        # Isaac Lab root_pos_w is (N, 3), root_quat_w is (N, 4) in wxyz
        pos = self.object_rigid.data.root_pos_w[0].cpu().numpy()
        quat_wxyz = self.object_rigid.data.root_quat_w[0].cpu().numpy()
        quat_xyzw = wxyz_to_xyzw(quat_wxyz)
        return np.concatenate([pos, quat_xyzw])

    def set_joint_position_targets(self, targets: np.ndarray):
        """Set joint position targets. Takes targets in JOINT_NAMES_ISAACGYM order."""
        import torch
        if self.permutation is not None:
            reordered = np.zeros_like(targets)
            for i, p in enumerate(self.permutation):
                reordered[p] = targets[i]
            targets = reordered
        targets_t = torch.tensor(targets, dtype=torch.float32, device=self.sim.device).unsqueeze(0)
        self.robot.set_joint_position_target(targets_t)
        self.robot.write_data_to_sim()

    def reset_robot_to_default_pose(self, render: bool = False):
        """Reset the robot joint state and targets to the trained default pose."""
        import torch

        default_targets = np.array(
            [DEFAULT_JOINT_POS.get(joint_name, 0.0) for joint_name in self.robot.joint_names],
            dtype=np.float32,
        )

        pos_t = torch.tensor(default_targets, dtype=torch.float32, device=self.sim.device).unsqueeze(0)
        vel_t = torch.zeros_like(pos_t)
        self.robot.write_joint_state_to_sim(pos_t, vel_t)
        self.robot.set_joint_position_target(pos_t)
        self.robot.write_data_to_sim()
        self.robot.update(self.sim.get_physics_dt())
        if render:
            self.step(render=True)

    def step(self, render: bool = True):
        for _ in range(PHYSICS_SUBSTEPS):
            self.sim.step(render=render)
        # Update articulation data buffers
        self.robot.update(self.sim.get_physics_dt())
        self.object_rigid.update(self.sim.get_physics_dt())

    def close(self):
        self.sim.stop()
        self.app.close()


def run_phase5_test():
    """Phase 5-6: Scene setup and stability tests."""
    # IMPORTANT: AppLauncher must be created BEFORE any isaaclab/omni imports.
    import argparse
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    _log("AppLauncher created")

    repo_root = Path(__file__).parent.parent

    robot_urdf = str(repo_root / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf")
    table_urdf = str(repo_root / "assets/urdf/fabrica/beam/environments/2/scene_coacd.urdf")

    # Find object URDF (these imports are safe — no omni/isaaclab deps)
    sys.path.insert(0, str(repo_root))
    import peg_in_hole_dynamic.fabrica.objects  # noqa: registers fabrica parts
    from dextoolbench.objects import NAME_TO_OBJECT
    obj_info = NAME_TO_OBJECT.get("beam_2_coacd")
    if obj_info:
        object_urdf = obj_info.urdf_path
    else:
        object_urdf = str(repo_root / "assets/urdf/objects/cube_multicolor.urdf")

    _log(f"Robot URDF:  {robot_urdf}")
    _log(f"Table URDF:  {table_urdf}")
    _log(f"Object URDF: {object_urdf}")

    env = IsaacSimEnv(
        robot_urdf=robot_urdf,
        table_urdf=table_urdf,
        object_urdf=object_urdf,
        headless=True,
        app=app,
    )

    # Place object on the table before tests
    import json
    traj_path = repo_root / "assets/urdf/fabrica/beam/trajectories/2/pick_place.json"
    with open(traj_path) as f:
        traj = json.load(f)
    start_pose = traj["start_pose"]  # [x,y,z,qx,qy,qz,qw]
    env.set_object_pose(
        pos_xyz=np.array(start_pose[:3], dtype=np.float32),
        quat_xyzw=np.array(start_pose[3:7], dtype=np.float32),
    )
    env.step(render=False)  # Apply the pose
    _log(f"Object placed at start pose: {start_pose}")

    # Stability test (Phase 5e)
    _log("\n--- Stability test: robot should hold default pose for 5s ---")
    q_initial = env.get_robot_state()[0].copy()
    for i in range(300):
        env.step(render=False)
    q_final = env.get_robot_state()[0]
    drift = np.abs(q_final - q_initial).max()
    _log(f"Max joint drift after 5s: {drift:.6f} rad")
    _log("PASS: Robot holds pose stably" if drift < 0.01 else f"WARN: Drift {drift:.4f} > 0.01")

    # Object stability test (Phase 6d)
    _log("\n--- Object stability test ---")
    obj_pose = env.get_object_pose_xyzw()
    _log(f"Object pose: {obj_pose}")
    for i in range(120):
        env.step(render=False)
    obj_pose_after = env.get_object_pose_xyzw()
    _log(f"Object pose after 2s: {obj_pose_after}")
    _log(f"PASS: Object stable at z={obj_pose_after[2]:.3f}" if obj_pose_after[2] > 0.3
         else f"FAIL: Object fell! z={obj_pose_after[2]:.3f}")

    # Manual action test (Phase 8d)
    _log("\n--- Manual action test: move arm joint 1 by 0.1 rad ---")
    q0 = env.get_robot_state()[0].copy()
    test_targets = q0.copy()
    test_targets[0] += 0.1
    env.set_joint_position_targets(test_targets)
    for _ in range(60):
        env.step(render=False)
    q_after = env.get_robot_state()[0]
    error = abs(q_after[0] - test_targets[0])
    _log(f"Commanded: {test_targets[0]:.3f}, Achieved: {q_after[0]:.3f}, Error: {error:.4f}")
    _log("PASS: Joint tracking OK" if error < 0.01 else f"WARN: Tracking error {error:.4f} > 0.01")

    env.close()
    _log("\n=== Phase 5-6 tests complete ===")


if __name__ == "__main__":
    run_phase5_test()
