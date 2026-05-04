"""PegInHoleDynamicEnv: SimToolReal subclass with dynamic hole placement.

Loads a bare table (``table_narrow.urdf``) and a separate hole fixture as
independent actors. The hole position is randomized on the table each
episode and goal poses (pre-insert + final) are computed dynamically from
the hole position — no pre-baked ``scenes.npz`` dependency.

Supports co-training: with ``randomGoalFraction > 0``, a fraction of envs
per reset become random-goal episodes (SimToolReal-style free-space reaching)
while the rest do insertion. For random-goal episodes the hole is moved out
of the workspace so it doesn't interfere with the peg.
"""

from __future__ import annotations

import os
import tempfile
from copy import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from isaacgym import gymapi
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.tasks.simtoolreal.utils import populate_dof_properties
from isaacgymenvs.utils.torch_jit_utils import get_axis_params, to_torch, torch_rand_float


VALID_GOAL_MODES = ("preInsertAndFinal", "finalGoalOnly")

TABLE_HALF_HEIGHT = 0.15


class PegInHoleDynamicEnv(SimToolReal):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        # ── Config ──
        self.enable_retract = cfg["env"].get("enableRetract", False)
        self.retract_reward_scale = cfg["env"].get("retractRewardScale", 1.0)
        self.retract_distance_threshold = cfg["env"].get("retractDistanceThreshold", 0.1)
        # retract_success_bonus is derived in code (see post-super() block)
        self.retract_success_tolerance = cfg["env"].get("retractSuccessTolerance", 0.005)

        self.random_goal_fraction = cfg["env"].get("randomGoalFraction", 0.0)
        self.random_goal_max_successes = cfg["env"].get("randomGoalMaxSuccesses", 5)

        self.insertion_success_tolerance = cfg["env"].get("insertionSuccessTolerance", 0.01)
        self.lift_bonus_fade_threshold = cfg["env"].get("liftBonusFadeThreshold", 2.0)
        self.random_goal_curriculum_success_threshold = cfg["env"].get(
            "randomGoalCurriculumSuccessThreshold", 2.0
        )

        self.goal_mode = cfg["env"].get("goalMode", "preInsertAndFinal")
        assert self.goal_mode in VALID_GOAL_MODES, (
            f"goalMode must be one of {VALID_GOAL_MODES}, got {self.goal_mode!r}"
        )
        self._num_insertion_goals = 2 if self.goal_mode == "preInsertAndFinal" else 1

        # Insertion pose relative to hole origin [x, y, z, qx, qy, qz, qw]
        insert_pose_rel = cfg["env"].get(
            "insertPoseRelHole", [0.0, 0.0, 0.136, 0.0, -0.70710678, 0.0, 0.70710678]
        )
        self._insert_pos_rel = torch.tensor(insert_pose_rel[:3], dtype=torch.float32)
        self._insert_quat_xyzw = torch.tensor(insert_pose_rel[3:7], dtype=torch.float32)

        # Approach direction + pre-insert back-off distance
        ins_dir = cfg["env"].get("insertionDirection", [0.0, 0.0, -1.0])
        self._insertion_dir = torch.tensor(ins_dir, dtype=torch.float32)
        self._insertion_dir = self._insertion_dir / self._insertion_dir.norm()
        self.pre_insert_offset = cfg["env"].get("preInsertOffset", 0.05)

        self._hole_urdf = cfg["env"].get("holeUrdf", "urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf")

        hole_x_range = cfg["env"].get("holeXRange", [-0.1875, 0.1875])
        hole_y_range = cfg["env"].get("holeYRange", [-0.1, 0.2])
        self.hole_x_min, self.hole_x_max = float(hole_x_range[0]), float(hole_x_range[1])
        self.hole_y_min, self.hole_y_max = float(hole_y_range[0]), float(hole_y_range[1])
        self.hole_z_offset = float(cfg["env"].get("holeZOffset", 0.0))

        # Parent init placeholders (don't override objectName — respect config/CLI)
        cfg["env"]["useFixedGoalStates"] = True
        cfg["env"]["useFixedInitObjectPose"] = True

        table_z = cfg["env"]["tableResetZ"]
        table_top_z = table_z + TABLE_HALF_HEIGHT
        # Placeholder goal at table center for parent init
        insert_world_z = table_top_z + float(self._insert_pos_rel[2])
        insert_quat = insert_pose_rel[3:7]
        pre_insert_pos = [
            float(self._insert_pos_rel[i]) - float(self._insertion_dir[i]) * self.pre_insert_offset
            for i in range(3)
        ]
        pre_insert_world_z = table_top_z + pre_insert_pos[2]

        cfg["env"]["objectStartPose"] = [0.0, 0.0, table_top_z + 0.1, 0, 0, 0, 1]
        if self.goal_mode == "preInsertAndFinal":
            cfg["env"]["fixedGoalStates"] = [
                [0.0, 0.0, pre_insert_world_z] + list(insert_quat),
                [0.0, 0.0, insert_world_z] + list(insert_quat),
            ]
        else:
            cfg["env"]["fixedGoalStates"] = [
                [0.0, 0.0, insert_world_z] + list(insert_quat),
            ]

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        # ── State tensors ──
        self.hole_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.is_random_goal_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_episode_is_random_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.env_max_goals = torch.full(
            (self.num_envs,), self._num_insertion_goals, dtype=torch.long, device=self.device
        )
        self.prev_episode_env_max_goals = self.env_max_goals.clone()

        # Fixed-at-init allocation: each env is insertion or reaching for the
        # full run. random_goal_fraction governs the per-env Bernoulli.
        if self.random_goal_fraction > 0:
            coin = torch.rand(self.num_envs, device=self.device)
            self.is_random_goal_env[:] = coin < self.random_goal_fraction
            self.env_max_goals[:] = torch.where(
                self.is_random_goal_env,
                torch.full_like(self.env_max_goals, self.random_goal_max_successes),
                torch.full_like(self.env_max_goals, self._num_insertion_goals),
            )
        else:
            self.is_random_goal_env[:] = False
            self.env_max_goals[:] = self._num_insertion_goals
        self.prev_episode_is_random_goal[:] = self.is_random_goal_env
        self.prev_episode_env_max_goals[:] = self.env_max_goals

        # Retract bonus is derived so that a successful insertion+retract
        # episode is worth at least as much as a fully-completed reaching
        # episode (random_goal_max_successes * reach_goal_bonus).
        self.retract_success_bonus = float(
            self.random_goal_max_successes * self.reach_goal_bonus
        )

        # One-way fade gate for the lift bonus on reaching envs (see
        # compute_kuka_reward). Insertion envs never get the lift bonus.
        self.lift_bonus_active = True

        self.retract_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.retract_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.rewards_episode["retract_rew"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        self.goal_pos_obs_noise = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device,
        )

        # Move insertion geometry tensors to device
        self._insert_pos_rel = self._insert_pos_rel.to(self.device)
        self._insert_quat_xyzw = self._insert_quat_xyzw.to(self.device)
        self._insertion_dir = self._insertion_dir.to(self.device)

        # Compute obs_buf slice for keypoints_rel_goal
        n_dof = self.num_hand_arm_dofs
        nk = self.num_keypoints
        nft = self.num_fingertips
        _sz = {
            "joint_pos": n_dof, "joint_vel": n_dof, "prev_action_targets": n_dof,
            "palm_pos": 3, "palm_rot": 4, "palm_vel": 6,
            "object_rot": 4, "object_vel": 6,
            "fingertip_pos_rel_palm": 3 * nft,
            "keypoints_rel_palm": 3 * nk, "keypoints_rel_goal": 3 * nk,
            "object_scales": 3,
        }
        _off = 0
        for _k in self.obs_list:
            if _k == "keypoints_rel_goal":
                break
            _off += _sz[_k]
        self._goal_kp_obs_slice = slice(_off, _off + 3 * nk)

    # ────────────────────────────────────────────────────────────────
    # Env creation
    # ────────────────────────────────────────────────────────────────

    def _create_envs(self, num_envs, spacing, num_per_row):
        if self.should_load_initial_states:
            self.load_initial_states()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        object_asset_root = asset_root
        tmp_assets_dir = tempfile.TemporaryDirectory()
        self.object_asset_files, self.object_asset_scales, self.object_need_vhacds = (
            self._main_object_assets_and_scales(object_asset_root, tmp_assets_dir.name)
        )

        # ── Robot asset ──
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.0 if self.cfg["env"].get("useSDF", False) else 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading robot asset {self.robot_asset_file} from {asset_root}")
        robot_asset = self.gym.load_asset(
            self.sim, asset_root, self.robot_asset_file, asset_options
        )

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(robot_asset)
        assert self.num_hand_arm_dofs == num_hand_arm_dofs

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(robot_dof_props["upper"][i])
        self.arm_hand_dof_lower_limits = to_torch(
            self.arm_hand_dof_lower_limits, device=self.device
        )
        self.arm_hand_dof_upper_limits = to_torch(
            self.arm_hand_dof_upper_limits, device=self.device
        )

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(0.0, 0.8, 0)
        robot_pose.r = gymapi.Quat(0, 0, 0, 1)

        # ── Object (peg) asset ──
        object_assets, object_rb_count, object_shapes_count = self._load_main_object_asset()

        # ── Table asset (single, shared) ──
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.collapse_fixed_joints = True
        if self.cfg["env"].get("useSDF", False):
            table_asset_options.thickness = 0.0

        table_asset = self.gym.load_asset(
            self.sim, asset_root, self.cfg["env"]["asset"]["table"], table_asset_options
        )
        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)

        if self.with_table_force_sensor:
            table_sensor_pose = gymapi.Transform()
            table_sensor_props = gymapi.ForceSensorProperties()
            table_sensor_props.enable_constraint_solver_forces = True
            table_sensor_props.enable_forward_dynamics_forces = False
            table_sensor_props.use_world_frame = True
            self.table_sensor_idx = self.gym.create_asset_force_sensor(
                asset=table_asset, body_idx=0,
                local_pose=table_sensor_pose, props=table_sensor_props,
            )
            if self.table_sensor_idx == -1:
                raise ValueError("Failed to create table force sensor")

        # ── Hole asset (single, shared, repositioned per reset) ──
        hole_asset_options = gymapi.AssetOptions()
        hole_asset_options.disable_gravity = True
        hole_asset_options.fix_base_link = True
        hole_asset_options.collapse_fixed_joints = True
        if self.cfg["env"].get("useSDF", False):
            hole_asset_options.thickness = 0.0

        hole_asset = self.gym.load_asset(
            self.sim, asset_root, self._hole_urdf, hole_asset_options
        )
        hole_rb_count = self.gym.get_asset_rigid_body_count(hole_asset)
        hole_shapes_count = self.gym.get_asset_rigid_shape_count(hole_asset)
        print(
            f"[PegInHoleDynamicEnv] Loaded hole asset {self._hole_urdf} "
            f"({hole_rb_count} bodies, {hole_shapes_count} shapes)"
        )

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = robot_pose.p.x
        table_pose_dy, table_pose_dz = -0.8, self.cfg["env"]["tableResetZ"]
        table_pose.p.y = robot_pose.p.y + table_pose_dy
        table_pose.p.z = robot_pose.p.z + table_pose_dz

        # Initial hole pose: on table surface (+ optional Z offset for centered meshes)
        hole_pose = gymapi.Transform()
        hole_pose.p = gymapi.Vec3(
            table_pose.p.x, table_pose.p.y,
            table_pose.p.z + TABLE_HALF_HEIGHT + self.hole_z_offset,
        )
        hole_pose.r = gymapi.Quat(0, 0, 0, 1)

        # ── Goal (marker) asset ──
        additional_rb, additional_shapes = self._load_additional_assets(
            object_asset_root, robot_pose
        )

        # Aggregate counts (robot + peg + table + hole + goal_marker)
        max_agg_bodies = (
            self.num_hand_arm_bodies + object_rb_count + table_rb_count
            + hole_rb_count + additional_rb
        )
        max_agg_shapes = (
            self.num_hand_arm_shapes + object_shapes_count + table_shapes_count
            + hole_shapes_count + additional_shapes
        )

        # Object start pose placeholder (on table, will be randomized on reset)
        obj_start_pose = gymapi.Transform()
        obj_start_pose.p = gymapi.Vec3(
            0.0, 0.0, table_pose.p.z + TABLE_HALF_HEIGHT + 0.1
        )
        obj_start_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.object_start_pose = obj_start_pose

        # ── Env loop ──
        self.robots = []
        self.envs = []
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robots = []
        self.objects = []

        object_init_state = []
        table_init_state = []
        self.rigid_body_name_to_idx = {}
        self.robot_indices = []
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = []
        object_indices = []
        table_indices = []
        hole_indices = []
        object_scales = []
        object_keypoint_offsets = []
        object_keypoint_offsets_fixed_size = []

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        for name in self.fingertips:
            assert name in body_names
        assert "iiwa14_link_7" in body_names

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.fingertips
        ]

        if self.with_fingertip_force_sensors:
            finger_sensor_pose = gymapi.Transform()
            self.finger_sensor_idxs = [
                self.gym.create_asset_force_sensor(
                    asset=robot_asset, body_idx=ft_handle, local_pose=finger_sensor_pose
                )
                for ft_handle in self.fingertip_handles
            ]

        self.robot_name = "iiwa14"
        self.palm_handle = self.gym.find_asset_rigid_body_index(robot_asset, "iiwa14_link_7")

        self.object_rb_handles = list(
            range(self.num_hand_arm_bodies, self.num_hand_arm_bodies + object_rb_count)
        )
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.object_rb_handles = list(
                range(2 * self.num_hand_arm_bodies, 2 * self.num_hand_arm_bodies + object_rb_count)
            )

        # Asset frictions
        if self.cfg["env"]["modifyAssetFrictions"]:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset,
                friction=self.cfg["env"]["robotFriction"],
                fingertip_friction=self.cfg["env"]["fingerTipFriction"],
            )
            self.set_table_asset_rigid_shape_properties(
                table_asset=table_asset, friction=self.cfg["env"]["tableFriction"],
            )
            self.set_object_asset_rigid_shape_properties(
                object_asset=object_assets[0], friction=self.cfg["env"]["objectFriction"],
            )
        else:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset, friction=None, fingertip_friction=None,
            )

        print(f"[PegInHoleDynamicEnv] Creating {num_envs} envs with "
              f"max_agg_bodies={max_agg_bodies}, max_agg_shapes={max_agg_shapes}")

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 1. Robot
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_pose, "robot", i, -1, 0,
            )
            populate_dof_properties(robot_dof_props, self.num_arm_dofs, self.num_hand_dofs)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            robot_idx = self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM)
            self.robot_indices.append(robot_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, robot_actor):
                self.rigid_body_name_to_idx["robot/" + name] = (
                    self.gym.find_actor_rigid_body_index(env_ptr, robot_actor, name, gymapi.DOMAIN_ENV)
                )

            if self.with_dof_force_sensors:
                self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)

            if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
                blue_robot_actor = self.gym.create_actor(
                    env_ptr, robot_asset, robot_pose, "blue_robot",
                    i + self.num_envs * 2, -1, 0,
                )
                self.gym.set_actor_dof_properties(env_ptr, blue_robot_actor, robot_dof_props)
                self.blue_robots.append(blue_robot_actor)
                self._set_actor_color(env_ptr, blue_robot_actor, (0, 0, 1))
                self.blue_robot_indices.append(
                    self.gym.get_actor_index(env_ptr, blue_robot_actor, gymapi.DOMAIN_SIM)
                )

            # 2. Object (peg)
            object_asset = object_assets[0]
            object_handle = self.gym.create_actor(
                env_ptr, object_asset, obj_start_pose, "object", i, 0, 0,
            )
            object_init_state.append([
                obj_start_pose.p.x, obj_start_pose.p.y, obj_start_pose.p.z,
                obj_start_pose.r.x, obj_start_pose.r.y, obj_start_pose.r.z, obj_start_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            object_indices.append(object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, object_handle):
                self.rigid_body_name_to_idx["object/" + name] = (
                    self.gym.find_actor_rigid_body_index(env_ptr, object_handle, name, gymapi.DOMAIN_ENV)
                )

            object_scale = self.object_asset_scales[0]
            object_scales.append(object_scale)
            object_offsets = []
            for keypoint in self.keypoints_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= (
                        object_scale[coord_idx] * self.object_base_size * self.keypoint_scale / 2
                    )
                object_offsets.append(keypoint)
            object_keypoint_offsets.append(object_offsets)

            object_scale_fixed_size = self.cfg["env"]["fixedSize"]
            object_offsets_fixed_size = []
            for keypoint in self.keypoints_offsets:
                kfs = copy(keypoint)
                for coord_idx in range(3):
                    kfs[coord_idx] *= (
                        object_scale_fixed_size[coord_idx] * self.keypoint_scale / 2
                    )
                object_offsets_fixed_size.append(kfs)
            object_keypoint_offsets_fixed_size.append(object_offsets_fixed_size)

            # 3. Table (shared asset)
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table_object", i, 0, 0,
            )
            table_init_state.append([
                table_pose.p.x, table_pose.p.y, table_pose.p.z,
                table_pose.r.x, table_pose.r.y, table_pose.r.z, table_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            table_indices.append(
                self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            )
            for name in self.gym.get_actor_rigid_body_names(env_ptr, table_handle):
                self.rigid_body_name_to_idx["table/" + name] = (
                    self.gym.find_actor_rigid_body_index(env_ptr, table_handle, name, gymapi.DOMAIN_ENV)
                )

            # 4. Hole (shared asset, repositioned per reset)
            hole_handle = self.gym.create_actor(
                env_ptr, hole_asset, hole_pose, "hole_object", i, 0, 0,
            )
            hole_indices.append(
                self.gym.get_actor_index(env_ptr, hole_handle, gymapi.DOMAIN_SIM)
            )

            # 5. Goal (marker)
            self._create_additional_objects(env_ptr, env_idx=i, object_asset_idx=0)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.objects.append(object_handle)

        # ── Post-env setup ──
        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.table_init_state = to_torch(
            table_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.robot_indices = to_torch(self.robot_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(table_indices, dtype=torch.long, device=self.device)
        self.hole_indices = to_torch(hole_indices, dtype=torch.long, device=self.device)
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = to_torch(self.blue_robot_indices, dtype=torch.long, device=self.device)

        self.object_scales = to_torch(object_scales, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets = to_torch(object_keypoint_offsets, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets_fixed_size = to_torch(
            object_keypoint_offsets_fixed_size, dtype=torch.float, device=self.device
        )

        self.joint_names = self.gym.get_actor_joint_names(env_ptr, robot_actor)
        props = self.gym.get_actor_dof_properties(env_ptr, robot_actor)
        self.joint_lower_limits = props["lower"]
        self.joint_upper_limits = props["upper"]

        if self.random_goal_fraction > 0:
            self.max_consecutive_successes = max(self._num_insertion_goals, self.random_goal_max_successes)
        else:
            self.max_consecutive_successes = self._num_insertion_goals

        self._after_envs_created()

        try:
            tmp_assets_dir.cleanup()
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────
    # Reset
    # ────────────────────────────────────────────────────────────────

    def reset_object_pose(self, env_ids, reset_buf_idxs=None, tensor_reset=True):
        if tensor_reset and len(env_ids) > 0 and reset_buf_idxs is None:
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]
            self.prev_episode_is_random_goal[env_ids] = self.is_random_goal_env[env_ids]

            # Sample hole XY on table
            table_top_z = self.table_init_state[env_ids, 2] + TABLE_HALF_HEIGHT
            hole_x = torch_rand_float(
                self.hole_x_min, self.hole_x_max, (len(env_ids), 1), self.device
            ).squeeze(-1)
            hole_y = torch_rand_float(
                self.hole_y_min, self.hole_y_max, (len(env_ids), 1), self.device
            ).squeeze(-1)
            self.hole_pos[env_ids, 0] = hole_x
            self.hole_pos[env_ids, 1] = hole_y
            self.hole_pos[env_ids, 2] = table_top_z + self.hole_z_offset

            # For random-goal envs: move hole out of workspace
            rg_mask = self.is_random_goal_env[env_ids]
            if rg_mask.any():
                rg_ids = env_ids[rg_mask]
                self.hole_pos[rg_ids, 0] = 0.0
                self.hole_pos[rg_ids, 1] = 0.0
                self.hole_pos[rg_ids, 2] = -1.0

            # Write hole root state
            self.root_state_tensor[self.hole_indices[env_ids], 0:3] = self.hole_pos[env_ids]
            self.root_state_tensor[self.hole_indices[env_ids], 3:7] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=self.device
            )
            self.root_state_tensor[self.hole_indices[env_ids], 7:13] = 0.0
            self.deferred_set_actor_root_state_tensor_indexed(
                [self.hole_indices[env_ids]]
            )

            # Goal obs noise: insertion envs get jitter, random-goal envs get none
            if self.random_goal_fraction > 0:
                ins_ids = env_ids[~rg_mask]
                if len(ins_ids) > 0:
                    self.goal_pos_obs_noise[ins_ids, 0:2] = torch_rand_float(
                        -self.cfg["env"]["goalXyObsNoise"],
                        self.cfg["env"]["goalXyObsNoise"],
                        (len(ins_ids), 2), device=self.device,
                    )
                rg_ids_for_noise = env_ids[rg_mask]
                if len(rg_ids_for_noise) > 0:
                    self.goal_pos_obs_noise[rg_ids_for_noise] = 0.0
            else:
                self.goal_pos_obs_noise[env_ids, 0:2] = torch_rand_float(
                    -self.cfg["env"]["goalXyObsNoise"],
                    self.cfg["env"]["goalXyObsNoise"],
                    (len(env_ids), 2), device=self.device,
                )

        super().reset_object_pose(env_ids, reset_buf_idxs, tensor_reset)

    def _reset_target(self, env_ids, reset_buf_idxs=None, tensor_reset=True, is_first_goal=True):
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            is_rg = self.is_random_goal_env[env_ids]
            ins_ids = env_ids[~is_rg]
            rg_ids = env_ids[is_rg]

            # ── Insertion envs: goal from hole position ──
            if len(ins_ids) > 0:
                subgoal_idx = (self.successes[ins_ids] % self.env_max_goals[ins_ids]).long()
                table_top_z = self.table_init_state[ins_ids, 2] + TABLE_HALF_HEIGHT

                # Final insertion pose = hole_pos + insertPoseRelHole
                final_pos = self.hole_pos[ins_ids] + self._insert_pos_rel

                # Pre-insert pose = final_pos backed off along -insertionDirection
                pre_insert_pos = final_pos - self._insertion_dir * self.pre_insert_offset

                if self.goal_mode == "finalGoalOnly":
                    self.goal_states[ins_ids, 0:3] = final_pos
                else:
                    is_pre = (subgoal_idx == 0).unsqueeze(-1).expand_as(final_pos)
                    self.goal_states[ins_ids, 0:3] = torch.where(is_pre, pre_insert_pos, final_pos)

                self.goal_states[ins_ids, 3:7] = self._insert_quat_xyzw

            # ── Random-goal envs ──
            if len(rg_ids) > 0:
                self._sample_random_goal(rg_ids, is_first_goal)

            # Update goal marker visual
            self.root_state_tensor[self.goal_object_indices[env_ids], 0:7] = (
                self.goal_states[env_ids, 0:7]
            )
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = (
                torch.zeros_like(
                    self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
                )
            )
            self.deferred_set_actor_root_state_tensor_indexed(
                [self.goal_object_indices[env_ids]]
            )

        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            rs_ofs = self.root_state_resets.shape[1]
            self.root_state_tensor[self.goal_object_indices[env_ids], :] = (
                self.root_state_resets[env_ids, rs_ofs:]
            )

    def _sample_random_goal(self, rg_ids, is_first_goal):
        if is_first_goal:
            tv_min = self.target_volume_origin + self.target_volume_extent[:, 0]
            tv_max = self.target_volume_origin + self.target_volume_extent[:, 1]
            tv_size = tv_max - tv_min
            rand_pos = torch_rand_float(0.0, 1.0, (len(rg_ids), 3), device=self.device)
            self.goal_states[rg_ids, 0:3] = tv_min + rand_pos * tv_size
            self.goal_states[rg_ids, 3:7] = self.get_random_quat(rg_ids)
            self._clip_goal_z(rg_ids)
        elif self.goal_sampling_type == "delta":
            self.goal_states[rg_ids, 0:7] = self._sample_delta_goal(
                self.goal_states[rg_ids, 0:7],
                self.delta_goal_distance,
                self.delta_rotation_degrees,
            )
        elif self.goal_sampling_type == "coin_flip":
            coin_flips = torch_rand_float(0.0, 1.0, (len(rg_ids), 1), device=self.device)
            trans_goals = self._sample_delta_goal(
                self.goal_states[rg_ids, 0:7], self.delta_goal_distance, 0.0
            )
            rot_goals = self._sample_delta_goal(
                self.goal_states[rg_ids, 0:7], 0.0, self.delta_rotation_degrees
            )
            self.goal_states[rg_ids, 0:7] = torch.where(
                coin_flips < 0.5, trans_goals, rot_goals,
            )
        else:
            tv_min = self.target_volume_origin + self.target_volume_extent[:, 0]
            tv_max = self.target_volume_origin + self.target_volume_extent[:, 1]
            tv_size = tv_max - tv_min
            rand_pos = torch_rand_float(0.0, 1.0, (len(rg_ids), 3), device=self.device)
            self.goal_states[rg_ids, 0:3] = tv_min + rand_pos * tv_size
            self.goal_states[rg_ids, 3:7] = self.get_random_quat(rg_ids)
            self._clip_goal_z(rg_ids)

    # ────────────────────────────────────────────────────────────────
    # Retract reward + reset logic (from PegInHoleEnv)
    # ────────────────────────────────────────────────────────────────

    def _curriculum_eligible_mask(self):
        # Reaching envs only — insertion uses a fixed tolerance.
        if self.random_goal_fraction <= 0:
            return None
        return self.is_random_goal_env

    def _curriculum_success_threshold(self):
        if self.random_goal_fraction <= 0:
            return None
        return float(self.random_goal_curriculum_success_threshold)

    def _compute_resets(self, is_success):
        if not self.enable_retract:
            return super()._compute_resets(is_success)

        ones = torch.ones_like(self.reset_buf)
        zeros = torch.zeros_like(self.reset_buf)

        object_z_low = torch.where(self.object_pos[:, 2] < 0.1, ones, zeros)

        if self.max_consecutive_successes > 0:
            self.progress_buf = torch.where(
                is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf
            )
            max_consecutive_successes_reached = torch.where(
                self.retract_succeeded, ones, zeros,
            )
            if self.random_goal_fraction > 0:
                random_goals_done = (self.successes >= self.env_max_goals) & self.is_random_goal_env
                max_consecutive_successes_reached = max_consecutive_successes_reached | random_goals_done
        else:
            max_consecutive_successes_reached = zeros

        max_episode_length_reached = torch.where(
            self.progress_buf >= self.max_episode_length - 1, ones, zeros
        )

        if self.with_table_force_sensor:
            table_force_threshold = float(
                self.cfg["env"].get("tableForceResetThreshold", 100.0)
            )
            table_force_too_high = torch.where(
                self.max_table_sensor_force_norm_smoothed > table_force_threshold,
                ones, zeros,
            )
        else:
            table_force_too_high = zeros

        hand_far_from_object = torch.where(
            (self.curr_fingertip_distances.max(dim=-1).values > 1.5) & ~self.retract_phase,
            ones, zeros,
        )

        if self.cfg["env"]["resetWhenDropped"]:
            dropped_z = self.object_init_state[:, 2]
            dropped = (
                torch.where(self.object_pos[:, 2] < dropped_z, ones, zeros) * self.lifted_object
            )
        else:
            dropped = zeros

        resets = (
            self.reset_buf
            | object_z_low
            | max_consecutive_successes_reached
            | max_episode_length_reached
            | table_force_too_high
            | hand_far_from_object
            | dropped
        )
        resets = self._extra_reset_rules(resets)

        self.retract_phase[resets.bool()] = False
        self.retract_succeeded[resets.bool()] = False

        return resets

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        if not self.enable_retract:
            return super().compute_kuka_reward()

        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()

        # Lift bonus policy:
        #   - Insertion envs never receive the lift bonus.
        #   - Reaching envs receive it until mean_rg_prev_ep_successes >= fade
        #     threshold, then it turns off (one-way).
        mean_rg_eps = 0.0
        if self.lift_bonus_active and self.is_random_goal_env.any():
            rg_successes = self.prev_episode_successes[self.is_random_goal_env].float()
            if rg_successes.numel() > 0:
                mean_rg_eps = rg_successes.mean().item()
                if mean_rg_eps >= self.lift_bonus_fade_threshold:
                    self.lift_bonus_active = False
        if self.lift_bonus_active:
            lift_mask = self.is_random_goal_env.float()
        else:
            lift_mask = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device,
            )
        lifting_rew = lifting_rew * lift_mask
        lift_bonus_rew = lift_bonus_rew * lift_mask

        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew, keypoint_rew_fixed_size = self._keypoint_reward(lifted_object)
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            keypoint_rew = keypoint_rew_fixed_size

        # Insertion envs always evaluate against insertion_success_tolerance
        # (default 0.01 m). Reaching envs follow the curriculum-driven
        # success_tolerance.
        fixed_ins_tol = self.insertion_success_tolerance * self.keypoint_scale
        curriculum_tol = self.success_tolerance * self.keypoint_scale
        keypoint_success_tolerance = torch.where(
            self.is_random_goal_env,
            torch.full_like(self.keypoints_max_dist, curriculum_tol),
            torch.full_like(self.keypoints_max_dist, fixed_ins_tol),
        )

        near_goal = self.keypoints_max_dist <= keypoint_success_tolerance
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            near_goal = self.keypoints_max_dist_fixed_size <= keypoint_success_tolerance

        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= self.success_steps
        is_success[self.retract_phase] = False
        goal_resets = is_success.clone()
        self.successes += is_success

        just_entered_retract = (self.successes >= self.env_max_goals) & ~self.retract_phase & ~self.is_random_goal_env
        self.retract_phase |= just_entered_retract
        self.successes.clamp_(max=self.max_consecutive_successes)

        goal_resets[self.retract_phase] = False
        self.reset_goal_buf[:] = goal_resets

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_linvel), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_angvel), dim=-1)

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew
        self.rewards_episode["raw_object_lin_vel_penalty"] += object_lin_vel_penalty
        self.rewards_episode["raw_object_ang_vel_penalty"] += object_ang_vel_penalty

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale
        object_lin_vel_penalty *= self.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            bonus_rew = is_success * self.reach_goal_bonus
        bonus_rew[self.retract_phase & ~just_entered_retract] = 0.0

        base_reward = (
            fingertip_delta_rew
            + hand_delta_penalty
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + kuka_actions_penalty
            + hand_actions_penalty
            + bonus_rew
            + object_lin_vel_penalty
            + object_ang_vel_penalty
        )

        retract_rew = torch.zeros_like(base_reward)
        if self.retract_phase.any():
            retract_tol = self.retract_success_tolerance * self.keypoint_scale
            object_at_goal = (self.keypoints_max_dist <= retract_tol).float()
            mean_fingertip_dist = self.curr_fingertip_distances.mean(dim=-1)
            retract_dist_rew = mean_fingertip_dist * self.retract_reward_scale * object_at_goal
            just_retracted = (
                (mean_fingertip_dist > self.retract_distance_threshold)
                & self.retract_phase
                & ~self.retract_succeeded
                & object_at_goal.bool()
            )
            self.retract_succeeded |= just_retracted
            retract_bonus = just_retracted.float() * self.retract_success_bonus
            retract_rew = (retract_dist_rew + retract_bonus) * self.retract_phase.float()

        already_in_retract = self.retract_phase & ~just_entered_retract
        retract_env_reward = kuka_actions_penalty + hand_actions_penalty + retract_rew
        reward = torch.where(already_in_retract, retract_env_reward, base_reward)

        self.rew_buf[:] = reward

        self.extras["retract_phase_ratio"] = self.retract_phase.float().mean().item()
        self.extras["retract_success_ratio"] = self.retract_succeeded.float().mean().item()
        self.extras["retract_success_tolerance"] = self.retract_success_tolerance
        self.extras["lift_bonus_active"] = float(self.lift_bonus_active)
        self.extras["mean_rg_prev_ep_successes"] = mean_rg_eps

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        if self.cfg["env"]["forceNoReset"]:
            self.reset_buf[:] = False

        self.extras["successes"] = self.prev_episode_successes
        self.extras["success_ratio"] = (
            self.prev_episode_successes / self.prev_episode_env_max_goals.float()
        ).mean().item()
        self.extras["all_goals_hit_ratio"] = (
            self.prev_episode_successes >= self.prev_episode_env_max_goals
        ).float().mean().item()
        self.extras["closest_keypoint_max_dist"] = self.prev_episode_closest_keypoint_max_dist
        self.extras["final_goal_tolerance"] = self.final_goal_success_tolerance
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        if self.random_goal_fraction > 0:
            ins_mask = ~self.prev_episode_is_random_goal
            rg_mask = self.prev_episode_is_random_goal
            prev_s = self.prev_episode_successes
            prev_mg = self.prev_episode_env_max_goals.float()
            if ins_mask.any():
                self.extras["insertion_success_ratio"] = (prev_s[ins_mask] / prev_mg[ins_mask]).mean().item()
                self.extras["insertion_all_goals_hit_ratio"] = (prev_s[ins_mask] >= self.prev_episode_env_max_goals[ins_mask]).float().mean().item()
            else:
                self.extras["insertion_success_ratio"] = 0.0
                self.extras["insertion_all_goals_hit_ratio"] = 0.0
            if rg_mask.any():
                self.extras["random_goal_success_ratio"] = (prev_s[rg_mask] / prev_mg[rg_mask]).mean().item()
                self.extras["random_goal_all_goals_hit_ratio"] = (prev_s[rg_mask] >= self.prev_episode_env_max_goals[rg_mask]).float().mean().item()
            else:
                self.extras["random_goal_success_ratio"] = 0.0
                self.extras["random_goal_all_goals_hit_ratio"] = 0.0
            self.extras["random_goal_frac"] = self.is_random_goal_env.float().mean().item()

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (hand_actions_penalty, "hand_actions_penalty"),
            (bonus_rew, "bonus_rew"),
            (object_lin_vel_penalty, "object_lin_vel_penalty"),
            (object_ang_vel_penalty, "object_ang_vel_penalty"),
            (retract_rew, "retract_rew"),
            (reward, "total_reward"),
        ]

        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        return self.rew_buf, is_success

    # ────────────────────────────────────────────────────────────────
    # Actor-only goal XY noise
    # ────────────────────────────────────────────────────────────────

    def populate_obs_and_states_buffers(self):
        super().populate_obs_and_states_buffers()
        self.obs_buf[:, self._goal_kp_obs_slice].view(
            self.num_envs, self.num_keypoints, 3
        ).sub_(self.goal_pos_obs_noise.unsqueeze(1))
