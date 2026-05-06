#!/usr/bin/env python3
"""Step 0 of the SDF+CoACD-hybrid plan: validate that Isaac Gym's URDF
importer accepts a single multi-link asset where one link has an SDF
collision (`<collision><geometry><mesh/></geometry><sdf resolution=.../></collision>`)
and another link has a plain `<box>` collision.

If this works:
  * the asset loads without warnings;
  * `get_asset_rigid_shape_count` returns 2 (one SDF mesh shape + one box);
  * the asset survives 60 steps of simulation while a small box drops on it.

If any of those fail, the per-link mix is unsupported and we need a
two-actor fallback (out of scope for this plan).

Run:
    .venv/bin/python -m peg_in_hole_dynamic.smoke_test_hybrid_urdf
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

# isaacgym must be imported before torch
from isaacgym import gymapi, gymtorch  # noqa: F401

import numpy as np
import torch
import trimesh


URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="hybrid_test">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>

  <link name="sdf_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="cube_a.obj" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="cube_a.obj" scale="1 1 1"/></geometry>
      <sdf resolution="256"/>
    </collision>
  </link>
  <joint name="base_to_sdf" type="fixed">
    <parent link="base_link"/><child link="sdf_link"/>
    <origin xyz="-0.06 0 0" rpy="0 0 0"/>
  </joint>

  <link name="box_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.05 0.05 0.05"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.05 0.05 0.05"/></geometry>
    </collision>
  </link>
  <joint name="base_to_box" type="fixed">
    <parent link="base_link"/><child link="box_link"/>
    <origin xyz="+0.06 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


def main() -> int:
    tmp_dir = Path(tempfile.mkdtemp(prefix="hybrid_smoke_"))
    try:
        # Tiny watertight cube — manifold mesh required for SDF.
        cube = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
        cube.export(str(tmp_dir / "cube_a.obj"))
        urdf_path = tmp_dir / "hybrid_test.urdf"
        urdf_path.write_text(URDF_TEMPLATE)
        print(f"[smoke] wrote test asset under {tmp_dir}")

        # ── Sim setup ──
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True
        sim_params.physx.max_gpu_contact_pairs = 1024 * 1024

        gym = gymapi.acquire_gym()
        sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane_params)

        # ── Load the hybrid asset (fixed base) ──
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        opts.collapse_fixed_joints = True
        hybrid_asset = gym.load_asset(sim, str(tmp_dir), urdf_path.name, opts)

        n_bodies = gym.get_asset_rigid_body_count(hybrid_asset)
        n_shapes = gym.get_asset_rigid_shape_count(hybrid_asset)
        print(f"[smoke] hybrid asset: bodies={n_bodies}  shapes={n_shapes}")
        if n_shapes < 2:
            print(f"[smoke] FAIL: expected ≥2 shapes (1 SDF mesh + 1 box), got {n_shapes}")
            print("[smoke]   the importer probably collapsed or dropped one collision shape.")
            return 1

        # Add a single env, place the asset on the table, drop a small box on top
        env = gym.create_env(
            sim, gymapi.Vec3(-0.5, -0.5, 0.0),
            gymapi.Vec3(+0.5, +0.5, 0.5), 1,
        )

        hybrid_pose = gymapi.Transform()
        hybrid_pose.p = gymapi.Vec3(0.0, 0.0, 0.10)
        hybrid_pose.r = gymapi.Quat(0, 0, 0, 1)
        gym.create_actor(env, hybrid_asset, hybrid_pose, "hybrid", 0, 0)

        # Falling probe box — checks the assets simulate without error
        probe_extents = (0.04, 0.04, 0.04)
        probe_options = gymapi.AssetOptions()
        probe_options.density = 200.0
        probe_asset = gym.create_box(sim, *probe_extents, probe_options)
        probe_pose = gymapi.Transform()
        probe_pose.p = gymapi.Vec3(-0.06, 0.0, 0.50)  # above the SDF link
        probe_pose.r = gymapi.Quat(0, 0, 0, 1)
        gym.create_actor(env, probe_asset, probe_pose, "probe", 0, 0)

        gym.prepare_sim(sim)
        for step in range(60):
            gym.simulate(sim)
            gym.fetch_results(sim, True)

        gym.destroy_sim(sim)
        print("[smoke] PASS: 60 steps simulated. Hybrid SDF + box URDF works.")
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
