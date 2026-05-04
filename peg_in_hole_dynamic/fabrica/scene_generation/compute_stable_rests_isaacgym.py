#!/usr/bin/env python3
"""Validate the 6 axis-aligned face-down resting orientations in Isaac Gym.

For cuboidal-ish parts (which all the fabrica beam parts are), the only
plausible stable resting orientations are the 6 ways to put one of the
canonical-frame axes pointing up: ``±x``, ``±y``, ``±z``. Random SO(3)
sampling never finds anything off this set in practice, so we just test
each one directly.

For each part, this script:
  1. Computes the 6 axis-aligned rotations.
  2. For each, places the part with that orientation just above the ground
     and lets gravity + contact dynamics settle it.
  3. Records the rest as "stable" iff the part-local "down" direction at
     settle matches the initial direction within a small angle tolerance
     (i.e., the part did NOT flip onto a different face — yaw drift is OK
     because yaw is randomized downstream by ``generate_scenes.py``).

Output format matches ``compute_stable_rests.py``:
  ``assets/urdf/fabrica/{assembly}/stable_rests/{part_id}.npz`` with keys
  - ``transforms``    : (N, 4, 4) float64
  - ``probabilities`` : (N,)      float64

Each transform places the canonical mesh on z=0 in the i-th rest.
Probabilities are uniform over the stable orientations (the per-rest
landing prior is supplied by the downstream sampler, not measured here).

Usage:
    python fabrica/scene_generation/compute_stable_rests_isaacgym.py --assembly beam
    python fabrica/scene_generation/compute_stable_rests_isaacgym.py --assembly beam --part 2
    python fabrica/scene_generation/compute_stable_rests_isaacgym.py --assembly beam --force --viewer
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch  # noqa: F401  (import side effects)
import torch  # noqa: E402,F401  (required after gymapi import)
import trimesh  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

# Drop / settle parameters
DROP_CLEARANCE = 0.005  # m, gap between part bottom and ground at start
SETTLE_STEPS = 360      # ~6 s @ dt=1/60
STABLE_ANGLE_DEG = 8.0  # max angle (part-local down init vs final) for "stable"


def load_assembly_parts(assembly):
    with open(ASSETS_DIR / assembly / "assembly_order.json") as f:
        return json.load(f)["steps"]


# --- The 6 axis-aligned rotations ---

def six_axis_rotations():
    """Return list of (label, R) for the 6 face-down orientations.

    Each R is a rotation matrix that maps the named part-local axis to
    world +z. Applying R to the canonical mesh thus orients the part with
    that face up; the opposite face touches the ground.
    """
    return [
        ("+z up",  np.eye(3, dtype=np.float64)),
        ("-z up",  np.array([[1, 0,  0],
                             [0, -1, 0],
                             [0, 0, -1]], dtype=np.float64)),
        ("+x up",  np.array([[0, 0, -1],
                             [0, 1,  0],
                             [1, 0,  0]], dtype=np.float64)),
        ("-x up",  np.array([[0, 0,  1],
                             [0, 1,  0],
                             [-1, 0, 0]], dtype=np.float64)),
        ("+y up",  np.array([[1, 0,  0],
                             [0, 0, -1],
                             [0, 1,  0]], dtype=np.float64)),
        ("-y up",  np.array([[1, 0,  0],
                             [0, 0,  1],
                             [0, -1, 0]], dtype=np.float64)),
    ]


def rotated_z_min(mesh_vertices, rot_matrix):
    """Apply rot_matrix to mesh vertices and return the minimum z value."""
    return float((mesh_vertices @ rot_matrix.T)[:, 2].min())


def rotmat_to_quat_xyzw(rot_matrix):
    return R.from_matrix(rot_matrix).as_quat()  # xyzw


def quat_xyzw_to_rotmat(q):
    return R.from_quat(q).as_matrix()


# --- Isaac Gym setup ---

def make_sim(headless, device_id):
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.solver_type = 1
    sim_params.physx.contact_offset = 0.002
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.2
    sim_params.physx.max_depenetration_velocity = 1.0

    graphics_device = device_id if not headless else -1
    sim = gym.create_sim(device_id, graphics_device, gymapi.SIM_PHYSX, sim_params)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 0.7
    plane_params.dynamic_friction = 0.7
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)

    return gym, sim


def test_axis_aligned_part(assembly, part_id, headless, device_id):
    """Drop the part in each of the 6 axis-aligned orientations and return
    the list of (label, R, z_lift) for the orientations that were stable."""

    # Load the canonical mesh once to compute rotated bbox z-mins.
    canonical_path = ASSETS_DIR / assembly / part_id / f"{part_id}_canonical.obj"
    mesh = trimesh.load_mesh(str(canonical_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    rests = six_axis_rotations()
    z_mins = [rotated_z_min(verts, R_i) for _, R_i in rests]
    # link-origin z when bottom touches z=0
    z_lifts = [-zm for zm in z_mins]

    gym, sim = make_sim(headless, device_id)

    asset_root = str(ASSETS_DIR / assembly / part_id / "coacd")
    asset_file = f"{assembly}_{part_id}_coacd.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.disable_gravity = False
    asset_options.fix_base_link = False
    asset_options.thickness = 0.0
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    spacing = 0.30
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing * 2)

    for i, ((label, R_i), z_lift) in enumerate(zip(rests, z_lifts)):
        env = gym.create_env(sim, env_lower, env_upper, len(rests))
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, z_lift + DROP_CLEARANCE)
        q = rotmat_to_quat_xyzw(R_i)
        pose.r = gymapi.Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        h = gym.create_actor(env, asset, pose, f"part_{i}", i, 0, 0)
        props = gym.get_actor_rigid_shape_properties(env, h)
        for p in props:
            p.friction = 0.7
            p.rolling_friction = 0.001
            p.torsion_friction = 0.001
            p.restitution = 0.0
        gym.set_actor_rigid_shape_properties(env, h, props)

    viewer = None
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    gym.prepare_sim(sim)

    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state_tensor)  # [N, 13]

    for _ in range(SETTLE_STEPS):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    gym.refresh_actor_root_state_tensor(sim)
    final_states = root_states.cpu().numpy()

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    cos_thresh = float(np.cos(np.deg2rad(STABLE_ANGLE_DEG)))
    stable = []
    for i, ((label, R_init), z_lift) in enumerate(zip(rests, z_lifts)):
        final_quat_xyzw = final_states[i, 3:7]
        R_final = quat_xyzw_to_rotmat(final_quat_xyzw)
        # part-local down direction is yaw-invariant: -row 2 of R
        d_init = -R_init[2, :]
        d_final = -R_final[2, :]
        cos_a = float(np.dot(d_init, d_final))
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        is_stable = cos_a > cos_thresh
        marker = "STABLE  " if is_stable else "tipped  "
        print(
            f"    {marker}{label:8s}  angle={angle_deg:5.1f}°  "
            f"z_lift={z_lift:+.4f}"
        )
        if is_stable:
            stable.append((label, R_init, z_lift))

    return stable


# --- Output ---

def save_rests(assembly, part_id, stable):
    out_dir = ASSETS_DIR / assembly / "stable_rests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{part_id}.npz"

    n = len(stable)
    transforms = np.zeros((n, 4, 4), dtype=np.float64)
    probabilities = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    for i, (_, R_i, z_lift) in enumerate(stable):
        T = np.eye(4)
        T[:3, :3] = R_i
        T[:3, 3] = [0.0, 0.0, z_lift]
        transforms[i] = T

    np.savez(out_path, transforms=transforms, probabilities=probabilities)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Test the 6 axis-aligned face-down rests in Isaac Gym."
    )
    parser.add_argument("--assembly", type=str, required=True)
    parser.add_argument(
        "--part", type=str, default=None,
        help="Single part id; defaults to all parts in assembly_order.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--viewer", action="store_true",
        help="Show the Isaac Gym viewer (slow; for visual debugging).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing stable_rests/{part}.npz files.",
    )
    args = parser.parse_args()

    if args.part is not None:
        parts = [args.part]
    else:
        parts = load_assembly_parts(args.assembly)

    print(f"Assembly: {args.assembly}")
    print(f"Parts: {parts}\n")

    for pid in parts:
        out_path = ASSETS_DIR / args.assembly / "stable_rests" / f"{pid}.npz"
        if out_path.exists() and not args.force:
            print(f"Part {pid}: skipping (exists, use --force)")
            continue

        print(f"Part {pid}:")
        stable = test_axis_aligned_part(
            args.assembly, pid,
            headless=not args.viewer, device_id=args.device_id,
        )
        if not stable:
            print(f"  WARNING part {pid}: 0 stable rests found")
            continue

        saved = save_rests(args.assembly, pid, stable)
        print(
            f"  → {len(stable)} stable rests, saved to "
            f"{saved.relative_to(REPO_ROOT)}\n"
        )


if __name__ == "__main__":
    main()
