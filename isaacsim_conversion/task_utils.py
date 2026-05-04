from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array([w, x, y, z], dtype=np.float64)


@dataclass
class CameraPose:
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    convention: str = "ros"
    mount: str = "world"
    link_name: str | None = None


@dataclass
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    focal_length: float = 24.0
    horizontal_aperture: float = 20.955
    focus_distance: float = 400.0
    clipping_range: tuple[float, float] = (0.1, 100.0)


@dataclass
class TaskSpec:
    robot_urdf: str
    table_urdf: str
    object_urdf: str
    object_name: str
    object_scales: np.ndarray
    start_pose: np.ndarray
    goals: list[np.ndarray]
    success_steps: int
    keypoint_tolerance: float
    camera_pose: CameraPose


REAL_CAMERA_T_W_R = np.eye(4)
REAL_CAMERA_T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0], dtype=np.float64)
REAL_CAMERA_T_R_C = np.array(
    [
        [0.9552763064728893, -0.17920451516639435, 0.2352295050275207, -0.5002050422666431],
        [-0.2889023075483251, -0.3958074425064433, 0.8717063296487887, -1.4385715691360608],
        [-0.06310812138518884, -0.9006787497218348, -0.42987806970668574, 1.0201893282998005],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def default_real_camera_pose() -> CameraPose:
    t_w_c = REAL_CAMERA_T_W_R @ REAL_CAMERA_T_R_C
    quat_xyzw = R.from_matrix(t_w_c[:3, :3]).as_quat()
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)
    return CameraPose(
        pos=tuple(float(x) for x in t_w_c[:3, 3]),
        quat_wxyz=tuple(float(x) for x in quat_wxyz),
        convention="ros",
        mount="world",
    )


def default_real_camera_transform() -> np.ndarray:
    return REAL_CAMERA_T_W_R @ REAL_CAMERA_T_R_C


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_task_spec(
    repo_root: Path,
    task_source: str,
    assembly: str,
    part_id: str,
    collision_method: str,
    object_category: str,
    object_name: str,
    task_name: str,
    teacher_config_path: str | Path,
) -> TaskSpec:
    from dextoolbench.objects import NAME_TO_OBJECT

    robot_urdf = str(
        repo_root / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )

    if task_source == "fabrica":
        table_urdf = str(
            repo_root
            / f"assets/urdf/fabrica/{assembly}/environments/{part_id}/scene_{collision_method}.urdf"
        )
        traj_path = repo_root / f"assets/urdf/fabrica/{assembly}/trajectories/{part_id}/pick_place.json"
        object_name = f"{assembly}_{part_id}_{collision_method}"
        import peg_in_hole_dynamic.fabrica.objects  # noqa: F401
    else:
        table_urdf = str(
            repo_root
            / f"assets/urdf/dextoolbench/environments/{object_category}/{object_name}/{task_name}.urdf"
        )
        traj_path = repo_root / f"dextoolbench/trajectories/{object_category}/{object_name}/{task_name}.json"

    with open(traj_path) as f:
        traj = json.load(f)

    obj_info = NAME_TO_OBJECT[object_name]
    policy_cfg = load_yaml(teacher_config_path)
    if task_source == "fabrica":
        fixed_size = policy_cfg.get("task", {}).get("env", {}).get("fixedSize", [0.141, 0.03025, 0.0271])
        object_scales = np.array([fixed_size], dtype=np.float32)
    else:
        object_scales = np.array([obj_info.scale], dtype=np.float32)

    return TaskSpec(
        robot_urdf=robot_urdf,
        table_urdf=table_urdf,
        object_urdf=str(obj_info.urdf_path),
        object_name=object_name,
        object_scales=object_scales,
        start_pose=np.array(traj["start_pose"], dtype=np.float32),
        goals=[np.array(goal, dtype=np.float32) for goal in traj["goals"]],
        success_steps=10,
        keypoint_tolerance=0.015,
        camera_pose=default_real_camera_pose(),
    )
