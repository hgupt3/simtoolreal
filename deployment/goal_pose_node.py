#!/usr/bin/env python
import json
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import rospy
import tyro
from geometry_msgs.msg import Pose, PoseStamped
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    _compute_keypoint_positions,
)
from isaacgymenvs.utils.utils import get_repo_root_dir


def info(message: str):
    print(colored(message, "green"))


def warn(message: str):
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float, key=None):
    """
    Print a warning message at most once every n_seconds per unique key.
    Stores state inside the function itself (no globals).
    """
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}  # create on first call

    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0)

    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


def signed_angle_projected_new_z_to_world_x_deg(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q

    new_z_x = 2 * (x * z + y * w)
    new_z_y = 2 * (y * z - x * w)

    if np.hypot(new_z_x, new_z_y) < 1e-8:
        raise ValueError("Projected new z axis is near zero, angle undefined")

    return float(np.rad2deg(np.arctan2(new_z_y, new_z_x)))

def keypoint_distance(
    pose1_xyzw: np.ndarray, pose2_xyzw: np.ndarray, object_scales: np.ndarray
) -> float:
    """Compute the distance between two keypoints."""
    object_keypoint_positions = _compute_keypoint_positions(
        pose=pose1_xyzw[None], scales=object_scales[None]
    )
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=pose2_xyzw[None], scales=object_scales[None]
    )
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    N_KEYPOINTS = 4
    N = 1
    assert keypoints_rel_goal.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_goal.shape: {keypoints_rel_goal.shape}, expected: (N, N_KEYPOINTS, 3)"
    )
    keypoint_distances_l2 = np.linalg.norm(keypoints_rel_goal, axis=-1).max(axis=-1)
    return keypoint_distances_l2


class GoalPoseNode:
    def __init__(
        self,
        goal_poses_robot_frame: np.ndarray,  # Assumes xyzw quat convention and robot frame
        object_scales: np.ndarray,
        success_threshold: float,
        success_steps: int,
        force_open_loop: bool = False,
        force_fixed_orientation: bool = False,
    ):
        # ROS setup
        rospy.init_node("goal_pose_node")

        KEYPOINT_SCALE = 1.5
        self.object_scales = object_scales
        self.success_threshold = success_threshold
        self.keypoint_success_threshold = success_threshold * KEYPOINT_SCALE
        self.success_steps = success_steps
        self.force_fixed_orientation = force_fixed_orientation
        self.current_success_steps = 0

        # Goal object pose
        self.goal_object_poses = goal_poses_robot_frame
        N = len(self.goal_object_poses)
        assert self.goal_object_poses.shape == (N, 7), (
            f"goal_object_poses.shape: {self.goal_object_poses.shape}, expected: (N, 7)"
        )

        # State
        self.current_goal_object_pose_index = 0

        # ROS msgs
        self.latest_current_object_pose = None

        # Publisher and subscriber
        self.goal_object_pose_pub = rospy.Publisher(
            "/robot_frame/goal_object_pose", Pose, queue_size=1
        )
        self.current_object_pose_sub = rospy.Subscriber(
            "/robot_frame/current_object_pose",
            PoseStamped,
            self.current_object_pose_callback,
            queue_size=1,
        )

        # Set control rate to 60Hz
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

        # Force open loop mode, i.e., do not use the current object pose to update the goal pose, but update it at a fixed rate.
        if force_open_loop:
            self.latest_current_object_pose = Pose()
            self.success_threshold = 10.0
            self.keypoint_success_threshold = self.success_threshold * KEYPOINT_SCALE
            self.success_steps = 30

    def current_object_pose_callback(self, msg: PoseStamped):
        """Callback to update the current object pose."""
        self.latest_current_object_pose = msg.pose

    def update_goal_object_pose(self):
        """Update the goal object pose."""
        num_goals = self.goal_object_poses.shape[0]
        if self.current_goal_object_pose_index >= num_goals:
            print(colored("Reached end of goal object poses", "blue"))
            print(
                colored(
                    f"self.current_goal_object_pose_index/num_goals: {self.current_goal_object_pose_index}/{num_goals} = {self.current_goal_object_pose_index / num_goals:.2%}",
                    "blue",
                )
            )
            return

        latest_current_object_pose = deepcopy(self.latest_current_object_pose)
        p = latest_current_object_pose

        current_object_pose_xyzw = np.array(
            [
                p.position.x,
                p.position.y,
                p.position.z,
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ]
        )
        current_goal_object_pose_xyzw = self.goal_object_poses[
            self.current_goal_object_pose_index
        ]

        if self.force_fixed_orientation:
            # Overwrite with fixed orientation
            current_object_pose_xyzw = np.copy(current_object_pose_xyzw)
            current_goal_object_pose_xyzw = np.copy(current_goal_object_pose_xyzw)
            current_object_pose_xyzw[3:7] = np.array([0, 0, 0, 1])
            current_goal_object_pose_xyzw[3:7] = np.array([0, 0, 0, 1])

        distance = keypoint_distance(
            pose1_xyzw=current_object_pose_xyzw,
            pose2_xyzw=current_goal_object_pose_xyzw,
            object_scales=self.object_scales,
        )
        num_goals = self.goal_object_poses.shape[0]
        print(
            f"Distance: {distance}, self.current_goal_object_pose_index/num_goals: {self.current_goal_object_pose_index}/{num_goals} = {self.current_goal_object_pose_index / num_goals:.2%}"
        )
        print(f"current_object_pose_xyzw[3:] = {current_object_pose_xyzw[3:]}")
        # angle = signed_angle_projected_new_z_to_world_x_deg(current_object_pose_xyzw[3:])
        # print(f"angle = {angle}")

        if distance < self.keypoint_success_threshold:
            self.current_success_steps += 1
            if self.current_success_steps >= self.success_steps:
                info(
                    f"Success threshold reached, updating goal object pose index to {self.current_goal_object_pose_index + 1}"
                )
                self.current_success_steps = 0
                self.current_goal_object_pose_index += 1
                # if self.current_goal_object_pose_index >= self.goal_object_poses.shape[0]:
                #     self.current_goal_object_pose_index = self.goal_object_poses.shape[0] - 1
            else:
                info(
                    f"Success threshold reached, at {self.current_success_steps} of {self.success_steps} steps"
                )

    def publish_goal_object_pose(self):
        """Publish the goal object pose."""
        idx = self.current_goal_object_pose_index
        if idx >= self.goal_object_poses.shape[0]:
            idx = self.goal_object_poses.shape[0] - 1
        elif idx < 0:
            idx = 0

        current_goal_object_pose_xyzw = self.goal_object_poses[idx]
        goal_object_pose_msg = Pose()
        goal_object_pose_msg.position.x = current_goal_object_pose_xyzw[0]
        goal_object_pose_msg.position.y = current_goal_object_pose_xyzw[1]
        goal_object_pose_msg.position.z = current_goal_object_pose_xyzw[2]
        goal_object_pose_msg.orientation.x = current_goal_object_pose_xyzw[3]
        goal_object_pose_msg.orientation.y = current_goal_object_pose_xyzw[4]
        goal_object_pose_msg.orientation.z = current_goal_object_pose_xyzw[5]
        goal_object_pose_msg.orientation.w = current_goal_object_pose_xyzw[6]

        self.goal_object_pose_pub.publish(goal_object_pose_msg)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""

        # Wait for the current object pose to be received
        while not rospy.is_shutdown():
            if self.latest_current_object_pose is None:
                warn_every("Waiting for current object pose", n_seconds=1.0)
                time.sleep(0.1)
            else:
                info("Current object pose received, starting goal pose node")
                break  # All messages received, exit loop

        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the goal object pose
            self.update_goal_object_pose()

            # Publish the goal object pose
            self.publish_goal_object_pose()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            loop_no_sleep_dt = (before_sleep_time - start_time).to_sec()
            loop_no_sleep_dts.append(loop_no_sleep_dt)
            loop_dt = (after_sleep_time - start_time).to_sec()
            loop_dts.append(loop_dt)

            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.dt)
            if len(loop_dts) == PRINT_FPS_EVERY_N_STEPS:
                loop_dt_array = np.array(loop_dts)
                loop_no_sleep_dt_array = np.array(loop_no_sleep_dts)
                fps_array = 1.0 / loop_dt_array
                fps_no_sleep_array = 1.0 / loop_no_sleep_dt_array
                print("FPS with sleep:")
                print(f"  Mean: {np.mean(fps_array):.1f}")
                print(f"  Median: {np.median(fps_array):.1f}")
                print(f"  Max: {np.max(fps_array):.1f}")
                print(f"  Min: {np.min(fps_array):.1f}")
                print(f"  Std: {np.std(fps_array):.1f}")
                print("FPS without sleep:")
                print(f"  Mean: {np.mean(fps_no_sleep_array):.1f}")
                print(f"  Median: {np.median(fps_no_sleep_array):.1f}")
                print(f"  Max: {np.max(fps_no_sleep_array):.1f}")
                print(f"  Min: {np.min(fps_no_sleep_array):.1f}")
                print(f"  Std: {np.std(fps_no_sleep_array):.1f}")
                print()
                loop_no_sleep_dts, loop_dts = [], []


@dataclass
class GoalPoseNodeArgs:
    object_category: str = "hammer"
    object_name: str = "claw_hammer"
    task_name: str = "swing_down"

    success_threshold: float = 0.02
    """Success threshold in meters."""

    success_steps: int = 1
    """Number of steps to consider a success."""

    force_open_loop: bool = False
    """Force open loop mode, i.e., do not use the current object pose to update the goal pose, but update it at a fixed rate."""

    force_fixed_orientation: bool = False
    """Force fixed orientation mode, i.e., overwrite the orientation with a fixed one."""


def main():
    args: GoalPoseNodeArgs = tyro.cli(GoalPoseNodeArgs)

    # Load trajectory
    trajectory_path = (
        get_repo_root_dir()
        / "dextoolbench/trajectories"
        / args.object_category
        / args.object_name
        / f"{args.task_name}.json"
    )
    if not trajectory_path.exists():
        print(f"Trajectory file not found: {trajectory_path}, trying Peg-in-Hole trajectory")
        trajectory_path = (
            get_repo_root_dir()
            / "assets/urdf/peg_in_hole/holes/hole_tol10mm/trajectories/peg/pick_place.json"
            # / "assets/urdf/fabrica"
            # / "assets/urdf/"
            # / args.object_category
            # / "trajectories"
            # / args.object_name
            # / f"{args.task_name}.json"
        )
        print(f"Peg-in-Hole trajectory path: {trajectory_path}")
        assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    # Account for robot to world frame
    goal_poses_world_frame = traj_data["goals"]
    goal_poses_robot_frame = [
        [x, y - 0.8, z, qx, qy, qz, qw]
        for x, y, z, qx, qy, qz, qw in goal_poses_world_frame
    ]

    ONLY_LAST_TWO_GOALS = True
    if ONLY_LAST_TWO_GOALS:
        goal_poses_robot_frame = goal_poses_robot_frame[-2:]
        print(f"Lowering z")
        goal_poses_robot_frame = [
            # [x, y, z-0.005, qx, qy, qz, qw]
            # [x, y, z-0.0025, qx, qy, qz, qw]
            # [x, y, z-0.0075, qx, qy, qz, qw]
            # [x, y, z-0.01, qx, qy, qz, qw]
            # [x, y+0.002, z-0.01, qx, qy, qz, qw]
            # [x+0.005, y+0.005, z-0.015, qx, qy, qz, qw]
            # [x+0.0025, y+0.0025, z-0.015, qx, qy, qz, qw]
            # [x+0.0025, y+0.005, z-0.015, qx, qy, qz, qw]
            # [x+0.005, y+0.005, z-0.015, qx, qy, qz, qw]
            # [x+0.005, y+0.005, z-0.0175, qx, qy, qz, qw]
            # [x+0.005, y+0.005, z-0.015, qx, qy, qz, qw]
            # [x+0.005, y+0.005, z, qx, qy, qz, qw]
            [x+0.01, y+0.01, z-0.015, qx, qy, qz, qw]
            for x, y, z, qx, qy, qz, qw in goal_poses_robot_frame
        ]
        # goal_poses_robot_frame[0][2] -= 0.005
        # goal_poses_robot_frame[1][2] += 0.005

    ONLY_LAST_GOAL = False
    if ONLY_LAST_GOAL:
        goal_poses_robot_frame = [goal_poses_robot_frame[-1]]

    OVERWRITE_WITH_FIXED_GOALS = False
    if OVERWRITE_WITH_FIXED_GOALS:
        # Hardcoded goal poses for beam_2x_2
        DZ = 0.025 # 2.5cm up
        goal_poses_robot_frame = [
            # [x, y, z, qx, qy, qz, qw]
            [-0.13820114040414633, -0.7945828086288749, 0.6142241720692808 + DZ, 0.0, -0.707, 0.0, 0.707],
            [-0.13820114040414633, -0.7945828086288749, 0.6142241720692808, 0.0, -0.707, 0.0, 0.707],
        ]
        # header: 
        #   seq: 1323
        #   stamp: 
        #     secs: 1776816607
        #     nsecs: 262202978
        #   frame_id: "robot_frame"
        # pose: 
        #   position: 
        #     x: -0.13820114040414633
        #     y: -0.7945828086288749
        #     z: 0.6142241720692808
        #   orientation: 
        #     x: 0.05030480113277667
        #     y: -0.7076525950454406
        #     z: 0.04521812018367412
        #     w: 0.7033154010225426

        # Hardcoded goal poses for beam_2x_0
        # DZ = 0.025 # 2.5cm up
        # goal_poses_robot_frame = [
        #     # [x, y, z, qx, qy, qz, qw]
        #     [-0.1405, -0.7997102958422697, 0.6829572463789897 + DZ, 0.5, -0.5, -0.5, 0.5],
        #     [-0.1405, -0.7997102958422697, 0.6829572463789897, 0.5, -0.5, -0.5, 0.5],
        # ]
        # header: 
        #   seq: 1932
        #   stamp: 
        #     secs: 1776817378
        #     nsecs: 909326076
        #   frame_id: "robot_frame"
        # pose: 
        #   position: 
        #     x: -0.140522378825981
        #     y: -0.7997102958422697
        #     z: 0.6829572463789897
        #   orientation: 
        #     x: 0.522595659676649
        #     y: -0.47141524203758334
        #     z: -0.48390688127442444
        #     w: 0.5200918921854434
        # ---

        # Hardcoded goal poses for fmb_peg_46
        # DZ = 0.05 # 5cm up
        # DZ = 0.1 # 10cm up
        # goal_poses_robot_frame = [
        #     # [x, y, z, qx, qy, qz, qw]
        #     [0.1390772720043013, -0.6913498106166442, 0.6082892190227137 + DZ, 0.5, 0.5, -0.5, 0.5],
        #     # [0.1390772720043013, -0.6913498106166442, 0.6082892190227137, 0.5, 0.5, -0.5, 0.5],
        # ]
        # header: 
        #   seq: 948
        #   stamp: 
        #     secs: 1778119976
        #     nsecs: 966065168
        #   frame_id: "robot_frame"
        # pose: 
        #   position: 
        #     x: 0.1390772720043013
        #     y: -0.6913498106166442
        #     z: 0.6082892190227137
        #   orientation: 
        #     x: 0.45794604135415756
        #     y: 0.5304030214896714
        #     z: -0.4800834584256212
        #     w: 0.5277100822883507





    # Adjustments
    # goal_poses_robot_frame = [
    #     [x - 0.05, y, z, qx, qy, qz, qw]
    #     for x, y, z, qx, qy, qz, qw in goal_poses_robot_frame
    # ]

    try:
        # Create and run the GoalPoseNode
        node = GoalPoseNode(
            goal_poses_robot_frame=np.array(goal_poses_robot_frame),
            object_scales=np.array([0.141, 0.03025, 0.0271]) * 25,  # fixed size
            success_threshold=args.success_threshold,
            success_steps=args.success_steps,
            force_open_loop=args.force_open_loop,
            force_fixed_orientation=args.force_fixed_orientation,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
