#!/usr/bin/env python
"""Closed-loop goal pose node: VLA replans brush goals from live tool/material poses.

Publishes ``/robot_frame/goal_object_pose`` like ``goal_pose_node.py``, but goals
come from the ``closed_loop`` brush VLA instead of a fixed trajectory JSON.

Install closed_loop on the ROS python env::

    pip install -e /path/to/closed_loop

Run::

    python deployment/goal_pose_node_closed_loop.py \\
        --fixed-destination -0.365 -0.056 0.517 \\
        --control-frame blue_brush
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rospy
import tyro
from geometry_msgs.msg import Pose, PoseStamped
from termcolor import colored

try:
    import sys
    sys.path.append("/home/tylerlum/github_repos/closed_loop")
    from closed_loop import (
        ClosedLoopBrushPolicy,
        default_instruction_for_control_frame,
        load_closed_loop_policy,
    )
except ImportError as exc:
    raise SystemExit(
        "closed_loop package not found. Install with:\n"
        "  pip install -e /path/to/closed_loop\n"
        "from the SimToolReal machine (sibling of simtoolreal/)."
    ) from exc

from isaacgymenvs.utils.observation_action_utils_sharpa import _compute_keypoint_positions


def info(message: str):
    print(colored(message, "green"))


def warn(message: str):
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float, key=None):
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}
    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0)
    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


def pose_to_xyzw(msg: Pose) -> np.ndarray:
    return np.array(
        [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ],
        dtype=np.float64,
    )


def xyzw_to_pose(xyzw: np.ndarray) -> Pose:
    p = Pose()
    p.position.x = float(xyzw[0])
    p.position.y = float(xyzw[1])
    p.position.z = float(xyzw[2])
    p.orientation.x = float(xyzw[3])
    p.orientation.y = float(xyzw[4])
    p.orientation.z = float(xyzw[5])
    p.orientation.w = float(xyzw[6])
    return p


def keypoint_distance(
    pose1_xyzw: np.ndarray, pose2_xyzw: np.ndarray, object_scales: np.ndarray
) -> float:
    object_keypoint_positions = _compute_keypoint_positions(
        pose=pose1_xyzw[None], scales=object_scales[None]
    )
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=pose2_xyzw[None], scales=object_scales[None]
    )
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    keypoint_distances_l2 = np.linalg.norm(keypoints_rel_goal, axis=-1).max(axis=-1)
    return float(keypoint_distances_l2[0])


class GoalPoseNodeClosedLoop:
    def __init__(
        self,
        policy: ClosedLoopBrushPolicy,
        *,
        object_scales: np.ndarray,
        success_threshold: float,
        success_steps: int,
    ):
        rospy.init_node("goal_pose_node_closed_loop")

        KEYPOINT_SCALE = 1.5
        self.policy = policy
        self.object_scales = object_scales
        self.success_threshold = success_threshold
        self.keypoint_success_threshold = success_threshold * KEYPOINT_SCALE
        self.success_steps = success_steps
        self.current_success_steps = 0

        self.latest_tool_pose: Optional[Pose] = None
        self.latest_material_pose: Optional[Pose] = None
        self.goal_poses_robot: np.ndarray = np.zeros((0, 7), dtype=np.float64)
        self.current_goal_index = 0
        self.initialized = False

        self.goal_object_pose_pub = rospy.Publisher(
            "/robot_frame/goal_object_pose", Pose, queue_size=1
        )
        rospy.Subscriber(
            policy_tool_topic,
            PoseStamped,
            self._tool_pose_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            policy_material_topic,
            PoseStamped,
            self._material_pose_callback,
            queue_size=1,
        )

        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

    def _tool_pose_callback(self, msg: PoseStamped):
        self.latest_tool_pose = msg.pose

    def _material_pose_callback(self, msg: PoseStamped):
        self.latest_material_pose = msg.pose

    def _wait_for_observations(self):
        while not rospy.is_shutdown():
            if self.latest_tool_pose is None or self.latest_material_pose is None:
                warn_every(
                    "Waiting for tool and material poses",
                    n_seconds=1.0,
                )
                time.sleep(0.1)
            else:
                info("Tool and material poses received")
                break

    def _initialize_policy(self):
        tool = pose_to_xyzw(self.latest_tool_pose)
        mat = pose_to_xyzw(self.latest_material_pose)[:3]
        self.policy.reset(tool[:3], tool[3:7], mat)
        self._load_chunk_from_policy()
        self.initialized = True
        info(f"Policy initialized; chunk has {len(self.goal_poses_robot)} goals")

    def _load_chunk_from_policy(self):
        chunk = self.policy.plan_chunk()
        self.goal_poses_robot = np.array(
            [np.concatenate([xyz, quat]) for xyz, quat in chunk],
            dtype=np.float64,
        )
        self.current_goal_index = 0
        self.current_success_steps = 0

    def _maybe_replan(self):
        if self.policy.done:
            return
        tool = pose_to_xyzw(self.latest_tool_pose)
        mat = pose_to_xyzw(self.latest_material_pose)[:3]
        self.policy.observe(tool[:3], tool[3:7], mat)
        if not self.policy.done:
            self._load_chunk_from_policy()
            info("Replanned new goal chunk from VLA")

    def update_goal_object_pose(self):
        if not self.initialized or self.goal_poses_robot.shape[0] == 0:
            return
        if self.policy.done:
            return

        if self.current_goal_index >= self.goal_poses_robot.shape[0]:
            self._maybe_replan()
            if self.goal_poses_robot.shape[0] == 0:
                return
            if self.current_goal_index >= self.goal_poses_robot.shape[0]:
                self.current_goal_index = self.goal_poses_robot.shape[0] - 1

        tool_pose = pose_to_xyzw(deepcopy(self.latest_tool_pose))
        goal_pose = self.goal_poses_robot[self.current_goal_index]
        distance = keypoint_distance(tool_pose, goal_pose, self.object_scales)
        n = self.goal_poses_robot.shape[0]
        print(
            f"Distance: {distance:.4f}, goal {self.current_goal_index}/{n - 1}, "
            f"policy_done={self.policy.done}"
        )

        if distance < self.keypoint_success_threshold:
            self.current_success_steps += 1
            if self.current_success_steps >= self.success_steps:
                self.current_success_steps = 0
                self.current_goal_index += 1
                if self.current_goal_index >= self.goal_poses_robot.shape[0]:
                    info("Chunk complete; replanning")
                    self._maybe_replan()
                else:
                    info(f"Advanced to goal index {self.current_goal_index}")
            else:
                info(
                    f"Success threshold reached, at {self.current_success_steps}/{self.success_steps}"
                )

    def publish_goal_object_pose(self):
        if self.goal_poses_robot.shape[0] == 0:
            return
        idx = min(self.current_goal_index, self.goal_poses_robot.shape[0] - 1)
        # self.goal_object_pose_pub.publish(xyzw_to_pose(self.goal_poses_robot[idx]))
        pose = xyzw_to_pose(self.goal_poses_robot[idx])
        # pose.position.y -= 0.8
        self.goal_object_pose_pub.publish(pose)

    def run(self):
        self._wait_for_observations()
        self._initialize_policy()

        while not rospy.is_shutdown():
            if self.latest_tool_pose is not None and self.latest_material_pose is not None:
                self.update_goal_object_pose()
                self.publish_goal_object_pose()
            self.rate.sleep()


# Module-level topic names set from CLI before node construction.
policy_tool_topic = "/robot_frame/current_object_pose"
policy_material_topic = "/robot_frame/current_object_pose_2"


@dataclass
class GoalPoseNodeClosedLoopArgs:
    tool_topic: str = "/robot_frame/current_object_pose"
    material_topic: str = "/robot_frame/current_object_pose_2"
    fixed_destination: tuple[float, float, float] = (-0.365, -0.056, 0.517)
    model: Optional[str] = None
    """closed_loop model_registry key. When unset, uses the package default model."""
    control_frame: str = "blue_brush"
    instruction: Optional[str] = None
    """Override prompt. When unset, the basic per-task default for ``control_frame`` is used."""
    y_shift: float = 0.8
    chunk_size: int = 5
    device: str = "cuda"
    success_threshold: float = 0.02
    success_steps: int = 1
    tool_pose_is_root: bool = True


def main():
    global policy_tool_topic, policy_material_topic
    args: GoalPoseNodeClosedLoopArgs = tyro.cli(GoalPoseNodeClosedLoopArgs)
    policy_tool_topic = args.tool_topic
    policy_material_topic = args.material_topic

    instruction = (
        args.instruction
        if args.instruction
        else default_instruction_for_control_frame(args.control_frame)
    )
    info(
        f"model={args.model or '<default>'!r} "
        f"control_frame={args.control_frame!r} instruction={instruction!r}"
    )

    policy = load_closed_loop_policy(
        args.model,
        device=args.device,
        control_frame=args.control_frame,
        instruction=instruction,
        frame_shift=(0.0, args.y_shift, 0.0),
        chunk_size=args.chunk_size,
        tool_pose_is_root=args.tool_pose_is_root,
    )
    policy.set_destination(np.asarray(args.fixed_destination, dtype=np.float64))

    # Brush-like object scales (same order of magnitude as goal_pose_node defaults)
    object_scales = np.array([0.141, 0.03025, 0.0271]) * 25

    try:
        node = GoalPoseNodeClosedLoop(
            policy,
            object_scales=object_scales,
            success_threshold=args.success_threshold,
            success_steps=args.success_steps,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
