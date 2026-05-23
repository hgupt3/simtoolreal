"""Unified registry for peg-in-hole-dynamic problem instances.

A `Problem` is the triple `(insertion_object, receptive_object, insert_poses)`
that fully specifies a `PegInHoleDynamicEnv` configuration. Each sub-package
(`peg`, `fabrica`, `fmb`) registers its problems into `PROBLEM_REGISTRY` on
import; pulling in `peg_in_hole_dynamic` triggers all three.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Tuple


Pose7 = Tuple[float, float, float, float, float, float, float]


def make_pre_insert_sequence(
    final_pose: Pose7,
    insertion_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    pre_insert_offset: float = 0.05,
) -> Tuple[Pose7, Pose7]:
    """Return the standard insertion sequence: pre-insert pose, then final pose."""
    x, y, z, qx, qy, qz, qw = (float(v) for v in final_pose)
    dx, dy, dz = (float(v) for v in insertion_direction)
    norm = sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 0.0:
        raise ValueError("insertion_direction must be non-zero")
    scale = float(pre_insert_offset) / norm
    pre_insert = (
        x - dx * scale,
        y - dy * scale,
        z - dz * scale,
        qx,
        qy,
        qz,
        qw,
    )
    final = (x, y, z, qx, qy, qz, qw)
    return pre_insert, final


def make_transport_pre_insert_sequence(
    final_pose: Pose7,
    insertion_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    pre_insert_offset: float = 0.025,
    transport_offset: float = 0.10,
) -> Tuple[Pose7, Pose7, Pose7]:
    """Hole-frame trajectory tail: transport_above, pre_insert, final."""
    pre_insert, final = make_pre_insert_sequence(
        final_pose=final_pose,
        insertion_direction=insertion_direction,
        pre_insert_offset=pre_insert_offset,
    )
    px, py, pz, qx, qy, qz, qw = pre_insert
    dx, dy, dz = (float(v) for v in insertion_direction)
    norm = sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 0.0:
        raise ValueError("insertion_direction must be non-zero")
    scale = float(transport_offset) / norm
    transport_above = (
        px - dx * scale,
        py - dy * scale,
        pz - dz * scale,
        qx,
        qy,
        qz,
        qw,
    )
    return transport_above, pre_insert, final


@dataclass(frozen=True)
class Problem:
    """A single insertion problem.

    ``insert_pose_rel_receptive`` is the complete ordered insertion-subgoal
    sequence in the receptive frame. Plain insertion tasks use two poses:
    pre-insert, then assembled pose. Screw-style tasks can provide denser
    waypoints.
    """

    name: str                                          # registry key, e.g. "peg.tol0p5mm"
    insertion_object_name: str                         # key into NAME_TO_OBJECT
    receptive_urdf: str                                # path relative to assets/urdf/
    insert_pose_rel_receptive: Tuple[Pose7, ...]
    # World-frame Z offset for placing the receptive URDF root above the table
    # top. Typically half the receiver's canonical Z extent (so the receiver's
    # bottom face sits on the table top). The receptive URDF root is the
    # receiver's canonical-mesh origin (centered at its centroid by canonical-
    # mesh convention), so loading at z = table_top_z + hole_z_offset puts the
    # bottom face flush with the table.
    hole_z_offset: float = 0.0
    insertion_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    pre_insert_offset: float = 0.05
    # If set, env constructs a per-episode prelude of [lift_in_place, over_hole]
    # at this Z lift above the start pose (with pre-insert orientation), prepended
    # to the hole-frame insertion sequence. Only honored when goal_mode allows it.
    prelude_lift_offset: float = 0.0

    def __post_init__(self) -> None:
        poses = self._normalize_pose_sequence(self.insert_pose_rel_receptive)
        object.__setattr__(self, "insert_pose_rel_receptive", poses)

    @staticmethod
    def _normalize_pose_sequence(raw) -> Tuple[Pose7, ...]:
        if len(raw) == 0:
            raise ValueError("insert_pose_rel_receptive must contain at least one pose")

        poses = []
        for pose in raw:
            if len(pose) != 7:
                raise ValueError(
                    "each insert_pose_rel_receptive entry must be length 7; "
                    f"got {len(pose)} in problem pose {pose!r}"
                )
            poses.append(tuple(float(v) for v in pose))
        return tuple(poses)

    @property
    def final_insert_pose_rel_receptive(self) -> Pose7:
        return self.insert_pose_rel_receptive[-1]


PROBLEM_REGISTRY: Dict[str, Problem] = {}


# Trigger registration of per-domain problems by importing the subpackages —
# each one's __init__.py imports its `problems` module for the side effect of
# populating PROBLEM_REGISTRY.
from . import peg               # noqa: E402, F401
from . import fabrica           # noqa: E402, F401
from . import fmb               # noqa: E402, F401
from . import furniture_bench   # noqa: E402, F401
