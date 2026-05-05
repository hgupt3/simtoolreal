"""Unified registry for peg-in-hole-dynamic problem instances.

A `Problem` is the triple `(insertion_object, receptive_object, insert_pose)`
that fully specifies a `PegInHoleDynamicEnv` configuration. Each sub-package
(`peg`, `fabrica`, `fmb`) registers its problems into `PROBLEM_REGISTRY` on
import; pulling in `peg_in_hole_dynamic` triggers all three.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Problem:
    """A single (insertion_object, receptive_object, insert_pose) triple."""

    name: str                                          # registry key, e.g. "peg.tol0p5mm"
    insertion_object_name: str                         # key into NAME_TO_OBJECT
    receptive_urdf: str                                # path relative to assets/urdf/
    insert_pose_rel_receptive: Tuple[float, float, float, float, float, float, float]
    # World-frame Z offset for placing the receptive URDF root above the table
    # top. Typically half the receiver's canonical Z extent (so the receiver's
    # bottom face sits on the table top). The receptive URDF root is the
    # receiver's canonical-mesh origin (centered at its centroid by canonical-
    # mesh convention), so loading at z = table_top_z + hole_z_offset puts the
    # bottom face flush with the table.
    hole_z_offset: float = 0.0
    insertion_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    pre_insert_offset: float = 0.05


PROBLEM_REGISTRY: Dict[str, Problem] = {}


# Trigger registration of per-domain problems by importing the subpackages —
# each one's __init__.py imports its `problems` module for the side effect of
# populating PROBLEM_REGISTRY.
from . import peg      # noqa: E402, F401
from . import fabrica  # noqa: E402, F401
from . import fmb      # noqa: E402, F401
